"""UniDepth v2 ViT-S -> ONNX at a fixed input size. Inputs: image (1,3,H,W) ImageNet-normalised, K (1,3,3) at that size.
Output: metric depth (1,H,W). The pad/resize the model's infer() does is baked in for the fixed shape."""
import sys, time, torch, numpy as np, torch.nn.functional as F
# opset 17 has no antialiased bilinear resize; plain bilinear differs slightly at the downscale step
_orig_interp = F.interpolate
def _interp_no_aa(*a, **k):
    k.pop("antialias", None)
    return _orig_interp(*a, **k)
F.interpolate = _interp_no_aa
torch.nn.functional.interpolate = _interp_no_aa
H, W = int(sys.argv[1]), int(sys.argv[2]); out = sys.argv[3]; level = int(sys.argv[4]) if len(sys.argv) > 4 else 0
from unidepth.models import UniDepthV2
from unidepth.models.unidepthv2.unidepthv2 import get_paddings, get_resize_factor
from unidepth.utils.camera import Pinhole, BatchCamera
model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14").eval(); model.resolution_level = level
sc = model.shape_constraints; print("shape constraints", sc)
pb = [sc["pixels_min"], sc["pixels_max"]]; rng = (pb[1] - pb[0]) / 10; pb = (level * rng + pb[0], (level + 1) * rng + pb[0])
paddings, (pH, pW) = get_paddings((H, W), sc["ratio_bounds"]); rf, (nH, nW) = get_resize_factor((pH, pW), pb)
pl, pr, pt, pbm = paddings; print(f"input {H}x{W} -> padded {pH}x{pW} -> model {nH}x{nW} (resize {rf:.3f})")
class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x, K):
        rgb = F.pad(x, (pl, pr, pt, pbm), value=0.0); rgb = F.interpolate(rgb, size=(nH, nW), mode="bilinear", align_corners=False)
        cam = BatchCamera.from_camera(Pinhole(K=K)); cam = cam.crop(left=-pl, top=-pt, right=-pr, bottom=-pbm); cam = cam.resize(rf)
        _, o = self.m.encode_decode(inputs={"image": rgb, "camera": cam}, image_metas=[])
        d = o["points"][:, -1:]; d = F.interpolate(d, size=(pH, pW), mode="bilinear", align_corners=False)
        return d[:, 0, pt:pt + H, pl:pl + W]
w = Wrap(model); x = torch.randn(1, 3, H, W); K = torch.tensor([[[214.4, 0, W / 2], [0, 214.4, H / 2], [0, 0, 1.0]]])
with torch.no_grad(): ref = w(x, K); print("torch out", tuple(ref.shape), float(ref.min()), float(ref.max()))
torch.onnx.export(w, (x, K), out, input_names=["image", "K"], output_names=["depth"], opset_version=17, do_constant_folding=True, dynamo=False)
import onnx, onnxruntime as ort
onnx.checker.check_model(onnx.load(out)); s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
feed = {"image": x.numpy(), "K": K.numpy()}; s.run(None, feed); t = time.time(); o = s.run(None, feed); dt = time.time() - t
print(f"ort {dt*1000:.0f} ms, out {o[0].shape}, max|diff| {np.abs(o[0]-ref.numpy()).max():.4f}")
