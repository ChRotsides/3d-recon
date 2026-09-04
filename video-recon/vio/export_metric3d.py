"""Metric3D v2 ViT-Small -> ONNX. Input: normalised RGB (1,3,H,W). Output: canonical depth (1,H,W);
metric depth = output * fx_at_input / 1000 (Metric3D's canonical camera has fx = 1000 px)."""
import sys, time, torch, numpy as np
# the hub inference path calls .cuda(); export on CPU
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
torch.cuda.is_available = lambda: False
H, W = int(sys.argv[1]), int(sys.argv[2]); out = sys.argv[3]
torch.hub.set_dir("/home/charalambos/ai/phone/hub")
import os; os.chdir("/home/charalambos/ai/phone/hub/yvanyin_metric3d_main"); sys.path.insert(0, os.getcwd())   # the hub code resolves backbones relative to its own dir
model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True, trust_repo=True).eval()
class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        pred_depth, confidence, output_dict = self.m.inference({"input": x})
        return pred_depth[:, 0]
w = Wrap(model); x = torch.randn(1, 3, H, W)
with torch.no_grad(): ref = w(x); print("torch out", tuple(ref.shape), float(ref.min()), float(ref.max()))
torch.onnx.export(w, (x,), out, input_names=["image"], output_names=["depth"], opset_version=17, do_constant_folding=True, dynamo=False)
import onnx, onnxruntime as ort
onnx.checker.check_model(onnx.load(out)); s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
t = time.time(); o = s.run(None, {"image": x.numpy()}); dt = time.time() - t; t = time.time(); o = s.run(None, {"image": x.numpy()}); dt2 = time.time() - t
print(f"ort {dt2*1000:.0f} ms, out {o[0].shape}, max|diff| {np.abs(o[0]-ref.numpy()).max():.4f}")
