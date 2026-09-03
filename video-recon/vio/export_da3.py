"""Export DA3-Small's core network to ONNX: mono (1 view) and posed window (N views with camera tokens).
Inputs are ImageNet-normalised NCHW floats at a multiple of 14; outputs depth (B,N,H,W) [+ conf]."""
import sys, time, argparse
sys.path.insert(0, "/home/charalambos/Documents/Projects/3d-recon/harness")
import numpy as np, torch
# rope.py builds its position grid with cartesian_prod, which the ONNX exporter cannot translate; this is equivalent
def _cartesian_prod2(a, b):
    ya, xb = torch.meshgrid(a, b, indexing="ij"); return torch.stack([ya.reshape(-1), xb.reshape(-1)], dim=-1)
torch.cartesian_prod = _cartesian_prod2
torch.Tensor.mT = property(lambda self: self.transpose(-2, -1))   # aten::mT has no ONNX symbolic; transpose does
from recon.depth.da3 import DA3Config, DA3Depth
ap = argparse.ArgumentParser(); ap.add_argument("--views", type=int, default=1); ap.add_argument("--h", type=int, default=182); ap.add_argument("--w", type=int, default=322); ap.add_argument("--out", default="/home/charalambos/ai/phone/model/da3_small.onnx")
a = ap.parse_args()
ad = DA3Depth(DA3Config(variant="small", enforce_commercial_licence=False, anchor_to_sparse_points=False, device="cpu"))
api = ad._load().eval(); core = api.model.eval()
class Wrap(torch.nn.Module):
    def __init__(self, core, posed): super().__init__(); self.core = core; self.posed = posed
    def forward(self, x, ex=None, K=None):
        out = self.core(x, ex, K, [], False, False, "first") if self.posed else self.core(x, None, None, [], False, False, "first")
        d = out["depth"] if isinstance(out, dict) else out.depth
        c = out["depth_conf"] if isinstance(out, dict) else getattr(out, "depth_conf", None)
        return (d, c) if c is not None else d
N, H, W = a.views, a.h, a.w
x = torch.randn(1, N, 3, H, W)
posed = N > 1
ex = torch.eye(4).repeat(1, N, 1, 1); ex[0, :, 0, 3] = torch.linspace(0, 0.3, N)
K = torch.tensor([[214.4, 0, 161.0], [0, 214.4, 90.5], [0, 0, 1.0]]).repeat(1, N, 1, 1)
m = Wrap(core, posed)
with torch.no_grad():
    t0 = time.time(); ref = m(x, ex, K) if posed else m(x); print(f"torch cpu {time.time()-t0:.2f}s; out {[tuple(r.shape) for r in (ref if isinstance(ref, tuple) else (ref,))]}")
inputs = (x, ex, K) if posed else (x,)
names = ["image", "extrinsics", "intrinsics"][: len(inputs)]
torch.onnx.export(m, inputs, a.out, input_names=names, output_names=["depth", "conf"], opset_version=17, do_constant_folding=True, dynamo=False)
import onnx, onnxruntime as ort
mdl = onnx.load(a.out); onnx.checker.check_model(mdl); print("onnx ok,", sum(1 for _ in mdl.graph.node), "nodes,", round(sum(len(t.raw_data) for t in mdl.graph.initializer) / 1e6, 1), "MB weights")
s = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
feed = {n: t.numpy() for n, t in zip(names, inputs)}
t0 = time.time(); o = s.run(None, feed); dt = time.time() - t0
t0 = time.time(); o = s.run(None, feed); dt2 = time.time() - t0
r0 = (ref[0] if isinstance(ref, tuple) else ref).numpy()
print(f"ort cpu {dt:.2f}s first, {dt2:.2f}s second; depth shape {o[0].shape}; max|diff| vs torch {np.abs(o[0]-r0).max():.4f} (depth range {r0.min():.2f}..{r0.max():.2f})")
