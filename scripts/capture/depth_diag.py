"""Measure why the DA3 cloud is inflated, against the ARCore depth we already have.

The bundle carries a real 16-bit millimetre depth map per keyframe. It is
noisy and sparse, but it is metric and it is measured rather than predicted —
which makes it the right instrument for asking whether DA3's error is a single
global scale (fixable with one number), a per-frame scale (fixable with a
solve), or shape error (not fixable by scaling at all).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from recon.depth.da3 import DA3Config, DA3Depth
from recon.io.bundle import BundleDir

BUNDLE = Path(sys.argv[1])
bundle = BundleDir.load(BUNDLE)
kfs = bundle.bundle.keyframes

est = DA3Depth(DA3Config(variant="small"))
pred = est.estimate(bundle)
print(f"model {pred.model_name}\n", flush=True)

rows = []
for kf in kfs:
    if not kf.depth:
        continue
    d_pred = pred.depth.get(kf.id)
    if d_pred is None:
        continue
    # ARCore depth: 16-bit millimetres, 0 == no return.
    arr = np.asarray(Image.open(BUNDLE / kf.depth)).astype(np.float64) / 1000.0
    ph, pw = d_pred.shape
    ys = (np.arange(ph) * arr.shape[0] / ph).astype(np.int32)
    xs = (np.arange(pw) * arr.shape[1] / pw).astype(np.int32)
    d_ref = arr[ys][:, xs]

    ok = (d_ref > 0.15) & (d_ref < 12.0) & (d_pred > 0.05)
    if ok.sum() < 200:
        continue
    a, b = d_pred[ok], d_ref[ok]
    ratio = b / a
    rows.append({
        "id": kf.id,
        "n": int(ok.sum()),
        "scale": float(np.median(ratio)),
        "spread": float(np.percentile(ratio, 84) - np.percentile(ratio, 16)),
        "pred_med": float(np.median(a)),
        "ref_med": float(np.median(b)),
    })

if not rows:
    print("no frames had comparable depth")
    raise SystemExit(1)

scales = np.array([r["scale"] for r in rows])
spreads = np.array([r["spread"] for r in rows])
print(f"{len(rows)} frames compared against ARCore depth\n")
print(f"per-frame scale (ARCore / DA3):")
print(f"  median {np.median(scales):.3f}   min {scales.min():.3f}   max {scales.max():.3f}")
print(f"  spread across frames: {scales.max() / scales.min():.2f}x")
print(f"  within-frame ratio spread (68% band), median: {np.median(spreads):.3f}")
print()

# What each fix would buy. A single global scale can only remove the common
# factor; the residual after that is what a per-frame solve would remove.
resid = scales / np.median(scales)
print("after one global scale, frames still disagree by:")
print(f"  {np.percentile(np.abs(resid - 1) * 100, 50):.1f}% median, "
      f"{np.percentile(np.abs(resid - 1) * 100, 90):.1f}% at p90, "
      f"{(np.abs(resid - 1) * 100).max():.1f}% worst")
print()
print(f"{'frame':10} {'n':>7} {'DA3 med':>8} {'ARCore':>8} {'scale':>7} {'spread':>7}")
for r in rows:
    print(f"{r['id']:10} {r['n']:7d} {r['pred_med']:8.2f} {r['ref_med']:8.2f} "
          f"{r['scale']:7.3f} {r['spread']:7.3f}")
