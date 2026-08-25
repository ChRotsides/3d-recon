"""Measure a DA3 variant against the ARCore depth the capture already carries.

The bundle stores a real 16-bit millimetre depth map per keyframe. It is noisy
and sparse, but it is metric and *measured* rather than predicted — which makes
it the right instrument for asking whether DA3's error is one global scale
(fixable with a number), a per-frame scale (fixable with a solve), or shape
(not fixable by scaling at all).

It also asks whether the per-frame affine anchor in `recon.depth.anchor` earns
its place, by scoring the same prediction before and after it, in both the
spaces the correction could live in.

    python scripts/capture/depth_diag.py data/captures/bundle_<uuid> [variant]

The model runs **once**, with the reference anchor off; the corrections are
fitted offline from that one prediction. Anything else would be comparing two
different predictions and attributing the difference to the anchor.
"""
import sys
from pathlib import Path

import numpy as np
from recon.depth.anchor import REF_RANGE_M, fit_affine_depth
from recon.depth.da3 import DA3Config, DA3Depth
from recon.io.bundle import BundleDir

BUNDLE = Path(sys.argv[1])
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "small"
#: Held-out check: fit on one set of 8x8 tiles, score on the other. Tiles rather
#: than a pixel checkerboard because neighbouring pixels carry the same shape
#: error, and a split that keeps them together would show a generalisation gap
#: where there is none.
TILE = 8

bundle = BundleDir.load(BUNDLE)
kfs = bundle.bundle.keyframes

est = DA3Depth(DA3Config(variant=VARIANT, anchor_to_reference_depth=False))
pred = est.estimate(bundle)
print(f"\nmodel {pred.model_name} · {pred.runtime_s:.1f}s\n", flush=True)


def stats(ref: np.ndarray, d: np.ndarray) -> dict | None:
    """How one corrected frame compares with what the sensor measured."""
    lo, hi = REF_RANGE_M
    ok = (ref > lo) & (ref < hi) & (d > 0.05)
    if ok.sum() < 200:
        return None
    a, b = d[ok], ref[ok]
    ratio = b / a
    return {
        "n": int(ok.sum()),
        # Median ratio: what one more multiplier would still have to fix.
        "scale": float(np.median(ratio)),
        # 68% band of that ratio *within* one frame — the shape disagreement,
        # which no per-frame multiplier can remove.
        "spread": float(np.percentile(ratio, 84) - np.percentile(ratio, 16)),
        # Plain metres, because a ratio hides whether an error matters.
        "abs_m": float(np.median(np.abs(b - a))),
    }


rows = []
for kf in kfs:
    d_pred = pred.depth.get(kf.id)
    ref = bundle.depth(kf)
    if d_pred is None or ref is None:
        continue
    before = stats(ref, d_pred)
    if before is None:
        continue

    row = {"id": kf.id, "before": before}
    for space in ("depth", "disparity"):
        fit = fit_affine_depth(d_pred, ref, space=space)
        row[space] = stats(ref, fit.apply(d_pred)) if fit else None
        row[f"{space}_fit"] = fit

    # Held out: fit on alternating tiles, score on every pixel. A two-parameter
    # line cannot memorise 300k pixels, but saying so is weaker than showing it.
    yy, xx = np.mgrid[0 : ref.shape[0], 0 : ref.shape[1]]
    tiles = (((yy // TILE) + (xx // TILE)) % 2).astype(bool)
    held = fit_affine_depth(d_pred, ref, space="depth", fit_mask=tiles)
    row["heldout"] = stats(ref, held.apply(d_pred)) if held else None
    rows.append(row)

if not rows:
    print("no frames had comparable depth")
    raise SystemExit(1)


def summarise(key: str) -> str:
    got = [r[key] for r in rows if r.get(key)]
    if not got:
        return f"{key:12} —"
    scales = np.array([g["scale"] for g in got])
    spreads = np.array([g["spread"] for g in got])
    abs_m = np.array([g["abs_m"] for g in got])
    resid = scales / np.median(scales)
    return (
        f"{key:12} {len(got):3d}  {np.median(scales):6.3f}  "
        f"{scales.max() / scales.min():6.2f}x  {np.median(spreads):7.3f}  "
        f"{np.median(abs_m):7.3f}  {np.percentile(np.abs(resid - 1) * 100, 50):5.1f}%  "
        f"{(np.abs(resid - 1) * 100).max():5.1f}%"
    )


print(f"{len(rows)} frames compared against ARCore depth")
print(f"{'':12} {'n':>3}  {'scale':>6}  {'frame':>7}  {'within':>7}  "
      f"{'abs m':>7}  {'resid':>6} {'worst':>6}")
print(f"{'':12} {'':>3}  {'median':>6}  {'spread':>7}  {'spread':>7}  "
      f"{'median':>7}  {'med':>6} {'':>6}")
print("-" * 72)
print(summarise("before"))
print(summarise("depth"))
print(summarise("disparity"))
print(summarise("heldout"))
print()

fits = [r["depth_fit"] for r in rows if r.get("depth_fit")]
if fits:
    print(f"affine fit (depth space): {len(fits)}/{len(rows)} frames fitted")
    print(f"  a  median {np.median([f.a for f in fits]):.3f}  "
          f"min {min(f.a for f in fits):.3f}  max {max(f.a for f in fits):.3f}")
    print(f"  b  median {np.median([f.b for f in fits]):+.3f} m  "
          f"min {min(f.b for f in fits):+.3f}  max {max(f.b for f in fits):+.3f}")
    print(f"  inliers median {np.median([f.inlier_fraction for f in fits]):.2f}  ·  "
          f"residual median {np.median([f.residual_m for f in fits]):.3f} m")
print()

print(f"{'frame':10} {'scale':>7} {'->':>7} {'spread':>7} {'->':>7} "
      f"{'abs m':>7} {'->':>7} {'a':>6} {'b':>7}")
for r in rows:
    after = r.get("depth") or {}
    fit = r.get("depth_fit")
    print(
        f"{r['id']:10} {r['before']['scale']:7.3f} {after.get('scale', float('nan')):7.3f} "
        f"{r['before']['spread']:7.3f} {after.get('spread', float('nan')):7.3f} "
        f"{r['before']['abs_m']:7.3f} {after.get('abs_m', float('nan')):7.3f} "
        f"{(fit.a if fit else float('nan')):6.3f} {(fit.b if fit else float('nan')):+7.3f}"
    )
