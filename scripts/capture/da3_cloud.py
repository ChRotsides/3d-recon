"""Run DA3 on a bundle, back-project every frame, merge into one point cloud.

Merged exactly as the panorama was: each frame is placed by its own pose, and
nothing is discarded for disagreeing with its neighbours. The panorama showed
where that assumption strains — near objects ghost because the camera really
did translate — and a point cloud shows the same thing as thickness rather
than transparency, which is more honest and more useful.

Writes a flat binary the web viewer can upload straight into GPU buffers:
positions as float32 xyz, colours as uint8 rgb. No PLY parsing in JavaScript.
"""
import json
import sys
from pathlib import Path

import numpy as np

from recon.depth.da3 import DA3Config, DA3Depth
from recon.io.bundle import BundleDir

BUNDLE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
VARIANT = sys.argv[3] if len(sys.argv) > 3 else "small"
MAX_POINTS = int(sys.argv[4]) if len(sys.argv) > 4 else 900_000

bundle = BundleDir.load(BUNDLE)
kfs = bundle.bundle.keyframes
print(f"{len(kfs)} keyframes · DA3-{VARIANT}", flush=True)

est = DA3Depth(DA3Config(variant=VARIANT))
result = est.estimate(bundle)
print(f"depth done · model {result.model_name} · {result.runtime_s:.1f}s", flush=True)

# Per-frame budget, so one frame cannot dominate the cloud.
per_frame = max(1, MAX_POINTS // max(len(kfs), 1))
rng = np.random.default_rng(0)

positions, colours = [], []
for kf in kfs:
    d = result.depth.get(kf.id)
    if d is None:
        continue
    rgb = bundle.rgb(kf)
    ih, iw = d.shape
    if rgb.shape[:2] != (ih, iw):
        # Depth comes back at the model's working resolution; sample the image
        # at the same grid rather than resizing the depth, which would
        # interpolate across depth discontinuities and smear edges into space.
        ys = (np.arange(ih) * rgb.shape[0] / ih).astype(np.int32)
        xs = (np.arange(iw) * rgb.shape[1] / iw).astype(np.int32)
        rgb = rgb[ys][:, xs]

    intr = kf.intrinsics
    sx, sy = iw / intr.width, ih / intr.height
    fx, fy = intr.fx * sx, intr.fy * sy
    cx, cy = intr.cx * sx, intr.cy * sy

    v, u = np.nonzero(d > 0.05)
    if len(u) == 0:
        continue
    if len(u) > per_frame:
        pick = rng.choice(len(u), per_frame, replace=False)
        u, v = u[pick], v[pick]
    z = d[v, u].astype(np.float64)

    # Pixel -> camera ray. PLAN.md §5: +X right, +Y up, camera looks down -Z,
    # while image rows run downwards — hence the sign on Y.
    x = (u - cx) / fx * z
    y = -(v - cy) / fy * z
    pts_cam = np.stack([x, y, -z], axis=1)

    R = np.asarray(kf.pose.R, dtype=np.float64)
    t = np.asarray(kf.pose.t, dtype=np.float64)
    positions.append(pts_cam @ R.T + t)
    colours.append(rgb[v, u])

pos = np.concatenate(positions).astype(np.float32)
col = np.concatenate(colours).astype(np.uint8)
print(f"{len(pos):,} points from {len(positions)} frames", flush=True)

# Trim the far tail: monocular depth invents distance beyond the room, and a
# handful of runaway points would otherwise set the viewer's whole scale.
centre = np.median(pos, axis=0)
radius = np.linalg.norm(pos - centre, axis=1)
keep = radius <= np.percentile(radius, 99.0)
pos, col = pos[keep], col[keep]
print(f"{len(pos):,} points after trimming the far 1%", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "wb") as f:
    f.write(pos.tobytes())
    f.write(col.tobytes())

lo, hi = pos.min(axis=0), pos.max(axis=0)
meta = {
    "count": int(len(pos)),
    "min": lo.tolist(),
    "max": hi.tolist(),
    "centre": ((lo + hi) / 2).tolist(),
    "extent": (hi - lo).tolist(),
    "model": result.model_name,
    "keyframes": len(kfs),
}
OUT.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", flush=True)
print("extent m:", np.round(hi - lo, 2).tolist(), flush=True)
