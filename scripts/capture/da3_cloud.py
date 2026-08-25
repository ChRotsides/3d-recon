"""Run DA3, back-project every frame, and robustly fuse one point cloud.

Each frame is placed by its own pose. Depth-edge pixels are removed before
projection; after projection, frame-aware voxels give every source frame one
vote and discard cells that are both weakly supported and spatially isolated.
Coherent single-view surfaces remain, and no global distance percentile deletes
legitimate geometry merely because it lies beyond a doorway.

Writes a flat binary the web viewer can upload straight into GPU buffers:
positions as float32 xyz, colours as uint8 rgb. No PLY parsing in JavaScript.

    python scripts/capture/da3_cloud.py BUNDLE OUT [variant] [max_points]
        [anchor_space] [voxel_m] [min_frame_support] [min_neighbours]
        [max_spread_voxels] [--depth-cache DEPTH.npz] [--segments SEG.npz|none]

Uses out/segments.npz automatically when present; `--segments none` disables it.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from recon.cloud import choose_voxel_size, flying_pixel_mask, frame_aware_voxel_filter
from recon.depth.base import DepthResult
from recon.depth.da3 import DA3Config, DA3Depth
from recon.io.bundle import BundleDir
from recon.segments import align_segments_to_consensus, sample_labels_to_grid

args = sys.argv[1:]
#: SAM segments are a standard stage, not an extra. When no path is given the
#: builder picks up the default cache if it is there, so a rebuild keeps the
#: segmentation without having to remember the flag. `--segments none` opts out.
DEFAULT_SEGMENTS = Path("out/segments.npz")
SEGMENTS = DEFAULT_SEGMENTS if DEFAULT_SEGMENTS.exists() else None
if "--segments" in args:
    seg_at = args.index("--segments")
    if seg_at + 1 >= len(args):
        raise SystemExit("--segments requires a .npz path written by sam_masks.py")
    SEGMENTS = None if args[seg_at + 1] == "none" else Path(args[seg_at + 1])
    del args[seg_at : seg_at + 2]
DEPTH_CACHE = None
if "--depth-cache" in args:
    cache_at = args.index("--depth-cache")
    if cache_at + 1 >= len(args):
        raise SystemExit("--depth-cache requires a .npz path")
    DEPTH_CACHE = Path(args[cache_at + 1])
    del args[cache_at : cache_at + 2]
if len(args) < 2:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} BUNDLE OUT [variant] [max_points] ...")

BUNDLE = Path(args[0])
OUT = Path(args[1])
VARIANT = args[2] if len(args) > 2 else "small"
MAX_POINTS = int(args[3]) if len(args) > 3 else 900_000
#: Voxel size in metres. 0 means "pick the finest grid that fits MAX_POINTS",
#: which is usually what you want — the budget is what the browser downloads.
VOXEL_M = float(args[5]) if len(args) > 5 else 0.0
# Cleanup thresholds. A cell is kept when distinct frames corroborate it *or*
# when it has enough occupied neighbours to be part of a coherent surface.
MIN_FRAME_SUPPORT = int(args[6]) if len(args) > 6 else 2
MIN_NEIGHBOURS = int(args[7]) if len(args) > 7 else 3
MAX_SPREAD_VOXELS = float(args[8]) if len(args) > 8 else 0.75

bundle = BundleDir.load(BUNDLE)
kfs = bundle.bundle.keyframes
print(f"{len(kfs)} keyframes · DA3-{VARIANT}", flush=True)

SPACE = args[4] if len(args) > 4 else "depth"

if DEPTH_CACHE is None:
    est = DA3Depth(DA3Config(variant=VARIANT, reference_anchor_space=SPACE))
    result = est.estimate(bundle)
    print(f"depth done · model {result.model_name} · {result.runtime_s:.1f}s", flush=True)
else:
    cache = np.load(DEPTH_CACHE)
    result = DepthResult(
        depth={kf.id: np.asarray(cache[kf.id], dtype=np.float32) for kf in kfs if kf.id in cache},
        model_name=str(cache["model"]),
        diagnostics={},
    )
    print(f"depth cache · {DEPTH_CACHE} · model {result.model_name}", flush=True)

# Say how many frames were actually corrected against the measured depth. A
# frame the fit refuses keeps its sparse anchor, which is a quieter and much
# weaker correction — worth seeing rather than assuming.
# Windows overlap, so a frame can be fitted more than once — count keyframes,
# not fits, or the line reads "32/27 frames" and undermines the number it is
# there to report.
by_frame = {
    d["keyframe"]: d
    for d in result.diagnostics.get("reference_anchor", [])
    if d.get("fitted")
}
if by_frame:
    a = np.array([d["a"] for d in by_frame.values()])
    b = np.array([d["b"] for d in by_frame.values()])
    print(
        f"reference anchor ({SPACE}): {len(by_frame)}/{len(kfs)} frames · "
        f"a median {np.median(a):.3f} · b median {np.median(b):+.3f} m",
        flush=True,
    )

seg_cache = None
if SEGMENTS:
    seg_cache = np.load(SEGMENTS)
    print(f"segments · {SEGMENTS} · {str(seg_cache['model'])}", flush=True)
positions, colours, source_frames, source_segments = [], [], [], []
flying_total = 0
for frame_index, kf in enumerate(kfs):
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

    # Every pixel, and then the ones that are not surface removed by name.
    # The builder used to keep a uniform random 1-in-9 instead, which is the
    # single biggest cause of the cloud looking full of holes: a random sample
    # of a surface is gaps at every scale. Voxelising below spends the same
    # budget on covering the surface once.
    flying = flying_pixel_mask(d)
    flying_total += int(flying.sum())
    v, u = np.nonzero((d > 0.05) & ~flying)
    if len(u) == 0:
        continue
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
    source_frames.append(np.full(len(u), frame_index, dtype=np.int16))
    if seg_cache is not None:
        labels = sample_labels_to_grid(np.asarray(seg_cache[kf.id]), ih, iw)
        source_segments.append(labels[v, u].astype(np.int32))

pos = np.concatenate(positions).astype(np.float32)
col = np.concatenate(colours).astype(np.uint8)
frame_id = np.concatenate(source_frames)
segment_id = np.concatenate(source_segments) if source_segments else None
print(
    f"{len(pos):,} points from {len(positions)} frames "
    f"({flying_total:,} flying pixels dropped at depth edges)",
    flush=True,
)

# Bring each object onto the same object as the other frames saw. Frames
# disagree about where the room is by ~7.5 cm, and the error varies smoothly
# across a frame, so one rigid motion per frame only reaches 5.2 cm while one
# per segment reaches 2.4 — inside the voxel, for the first time. Runs before
# voxelising so the grid fuses surfaces that now coincide.
if segment_id is not None:
    pos, seg_stats = align_segments_to_consensus(pos, frame_id, segment_id)
    print(
        f"segments: {seg_stats.aligned_segments} aligned, "
        f"{seg_stats.unmatched_segments} without a partner, "
        f"{seg_stats.rejected_segments} rejected as too large · "
        f"median {seg_stats.median_translation_m * 100:.1f} cm / "
        f"{seg_stats.median_rotation_deg:.2f}° · "
        f"{seg_stats.carried_points:,} unlabelled points carried",
        flush=True,
    )

# One robust point per occupied cell, at the size that fits the budget. Each
# frame gets one vote in a voxel, then cells that are both single-view and
# isolated are discarded. The old unconditional far-1% trim is deliberately
# gone: on the measured capture that tail is a real doorway. Robust bounds are
# written below for viewer framing without deleting the underlying geometry.
voxel = VOXEL_M if VOXEL_M > 0 else choose_voxel_size(pos, MAX_POINTS)
pos, col, cleanup = frame_aware_voxel_filter(
    pos,
    col,
    frame_id,
    voxel,
    min_frame_support=MIN_FRAME_SUPPORT,
    min_neighbours=MIN_NEIGHBOURS,
    max_spread_m=MAX_SPREAD_VOXELS * voxel,
)
print(
    f"{len(pos):,} points after a {voxel * 100:.1f} cm frame-aware voxel grid · "
    f"{cleanup.dropped_isolated:,} isolated and {cleanup.dropped_spread:,} "
    "high-spread cells removed · "
    f"{cleanup.low_support_kept_by_neighbours:,} coherent low-support cells kept",
    flush=True,
)
if len(pos) == 0:
    raise RuntimeError("point-cloud cleanup removed every voxel")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "wb") as f:
    f.write(pos.tobytes())
    f.write(col.tobytes())

lo, hi = pos.min(axis=0), pos.max(axis=0)
# Viewer framing ignores coordinate-wise extreme percentiles, but those points
# remain in cloud.bin and the exact bounds remain in min/max below.
view_lo, view_hi = np.percentile(pos, [1.0, 99.0], axis=0)
meta = {
    "count": int(len(pos)),
    "min": lo.tolist(),
    "max": hi.tolist(),
    "centre": ((lo + hi) / 2).tolist(),
    "extent": (hi - lo).tolist(),
    "view_min": view_lo.tolist(),
    "view_max": view_hi.tolist(),
    "view_centre": ((view_lo + view_hi) / 2).tolist(),
    "view_extent": (view_hi - view_lo).tolist(),
    "model": result.model_name,
    "keyframes": len(kfs),
    # The viewer sizes its discs from this. A point smaller than the grid it
    # came from leaves a gap between every neighbour — holes the data does not
    # have, put back by the renderer.
    "voxel_m": float(voxel),
    "cleanup": asdict(cleanup),
    "segments": asdict(seg_stats) if segment_id is not None else None,
}
OUT.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", flush=True)
print("extent m:", np.round(hi - lo, 2).tolist(), flush=True)
