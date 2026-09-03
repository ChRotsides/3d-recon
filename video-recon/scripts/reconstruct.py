#!/usr/bin/env python3
"""Reconstruct a room from one phone recording (video + sensors).

    source ~/ai/env.sh kepler
    python video-recon/scripts/reconstruct.py data/recordings/rec_<stamp> out/video/<stamp>

Stages are cached in the output directory; pass --force to redo them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "harness"))

from videorecon.pipeline import Options, reconstruct  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("recording", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--orientation-sensor", choices=("gamerotv", "rotv"), default="gamerotv")
    p.add_argument(
        "--min-rotation",
        type=float,
        default=8.0,
        help="degrees turned before a new keyframe is due",
    )
    p.add_argument(
        "--max-interval",
        type=float,
        default=0.6,
        help="seconds before a keyframe is due regardless",
    )
    p.add_argument("--max-long-edge", type=int, default=1600)
    p.add_argument("--max-keyframes", type=int, default=450)
    p.add_argument(
        "--exhaustive-below",
        type=int,
        default=450,
        help="match all pairs up to this many keyframes",
    )
    p.add_argument(
        "--variant", default="base", help="DA3 checkpoint: small, base, large, metric-large"
    )
    p.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel size in metres")
    p.add_argument("--depth-trunc", type=float, default=6.0)
    p.add_argument("--no-depth", action="store_true", help="stop after poses and the bundle")
    p.add_argument("--no-fusion", action="store_true", help="predict depth but do not fuse")
    p.add_argument("--no-magnetic-heading", action="store_true")
    p.add_argument(
        "--camera-height", type=float, default=1.5, help="metres the phone was held above the floor"
    )
    p.add_argument("--scale-source", choices=("auto", "imu", "floor"), default="auto")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    opts = Options(
        orientation_sensor=a.orientation_sensor,
        min_rotation_deg=a.min_rotation,
        max_interval_s=a.max_interval,
        max_long_edge=a.max_long_edge,
        max_keyframes=a.max_keyframes,
        depth_variant=a.variant,
        run_depth=not a.no_depth,
        run_fusion=not a.no_fusion,
        voxel_m=a.voxel,
        depth_trunc_m=a.depth_trunc,
        force=a.force,
        use_magnetic_heading=not a.no_magnetic_heading,
    )
    reconstruct(a.recording, a.out, opts)


if __name__ == "__main__":
    main()
