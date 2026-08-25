#!/usr/bin/env python3
"""Write a standalone keyframe subset of a capture bundle.

The source is never modified. Pixel files are copied intact, while bundle.json
is reduced to the requested half-open keyframe range and sparse observations
that still refer to selected frames.

    PYTHONPATH=harness uv run python scripts/capture/subset_bundle.py \
        SOURCE OUTPUT START END
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from recon.io.bundle import BundleDir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("start", type=int, help="first keyframe index, inclusive")
    parser.add_argument("end", type=int, help="last keyframe index, exclusive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = BundleDir.load(args.source)
    total = len(source.bundle.keyframes)
    if not 0 <= args.start < args.end <= total:
        raise SystemExit(f"range must satisfy 0 <= start < end <= {total}")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace existing output: {args.output}")
    if source.root.resolve() in args.output.resolve().parents:
        raise ValueError("output must not be inside the source bundle")

    selected = source.bundle.keyframes[args.start : args.end]
    selected_ids = {keyframe.id for keyframe in selected}
    sparse = []
    for point in source.bundle.sparse_points:
        observations = [frame_id for frame_id in point.observations if frame_id in selected_ids]
        if observations:
            sparse.append(point.model_copy(update={"observations": observations}))

    subset = source.bundle.model_copy(deep=True)
    subset.session_id = f"{source.bundle.session_id}-frames-{args.start:04d}-{args.end - 1:04d}"
    subset.keyframes = list(selected)
    subset.sparse_points = sparse
    subset.depth_frame_count = sum(
        bool(keyframe.depth and (source.root / keyframe.depth).exists()) for keyframe in selected
    )

    shutil.copytree(source.root, args.output)
    BundleDir(root=args.output, bundle=subset).save()
    print(
        f"wrote {args.output}: frames {args.start}-{args.end - 1} "
        f"({len(selected)} keyframes, {subset.depth_frame_count} depth maps)"
    )


if __name__ == "__main__":
    main()
