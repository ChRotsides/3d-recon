#!/usr/bin/env python3
"""Stitch a capture from RGB features and verified visual overlap only.

Pose, depth, heading, tracking state, and all other AR fields are ignored.  The
largest strictly connected overlap component is rendered; diagnostics explain
which frames could and could not be joined.

    PYTHONPATH=harness python scripts/capture/feature_stitch.py BUNDLE OUT.png \
        [--diagnostics overlap.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
from recon.image_stitch import stitch_largest_component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="minimum SIFT/RANSAC pair confidence before pixel-level verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wire = json.loads((args.bundle / "bundle.json").read_text())
    keyframes = wire["keyframes"]
    images = [cv2.imread(str(args.bundle / keyframe["image"])) for keyframe in keyframes]
    missing = [keyframes[index]["id"] for index, image in enumerate(images) if image is None]
    if missing:
        raise SystemExit(f"could not read images for: {', '.join(missing)}")

    result = stitch_largest_component(images, confidence_threshold=args.confidence)
    selected = result.graph.selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".png":
        output_image = cv2.cvtColor(result.image, cv2.COLOR_BGR2BGRA)
        output_image[:, :, 3] = result.alpha
    else:
        output_image = result.image
    if not cv2.imwrite(str(args.output), output_image):
        raise SystemExit(f"could not write {args.output}")

    diagnostics = {
        "bundle": str(args.bundle),
        "algorithm": (
            "spatially balanced CLAHE RootSIFT + symmetric ratio matching + MAGSAC "
            "homography + inlier-spread/reprojection/pixel verification + image-only "
            "focal self-calibration + robust rotation graph + exposure compensation "
            "+ single-view seam ownership"
        ),
        "uses_ar_metadata": False,
        "confidence_threshold": args.confidence,
        "input_frames": len(keyframes),
        "selected_frames": [keyframes[index]["id"] for index in selected],
        "components": [
            [keyframes[index]["id"] for index in component]
            for component in result.graph.components
        ],
        "keypoints": {
            keyframes[index]["id"]: count
            for index, count in enumerate(result.graph.keypoint_counts)
        },
        "accepted_edges": [
            {
                **asdict(edge),
                "first_id": keyframes[edge.first]["id"],
                "second_id": keyframes[edge.second]["id"],
                "inlier_ratio": edge.inlier_ratio,
            }
            for edge in result.graph.edges
        ],
        "output_size": [int(result.image.shape[1]), int(result.image.shape[0])],
        "estimated_focal_px": result.focal_px,
        "rotation_tree": [
            [keyframes[first]["id"], keyframes[second]["id"]]
            for first, second in result.tree_edges
        ],
        "median_cycle_error_deg": result.cycle_error_deg,
    }
    diagnostics_path = args.diagnostics or args.output.with_suffix(".json")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(
        f"{len(selected)}/{len(keyframes)} frames in largest strict overlap component · "
        f"{len(result.graph.edges)} accepted pair edges"
    )
    print(f"wrote {args.output} {result.image.shape[1]}x{result.image.shape[0]}")
    print(f"wrote {diagnostics_path}")


if __name__ == "__main__":
    main()
