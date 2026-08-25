#!/usr/bin/env python3
"""Refine a captured bundle's poses from RGB features and ARCore depth.

The source bundle is never modified. A sibling bundle is written only when
held-out metric feature correspondences improve.

    PYTHONPATH=harness uv run python scripts/capture/refine_poses.py BUNDLE
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from recon.io.bundle import BundleDir
from recon.pose_refine import (
    estimate_pose_edge,
    refine_pose_graph,
    relative_pose_error,
)
from recon.schemas import Pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, help="refined bundle directory")
    parser.add_argument("--diagnostics", type=Path, help="diagnostics JSON path")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe match ratio")
    parser.add_argument("--max-features", type=int, default=2500)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.10)
    parser.add_argument("--max-pose-delta-m", type=float, default=0.80)
    parser.add_argument("--max-pose-delta-deg", type=float, default=25.0)
    return parser.parse_args()


def depth_at(depth: np.ndarray, pixel: np.ndarray) -> float | None:
    u, v = np.rint(pixel).astype(int)
    height, width = depth.shape
    if u < 1 or v < 1 or u >= width - 1 or v >= height - 1:
        return None
    patch = depth[v - 1 : v + 2, u - 1 : u + 2]
    valid = patch[(patch > 0.15) & (patch < 8.0) & np.isfinite(patch)]
    return float(np.median(valid)) if len(valid) >= 3 else None


def lift(pixel: np.ndarray, depth_m: float, intrinsics) -> np.ndarray:
    u, v = pixel
    return np.array(
        [
            (u - intrinsics.cx) / intrinsics.fx * depth_m,
            -(v - intrinsics.cy) / intrinsics.fy * depth_m,
            -depth_m,
        ],
        dtype=np.float64,
    )


def ratio_matches(matcher, first: np.ndarray, second: np.ndarray, ratio: float) -> dict[int, int]:
    if first is None or second is None or len(first) < 2 or len(second) < 2:
        return {}
    output: dict[int, int] = {}
    for pair in matcher.knnMatch(first, second, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            output[pair[0].queryIdx] = pair[0].trainIdx
    return output


def feature_correspondences(
    keypoints_i,
    descriptors_i: np.ndarray,
    keypoints_j,
    descriptors_j: np.ndarray,
    matcher,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Lowe's ratio already makes each source descriptor unambiguous. Requiring
    # the match to win in the reverse direction as well disconnected large,
    # repeated wall regions in real captures; the two RANSAC stages below are
    # the appropriate geometric filters for those candidates.
    matches = list(ratio_matches(matcher, descriptors_i, descriptors_j, ratio).items())
    if len(matches) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    points_i = np.float32([keypoints_i[i].pt for i, _ in matches])
    points_j = np.float32([keypoints_j[j].pt for _, j in matches])
    _, mask = cv2.findFundamentalMat(
        points_i, points_j, cv2.FM_RANSAC, 1.5, 0.995, maxIters=5000
    )
    if mask is None:
        return np.empty((0, 2)), np.empty((0, 2))
    keep = mask.ravel().astype(bool)
    return points_i[keep], points_j[keep]


def metric_correspondences(points_i, points_j, depth_i, depth_j, intr_i, intr_j):
    lifted_i = []
    lifted_j = []
    for pixel_i, pixel_j in zip(points_i, points_j, strict=True):
        z_i = depth_at(depth_i, pixel_i)
        z_j = depth_at(depth_j, pixel_j)
        if z_i is None or z_j is None:
            continue
        lifted_i.append(lift(pixel_i, z_i, intr_i))
        lifted_j.append(lift(pixel_j, z_j, intr_j))
    return np.asarray(lifted_i), np.asarray(lifted_j)


def correction_stats(before: list[np.ndarray], after: list[np.ndarray]) -> dict:
    translations = []
    rotations = []
    for first, second in zip(before, after, strict=True):
        translations.append(float(np.linalg.norm(second[:3, 3] - first[:3, 3])))
        delta = second[:3, :3] @ first[:3, :3].T
        angle = np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
        rotations.append(float(np.degrees(angle)))
    return {
        "translation_m": {
            "median": float(np.median(translations)),
            "p90": float(np.percentile(translations, 90)),
            "max": float(np.max(translations)),
        },
        "rotation_deg": {
            "median": float(np.median(rotations)),
            "p90": float(np.percentile(rotations, 90)),
            "max": float(np.max(rotations)),
        },
    }


def write_refined_bundle(source: BundleDir, output: Path, poses: list[np.ndarray]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if source.root.resolve() in output.resolve().parents:
        raise ValueError("output must not be inside the source bundle")
    shutil.copytree(source.root, output)
    refined = source.bundle.model_copy(deep=True)
    for keyframe, matrix in zip(refined.keyframes, poses, strict=True):
        keyframe.pose = Pose.from_matrix(matrix)
    BundleDir(root=output, bundle=refined).save()


def main() -> int:
    args = parse_args()
    source = BundleDir.load(args.bundle)
    keyframes = source.bundle.keyframes
    output = args.output or source.root.with_name(f"{source.root.name}-refined")
    diagnostics_path = args.diagnostics or output.with_name(f"{output.name}-pose-refine.json")

    sift = cv2.SIFT_create(nfeatures=args.max_features, contrastThreshold=0.02)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    features = []
    depths = []
    for index, keyframe in enumerate(keyframes):
        gray = np.asarray(source.gray(keyframe), dtype=np.uint8)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        features.append((keypoints, descriptors))
        depths.append(source.depth(keyframe))
        print(f"features {index + 1:02d}/{len(keyframes)}: {keyframe.id} {len(keypoints)}")

    edges = []
    rejected = Counter()
    pair_stats = []
    initial = [keyframe.pose.matrix() for keyframe in keyframes]
    for i, j in combinations(range(len(keyframes)), 2):
        if depths[i] is None or depths[j] is None:
            rejected["missing_depth"] += 1
            continue
        pixels_i, pixels_j = feature_correspondences(
            *features[i], *features[j], matcher, args.ratio
        )
        if len(pixels_i) < 8:
            rejected["few_2d_matches"] += 1
            continue
        points_i, points_j = metric_correspondences(
            pixels_i,
            pixels_j,
            depths[i],
            depths[j],
            keyframes[i].intrinsics,
            keyframes[j].intrinsics,
        )
        if len(points_i) < 11:
            rejected["few_depth_matches"] += 1
            continue
        edge = estimate_pose_edge(
            i,
            j,
            points_i,
            points_j,
            seed=7919 * i + j,
            threshold_m=args.ransac_threshold_m,
        )
        if edge is None:
            rejected["metric_ransac"] += 1
            continue
        captured_relative = np.linalg.inv(initial[i]) @ initial[j]
        delta_m, delta_deg = relative_pose_error(edge.relative_ij, captured_relative)
        if delta_m > args.max_pose_delta_m or delta_deg > args.max_pose_delta_deg:
            rejected["implausible_pose_delta"] += 1
            continue
        edges.append(edge)
        pair_stats.append(
            {
                "i": i,
                "j": j,
                "frame_i": keyframes[i].id,
                "frame_j": keyframes[j].id,
                "verified_2d": len(pixels_i),
                "metric_matches": len(points_i),
                "train_inliers": edge.inlier_count,
                "train_rmse_m": edge.train_rmse_m,
                "captured_delta_m": delta_m,
                "captured_delta_deg": delta_deg,
                "heldout": len(edge.heldout_i),
            }
        )
        print(
            f"edge {i:02d}-{j:02d}: {edge.inlier_count} inliers, "
            f"{edge.train_rmse_m * 100:.1f} cm fit, pose delta {delta_m * 100:.1f} cm"
        )

    result = refine_pose_graph(initial, edges)
    diagnostics = {
        "source_bundle": str(source.root),
        "output_bundle": str(output),
        "frame_count": len(keyframes),
        "edge_count": len(edges),
        "rejected_pairs": dict(rejected),
        "optimized_component": list(result.component),
        "optimized_frame_count": len(result.component),
        "heldout_before_m": result.heldout_before_m,
        "heldout_after_m": result.heldout_after_m,
        "heldout_improvement_fraction": (
            1.0 - result.heldout_after_m / result.heldout_before_m
            if np.isfinite(result.heldout_before_m) and result.heldout_before_m > 0
            else None
        ),
        "relative_translation_before_m": result.train_before_m,
        "relative_translation_after_m": result.train_after_m,
        "optimizer_success": result.optimizer_success,
        "optimizer_cost": result.optimizer_cost,
        "accepted": result.accepted,
        "corrections": correction_stats(initial, result.poses),
        "edges": pair_stats,
    }
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n")

    print(
        f"\nheld-out median: {result.heldout_before_m * 100:.1f} -> "
        f"{result.heldout_after_m * 100:.1f} cm"
    )
    print(f"component: {len(result.component)}/{len(keyframes)} frames; edges: {len(edges)}")
    print(f"diagnostics: {diagnostics_path}")
    if not result.accepted:
        print("REJECTED: held-out validation did not improve enough; no bundle written")
        return 2
    write_refined_bundle(source, output, result.poses)
    print(f"refined bundle: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
