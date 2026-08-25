#!/usr/bin/env python3
"""Reconstruct 3D from a complete non-overlapping cubemap of a 360 panorama.

Four side views plus ceiling and floor partition the sphere exactly into six
90-degree faces, sharing only boundary rays. Depth Anything predicts their
depth from RGB. Known virtual-camera rotations from the equirectangular
projection place all six depth maps back into one 360-degree coordinate system.

No capture depth, pose, intrinsics, sparse points, heading, or tracking data is
read by this script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from da3_rgb_scene import (
    _point_cloud_from_glb,
    _write_viewer_cloud,
)
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panorama", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", default="small")
    parser.add_argument("--face-width", type=int, default=512)
    parser.add_argument("--face-height", type=int, default=512)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--max-points", type=int, default=800_000)
    return parser.parse_args()


def perspective_from_equirectangular(
    panorama: np.ndarray,
    *,
    yaw_degrees: float = 0.0,
    width: int,
    height: int,
    horizontal_fov_degrees: float = 90.0,
    camera_to_world: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one RGB pinhole view and its synthetic K and world-to-camera."""

    focal = width / (2.0 * np.tan(np.radians(horizontal_fov_degrees) / 2.0))
    centre_x = (width - 1) / 2.0
    centre_y = (height - 1) / 2.0
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    camera_rays = np.stack(
        [
            (columns - centre_x) / focal,
            (rows - centre_y) / focal,
            np.ones_like(columns),
        ],
        axis=-1,
    )
    if camera_to_world is None:
        yaw = np.radians(yaw_degrees)
        camera_to_world = np.array(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ],
            dtype=np.float64,
        )
    world_rays = camera_rays @ camera_to_world.T
    longitude = np.arctan2(world_rays[..., 0], world_rays[..., 2])
    latitude = np.arctan2(
        world_rays[..., 1],
        np.hypot(world_rays[..., 0], world_rays[..., 2]),
    )
    pano_height, pano_width = panorama.shape[:2]
    map_x = ((longitude / (2.0 * np.pi) + 0.5) * pano_width).astype(np.float32)
    map_y = ((latitude / np.pi + 0.5) * pano_height).astype(np.float32)
    view = cv2.remap(
        panorama,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    intrinsic = np.array(
        [[focal, 0.0, centre_x], [0.0, focal, centre_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = camera_to_world.T
    return view, intrinsic, world_to_camera


def cubemap_cameras() -> list[tuple[str, np.ndarray]]:
    """Six OpenCV camera bases that tile every spherical direction once."""

    sides = []
    for name, yaw_degrees in (
        ("front", 0.0),
        ("right", 90.0),
        ("back", 180.0),
        ("left", 270.0),
    ):
        yaw = np.radians(yaw_degrees)
        sides.append(
            (
                name,
                np.array(
                    [
                        [np.cos(yaw), 0.0, np.sin(yaw)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(yaw), 0.0, np.cos(yaw)],
                    ],
                    dtype=np.float64,
                ),
            )
        )
    ceiling = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    floor = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    return [*sides, ("ceiling", ceiling), ("floor", floor)]


def main() -> None:
    args = parse_args()
    if args.face_width != args.face_height:
        raise SystemExit("a non-overlapping cubemap requires square faces")
    panorama = np.asarray(Image.open(args.panorama).convert("RGB"))
    args.output.mkdir(parents=True, exist_ok=True)
    cameras = cubemap_cameras()
    views: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    for index, (name, camera_to_world) in enumerate(cameras):
        view, intrinsic, extrinsic = perspective_from_equirectangular(
            panorama,
            width=args.face_width,
            height=args.face_height,
            camera_to_world=camera_to_world,
        )
        views.append(view)
        intrinsics.append(intrinsic)
        extrinsics.append(extrinsic)
        Image.fromarray(view).save(args.output / f"view-{index}-{name}.jpg", quality=94)

    import torch
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.export.glb import export_to_glb

    checkpoint = f"depth-anything/DA3-{args.variant.upper()}"
    print(
        f"6 non-overlapping cubemap views · {checkpoint} · no ARCore inputs",
        flush=True,
    )
    model = DepthAnything3.from_pretrained(checkpoint, local_files_only=True)
    model = model.to("cpu").eval()
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model.inference(
            image=views,
            extrinsics=None,
            intrinsics=None,
            align_to_input_ext_scale=False,
            ref_view_strategy="first",
            process_res=args.process_res,
        )

    processed_height, processed_width = prediction.depth.shape[1:]
    scale_x = processed_width / args.face_width
    scale_y = processed_height / args.face_height
    projected_intrinsics = np.stack(intrinsics)
    projected_intrinsics[:, 0] *= scale_x
    projected_intrinsics[:, 1] *= scale_y
    prediction.intrinsics = projected_intrinsics
    prediction.extrinsics = np.stack(extrinsics)[:, :3, :]

    export_to_glb(
        prediction,
        str(args.output),
        num_max_points=args.max_points,
        conf_thresh=0.0,
        conf_thresh_percentile=0.0,
        ensure_thresh_percentile=100.0,
        show_cameras=False,
        export_depth_vis=False,
    )
    runtime_s = time.perf_counter() - started
    np.savez_compressed(
        args.output / "prediction-panorama-only.npz",
        model=np.array(f"da3-{args.variant}-panorama-views"),
        face_names=np.asarray([name for name, _ in cameras]),
        depth=np.asarray(prediction.depth, dtype=np.float32),
        confidence=np.asarray(prediction.conf, dtype=np.float32),
        extrinsics=np.asarray(prediction.extrinsics, dtype=np.float32),
        intrinsics=np.asarray(prediction.intrinsics, dtype=np.float32),
        uses_arcore=np.array(False),
    )
    points, colours = _point_cloud_from_glb(args.output / "scene.glb")
    _write_viewer_cloud(
        args.output / "cloud",
        points,
        colours,
        frame_ids=[f"panorama-{name}" for name, _ in cameras],
        model_name=f"da3-{args.variant}-panorama-views",
        runtime_s=runtime_s,
        metadata_extra={
            "method": "six non-overlapping 90-degree cubemap views from wrapped panorama",
            "source_panorama": str(args.panorama),
            "virtual_faces": [name for name, _ in cameras],
            "virtual_camera_model": "rectilinear pinhole",
            "horizontal_fov_degrees": 90.0,
            "vertical_fov_degrees": 90.0,
            "shared_optical_center": True,
            "face_focal_px": args.face_width / 2.0,
            "face_principal_point_px": [
                (args.face_width - 1) / 2.0,
                (args.face_height - 1) / 2.0,
            ],
            "spherical_coverage": "complete",
            "confidence_filtering": "disabled",
            "uses_arcore": False,
            "inputs": ["wrapped_panorama_rgb"],
        },
    )
    diagnostics = {
        "uses_arcore": False,
        "source": str(args.panorama),
        "views": 6,
        "horizontal_overlap_degrees": 0.0,
        "faces": [name for name, _ in cameras],
        "spherical_coverage": "complete",
        "confidence_filtering": "disabled",
        "virtual_camera": {
            "model": "rectilinear pinhole",
            "horizontal_fov_degrees": 90.0,
            "vertical_fov_degrees": 90.0,
            "shared_optical_center": True,
            "focal_px": args.face_width / 2.0,
            "principal_point_px": [
                (args.face_width - 1) / 2.0,
                (args.face_height - 1) / 2.0,
            ],
        },
        "face_size": [args.face_width, args.face_height],
        "prediction_size": [processed_width, processed_height],
        "depth_medians": [float(np.median(depth)) for depth in prediction.depth],
        "runtime_s": runtime_s,
        "points": int(len(points)),
    }
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(
        f"wrote {args.output / 'scene.glb'} and {len(points):,}-point viewer cloud "
        f"in {runtime_s:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
