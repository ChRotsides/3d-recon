"""Batch wrapper around the established RGB panorama/cubemap pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from recon.image_stitch import stitch_largest_component, wrap_equirectangular
from recon.live_rgb.replay import ReplayFrame

FACE_NAMES = ("front", "right", "back", "left", "ceiling", "floor")
CLOUD_SCHEMA = "live-rgb-cloud-v1"
MAX_STITCH_LONG_EDGE = 1024


def cubemap_cameras() -> list[tuple[str, np.ndarray]]:
    sides: list[tuple[str, np.ndarray]] = []
    for name, yaw_degrees in zip(FACE_NAMES[:4], (0.0, 90.0, 180.0, 270.0), strict=True):
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
    ceiling = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    floor = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    return [*sides, ("ceiling", ceiling), ("floor", floor)]


def render_face(
    panorama: np.ndarray, camera_to_world: np.ndarray, size: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    focal = size / 2.0
    centre = (size - 1) / 2.0
    columns, rows = np.meshgrid(np.arange(size), np.arange(size))
    rays = np.stack(
        [(columns - centre) / focal, (rows - centre) / focal, np.ones_like(columns)], axis=-1
    )
    world = rays @ camera_to_world.T
    longitude = np.arctan2(world[..., 0], world[..., 2])
    latitude = np.arctan2(world[..., 1], np.hypot(world[..., 0], world[..., 2]))
    height, width = panorama.shape[:2]
    map_x = ((longitude / (2 * np.pi) + 0.5) * width).astype(np.float32)
    map_y = ((latitude / np.pi + 0.5) * height).astype(np.float32)
    view = cv2.remap(
        panorama, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    )
    intrinsic = np.array(
        [[focal, 0.0, centre], [0.0, focal, centre], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = camera_to_world.T
    return view, intrinsic, extrinsic


def _stub_depth(views: list[np.ndarray], process_res: int) -> tuple[np.ndarray, np.ndarray, str]:
    depths = []
    confidences = []
    for view in views:
        resized = cv2.resize(view, (process_res, process_res), interpolation=cv2.INTER_AREA)
        luminance = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        depths.append(0.75 + luminance * 0.5)
        confidences.append(np.ones_like(luminance, dtype=np.float32))
    return np.stack(depths), np.stack(confidences), "deterministic-depth-stub-v1"


def _da3_depth(
    views: list[np.ndarray], process_res: int
) -> tuple[np.ndarray, np.ndarray, str, str]:
    import sys

    import torch
    from recon.depth.da3 import _install_full_precision_forward, _usable_device
    from recon.torch_compat import (
        compatible_dtype,
        configure_cuda_for_legacy_device,
        cuda_requires_fp32,
    )

    checkpoint = "depth-anything/DA3-SMALL"
    device = _usable_device(torch)
    needs_fp32 = cuda_requires_fp32(torch, device)
    configure_cuda_for_legacy_device(torch, device)
    if needs_fp32:
        # The shared web dependency path may contain a modern xFormers wheel.
        # It cannot run against the custom sm_35 PyTorch build; DA3's optional
        # import correctly falls back to math attention when it is unavailable.
        sys.modules["xformers"] = None
    from depth_anything_3.api import DepthAnything3

    requested_dtype = "float16" if device.startswith("cuda") else "float32"
    dtype = compatible_dtype(torch, device, requested_dtype)
    print(f"[live-rgb] DA3 loading on {device}:{dtype}", flush=True)
    model = DepthAnything3.from_pretrained(checkpoint, local_files_only=True).to(device)
    if dtype == "float16":
        model = model.half()
    elif dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
    else:
        model = model.float()
    if needs_fp32:
        _install_full_precision_forward(model, torch)
    model.eval()
    with torch.inference_mode():
        prediction = model.inference(
            image=views,
            extrinsics=None,
            intrinsics=None,
            align_to_input_ext_scale=False,
            ref_view_strategy="first",
            process_res=process_res,
        )
    return (
        np.asarray(prediction.depth, dtype=np.float32),
        np.asarray(prediction.conf, dtype=np.float32),
        checkpoint,
        f"{device}:{dtype}",
    )


def _prepare_stitch_images(images: list[np.ndarray]) -> list[np.ndarray]:
    """Bound live phone images before all-pairs feature analysis and warping."""

    if not images:
        return []
    height, width = images[0].shape[:2]
    longest = max(height, width)
    if longest <= MAX_STITCH_LONG_EDGE:
        return images
    scale = MAX_STITCH_LONG_EDGE / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return [cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in images]


def _backproject(
    views: list[np.ndarray],
    depths: np.ndarray,
    intrinsics: list[np.ndarray],
    cameras: list[tuple[str, np.ndarray]],
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_points: list[np.ndarray] = []
    all_colours: list[np.ndarray] = []
    for index, ((_, camera_to_world), depth) in enumerate(zip(cameras, depths, strict=True)):
        height, width = depth.shape
        scale_x = width / views[index].shape[1]
        scale_y = height / views[index].shape[0]
        k = intrinsics[index].copy()
        k[0] *= scale_x
        k[1] *= scale_y
        columns, rows = np.meshgrid(np.arange(width), np.arange(height))
        rays = np.stack(
            [(columns - k[0, 2]) / k[0, 0], (rows - k[1, 2]) / k[1, 1], np.ones_like(depth)],
            axis=-1,
        )
        valid = np.isfinite(depth) & (depth > 0)
        points = rays[valid] * depth[valid, None]
        points = points @ camera_to_world.T
        colours = cv2.resize(views[index], (width, height), interpolation=cv2.INTER_AREA)[valid]
        all_points.append(points.astype(np.float32))
        all_colours.append(colours.astype(np.uint8))
    points = np.concatenate(all_points)
    colours = np.concatenate(all_colours)
    if len(points) > max_points:
        selected = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points, colours = points[selected], colours[selected]
    return points, colours


def _write_cloud(base: Path, points: np.ndarray, colours: np.ndarray, metadata: dict) -> dict:
    points = np.ascontiguousarray(points, dtype=np.float32)
    colours = np.ascontiguousarray(colours, dtype=np.uint8)
    payload = points.tobytes() + colours.tobytes()
    base.with_suffix(".bin").write_bytes(payload)
    low, high = points.min(axis=0), points.max(axis=0)
    cloud = {
        "schema": CLOUD_SCHEMA,
        "count": int(len(points)),
        "stride": {"position_bytes": 12, "colour_bytes": 3, "layout": "positions_then_rgb"},
        "min": low.tolist(),
        "max": high.tolist(),
        "centre": ((low + high) / 2).tolist(),
        "extent": (high - low).tolist(),
        "asset_sha256": hashlib.sha256(payload).hexdigest(),
        "asset_bytes": len(payload),
        **metadata,
    }
    base.with_suffix(".json").write_text(json.dumps(cloud, indent=2, sort_keys=True) + "\n")
    return cloud


def reconstruct(
    frames: list[ReplayFrame],
    output: Path,
    *,
    depth_backend: str = "stub",
    max_points: int = 800_000,
    face_size: int = 512,
    process_res: int = 504,
) -> dict:
    if not frames:
        raise ValueError("at least one RGB frame is required")
    started = time.perf_counter()

    def progress(message: str) -> None:
        elapsed = time.perf_counter() - started
        print(f"[live-rgb +{elapsed:7.1f}s] {message}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(frames, key=lambda frame: (frame.timestamp_ns, frame.frame_id))
    progress(f"loading {len(ordered)} RGB frames")
    source_images = [cv2.imread(str(frame.jpeg_path), cv2.IMREAD_COLOR) for frame in ordered]
    if any(image is None for image in source_images):
        raise ValueError("every frame must be a readable JPEG")
    source_size = [int(source_images[0].shape[1]), int(source_images[0].shape[0])]
    images = _prepare_stitch_images(source_images)
    del source_images
    stitch_size = [int(images[0].shape[1]), int(images[0].shape[0])]
    progress(f"stitch inputs prepared at {stitch_size[0]}x{stitch_size[1]}")
    stitch = stitch_largest_component(images, pair_window=5, progress=progress)
    progress("camera graph and spherical stitch complete")
    panorama = wrap_equirectangular(stitch.image, stitch.alpha)
    # A source stitch may have odd width (the 065a reference is 2667 px).
    # A discrete equirectangular image can only be exactly 2:1 at even width;
    # normalize the single longitude pixel before any virtual camera is rendered.
    exact_width = panorama.shape[0] * 2
    if panorama.shape[1] != exact_width:
        panorama = cv2.resize(
            panorama,
            (exact_width, panorama.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if panorama.shape[1] != panorama.shape[0] * 2:
        raise AssertionError("wrapped panorama is not exactly 2:1")
    Image.fromarray(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)).save(output / "panorama.png")
    progress(f"wrote {panorama.shape[1]}x{panorama.shape[0]} panorama")
    panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    cameras = cubemap_cameras()
    views: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    for index, (name, basis) in enumerate(cameras):
        view, intrinsic, extrinsic = render_face(panorama_rgb, basis, face_size)
        views.append(view)
        intrinsics.append(intrinsic)
        extrinsics.append(extrinsic)
        Image.fromarray(view).save(output / f"view-{index}-{name}.jpg", quality=94)
    progress("rendered six cubemap views")
    if depth_backend == "stub":
        depths, confidence, model_revision = _stub_depth(views, process_res)
        depth_device = "cpu:float32"
    elif depth_backend == "da3-small":
        progress("starting DA3 depth inference")
        depths, confidence, model_revision, depth_device = _da3_depth(views, process_res)
    else:
        raise ValueError(f"unknown depth backend: {depth_backend}")
    progress(f"depth inference complete on {depth_device}")
    progress("backprojecting depth into the viewer point cloud")
    points, colours = _backproject(views, depths, intrinsics, cameras, max_points)
    selected_ids = [ordered[index].frame_id for index in stitch.graph.selected]
    graph = {
        "selected_frames": selected_ids,
        "selected_count": len(selected_ids),
        "accepted_edge_count": len(stitch.graph.edges),
        "accepted_edges": [asdict(edge) for edge in stitch.graph.edges],
        "median_cycle_error_deg": stitch.cycle_error_deg,
        "estimated_focal_px": stitch.focal_px,
    }
    virtual = {
        "face_names": list(FACE_NAMES),
        "face_size": [face_size, face_size],
        "prediction_size": [int(depths.shape[2]), int(depths.shape[1])],
        "horizontal_fov_degrees": 90.0,
        "vertical_fov_degrees": 90.0,
        "shared_optical_center": True,
        "intrinsics": [matrix.tolist() for matrix in intrinsics],
        "extrinsics": [matrix.tolist() for matrix in extrinsics],
        "da3_input_extrinsics": None,
        "da3_input_intrinsics": None,
        "exported_camera_source": "exact_virtual_cubemap",
    }
    cloud = _write_cloud(
        output / "cloud",
        points,
        colours,
        {
            "model_revision": model_revision,
            "uses_arcore": False,
            "inputs": ["rgb_jpeg"],
            "confidence_filtering": "disabled",
            "finite_positive_depth_count": int(
                np.count_nonzero(np.isfinite(depths) & (depths > 0))
            ),
            # All finite positive samples are eligible for projection. The
            # separate max-point budget may deterministically subsample the
            # cloud, but confidence never excludes a direction.
            "exported_depth_count": int(
                np.count_nonzero(np.isfinite(depths) & (depths > 0))
            ),
        },
    )
    diagnostics = {
        "schema": "live-rgb-diagnostics-v1",
        "uses_arcore": False,
        "input_frame_count": len(ordered),
        "input_image_size": source_size,
        "stitch_image_size": stitch_size,
        "graph": graph,
        "panorama": {
            "size": [int(panorama.shape[1]), int(panorama.shape[0])],
            "exactly_2_to_1": True,
            "transparent_pixels": 0,
            "horizontal_seam_max_abs": int(
                np.abs(panorama[:, 0].astype(np.int16) - panorama[:, -1].astype(np.int16)).max()
            ),
        },
        "virtual_cameras": virtual,
        "depth": {
            "backend": depth_backend,
            "model_revision": model_revision,
            "device": depth_device,
            "confidence_filtering": "disabled",
            "finite_positive": int(np.count_nonzero(np.isfinite(depths) & (depths > 0))),
            "confidence_shape": list(confidence.shape),
        },
        "cloud": cloud,
        "runtime_s": time.perf_counter() - started,
    }
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )
    progress(f"complete: wrote {len(points):,} points")
    return diagnostics
