#!/usr/bin/env python3
"""Build a Depth Anything 3 scene from RGB images only.

No captured depth, pose, intrinsics, sparse points, heading, or tracking fields
are read. DA3 estimates relative depth and cameras jointly from the selected
images, then exports a colored GLB and the flat point buffer used by the local
WebGL viewer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--frame-ids",
        required=True,
        help="comma-separated keyframe IDs chosen from image overlap",
    )
    parser.add_argument("--variant", default="small")
    parser.add_argument("--process-res", type=int, default=336)
    parser.add_argument("--max-points", type=int, default=600_000)
    return parser.parse_args()


def _point_cloud_from_glb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scene = trimesh.load(path, force="scene")
    positions: list[np.ndarray] = []
    colours: list[np.ndarray] = []
    for geometry in scene.geometry.values():
        vertices = np.asarray(getattr(geometry, "vertices", ()), dtype=np.float32)
        if not len(vertices):
            continue
        visual = getattr(geometry, "visual", None)
        vertex_colours = np.asarray(
            getattr(visual, "vertex_colors", ()), dtype=np.uint8
        )
        if len(vertex_colours) != len(vertices):
            vertex_colours = np.full((len(vertices), 4), 210, dtype=np.uint8)
        # DA3 applies its alignment before constructing the GLB point cloud, so
        # exported vertices already live in the scene frame.
        positions.append(vertices)
        colours.append(vertex_colours[:, :3])
    if not positions:
        raise RuntimeError("DA3 GLB contains no point geometry")
    return np.concatenate(positions), np.concatenate(colours)


def _write_viewer_cloud(
    output: Path,
    positions: np.ndarray,
    colours: np.ndarray,
    *,
    frame_ids: list[str],
    model_name: str,
    runtime_s: float,
    metadata_extra: dict[str, object] | None = None,
) -> None:
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    colours = np.ascontiguousarray(colours, dtype=np.uint8)
    output.with_suffix(".bin").write_bytes(positions.tobytes() + colours.tobytes())
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    quantile_low = np.quantile(positions, 0.01, axis=0)
    quantile_high = np.quantile(positions, 0.99, axis=0)
    metadata = {
        "count": int(len(positions)),
        "min": low.tolist(),
        "max": high.tolist(),
        "centre": ((low + high) / 2).tolist(),
        "extent": (high - low).tolist(),
        "view_min": quantile_low.tolist(),
        "view_max": quantile_high.tolist(),
        "view_centre": ((quantile_low + quantile_high) / 2).tolist(),
        "view_extent": (quantile_high - quantile_low).tolist(),
        "model": model_name,
        "keyframes": len(frame_ids),
        "frame_ids": frame_ids,
        "units": "relative",
        "runtime_s": runtime_s,
        "uses_arcore": False,
        "inputs": ["rgb"],
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    requested = [item.strip() for item in args.frame_ids.split(",") if item.strip()]
    wire = json.loads((args.bundle / "bundle.json").read_text())
    image_by_id = {
        keyframe["id"]: args.bundle / keyframe["image"]
        for keyframe in wire["keyframes"]
    }
    missing = [frame_id for frame_id in requested if frame_id not in image_by_id]
    if missing:
        raise SystemExit(f"unknown frame IDs: {', '.join(missing)}")
    images = [
        np.asarray(Image.open(image_by_id[frame_id]).convert("RGB"))
        for frame_id in requested
    ]

    # Import after argument/input validation: loading DA3 initialises PyTorch and
    # should not be the price of discovering a typo in a frame ID.
    import torch
    from depth_anything_3.api import DepthAnything3

    checkpoint = f"depth-anything/DA3-{args.variant.upper()}"
    print(
        f"{len(images)} RGB frames · {checkpoint} · {args.process_res}px · "
        "no ARCore inputs",
        flush=True,
    )
    model = DepthAnything3.from_pretrained(checkpoint, local_files_only=True)
    model = model.to("cpu").eval()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model.inference(
            image=images,
            extrinsics=None,
            intrinsics=None,
            align_to_input_ext_scale=False,
            process_res=args.process_res,
            export_dir=str(args.output),
            export_format="glb",
            conf_thresh_percentile=45.0,
            num_max_points=args.max_points,
            show_cameras=False,
            export_kwargs={"glb": {"export_depth_vis": False}},
        )
    runtime_s = time.perf_counter() - started

    np.savez_compressed(
        args.output / "prediction-rgb-only.npz",
        model=np.array(f"da3-{args.variant}-rgb-only"),
        frame_ids=np.asarray(requested),
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
        frame_ids=requested,
        model_name=f"da3-{args.variant}-rgb-only",
        runtime_s=runtime_s,
    )
    print(
        f"wrote {args.output / 'scene.glb'} and {len(points):,}-point viewer cloud "
        f"in {runtime_s:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
