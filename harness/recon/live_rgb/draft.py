"""Versioned RGB-only draft reconstruction and transport.

Drafts are complete immutable snapshots.  A newer request may supersede queued or
running work, but point records from different source panorama versions are never
combined.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import tracemalloc
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from recon.image_stitch import stitch_largest_component, wrap_equirectangular
from recon.live_rgb.pipeline import _da3_depth, _prepare_stitch_images, _stub_depth
from recon.live_rgb.replay import ReplayFrame

DRAFT_SCHEMA = "live-rgb-draft-cloud-v3"
DRAFT_EVENT_SCHEMA = "live-rgb-draft-event-v1"
FACE_NAMES = ("front", "right", "back", "left", "ceiling", "floor")
FACE_SIZE = 256
DEFAULT_PROCESS_RES = 252
MAX_POINTS = 250_000
MAX_TRANSPORT_BYTES = 12 * 1024 * 1024
RECORD_DTYPE = np.dtype(
    [
        ("position", "<f4", (3,)),
        ("rgb", "u1", (3,)),
        ("padding", "u1"),
        ("confidence", "<f4"),
        ("panorama_support", "<f4"),
        ("source_version", "<u4"),
    ],
    align=False,
)
assert RECORD_DTYPE.itemsize == 28


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class VirtualCamera:
    name: str
    camera_to_world: np.ndarray
    rendered_intrinsic: np.ndarray
    processed_intrinsic: np.ndarray
    rendered_size: tuple[int, int]
    processed_size: tuple[int, int]

    @property
    def extrinsic(self) -> np.ndarray:
        """Exact world-to-camera matrix with the shared origin at (0, 0, 0)."""

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = self.camera_to_world.T
        return matrix


def _intrinsic(size: int) -> np.ndarray:
    # Coordinates are evaluated at pixel centres (column + 0.5), so a sensor
    # spanning [-size/2, size/2] with f=size/2 has exactly a 90 degree FOV.
    return np.array(
        [[size / 2.0, 0.0, size / 2.0], [0.0, size / 2.0, size / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def rescale_intrinsic(
    intrinsic: np.ndarray,
    rendered_size: tuple[int, int],
    processed_size: tuple[int, int],
) -> np.ndarray:
    """Scale fx/cx and fy/cy by the exact independent processing ratios."""

    rendered_width, rendered_height = rendered_size
    processed_width, processed_height = processed_size
    scaled = np.asarray(intrinsic, dtype=np.float32).copy()
    scaled[0, :] *= processed_width / rendered_width
    scaled[1, :] *= processed_height / rendered_height
    return scaled


def cubemap_cameras(
    face_size: int = FACE_SIZE, process_res: int = DEFAULT_PROCESS_RES
) -> tuple[VirtualCamera, ...]:
    if face_size <= 0 or process_res <= 0:
        raise ValueError("cubemap resolutions must be positive")
    rotations = []
    for name, yaw_degrees in zip(FACE_NAMES[:4], (0.0, 90.0, 180.0, 270.0), strict=True):
        yaw = np.radians(yaw_degrees)
        rotations.append(
            (
                name,
                np.array(
                    [
                        [np.cos(yaw), 0.0, np.sin(yaw)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(yaw), 0.0, np.cos(yaw)],
                    ],
                    dtype=np.float32,
                ),
            )
        )
    # This pipeline uses image coordinates (+Y down): ceiling looks toward -Y
    # and floor toward +Y while preserving a right-handed camera basis.
    rotations.extend(
        [
            (
                "ceiling",
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
            ),
            (
                "floor",
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                    dtype=np.float32,
                ),
            ),
        ]
    )
    rendered = _intrinsic(face_size)
    processed = rescale_intrinsic(rendered, (face_size, face_size), (process_res, process_res))
    return tuple(
        VirtualCamera(
            name=name,
            camera_to_world=rotation,
            rendered_intrinsic=rendered.copy(),
            processed_intrinsic=processed.copy(),
            rendered_size=(face_size, face_size),
            processed_size=(process_res, process_res),
        )
        for name, rotation in rotations
    )


def direction_face(directions: np.ndarray) -> np.ndarray:
    """Assign every non-zero sphere direction to one face with stable tie breaks."""

    values = np.asarray(directions, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("directions must end in XYZ")
    flat = values.reshape(-1, 3)
    if np.any(~np.isfinite(flat)) or np.any(np.linalg.norm(flat, axis=1) == 0):
        raise ValueError("directions must be finite and non-zero")
    x, y, z = flat.T
    absolute = np.abs(flat)
    axis = np.argmax(absolute[:, [2, 0, 1]], axis=1)  # boundary priority Z, X, Y
    result = np.empty(len(flat), dtype=np.int8)
    z_axis, x_axis, y_axis = axis == 0, axis == 1, axis == 2
    result[z_axis] = np.where(z[z_axis] >= 0, 0, 2)
    result[x_axis] = np.where(x[x_axis] >= 0, 1, 3)
    result[y_axis] = np.where(y[y_axis] < 0, 4, 5)
    return result.reshape(values.shape[:-1])


def project_direction(direction: np.ndarray, camera: VirtualCamera) -> np.ndarray:
    camera_ray = camera.camera_to_world.T @ np.asarray(direction, dtype=np.float64)
    if camera_ray[2] <= 0:
        raise ValueError("direction is behind this camera")
    k = camera.processed_intrinsic.astype(np.float64)
    return np.array(
        [
            k[0, 0] * camera_ray[0] / camera_ray[2] + k[0, 2],
            k[1, 1] * camera_ray[1] / camera_ray[2] + k[1, 2],
        ]
    )


def backproject_direction(pixel: np.ndarray, camera: VirtualCamera) -> np.ndarray:
    k = camera.processed_intrinsic.astype(np.float64)
    uv = np.asarray(pixel, dtype=np.float64)
    ray = np.array([(uv[0] - k[0, 2]) / k[0, 0], (uv[1] - k[1, 2]) / k[1, 1], 1.0])
    world = camera.camera_to_world.astype(np.float64) @ ray
    return world / np.linalg.norm(world)


def _face_maps(camera: VirtualCamera, size: int) -> tuple[np.ndarray, np.ndarray]:
    columns, rows = np.meshgrid(
        np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32)
    )
    k = camera.rendered_intrinsic
    rays = np.stack(
        [
            (columns + 0.5 - k[0, 2]) / k[0, 0],
            (rows + 0.5 - k[1, 2]) / k[1, 1],
            np.ones_like(columns),
        ],
        axis=-1,
    )
    world = rays @ camera.camera_to_world.T
    longitude = np.arctan2(world[..., 0], world[..., 2])
    latitude = np.arctan2(world[..., 1], np.hypot(world[..., 0], world[..., 2]))
    return longitude, latitude


def render_cubemap(
    panorama_rgb: np.ndarray,
    panorama_support: np.ndarray | None = None,
    *,
    face_size: int = FACE_SIZE,
    process_res: int = DEFAULT_PROCESS_RES,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[VirtualCamera, ...]]:
    if panorama_rgb.ndim != 3 or panorama_rgb.shape[2] != 3:
        raise ValueError("panorama must be RGB")
    height, width = panorama_rgb.shape[:2]
    if width != height * 2:
        raise ValueError("panorama must be exactly 2:1")
    if panorama_support is None:
        panorama_support = np.ones((height, width), dtype=np.float32)
    if panorama_support.shape != (height, width):
        raise ValueError("support must match the panorama")
    cameras = cubemap_cameras(face_size, process_res)
    views, supports = [], []
    for camera in cameras:
        longitude, latitude = _face_maps(camera, face_size)
        map_x = ((longitude / (2.0 * np.pi) + 0.5) * width - 0.5).astype(np.float32)
        map_y = ((latitude / np.pi + 0.5) * height - 0.5).astype(np.float32)
        views.append(
            cv2.remap(
                panorama_rgb,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
        )
        supports.append(
            np.clip(
                cv2.remap(
                    panorama_support.astype(np.float32),
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP,
                ),
                0.0,
                1.0,
            )
        )
    return views, supports, cameras


def _exact_panorama(frames: Iterable[ReplayFrame]) -> tuple[np.ndarray, np.ndarray, dict]:
    ordered = sorted(frames, key=lambda frame: (frame.timestamp_ns, frame.frame_id))
    images = [cv2.imread(str(frame.jpeg_path), cv2.IMREAD_COLOR) for frame in ordered]
    if not images or any(image is None for image in images):
        raise ValueError("draft requires readable RGB frames")
    prepared = _prepare_stitch_images(images)
    stitch = stitch_largest_component(prepared, pair_window=5)
    panorama = wrap_equirectangular(stitch.image, stitch.alpha)
    width = panorama.shape[0] * 2
    if panorama.shape[1] != width:
        panorama = cv2.resize(panorama, (width, panorama.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Alpha is only an evidence channel.  Resize it into equirectangular space;
    # geometry is always retained even where this evidence is weak.
    alpha = np.asarray(stitch.alpha, dtype=np.float32)
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    support = cv2.resize(alpha, (width, panorama.shape[0]), interpolation=cv2.INTER_LINEAR)
    maximum = float(np.max(support))
    support = np.clip(support / maximum if maximum > 0 else support, 0.0, 1.0)
    return (
        cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB),
        support,
        {
            "selected_count": len(stitch.graph.selected),
            "accepted_edge_count": len(stitch.graph.edges),
            "cycle_error_deg": float(stitch.cycle_error_deg),
        },
    )


def backproject_evidence(
    views: list[np.ndarray],
    supports: list[np.ndarray],
    depths: np.ndarray,
    confidences: np.ndarray,
    cameras: tuple[VirtualCamera, ...],
    source_version: int,
    max_points: int = MAX_POINTS,
) -> np.ndarray:
    records = []
    for view, support, depth, confidence, camera in zip(
        views, supports, depths, confidences, cameras, strict=True
    ):
        height, width = depth.shape
        expected = camera.processed_size
        if (width, height) != expected:
            k = rescale_intrinsic(camera.rendered_intrinsic, camera.rendered_size, (width, height))
        else:
            k = camera.processed_intrinsic
        columns, rows = np.meshgrid(np.arange(width), np.arange(height))
        rays = np.stack(
            [
                (columns + 0.5 - k[0, 2]) / k[0, 0],
                (rows + 0.5 - k[1, 2]) / k[1, 1],
                np.ones_like(depth),
            ],
            axis=-1,
        )
        valid = np.isfinite(depth) & (depth > 0)
        count = int(np.count_nonzero(valid))
        record = np.zeros(count, dtype=RECORD_DTYPE)
        record["position"] = ((rays[valid] * depth[valid, None]) @ camera.camera_to_world.T).astype(
            np.float32
        )
        record["rgb"] = cv2.resize(view, (width, height), interpolation=cv2.INTER_AREA)[valid]
        resized_confidence = cv2.resize(
            np.asarray(confidence, dtype=np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_support = cv2.resize(
            np.asarray(support, dtype=np.float32), (width, height), interpolation=cv2.INTER_LINEAR
        )
        record["confidence"] = np.nan_to_num(resized_confidence[valid], nan=0.0)
        record["panorama_support"] = np.clip(
            np.nan_to_num(resized_support[valid], nan=0.0), 0.0, 1.0
        )
        record["source_version"] = source_version
        records.append(record)
    cloud = np.concatenate(records)
    if len(cloud) > max_points:
        selected = np.linspace(0, len(cloud) - 1, max_points, dtype=np.int64)
        cloud = cloud[selected]
    return np.ascontiguousarray(cloud)


def write_draft_asset(base: Path, records: np.ndarray, metadata: dict) -> dict:
    records = np.ascontiguousarray(records, dtype=RECORD_DTYPE)
    if not 0 < len(records) <= MAX_POINTS:
        raise ValueError("invalid draft point count")
    positions = records["position"]
    if not np.isfinite(positions).all():
        raise ValueError("draft contains non-finite coordinates")
    if np.any(records["panorama_support"] < 0) or np.any(records["panorama_support"] > 1):
        raise ValueError("draft support is outside [0, 1]")
    raw = records.tobytes()
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    if len(compressed) > MAX_TRANSPORT_BYTES:
        raise ValueError("draft transport exceeds 12 MiB")
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".bin.gz").write_bytes(compressed)
    low, high = positions.min(axis=0), positions.max(axis=0)
    source_versions = np.unique(records["source_version"])
    if len(source_versions) != 1:
        raise ValueError("mixed source versions are forbidden")
    payload = {
        "schema": DRAFT_SCHEMA,
        "version": int(metadata["version"]),
        "source_panorama_version": int(source_versions[0]),
        "count": int(len(records)),
        "record_bytes": RECORD_DTYPE.itemsize,
        "layout": "interleaved_xyz_rgb_pad_confidence_support_source_version_le",
        "compression": "gzip",
        "asset_bytes": len(compressed),
        "decoded_bytes": len(raw),
        "asset_sha256": hashlib.sha256(compressed).hexdigest(),
        "decoded_sha256": hashlib.sha256(raw).hexdigest(),
        "min": low.tolist(),
        "max": high.tolist(),
        "uses_arcore": False,
        "inputs": ["rgb_panorama"],
        **metadata,
    }
    _atomic_json(base.with_suffix(".json"), payload)
    validate_draft_asset(base.with_suffix(".json"), base.with_suffix(".bin.gz"))
    return payload


def validate_draft_asset(metadata_path: Path, asset_path: Path) -> np.ndarray:
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != DRAFT_SCHEMA:
        raise ValueError("unsupported draft schema")
    count = int(metadata["count"])
    if not 0 < count <= MAX_POINTS or int(metadata["record_bytes"]) != RECORD_DTYPE.itemsize:
        raise ValueError("invalid draft point contract")
    compressed = asset_path.read_bytes()
    if len(compressed) != int(metadata["asset_bytes"]) or len(compressed) > MAX_TRANSPORT_BYTES:
        raise ValueError("invalid draft asset size")
    if hashlib.sha256(compressed).hexdigest() != metadata["asset_sha256"]:
        raise ValueError("draft asset checksum mismatch")
    raw = gzip.decompress(compressed)
    if len(raw) != count * RECORD_DTYPE.itemsize or len(raw) != int(metadata["decoded_bytes"]):
        raise ValueError("invalid decoded draft size")
    if hashlib.sha256(raw).hexdigest() != metadata["decoded_sha256"]:
        raise ValueError("decoded draft checksum mismatch")
    records = np.frombuffer(raw, dtype=RECORD_DTYPE)
    positions = records["position"]
    low, high = np.asarray(metadata["min"]), np.asarray(metadata["max"])
    if (
        not np.isfinite(positions).all()
        or np.any(positions < low - 1e-5)
        or np.any(positions > high + 1e-5)
    ):
        raise ValueError("draft coordinates violate declared bounds")
    if not np.all(records["source_version"] == int(metadata["source_panorama_version"])):
        raise ValueError("mixed draft source versions")
    return records


def reconstruct_draft(
    frames: list[ReplayFrame],
    output: Path,
    *,
    source_version: int,
    depth_backend: str = "stub",
    face_size: int = FACE_SIZE,
    process_res: int = DEFAULT_PROCESS_RES,
    max_points: int = MAX_POINTS,
) -> dict:
    started = time.perf_counter()
    tracemalloc.start()
    panorama, panorama_support, graph = _exact_panorama(frames)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panorama).save(output / "panorama.png")
    views, supports, cameras = render_cubemap(
        panorama, panorama_support, face_size=face_size, process_res=process_res
    )
    for index, (camera, view) in enumerate(zip(cameras, views, strict=True)):
        Image.fromarray(view).save(output / f"view-{index}-{camera.name}.jpg", quality=90)
    inference_started = time.perf_counter()
    if depth_backend == "stub":
        depths, confidences, model_revision = _stub_depth(views, process_res)
        device = "cpu:float32"
    elif depth_backend == "da3-small":
        depths, confidences, model_revision, device = _da3_depth(views, process_res)
    else:
        raise ValueError(f"unknown depth backend: {depth_backend}")
    inference_s = time.perf_counter() - inference_started
    records = backproject_evidence(
        views, supports, depths, confidences, cameras, source_version, max_points
    )
    cloud = write_draft_asset(
        output / "draft",
        records,
        {
            "version": source_version,
            "model_revision": model_revision,
            "confidence_filtering": "disabled",
            "eligible_finite_positive_depths": int(
                np.count_nonzero(np.isfinite(depths) & (depths > 0))
            ),
        },
    )
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    diagnostics = {
        "schema": "live-rgb-draft-diagnostics-v1",
        "version": source_version,
        "source_panorama_version": source_version,
        "uses_arcore": False,
        "graph": graph,
        "panorama_size": [int(panorama.shape[1]), int(panorama.shape[0])],
        "virtual_cameras": {
            "face_names": [camera.name for camera in cameras],
            "rendered_size": [face_size, face_size],
            "processed_size": [process_res, process_res],
            "horizontal_fov_degrees": 90.0,
            "vertical_fov_degrees": 90.0,
            "shared_optical_center": [0.0, 0.0, 0.0],
            "rendered_intrinsics": [camera.rendered_intrinsic.tolist() for camera in cameras],
            "processed_intrinsics": [camera.processed_intrinsic.tolist() for camera in cameras],
            "extrinsics": [camera.extrinsic.tolist() for camera in cameras],
            "predicted_cameras_exported": False,
            "exported_camera_source": "exact_virtual_cubemap",
        },
        "depth": {
            "backend": depth_backend,
            "model_revision": model_revision,
            "device": device,
            "inference_s": inference_s,
            "confidence_filtering": "disabled",
        },
        "cloud": cloud,
        "metrics": {
            "stable_panorama_to_published_draft_s": time.perf_counter() - started,
            "inference_s": inference_s,
            "transport_bytes": cloud["asset_bytes"],
            "point_count": cloud["count"],
            "peak_worker_memory_bytes": peak_memory,
        },
    }
    _atomic_json(output / "diagnostics.json", diagnostics)
    return diagnostics


@dataclass(frozen=True)
class DraftRequest:
    station: Path
    source_version: int
    frames: tuple[ReplayFrame, ...]
    queued_at: float


class DraftWorker:
    """Single inference worker whose pending queue coalesces per station."""

    def __init__(
        self,
        root: Path,
        *,
        depth_backend: str = "stub",
        builder: Callable[..., dict] = reconstruct_draft,
    ) -> None:
        self.root = root
        self.depth_backend = depth_backend
        self.builder = builder
        self._condition = threading.Condition()
        self._pending: dict[str, DraftRequest] = {}
        self._latest_requested: dict[str, int] = {}
        self._running: DraftRequest | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="live-rgb-draft", daemon=True)
        self._thread.start()

    @staticmethod
    def _key(station: Path) -> str:
        return str(station.resolve())

    def submit(self, station: Path, source_version: int, frames: Iterable[ReplayFrame]) -> bool:
        request = DraftRequest(station, source_version, tuple(frames), time.perf_counter())
        if not request.frames:
            return False
        key = self._key(station)
        with self._condition:
            if source_version <= self._latest_requested.get(key, 0):
                return False
            self._latest_requested[key] = source_version
            self._pending[key] = request  # replaces stale queued work for this station
            self._condition.notify_all()
        return True

    def status(self, station: Path) -> dict:
        current = station / "drafts/current.json"
        with self._condition:
            pending = self._latest_requested.get(self._key(station))
            running = self._running
        if current.exists():
            event = json.loads(current.read_text())
            event["state"] = "ready"
            event["latest_requested_version"] = pending or event["version"]
            return event
        if pending is not None:
            return {
                "schema": DRAFT_EVENT_SCHEMA,
                "state": "running"
                if running and self._key(running.station) == self._key(station)
                else "queued",
                "latest_requested_version": pending,
            }
        return {"schema": DRAFT_EVENT_SCHEMA, "state": "unavailable"}

    def wait_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._pending or self._running is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                key = min(self._pending, key=lambda item: self._pending[item].queued_at)
                request = self._pending.pop(key)
                self._running = request
            drafts = request.station / "drafts"
            temporary = drafts / "working" / f"v{request.source_version:06d}"
            version_path = drafts / "versions" / f"v{request.source_version:06d}"
            try:
                if temporary.exists():
                    shutil.rmtree(temporary)
                diagnostics = self.builder(
                    list(request.frames),
                    temporary,
                    source_version=request.source_version,
                    depth_backend=self.depth_backend,
                )
                with self._condition:
                    current_allowed = self._latest_requested.get(key) == request.source_version
                if current_allowed:
                    version_path.parent.mkdir(parents=True, exist_ok=True)
                    if version_path.exists():
                        shutil.rmtree(version_path)
                    os.replace(temporary, version_path)
                    cloud = diagnostics["cloud"]
                    event = {
                        "schema": DRAFT_EVENT_SCHEMA,
                        "state": "ready",
                        "version": request.source_version,
                        "source_panorama_version": request.source_version,
                        "metadata_name": "draft.json",
                        "asset_name": "draft.bin.gz",
                        "asset_sha256": cloud["asset_sha256"],
                        "asset_bytes": cloud["asset_bytes"],
                        "point_count": cloud["count"],
                    }
                    _atomic_json(drafts / "events" / f"v{request.source_version:06d}.json", event)
                    _atomic_json(drafts / "current.json", event)
                elif temporary.exists():
                    shutil.rmtree(temporary)
            except Exception as error:
                _atomic_json(
                    drafts / "failures" / f"v{request.source_version:06d}.json",
                    {
                        "schema": DRAFT_EVENT_SCHEMA,
                        "state": "failed",
                        "version": request.source_version,
                        "error": str(error),
                    },
                )
                if temporary.exists():
                    shutil.rmtree(temporary)
            finally:
                with self._condition:
                    self._running = None
                    self._condition.notify_all()
