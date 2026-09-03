"""Dense depth for the keyframes, through the repo's existing DA3 adapter.

The metric, sensor-aligned poses are written as a `SessionBundle` — the wire
format the rest of the repository already reads — with COLMAP's sparse points
as the bundle's `sparse_points`. That makes `recon.depth.da3.DA3Depth` do the
right thing unchanged: pose-conditioned multi-view depth, then a per-frame
scale anchor against the (now metric) sparse points.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from videorecon.imu import CV_TO_GL
from videorecon.recording import Intrinsics

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "harness"


def _ensure_harness_importable() -> None:
    try:
        import recon  # noqa: F401
    except ImportError:
        if HARNESS.is_dir():
            sys.path.insert(0, str(HARNESS))


@dataclass(frozen=True)
class BundleFrame:
    id: str
    image_name: str
    timestamp_ns: int
    #: world<-camera in the repo frame (Y up, camera looks -Z).
    rotation: np.ndarray
    translation: np.ndarray
    blur_score: float = 0.0
    heading_deg: float | None = None


def write_bundle(
    out_dir: Path,
    images_dir: Path,
    frames: list[BundleFrame],
    intrinsics: Intrinsics,
    points: np.ndarray,
    observations: list[list[str]],
    *,
    session_id: str,
    device: dict,
    extra: dict | None = None,
) -> Path:
    """Write `bundle.json` and copy the keyframe images next to it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / "images"
    img_out.mkdir(exist_ok=True)
    name_to_id = {f.image_name: f.id for f in frames}
    keyframes = []
    for f in frames:
        src = images_dir / f.image_name
        dst = img_out / f.image_name
        if not dst.exists():
            shutil.copyfile(src, dst)
        keyframes.append(
            {
                "id": f.id,
                "image": f"images/{f.image_name}",
                "timestamp_ns": int(f.timestamp_ns),
                "pose": {
                    "R": np.asarray(f.rotation).tolist(),
                    "t": np.asarray(f.translation).tolist(),
                },
                "intrinsics": {
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "cx": intrinsics.cx,
                    "cy": intrinsics.cy,
                    "width": intrinsics.width,
                    "height": intrinsics.height,
                    "distortion": list(intrinsics.distortion),
                },
                "tracking_state": "normal",
                "blur_score": float(f.blur_score),
                "exposure_clip_ratio": 0.0,
                "depth": None,
                "heading_deg": f.heading_deg,
            }
        )
    sparse = []
    for xyz, obs in zip(points, observations, strict=True):
        ids = [name_to_id[n] for n in obs if n in name_to_id]
        if len(ids) >= 2:
            sparse.append({"xyz": [float(v) for v in xyz], "observations": ids})
    bundle = {
        "session_id": session_id,
        "device": {
            "model": str(device.get("model", "unknown")),
            "os": str(device.get("os", "android")),
            # The schema's platform enum predates this path; the poses here come
            # from video SfM + IMU, which is recorded in `ar_version`.
            "ar_platform": "synthetic",
            "ar_version": "video-imu-sfm",
        },
        "room_id": session_id,
        "gravity": [0.0, -1.0, 0.0],
        "capture_mode": "rgb_only",
        "frame": {
            "handedness": "right",
            "up_axis": "y_up",
            "camera_forward": "-z",
            "name": "recon_world_v1",
        },
        "keyframes": keyframes,
        "sparse_points": sparse,
        "schema_version": 1,
    }
    (out_dir / "bundle.json").write_text(json.dumps(bundle, indent=1))
    if extra:
        (out_dir / "provenance.json").write_text(json.dumps(extra, indent=2, default=str))
    return out_dir


def run_da3(
    bundle_dir: Path,
    out_npz: Path,
    *,
    variant: str = "small",
    anchor_to_sparse_points: bool = True,
    window: int | None = None,
    process_res: int | None = None,
    log=print,
) -> dict:
    """Predict depth for every keyframe and cache it as `out_npz`.

    Returns the adapter's diagnostics. The cache format matches
    `scripts/capture/da3_cloud.py --depth-cache`: one array per keyframe id plus
    a `model` string.
    """
    _ensure_harness_importable()
    from recon.depth.da3 import DA3Config, DA3Depth
    from recon.io.bundle import BundleDir

    bundle = BundleDir.load(bundle_dir)
    kwargs = dict(
        variant=variant,
        anchor_to_sparse_points=anchor_to_sparse_points,
        anchor_to_reference_depth=False,
        enforce_commercial_licence=False,
    )
    if window is not None:
        kwargs["window"] = window
        kwargs["window_overlap"] = max(1, window // 4)
    if process_res is not None:
        kwargs["process_res"] = process_res
    estimator = DA3Depth(DA3Config(**kwargs))
    if window is not None:
        # The adapter clamps Kepler to 4-frame windows for FP32 attention memory;
        # DA3-small at 8 frames measured 2.2 GB of the 6 GB, so an explicit
        # request wins over the clamp.
        estimator._load()
        estimator._legacy_cuda = False
    log(
        f"depth: DA3-{variant} on {len(bundle.bundle.keyframes)} keyframes, "
        f"{len(bundle.bundle.sparse_points)} sparse anchor points"
    )
    result = estimator.estimate(bundle)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        kf.id: np.asarray(result.depth[kf.id], dtype=np.float32)
        for kf in bundle.bundle.keyframes
        if kf.id in result.depth
    }
    for kf in bundle.bundle.keyframes:
        if kf.id in result.confidence:
            arrays[f"conf_{kf.id}"] = np.asarray(result.confidence[kf.id], dtype=np.float32)
    np.savez_compressed(out_npz, model=np.array(result.model_name), **arrays)
    log(
        f"depth: {len(arrays)} maps in {result.runtime_s:.0f}s, "
        f"peak VRAM {result.peak_vram_gb:.1f} GB -> {out_npz}"
    )
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "model": result.model_name,
            "runtime_s": result.runtime_s,
            "peak_vram_gb": result.peak_vram_gb,
        }
    )
    return diagnostics


def _image_rotation_matrix(rotation_deg: int) -> np.ndarray:
    """New OpenCV camera axes expressed in the old ones, for a clockwise image rotation."""
    r = rotation_deg % 360
    if r == 90:  # u' = H-1-v, v' = u
        return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    if r == 270:  # u' = v, v' = W-1-u
        return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    if r == 180:
        return np.diag([-1.0, -1.0, 1.0])
    return np.eye(3)


def rotate_intrinsics(intr: Intrinsics, rotation_deg: int) -> Intrinsics:
    r = rotation_deg % 360
    w, h = intr.width, intr.height
    if r == 90:
        return Intrinsics(
            intr.fy, intr.fx, (h - 1) - intr.cy, intr.cx, h, w, intr.distortion, intr.source
        )
    if r == 270:
        return Intrinsics(
            intr.fy, intr.fx, intr.cy, (w - 1) - intr.cx, h, w, intr.distortion, intr.source
        )
    if r == 180:
        return Intrinsics(
            intr.fx,
            intr.fy,
            (w - 1) - intr.cx,
            (h - 1) - intr.cy,
            w,
            h,
            intr.distortion,
            intr.source,
        )
    return intr


def rotate_pose(rotation_gl: np.ndarray, rotation_deg: int) -> np.ndarray:
    """World<-camera rotation (GL convention) after rotating the image clockwise."""
    m = _image_rotation_matrix(rotation_deg)
    r_cv = np.asarray(rotation_gl) @ CV_TO_GL
    return r_cv @ m @ CV_TO_GL


def upright_for_depth(
    images_dir: Path,
    out_dir: Path,
    frames: list[BundleFrame],
    intr: Intrinsics,
    rotation_deg: int,
    *,
    jpeg_quality: int = 94,
) -> tuple[list[BundleFrame], Intrinsics]:
    """Rotate keyframes upright (by the sensor orientation) with matching intrinsics and poses.

    Measured on the reference walk: the depth model given sideways frames had a
    near/far slope of -0.38 and 17.5% of pixels disagreeing with a frame 20
    keyframes later by more than 25%; upright, -0.18 and 8.6%.
    """
    import cv2

    r = rotation_deg % 360
    if r == 0:
        return frames, intr
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[
        r
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    rotated = []
    for f in frames:
        dst = out_dir / f.image_name
        if not dst.exists():
            img = cv2.imread(str(images_dir / f.image_name))
            if img is None:
                raise FileNotFoundError(images_dir / f.image_name)
            cv2.imwrite(
                str(dst), cv2.rotate(img, code), [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
        rotated.append(
            BundleFrame(
                f.id,
                f.image_name,
                f.timestamp_ns,
                rotate_pose(f.rotation, r),
                f.translation,
                f.blur_score,
                f.heading_deg,
            )
        )
    return rotated, rotate_intrinsics(intr, r)
