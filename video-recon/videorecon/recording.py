"""Read what the phone wrote: `meta.json`, `video.mp4`, `frames.csv`, `imu.csv`.

Layout of one recording directory (written by the Android app in this folder):

    meta.json    device, camera characteristics, intrinsics, timestamp clock
    video.mp4    H.264, sensor-native orientation, no stabilisation
    frames.csv   frame_number, sensor_timestamp_ns, exposure_ns, focus_diopters, iso
    imu.csv      timestamp_ns, sensor, x, y, z, w

Every timestamp is on the same monotonic clock (`SystemClock.elapsedRealtimeNanos`)
when the camera reports `SENSOR_INFO_TIMESTAMP_SOURCE == REALTIME`; the loader
refuses anything else rather than guessing at an offset.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from videorecon.imu import OrientationTrack

SENSOR_NAMES = ("acc", "gyr", "mag", "rotv", "gamerotv", "grav", "linacc")


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for an image of `width` x `height` pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    #: OpenCV order [k1, k2, p1, p2, k3]; empty when unknown.
    distortion: tuple[float, ...] = ()
    source: str = "unknown"

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def scaled(self, width: int, height: int) -> Intrinsics:
        """The same camera at another resolution, assuming a uniform resample."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(
            self.fx * sx,
            self.fy * sy,
            self.cx * sx,
            self.cy * sy,
            width,
            height,
            self.distortion,
            self.source,
        )


def intrinsics_from_meta(meta: dict) -> Intrinsics:
    """Video-frame intrinsics from the Camera2 characteristics the app recorded.

    `LENS_INTRINSIC_CALIBRATION` is expressed in pixels of the pre-correction
    active array. The video stream is the active array centre-cropped to the
    video aspect ratio and scaled, which is what every Android HAL does for a
    stream whose aspect differs from the sensor's. When the calibration is
    missing, the focal length falls back to `f_mm / sensor_width_mm * width_px`,
    which is right to a few percent; SfM refines it either way.
    """
    cam = meta["camera"]
    video_w, video_h = int(meta["video"]["width"]), int(meta["video"]["height"])
    array_w, array_h = int(cam["active_array"]["width"]), int(cam["active_array"]["height"])
    calib = cam.get("intrinsic_calibration")
    if calib and len(calib) >= 4 and calib[0] > 0:
        pre = cam.get("pre_correction_active_array") or cam["active_array"]
        base_w, base_h = int(pre["width"]), int(pre["height"])
        fx, fy, cx, cy = (float(v) for v in calib[:4])
        source = "LENS_INTRINSIC_CALIBRATION"
    else:
        f_mm = float(cam["focal_length_mm"])
        sensor_w_mm = float(cam["physical_size_mm"]["width"])
        base_w, base_h = array_w, array_h
        fx = fy = f_mm / sensor_w_mm * base_w
        cx, cy = base_w / 2.0, base_h / 2.0
        source = "focal_length_mm/physical_size"
    # Centre crop to the video aspect, then a uniform scale.
    video_aspect = video_w / video_h
    base_aspect = base_w / base_h
    if video_aspect >= base_aspect:
        crop_w, crop_h = base_w, base_w / video_aspect
    else:
        crop_w, crop_h = base_h * video_aspect, base_h
    off_x, off_y = (base_w - crop_w) / 2.0, (base_h - crop_h) / 2.0
    scale = video_w / crop_w
    # Android's LENS_DISTORTION is a *correction* in normalised coordinates, the
    # inverse of OpenCV's model; it is kept in meta.json for reference only.
    dist = tuple(float(v) for v in cam.get("distortion_opencv", []) or ())
    return Intrinsics(
        fx=fx * scale,
        fy=fy * scale,
        cx=(cx - off_x) * scale,
        cy=(cy - off_y) * scale,
        width=video_w,
        height=video_h,
        distortion=dist,
        source=source,
    )


@dataclass
class Recording:
    root: Path
    meta: dict
    #: (N, 2): capture frame number, sensor timestamp ns — every frame the camera produced.
    frames: np.ndarray
    imu: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def load(cls, root: str | Path) -> Recording:
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text())
        source = meta["camera"].get("timestamp_source")
        if source != "REALTIME":
            raise ValueError(
                f"camera timestamps are on clock {source!r}, not REALTIME; the IMU clock "
                "cannot be matched to the video without a calibrated offset"
            )
        frames = _read_frames_csv(root / "frames.csv")
        imu = _read_imu_csv(root / "imu.csv")
        return cls(root, meta, frames, imu)

    @property
    def video_path(self) -> Path:
        return self.root / "video.mp4"

    @property
    def sensor_orientation_deg(self) -> int:
        return int(self.meta["camera"]["sensor_orientation_deg"])

    def intrinsics(self) -> Intrinsics:
        return intrinsics_from_meta(self.meta)

    def orientation_track(self, sensor: str = "gamerotv") -> OrientationTrack:
        """Orientation over time. `gamerotv` (no magnetometer) is smooth and
        drift-free in pitch/roll; `rotv` adds the magnetometer for absolute yaw."""
        if sensor not in self.imu:
            available = sorted(self.imu)
            raise KeyError(f"no {sensor!r} samples in imu.csv; have {available}")
        t, v = self.imu[sensor]
        return OrientationTrack.from_samples(t, v[:, :4])

    def has_sensor(self, sensor: str) -> bool:
        return sensor in self.imu and len(self.imu[sensor][0]) > 1

    def sensor_rates_hz(self) -> dict[str, float]:
        out = {}
        for name, (t, _) in self.imu.items():
            if len(t) > 1:
                span = (t[-1] - t[0]) * 1e-9
                out[name] = float((len(t) - 1) / span) if span > 0 else 0.0
        return out


def _read_frames_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["frame_number"]), int(row["sensor_timestamp_ns"])))
    if not rows:
        raise ValueError(f"{path} has no frames")
    arr = np.asarray(rows, dtype=np.int64)
    return arr[np.argsort(arr[:, 1], kind="stable")]


def _read_imu_csv(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    buckets: dict[str, list[tuple[int, float, float, float, float]]] = {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return {}
        for row in reader:
            if len(row) < 5:
                continue
            w = float(row[5]) if len(row) > 5 and row[5] != "" else 0.0
            buckets.setdefault(row[1], []).append(
                (int(row[0]), float(row[2]), float(row[3]), float(row[4]), w)
            )
    out = {}
    for name, rows in buckets.items():
        arr = np.asarray(rows, dtype=np.float64)
        order = np.argsort(arr[:, 0], kind="stable")
        arr = arr[order]
        out[name] = (arr[:, 0].astype(np.int64), arr[:, 1:5])
    return out


# --------------------------------------------------------------------------
# Video frame timing
# --------------------------------------------------------------------------


def video_frame_pts_ns(video: Path) -> np.ndarray:
    """Presentation timestamp of every video frame, in ns from the first frame.

    `ffprobe` reads the container's own timestamps, which the muxer derived from
    the camera's sensor timestamps; the first frame is normalised to zero.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pts_time,best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(video),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    pts = []
    for line in out.splitlines():
        parts = [p for p in line.strip().split(",") if p not in ("", "N/A")]
        if parts:
            pts.append(float(parts[0]))
    if not pts:
        raise ValueError(f"ffprobe found no video frames in {video}")
    pts_ns = np.round(np.asarray(pts, dtype=np.float64) * 1e9).astype(np.int64)
    return pts_ns - pts_ns[0]


@dataclass(frozen=True)
class FrameTiming:
    """Which sensor timestamp each video frame was captured at."""

    #: (V,) sensor timestamp for each video frame index.
    timestamps_ns: np.ndarray
    #: (V,) residual between the container PTS and the matched capture result.
    residual_ns: np.ndarray
    matched_fraction: float
    frame_interval_ns: float


def match_video_to_capture_results(pts_ns: np.ndarray, capture_ts_ns: np.ndarray) -> FrameTiming:
    """Pair container PTS with the capture-result timestamps the app logged.

    Both are the sensor clock, offset by the first *recorded* frame. The
    recorder may drop the odd frame, so the pairing is nearest-neighbour with
    the offset fitted robustly rather than assumed equal to the first log entry.
    """
    pts_ns = np.asarray(pts_ns, dtype=np.int64)
    cap = np.sort(np.asarray(capture_ts_ns, dtype=np.int64))
    if len(cap) < 2 or len(pts_ns) < 1:
        raise ValueError("need at least two capture results and one video frame")
    interval = float(np.median(np.diff(cap)))
    # Candidate offsets: the first few logged frames could be the first encoded one.
    best = None
    for start in cap[: min(len(cap), 30)]:
        candidate = pts_ns + start
        idx = np.searchsorted(cap, candidate)
        idx = np.clip(idx, 1, len(cap) - 1)
        left, right = cap[idx - 1], cap[idx]
        nearest = np.where(np.abs(candidate - left) <= np.abs(candidate - right), left, right)
        residual = candidate - nearest
        matched = np.abs(residual) <= 0.25 * interval
        score = (int(matched.sum()), -float(np.median(np.abs(residual))))
        if best is None or score > best[0]:
            best = (score, nearest, residual, matched)
    _, nearest, residual, matched = best
    return FrameTiming(
        timestamps_ns=nearest.astype(np.int64),
        residual_ns=residual.astype(np.int64),
        matched_fraction=float(matched.mean()),
        frame_interval_ns=interval,
    )
