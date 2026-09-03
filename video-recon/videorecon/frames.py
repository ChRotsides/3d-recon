"""Turn a video into the still images the rest of the pipeline wants.

Keyframes are chosen by *orientation change* measured by the sensors, not by
wall-clock: a frame is due when the camera has turned by `min_rotation_deg`
since the last keyframe, or `max_interval_s` has elapsed (which is what catches
pure translation, since the IMU cannot see it). Within the few frames around a
due point the sharpest one wins, so motion blur from a hand-held walk does not
end up in the reconstruction.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from videorecon.recording import Intrinsics


@dataclass(frozen=True)
class KeyframeSelection:
    #: Video frame indices chosen, ascending.
    indices: np.ndarray
    #: Blur score (variance of the Laplacian on the downscaled grey image) per chosen frame.
    blur: np.ndarray
    #: Why each frame was due: "rotation", "interval" or "first".
    reasons: tuple[str, ...]


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_keyframes(
    frame_times_ns: np.ndarray,
    r_wc: np.ndarray,
    blur: np.ndarray,
    *,
    min_rotation_deg: float = 8.0,
    max_interval_s: float = 0.6,
    sharpest_window: int = 3,
    min_blur: float = 0.0,
) -> KeyframeSelection:
    """Pick keyframe indices from per-frame times, camera rotations and blur.

    `r_wc` is (V, 3, 3), any consistent frame. Returned indices are unique and
    ascending. The sharpest-of-window rule may pull a keyframe a few frames
    earlier or later than the due point; that is intended.
    """
    n = len(frame_times_ns)
    if n == 0:
        return KeyframeSelection(np.zeros(0, dtype=np.int64), np.zeros(0), ())
    rot = Rotation.from_matrix(r_wc)
    chosen: list[int] = []
    reasons: list[str] = []
    last_index = -1
    for i in range(n):
        if last_index < 0:
            due, why = True, "first"
        else:
            angle = np.rad2deg((rot[last_index].inv() * rot[i]).magnitude())
            elapsed = (frame_times_ns[i] - frame_times_ns[last_index]) * 1e-9
            if angle >= min_rotation_deg:
                due, why = True, "rotation"
            elif elapsed >= max_interval_s:
                due, why = True, "interval"
            else:
                due, why = False, ""
        if not due:
            continue
        lo, hi = max(0, i - sharpest_window), min(n, i + sharpest_window + 1)
        if chosen:
            lo = max(lo, chosen[-1] + 1)
        window = np.arange(lo, hi)
        if len(window) == 0:
            continue
        best = int(window[np.argmax(blur[window])])
        if blur[best] < min_blur:
            continue
        chosen.append(best)
        reasons.append(why)
        last_index = best
    idx = np.asarray(chosen, dtype=np.int64)
    return KeyframeSelection(idx, blur[idx] if len(idx) else np.zeros(0), tuple(reasons))


def _iter_video_frames(video: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    # Never let OpenCV apply the container's rotation flag: the recorder writes
    # none, and the intrinsics are for the sensor-native orientation.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


def video_frame_count(video: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "csv=p=0",
        str(video),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    return int(out.split(",")[0])


def score_all_frames(video: Path, *, analysis_width: int = 320) -> np.ndarray:
    """Blur score of every frame, on a downscaled grey copy. One decode pass."""
    scores = []
    for _, frame in _iter_video_frames(video):
        h, w = frame.shape[:2]
        scale = analysis_width / max(w, 1)
        small = cv2.resize(
            frame, (analysis_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        scores.append(blur_score(gray))
    return np.asarray(scores, dtype=np.float64)


@dataclass(frozen=True)
class ExtractedFrames:
    directory: Path
    #: Video frame index -> written file name.
    names: dict[int, str]
    intrinsics: Intrinsics
    width: int
    height: int


def extract_frames(
    video: Path,
    indices: np.ndarray,
    out_dir: Path,
    intrinsics: Intrinsics,
    *,
    max_long_edge: int = 1280,
    jpeg_quality: int = 94,
) -> ExtractedFrames:
    """Decode the chosen frames, downscale uniformly, write JPEGs, scale intrinsics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(int(i) for i in indices)
    names: dict[int, str] = {}
    width = height = None
    scaled = intrinsics
    for index, frame in _iter_video_frames(video):
        if index not in wanted:
            continue
        h, w = frame.shape[:2]
        if (w, h) != (intrinsics.width, intrinsics.height):
            raise ValueError(
                f"video frame is {w}x{h} but meta.json says {intrinsics.width}x{intrinsics.height}"
            )
        if max(w, h) > max_long_edge:
            s = max_long_edge / max(w, h)
            frame = cv2.resize(
                frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA
            )
        h, w = frame.shape[:2]
        if width is None:
            width, height = w, h
            scaled = intrinsics.scaled(w, h)
        name = f"frame_{index:06d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        names[index] = name
        if len(names) == len(wanted):
            break
    missing = wanted - set(names)
    if missing:
        raise ValueError(f"{len(missing)} requested frames were beyond the end of the video")
    if width is None:
        raise ValueError("no frames extracted")
    return ExtractedFrames(out_dir, names, scaled, width, height)


def undistort_frames(
    src_dir: Path, dst_dir: Path, intr: Intrinsics, names, *, jpeg_quality: int = 94
) -> Intrinsics:
    """Remap distorted keyframes to a pinhole camera, cropped to the valid area.

    Everything after SfM assumes a pinhole model. The OpenCV coefficients come
    from COLMAP's OPENCV camera, in the same [k1, k2, p1, p2] order.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    k = intr.matrix()
    dist = np.asarray(list(intr.distortion) + [0.0] * 5, dtype=np.float64)[:5]
    size = (intr.width, intr.height)
    new_k, roi = cv2.getOptimalNewCameraMatrix(k, dist, size, 0, size)
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        new_k, (x, y, w, h) = k, (0, 0, intr.width, intr.height)
    map1, map2 = cv2.initUndistortRectifyMap(k, dist, None, new_k, size, cv2.CV_32FC1)
    for name in names:
        out = dst_dir / name
        if out.exists():
            continue
        img = cv2.imread(str(src_dir / name))
        if img is None:
            raise FileNotFoundError(src_dir / name)
        fixed = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)[y : y + h, x : x + w]
        cv2.imwrite(str(out), fixed, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    return Intrinsics(
        fx=float(new_k[0, 0]),
        fy=float(new_k[1, 1]),
        cx=float(new_k[0, 2] - x),
        cy=float(new_k[1, 2] - y),
        width=int(w),
        height=int(h),
        distortion=(),
        source=intr.source + "+undistorted",
    )
