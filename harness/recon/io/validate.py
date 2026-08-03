"""Session-bundle conformance checks.

The pose-logging clients (plan task 3) exist to unblock real data collection.
That only works if a bundle that came off a phone is known-good *before* someone
drives to a house and captures fifteen rooms with it. This module is the
contract test: schema, frame, poses, intrinsics, files, and capture health.

Run it against a bundle straight off a device:

    recon validate /path/to/bundle_<uuid>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recon.schemas import CANONICAL_FRAME, SessionBundle

#: Plan §7 hard cap, and the cost control.
MAX_KEYFRAMES = 150
#: Below this a room is very unlikely to reconstruct.
MIN_KEYFRAMES = 20


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_bundle(root: Path) -> Report:
    r = Report()
    root = Path(root)
    path = root / "bundle.json" if root.is_dir() else root
    if not path.exists():
        r.error(f"no bundle.json at {path}")
        return r
    root = path.parent

    try:
        b = SessionBundle.model_validate_json(path.read_text())
    except Exception as exc:  # noqa: BLE001 - report, don't raise
        r.error(f"schema validation failed: {exc}")
        return r

    _check_frame(b, r)
    _check_keyframes(b, root, r)
    _check_poses(b, r)
    _check_measurements(b, r)

    r.stats.update(
        session_id=b.session_id,
        room_id=b.room_id,
        device=f"{b.device.model} / {b.device.os}",
        platform=b.device.ar_platform.value,
        keyframes=len(b.keyframes),
        sparse_points=len(b.sparse_points),
        measurements=len(b.reference_measurements),
    )
    return r


def _check_frame(b: SessionBundle, r: Report) -> None:
    if not b.frame.matches(CANONICAL_FRAME):
        r.error(
            f"bundle frame {b.frame.name!r} is not canonical "
            f"({b.frame.handedness.value}-handed, {b.frame.up_axis.value}, "
            f"camera {b.frame.camera_forward}). Convert at the client boundary."
        )
    g = np.asarray(b.gravity, dtype=float)
    up = -g / max(np.linalg.norm(g), 1e-9)
    if abs(float(up @ np.array([0.0, 1.0, 0.0])) - 1.0) > 0.05:
        r.warn(
            f"gravity {b.gravity} is not -Y. Legal, but every other component "
            "assumes a gravity-aligned session; check the AR world alignment."
        )


def _check_keyframes(b: SessionBundle, root: Path, r: Report) -> None:
    n = len(b.keyframes)
    if n == 0:
        r.error("bundle has no keyframes")
        return
    if n > MAX_KEYFRAMES:
        r.error(f"{n} keyframes exceeds the {MAX_KEYFRAMES} cap (plan §7 cost control)")
    elif n < MIN_KEYFRAMES:
        r.warn(f"only {n} keyframes; reconstruction quality will be poor")

    ids = [k.id for k in b.keyframes]
    if len(set(ids)) != len(ids):
        r.error("duplicate keyframe ids")

    missing = [k.id for k in b.keyframes if not (root / k.image).exists()]
    if missing:
        r.error(f"{len(missing)} keyframe images missing, e.g. {missing[:3]}")

    for k in b.keyframes:
        i = k.intrinsics
        if i.fx <= 0 or i.fy <= 0:
            r.error(f"{k.id}: non-positive focal length")
        if not (0 < i.cx < i.width and 0 < i.cy < i.height):
            r.error(f"{k.id}: principal point ({i.cx}, {i.cy}) outside the image")
        # A principal point far from centre usually means intrinsics were not
        # rescaled after downscaling the image — the classic client bug.
        if abs(i.cx / i.width - 0.5) > 0.25 or abs(i.cy / i.height - 0.5) > 0.25:
            r.warn(
                f"{k.id}: principal point is far off-centre; were intrinsics "
                "rescaled to match the saved image size?"
            )
        if i.fx / i.width < 0.3 or i.fx / i.width > 3.0:
            r.warn(f"{k.id}: fx/width = {i.fx/i.width:.2f} is outside the plausible range")

    if len({(k.intrinsics.width, k.intrinsics.height) for k in b.keyframes}) > 1:
        r.warn("keyframes have mixed resolutions; supported, but unusual for one session")

    bad = [k.id for k in b.keyframes if k.tracking_state.value != "normal"]
    if bad:
        r.warn(f"{len(bad)} keyframes captured with degraded tracking")

    blur = np.array([k.blur_score for k in b.keyframes])
    if (blur <= 0).all():
        r.warn("all blur scores are zero; the client is not scoring sharpness")
    r.stats["blur_median"] = float(np.median(blur))

    ts = np.array([k.timestamp_ns for k in b.keyframes], dtype=np.int64)
    if (np.diff(ts) <= 0).any():
        r.warn("keyframe timestamps are not strictly increasing")
    span_s = float(ts.max() - ts.min()) / 1e9 if len(ts) > 1 else 0.0
    r.stats["capture_seconds"] = round(span_s, 1)
    if span_s > 300:
        r.warn(f"capture spans {span_s/60:.1f} minutes; drift risk is high")


def _check_poses(b: SessionBundle, r: Report) -> None:
    """Baseline and rotation coverage — the two things that decide whether a
    capture can reconstruct at all."""
    if len(b.keyframes) < 2:
        return
    pos = np.array([k.pose.position for k in b.keyframes])
    fwd = np.array([k.pose.forward for k in b.keyframes])

    extent = pos.max(axis=0) - pos.min(axis=0)
    baseline = float(np.linalg.norm(extent))
    r.stats["camera_extent_m"] = [round(float(x), 2) for x in extent]
    if baseline < 0.5:
        r.error(
            f"cameras span only {baseline:.2f} m — a pure rotation has no parallax "
            "and will not reconstruct. The user needs to walk around."
        )
    elif baseline < 1.5:
        r.warn(f"cameras span only {baseline:.2f} m; expect weak geometry")

    # Yaw coverage: a sweep that never turns around leaves whole walls unseen.
    yaw = np.degrees(np.arctan2(fwd[:, 0], -fwd[:, 2])) % 360.0
    occupied = np.zeros(12, dtype=bool)
    occupied[(yaw / 30.0).astype(int) % 12] = True
    r.stats["yaw_sectors_seen"] = int(occupied.sum())
    if occupied.sum() < 6:
        r.warn(
            f"only {occupied.sum()}/12 yaw sectors observed; parts of the room were "
            "never faced"
        )

    pitch = np.degrees(np.arcsin(np.clip(fwd[:, 1], -1, 1)))
    r.stats["pitch_range_deg"] = [round(float(pitch.min()), 1), round(float(pitch.max()), 1)]
    if pitch.max() < 10:
        r.warn("the camera never tilted up; the ceiling will be missing")
    if pitch.min() > -10:
        r.warn("the camera never tilted down; the floor will be missing, and the "
               "floor plan depends on it")

    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    if len(steps) and steps.max() > 2.0:
        r.warn(f"a {steps.max():.1f} m jump between consecutive keyframes suggests "
               "a tracking break")


def _check_measurements(b: SessionBundle, r: Report) -> None:
    kinds = {m.type.value for m in b.reference_measurements}
    if "vio_scale" not in kinds:
        r.error(
            "no VIO scale prior. Plan §6.2 requires it as the anchor that keeps a "
            "bad tape entry from rescaling the room; the solver refuses without it."
        )
    if not (kinds - {"vio_scale"}):
        r.warn("no reference measurement beyond the VIO prior; expect ~3-5% scale error")
    for m in b.reference_measurements:
        if m.measured_value_m <= 0:
            r.error(f"{m.type.value}: non-positive measurement {m.measured_value_m}")
        if m.type.value == "user_tape" and not (0.2 <= m.measured_value_m <= 15.0):
            r.error(
                f"user tape reading {m.measured_value_m} m is implausible — a units "
                "mistake (cm entered as m?) would rescale the whole room"
            )
