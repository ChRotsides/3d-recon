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

from recon.schemas import CANONICAL_FRAME, CaptureMode, SessionBundle

#: Plan §7 hard cap, and the cost control.
MAX_KEYFRAMES = 150
#: Below this a room is very unlikely to reconstruct.
MIN_KEYFRAMES = 24

#: Pitch above which a frame is credited as having seen the ceiling.
#:
#: Was 10 degrees, which is not a ceiling shot: from 1.5 m eye height the
#: centre ray of a 10-degree look-up meets a 2.5 m ceiling (2.5-1.5)/tan(10)
#: = 5.7 m away, further than most rooms are wide, so the frame images upper
#: wall. At 25 degrees the same ray lands at 2.1 m, inside the room.
CEILING_PITCH_DEG = 25.0

#: Frames a yaw sector needs before its support counts as real rather than
#: incidental. Guards the sector counts against a few stray frames spreading
#: themselves thinly enough to satisfy a coverage requirement.
MIN_FRAMES_PER_SECTOR = 3

#: Ceiling shots that may be abandoned before the height is unmeasurable. The
#: script asks for one per direction, so a couple of blocked corners is a
#: furnished room; most of them missing is a capture with no ceiling in it.
CEILING_SKIPS_BEFORE_ERROR = 4


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)
    schema_valid: bool = True

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def reconstruction_ready(self) -> bool:
        return self.schema_valid and not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_bundle(root: Path) -> Report:
    r = Report()
    root = Path(root)
    path = root / "bundle.json" if root.is_dir() else root
    if not path.exists():
        r.schema_valid = False
        r.error(f"no bundle.json at {path}")
        return r
    root = path.parent

    try:
        b = SessionBundle.model_validate_json(path.read_text())
    except Exception as exc:  # noqa: BLE001 - report, don't raise
        r.schema_valid = False
        r.error(f"schema validation failed: {exc}")
        return r

    _check_frame(b, r)
    _check_keyframes(b, root, r)
    _check_poses(b, r)
    _check_measurements(b, r)
    _check_skipped_shots(b, r)
    _check_capture_mode(b, root, r)

    r.stats.update(
        session_id=b.session_id,
        room_id=b.room_id,
        device=f"{b.device.model} / {b.device.os}",
        platform=b.device.ar_platform.value,
        capture_mode=b.capture_mode.value,
        keyframes=len(b.keyframes),
        sparse_points=len(b.sparse_points),
        measurements=len(b.reference_measurements),
        schema_valid=r.schema_valid,
        reconstruction_ready=r.reconstruction_ready,
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
        r.error(
            f"only {n} keyframes; at least {MIN_KEYFRAMES} are required for "
            "angular overlap and multiframe surface support"
        )

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

    relocalizing = [
        k.id for k in b.keyframes if k.tracking_state.value == "relocalizing"
    ]
    if relocalizing:
        r.error(
            f"{len(relocalizing)} tracking discontinuities are tagged at "
            f"{', '.join(relocalizing)}; split the capture before reconstruction"
        )
    limited = [k.id for k in b.keyframes if k.tracking_state.value == "limited"]
    if limited:
        r.warn(f"{len(limited)} keyframes captured with limited tracking")

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

    horizontal = pos[:, [0, 2]]
    max_horizontal = float(
        np.sqrt(np.max(np.sum((horizontal[:, None] - horizontal[None, :]) ** 2, axis=2)))
    )
    r.stats["max_horizontal_baseline_m"] = round(max_horizontal, 2)
    if max_horizontal < 0.65:
        r.error(
            f"maximum horizontal baseline is only {max_horizontal:.2f} m; guided "
            "targets were completed without a useful translated station"
        )

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
    if pitch.max() < CEILING_PITCH_DEG:
        r.warn("the camera never tilted up; the ceiling will be missing")
    if pitch.min() > -10:
        r.warn("the camera never tilted down; the floor will be missing, and the "
               "floor plan depends on it")

    if b.capture_mode != CaptureMode.UNKNOWN:
        surface_masks = {
            "floor": pitch <= -10.0,
            "walls": np.abs(pitch) < 30.0,
            "ceiling": pitch >= CEILING_PITCH_DEG,
        }
        minimum = {"floor": 4, "walls": 8, "ceiling": 4}
        for surface, mask in surface_masks.items():
            sectors = np.zeros(12, dtype=bool)
            sectors[(yaw[mask] / 30.0).astype(int) % 12] = True
            count = int(sectors.sum())
            r.stats[f"{surface}_yaw_sectors"] = count
            r.stats[f"{surface}_frames"] = int(mask.sum())
            if count < minimum[surface]:
                r.error(
                    f"{surface} has support in only {count}/12 yaw sectors; "
                    f"at least {minimum[surface]} are required"
                )
            # Sector spread alone is satisfiable by a handful of stray frames.
            # A real capture of bundle_5af1214c painted 8 of 12 ceiling sectors
            # with 8 frames out of 150, passed this gate, and then reconstructed
            # a tabletop as its ceiling. Density is the other half of the test:
            # a sector's worth of evidence, not one frame that glanced through it.
            elif mask.sum() < count * MIN_FRAMES_PER_SECTOR:
                r.error(
                    f"{surface} has only {int(mask.sum())} frames spread over "
                    f"{count} yaw sectors; at least {MIN_FRAMES_PER_SECTOR} per "
                    "sector are required for the support to be real"
                )

    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    if len(steps) and steps.max() > 2.5:
        r.error(
            f"a {steps.max():.1f} m jump between consecutive keyframes is a tracking "
            "break; split the capture before reconstruction"
        )


def _check_skipped_shots(b: SessionBundle, r: Report) -> None:
    """Report what the operator gave up on.

    A skip is a fact about the room, not a fault: some corners genuinely cannot
    be shot. What it must not do is pass unmentioned, because every skip is a
    direction the reconstruction will be guessing at, and a reader comparing two
    bundles has no other way to tell which one was actually captured.
    """
    skipped = b.skipped_shots
    r.stats["skipped_shots"] = len(skipped)
    if not skipped:
        return

    by_height: dict[str, int] = {}
    for s in skipped:
        by_height[s.height.value] = by_height.get(s.height.value, 0) + 1
    r.stats["skipped_by_height"] = by_height

    detail = ", ".join(f"{n} {h}" for h, n in sorted(by_height.items()))
    r.warn(f"{len(skipped)} shots reported unobtainable ({detail})")

    # The ceiling is the one users skip unprompted, and it is also the one the
    # layout cannot infer from anything else: floor and walls are constrained
    # by the footprint, ceiling height is only ever measured.
    if by_height.get("top", 0) >= CEILING_SKIPS_BEFORE_ERROR:
        r.error(
            f"{by_height['top']} ceiling shots were skipped; ceiling height "
            "cannot be measured from the remaining views"
        )

    at_station = sum(1 for s in skipped if s.at_station)
    if at_station:
        r.warn(
            f"{at_station} of them were station shots, which is where the "
            "capture's baseline comes from"
        )


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


def _check_capture_mode(b: SessionBundle, root: Path, r: Report) -> None:
    if b.capture_mode == CaptureMode.UNKNOWN:
        message = (
            "legacy bundle has no capture_mode; schema-valid, but depth provenance "
            "cannot be established"
        )
        if b.device.ar_platform.value == "synthetic":
            r.warn(message)
        else:
            r.error(message)
        return
    if b.capture_mode == CaptureMode.RGB_ONLY:
        r.warn(
            "capture is explicitly RGB-only; evaluate it against RGB-only criteria, "
            "not the production depth path"
        )
        return
    if b.capture_mode == CaptureMode.SYNTHETIC_DEPTH:
        return

    self_test = root / b.depth_self_test if b.depth_self_test else None
    if self_test is None or not self_test.exists():
        r.error("ARCore depth capture has no successful startup depth self-test PNG")
    else:
        try:
            from PIL import Image

            proof = np.asarray(Image.open(self_test))
            if proof.dtype != np.uint16 or proof.ndim != 2 or not np.any(proof > 0):
                r.error("startup depth self-test is not a non-empty 16-bit grayscale PNG")
        except Exception as exc:  # noqa: BLE001 - validation report, not a crash
            r.error(f"startup depth self-test PNG cannot be decoded: {exc}")
    depth_keyframes = [kf for kf in b.keyframes if kf.depth and (root / kf.depth).exists()]
    r.stats["depth_keyframes"] = len(depth_keyframes)
    if b.depth_frame_count != len(depth_keyframes):
        r.error(
            f"bundle reports {b.depth_frame_count} depth frames but contains "
            f"{len(depth_keyframes)} readable depth PNG paths"
        )
    if len(depth_keyframes) != len(b.keyframes):
        r.error(
            f"ARCore depth capture has depth for {len(depth_keyframes)}/{len(b.keyframes)} "
            "keyframes"
        )
