import json

import numpy as np
import pytest
from pydantic import ValidationError
from recon.io.bundle import BundleDir, FrameMismatch
from recon.schemas import (
    ARPlatform,
    CoordinateFrame,
    Device,
    Handedness,
    Intrinsics,
    Keyframe,
    Pose,
    SessionBundle,
)


def _pose(yaw: float = 0.0) -> Pose:
    c, s = np.cos(yaw), np.sin(yaw)
    return Pose(R=[[c, 0, s], [0, 1, 0], [-s, 0, c]], t=[1.0, 1.5, 2.0])


def test_pose_rejects_non_rotation():
    with pytest.raises(ValidationError):
        Pose(R=[[2, 0, 0], [0, 1, 0], [0, 0, 1]], t=[0, 0, 0])


def test_pose_rejects_mirrored_rotation():
    """A left-handed rotation sneaking in is the flipped-scene bug (PLAN §7)."""
    with pytest.raises(ValidationError):
        Pose(R=[[-1, 0, 0], [0, 1, 0], [0, 0, 1]], t=[0, 0, 0])


def test_pose_inverse_round_trips():
    p = _pose(0.7)
    assert np.allclose(p.matrix() @ p.inverse_matrix(), np.eye(4), atol=1e-12)


def test_pose_forward_is_minus_z():
    p = _pose(0.0)
    assert np.allclose(p.forward, [0.0, 0.0, -1.0])


def test_gravity_must_be_unit():
    with pytest.raises(ValidationError):
        SessionBundle(
            session_id="s", room_id="r", gravity=[0.0, -2.0, 0.0],
            device=Device(model="m", os="o", ar_platform=ARPlatform.SYNTHETIC),
        )


def test_bundle_round_trips_through_json():
    b = SessionBundle(
        session_id="s", room_id="r",
        device=Device(model="Pixel 9", os="Android 15", ar_platform=ARPlatform.ARCORE),
        keyframes=[
            Keyframe(
                id="kf_0000", image="images/kf_0000.png", timestamp_ns=1,
                pose=_pose(0.3),
                intrinsics=Intrinsics(fx=500, fy=500, cx=160, cy=120, width=320, height=240),
            )
        ],
    )
    again = SessionBundle.model_validate_json(b.model_dump_json())
    assert again.keyframes[0].id == "kf_0000"
    assert again.device.ar_platform is ARPlatform.ARCORE


def test_extra_fields_are_rejected():
    """A client sending a field we don't know about should fail loudly, not
    have it silently dropped on the way into reconstruction."""
    payload = json.loads(
        SessionBundle(
            session_id="s", room_id="r",
            device=Device(model="m", os="o", ar_platform=ARPlatform.SYNTHETIC),
        ).model_dump_json()
    )
    payload["mystery_field"] = 1
    with pytest.raises(ValidationError):
        SessionBundle.model_validate(payload)


def test_left_handed_bundle_is_refused(tmp_path):
    b = SessionBundle(
        session_id="s", room_id="r",
        device=Device(model="m", os="o", ar_platform=ARPlatform.ARCORE),
        frame=CoordinateFrame(handedness=Handedness.LEFT),
    )
    (tmp_path / "bundle.json").write_text(b.model_dump_json())
    with pytest.raises(FrameMismatch):
        BundleDir.load(tmp_path)
    # ...but an explicit opt-out still lets tooling inspect it.
    assert BundleDir.load(tmp_path, strict_frame=False).bundle.session_id == "s"


def test_intrinsics_matrix():
    k = Intrinsics(fx=500, fy=501, cx=160, cy=120, width=320, height=240).matrix()
    assert k[0, 0] == 500 and k[1, 1] == 501 and k[0, 2] == 160 and k[2, 2] == 1


def test_heading_is_optional_so_older_bundles_still_load():
    """Every capture so far predates the compass field; none may become invalid."""
    from recon.schemas import Keyframe

    without = Keyframe.model_validate(
        {
            "id": "kf_0000",
            "image": "images/kf_0000.jpg",
            "timestamp_ns": 1,
            "pose": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0]},
            "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "width": 2, "height": 2},
        }
    )
    assert without.heading_deg is None

    with_heading = without.model_copy(update={"heading_deg": 187.5})
    assert with_heading.heading_deg == 187.5


# ---------------------------------------------------------------------------
# The footprint the user taps out during capture.
# ---------------------------------------------------------------------------


def test_floor_outline_area_and_perimeter():
    from recon.schemas import FloorOutline

    room = FloorOutline(corners=[[0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 0, 3]], closed=True)
    assert room.area_m2 == pytest.approx(12.0)
    assert room.perimeter_m == pytest.approx(14.0)


def test_floor_outline_handles_non_convex_rooms():
    """An L is a room. Only a *self-crossing* outline is a mis-tap."""
    from recon.schemas import FloorOutline

    ell = FloorOutline(
        corners=[[0, 0, 0], [4, 0, 0], [4, 0, 2], [2, 0, 2], [2, 0, 4], [0, 0, 4]],
        closed=True,
    )
    assert ell.area_m2 == pytest.approx(12.0)


def test_floor_outline_winding_does_not_flip_the_area():
    from recon.schemas import FloorOutline

    cw = FloorOutline(corners=[[0, 0, 0], [0, 0, 3], [4, 0, 3], [4, 0, 0]], closed=True)
    assert cw.area_m2 == pytest.approx(12.0)


def test_an_open_outline_bounds_no_area():
    """A trace the user abandoned must never be used as a room boundary."""
    from recon.schemas import FloorOutline

    partial = FloorOutline(corners=[[0, 0, 0], [4, 0, 0], [4, 0, 3]], closed=False)
    assert partial.area_m2 == 0.0
    assert partial.perimeter_m == pytest.approx(7.0)


def test_floor_outline_ignores_corner_height_when_measuring():
    """Skirting boards and sloping floors must not inflate the footprint."""
    from recon.schemas import FloorOutline

    uneven = FloorOutline(
        corners=[[0, 0.00, 0], [4, 0.04, 0], [4, -0.03, 3], [0, 0.02, 3]], closed=True
    )
    assert uneven.area_m2 == pytest.approx(12.0)


def test_floor_outline_keeps_reference_point_motion_for_later_correction():
    from recon.schemas import FloorOutline

    outline = FloorOutline.model_validate(
        {
            "corners": [[0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 0, 3]],
            "corner_ids": [1, 2, 3, 4],
            "closed": True,
            "reference_point_adjustments": [
                {
                    "corner_id": 2,
                    "from": [4, 0, 0],
                    "to": [4.25, 0, -0.1],
                    "delta": [0.25, 0, -0.1],
                    "started_at_ns": 10,
                    "ended_at_ns": 20,
                }
            ],
        }
    )
    adjustment = outline.reference_point_adjustments[0]
    assert adjustment.corner_id == 2
    assert adjustment.from_ == [4.0, 0.0, 0.0]
    assert adjustment.delta == [0.25, 0.0, -0.1]
    assert outline.corner_ids == [1, 2, 3, 4]


def test_reference_point_motion_cannot_end_before_it_started():
    from recon.schemas import ReferencePointAdjustment

    with pytest.raises(ValidationError, match="ends before"):
        ReferencePointAdjustment(
            corner_id=1,
            from_=[0, 0, 0],
            to=[1, 0, 0],
            delta=[1, 0, 0],
            started_at_ns=20,
            ended_at_ns=10,
        )


def test_a_two_corner_footprint_is_refused():
    from recon.schemas import FloorOutline

    with pytest.raises(ValidationError, match="needs 3 corners"):
        FloorOutline(corners=[[0, 0, 0], [1, 0, 0]], closed=True)


def test_bundles_without_a_footprint_still_load():
    """Every capture made before this feature existed; none may become invalid."""
    b = SessionBundle(
        session_id="s",
        room_id="r",
        device=Device(model="m", os="o", ar_platform=ARPlatform.SYNTHETIC),
    )
    assert b.floor_outline is None
