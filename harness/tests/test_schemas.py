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
