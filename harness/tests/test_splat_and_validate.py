"""Tests for the GPU-free parts of the splat path and the bundle validator."""

from __future__ import annotations

import numpy as np
import pytest
from recon import synth
from recon.depth.base import GroundTruthDepth
from recon.io.bundle import BundleDir
from recon.io.validate import validate_bundle
from recon.masking.base import GroundTruthMasker
from recon.schemas import Intrinsics, Keyframe, Pose, SessionBundle
from recon.splat.base import holdout_split, seed_points_from_depth
from recon.splat.pgsr import PGSRTrainer, _mat_to_quat_wxyz


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("splat")
    # Two stations, not one: a single station has zero parallax and the
    # validator correctly refuses it.
    cfg = synth.CaptureConfig(width=96, height=72, n_stations=2, yaw_step_deg=60,
                              pitches_deg=(0.0, -20.0), max_keyframes=24)
    synth.generate(synth.preset("empty_rect"), d / "s", cfg)
    return BundleDir.load(d / "s")


def _kf(i: int) -> Keyframe:
    return Keyframe(
        id=f"kf_{i:04d}", image="x.png", timestamp_ns=i,
        pose=Pose(R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], t=[0, 0, 0]),
        intrinsics=Intrinsics(fx=1, fy=1, cx=1, cy=1, width=2, height=2),
    )


def test_holdout_split_is_strided_not_random():
    """A random split leaves each test frame between two near-identical training
    frames, so 'novel view' PSNR measures nothing."""
    kfs = [_kf(i) for i in range(32)]
    train, test = holdout_split(kfs, every=8)
    assert len(train) + len(test) == 32
    assert len(test) == 4
    idx = sorted(int(k.id.split("_")[1]) for k in test)
    assert np.all(np.diff(idx) == 8), idx
    assert not set(k.id for k in train) & set(k.id for k in test)


def test_seed_points_land_on_the_room_surface(scene):
    depth = GroundTruthDepth().estimate(scene)
    masks = GroundTruthMasker().mask(scene)
    pts, cols = seed_points_from_depth(scene, depth, masks, max_points=20_000)

    assert len(pts) > 1000
    assert len(pts) == len(cols)
    gt = scene.ground_truth()
    poly = np.asarray(gt["footprint"])
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    # Every seed point should be inside the room's bounding box, with slack for
    # the wall surface itself.
    assert pts[:, 0].min() > lo[0] - 0.1 and pts[:, 0].max() < hi[0] + 0.1
    assert pts[:, 2].min() > lo[1] - 0.1 and pts[:, 2].max() < hi[1] + 0.1
    assert pts[:, 1].min() > -0.1
    assert pts[:, 1].max() < gt["ceiling_height_m"] + 0.1
    assert (cols >= 0).all() and (cols <= 1).all()


def test_seed_points_respect_the_scale_correction(scene):
    depth = GroundTruthDepth().estimate(scene)
    masks = GroundTruthMasker().mask(scene)
    a, _ = seed_points_from_depth(scene, depth, masks, scale=1.0, max_points=5_000, seed=1)
    b, _ = seed_points_from_depth(scene, depth, masks, scale=2.0, max_points=5_000, seed=1)
    assert np.allclose(b, a * 2.0, atol=1e-5)


def test_pgsr_prepare_produces_a_usable_starting_state(scene):
    depth = GroundTruthDepth().estimate(scene)
    masks = GroundTruthMasker().mask(scene)
    prep = PGSRTrainer().prepare(scene, depth, masks)

    assert prep["train_keyframes"] and prep["holdout_keyframes"]
    assert not set(prep["train_keyframes"]) & set(prep["holdout_keyframes"])
    assert len(prep["seed_points"]) > 0
    for kf_id, w in prep["geometric_weight"].items():
        assert w.min() >= 0.0 and w.max() <= 1.0, kf_id


def test_pgsr_train_refuses_rather_than_faking_a_result(scene):
    """A benchmark row filled in by never-executed code is worse than a gap."""
    depth = GroundTruthDepth().estimate(scene)
    masks = GroundTruthMasker().mask(scene)
    with pytest.raises((NotImplementedError, RuntimeError)):
        PGSRTrainer().train(scene, depth, masks)


def test_quaternion_conversion_round_trips():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(size=(3, 3))
        q_, _ = np.linalg.qr(a)
        if np.linalg.det(q_) < 0:
            q_[:, 0] *= -1
        w, x, y, z = _mat_to_quat_wxyz(q_)
        assert abs(np.linalg.norm([w, x, y, z]) - 1.0) < 1e-9
        back = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        assert np.allclose(back, q_, atol=1e-8)


def test_quaternion_handles_trace_near_minus_one():
    """A 180 degree rotation is where the naive formula divides by zero."""
    r = np.diag([1.0, -1.0, -1.0])
    w, x, y, z = _mat_to_quat_wxyz(r)
    assert abs(np.linalg.norm([w, x, y, z]) - 1.0) < 1e-9
    assert abs(w) < 1e-9 and abs(abs(x) - 1.0) < 1e-9


# -- validator --------------------------------------------------------------


def test_validator_accepts_a_good_bundle(scene):
    rep = validate_bundle(scene.root)
    assert rep.ok, rep.errors
    assert rep.stats["keyframes"] > 0


def test_validator_rejects_a_pure_rotation_capture(tmp_path):
    """No parallax, no reconstruction — this must be caught before someone
    drives to a house and captures fifteen rooms this way."""
    b = SessionBundle.model_validate_json(
        (
            SessionBundle(
                session_id="s", room_id="r",
                device={"model": "m", "os": "o", "ar_platform": "synthetic"},
                keyframes=[_kf(i) for i in range(30)],
                reference_measurements=[
                    {"type": "vio_scale", "measured_value_m": 1.0, "sigma_m": 0.03}
                ],
            )
        ).model_dump_json()
    )
    (tmp_path / "bundle.json").write_text(b.model_dump_json())
    (tmp_path / "images").mkdir()
    for kf in b.keyframes:
        (tmp_path / kf.image).write_bytes(b"")
    rep = validate_bundle(tmp_path)
    assert not rep.ok
    assert any("parallax" in e for e in rep.errors), rep.errors


def test_validator_requires_the_vio_prior(tmp_path):
    b = SessionBundle(
        session_id="s", room_id="r",
        device={"model": "m", "os": "o", "ar_platform": "synthetic"},
    )
    (tmp_path / "bundle.json").write_text(b.model_dump_json())
    rep = validate_bundle(tmp_path)
    assert any("VIO scale prior" in e for e in rep.errors), rep.errors


def test_validator_catches_a_units_mistake_in_the_tape_measure(scene):
    """0.83 m typed as 83 would rescale the whole house."""
    import copy

    bundle = copy.deepcopy(scene.bundle)
    bundle.reference_measurements.append(
        type(bundle.reference_measurements[0])(
            type="user_tape", point_a=[0, 0, 0], point_b=[1, 0, 0],
            measured_value_m=83.0, sigma_m=0.005,
        )
    )
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path

        p = Path(d)
        (p / "bundle.json").write_text(bundle.model_dump_json())
        (p / "images").mkdir()
        for kf in bundle.keyframes:
            (p / kf.image).write_bytes(b"")
        rep = validate_bundle(p)
    assert any("implausible" in e for e in rep.errors), rep.errors
