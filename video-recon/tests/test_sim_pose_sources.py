"""Pose sources: the ground-truth passthrough, the drifting odometry, and (if it was built)
the real OpenVINS stream."""

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sim import pose_sources
from sim.sensors import Camera, ImuSim
from sim.vehicle import Vehicle
from sim.world import World

VIO_BIN = Path(os.path.expanduser("~/ai/vio/open_vins/ov_msckf/build/vio_stream"))


def _track(seed=0, n=400, source=None, **kw):
    """Drive a short S and return (gt centres, estimated centres)."""
    world = World.generate(4, "open")
    cam = Camera()
    veh = Vehicle(world, world.start[0], world.start[1], 0.2, rate=200.0, seed=seed)
    src = source or pose_sources.make("gt_noisy", seed=seed, **kw)
    gt, est = [], []
    for k in range(n * 10):
        veh.step(0.5, 0.3 * np.sin(k / 400.0))
        if k % 10 == 0:
            R_wc, c_w = cam.pose(*veh.body_pose())
            _, R_e, c_e, ok = src.update(k / 200.0, R_wc, c_w)
            gt.append(c_w)
            est.append(c_e.copy())
    return np.asarray(gt), np.asarray(est)


def test_gt_source_is_the_truth():
    src = pose_sources.make("gt")
    R = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()
    c = np.array([1.0, 2.0, 0.5])
    t, R2, c2, ok = src.update(3.0, R, c)
    assert ok and t == 3.0
    assert np.array_equal(R, R2) and np.array_equal(c, c2)


def test_gt_noisy_is_deterministic_per_seed():
    a = _track(seed=5)[1]
    b = _track(seed=5)[1]
    assert np.array_equal(a, b)
    c = _track(seed=6)[1]
    assert not np.allclose(a, c)


def test_gt_noisy_drift_grows_with_distance():
    gt, est = _track(seed=2, n=600)
    err = np.linalg.norm(est - gt, axis=1)
    travelled = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
    assert err[0] < 1e-9, "the first pose is the truth"
    q = len(err) // 4
    assert err[:q].mean() < err[-q:].mean(), "drift must accumulate"
    assert err[-1] > 0.02, f"a {travelled[-1]:.1f} m drive should drift more than 2 cm, got {err[-1]:.3f}"
    assert err[-1] < 0.35 * travelled[-1], "2 % odometry must not drift by a third of the path"


def test_gt_noisy_scale_error_shrinks_or_stretches_the_path():
    _, small = _track(seed=3, scale_error=1.0)
    _, big = _track(seed=3, scale_error=1.20)
    # the per-step noise inflates path length, so compare extents, which it does not bias
    ext_small = np.linalg.norm(small.max(0) - small.min(0))
    ext_big = np.linalg.norm(big.max(0) - big.min(0))
    assert 1.15 < ext_big / ext_small < 1.25


def test_pose_sources_share_one_interface():
    for kind in ("gt", "gt_noisy"):
        src = pose_sources.make(kind, seed=1)
        out = src.update(0.0, np.eye(3), np.zeros(3))
        assert len(out) == 4
        t, R, c, ok = out
        assert R.shape == (3, 3) and c.shape == (3,) and isinstance(ok, (bool, np.bool_))


@pytest.mark.slow
@pytest.mark.skipif(not VIO_BIN.exists(), reason="OpenVINS vio_stream not built")
def test_openvins_initialises_on_fifteen_seconds_of_simulated_data(tmp_path):
    """Stream the initialisation manoeuvre into the real filter and check it converges."""
    import cv2
    from sim.run import init_manoeuvre

    world = World.generate(3, "apartment")
    cam = Camera()
    veh = Vehicle(world, *world.start, rate=200.0, seed=3)
    imu = ImuSim(200.0, noise=1.0, seed=3)
    rng = np.random.default_rng(3)
    # the configuration the sweep and the findings recommend: online calibration off, which is
    # what keeps the filter bounded on this rig (docs/sim-findings-2026-09-04.md)
    src = pose_sources.VioPose(tmp_path, cam.R_bc, cam.p_bc, seed=3,
                               config="estimator_config_nocalib.yaml",
                               K=cam.K, width=cam.width, height=cam.height)
    buf, gt, est, t = [], [], [], 0.0
    try:
        while t < 15.0:
            veh.step(*init_manoeuvre(t))
            R_wb, p_wb = veh.body_pose()
            s = imu.sample(t, R_wb, p_wb)
            if s:
                buf.append(s)
            if round(t * 200) % 10 == 0:
                R_wc, c_w = cam.pose(R_wb, p_wb)
                img, _ = cam.render(world, R_wc, c_w, rng)
                _, R_e, c_e, ok = src.update(t, R_wc, c_w, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), buf)
                buf = []
                if ok:
                    gt.append(c_w)
                    est.append(c_e.copy())
            t += 1 / 200.0
    finally:
        src.close()
    assert src.initialised, "OpenVINS did not initialise on 15 s of simulated data"
    assert src.init_time < 8.0, f"initialisation took {src.init_time:.1f} s"
    assert len(est) > 50
    gt, est = np.asarray(gt), np.asarray(est)
    # Deliberately not a scale check. Fifteen seconds of the initialisation manoeuvre is the
    # worst case for a monocular filter - the robot ends where it started, so there is almost no
    # net translation to fix the scale on - and the estimate does drift by metres here. See
    # docs/sim-findings-2026-09-04.md. What is asserted is what the rest of the system relies on:
    # the filter initialises fast, keeps producing poses, does not jump every frame, and stays
    # inside the envelope run.py's divergence guard tolerates.
    from sim.run import MAX_FROM_START

    span_gt = np.linalg.norm(gt.max(0) - gt.min(0))
    span_est = np.linalg.norm(est.max(0) - est.min(0))
    assert span_gt > 0.2, "the manoeuvre must actually move the robot"
    assert span_est < MAX_FROM_START, f"span {span_est:.1f} m exceeds the guard's {MAX_FROM_START} m"
    steps = np.linalg.norm(np.diff(est, axis=0), axis=1)
    assert np.median(steps) < 0.2, "a healthy filter does not jump every frame"
