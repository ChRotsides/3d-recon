"""The closed loop, end to end, with perfect pose and depth: the robot must explore and not crash."""

import json

import numpy as np
import pytest
from sim.depth_sources import FUSE_H, FUSE_W, fusion_K
from sim.depth_sources import make as make_depth
from sim.sensors import Camera


def test_gt_depth_round_trips_to_the_fusion_grid():
    cam = Camera()
    src = make_depth("gt", cam.K, cam.width, cam.height)
    d = np.full((cam.height, cam.width), 2.5, np.float32)
    mm, info = src.compute(None, d, np.eye(3))
    assert mm.shape == (FUSE_H, FUSE_W) and mm.dtype == np.uint16
    assert abs(mm.mean() - 2500) < 1
    K = fusion_K(cam.K, cam.width, cam.height)
    assert abs(K[0] - cam.K[0] * FUSE_W / cam.width) < 1e-3   # K is float32
    assert abs(K[2] - cam.K[2] * FUSE_W / cam.width) < 1e-3


def test_gt_noisy_depth_is_noisy_but_unbiased():
    cam = Camera()
    src = make_depth("gt_noisy", cam.K, cam.width, cam.height, seed=0)
    d = np.full((cam.height, cam.width), 3.0, np.float32)
    mm, _ = src.compute(None, d, np.eye(3))
    z = mm.astype(float) / 1000.0
    assert abs(z.mean() - 3.0) < 0.02
    assert 0.02 < z.std() / 3.0 < 0.08


@pytest.mark.slow
def test_closed_loop_explores_without_collisions(tmp_path):
    from sim.run import main

    sim = main(["--world", "apartment", "--seed", "1", "--pose", "gt", "--depth", "gt",
                "--duration", "20", "--out", str(tmp_path / "run"), "--headless"])
    out = tmp_path / "run"
    rep = json.loads((out / "report.json").read_text())
    stats = json.loads((out / "stats.json").read_text())

    assert rep["explored_final"] > 0.10, f"explored only {100 * rep['explored_final']:.1f} %"
    assert stats["collisions"] == 0, f"{stats['collisions']} collisions with a perfect pose and map"
    assert rep["ate_se3"]["max_cm"] < 1e-6, "pose gt must equal the truth exactly"
    assert rep["scale_error"] is None or abs(rep["scale_error"] - 1.0) < 1e-6

    for name in ("gt.txt", "est.txt", "commands.csv", "events.csv", "stats.json",
                 "navgrid.npz", "tsdf_surface.ply", "report.json", "plots.png"):
        assert (out / name).exists(), f"missing output {name}"
    assert len(list((out / "costmaps").glob("*.png"))) >= 2
    assert len(list((out / "frames").glob("rgb_*.jpg"))) >= 5

    gt = np.loadtxt(out / "gt.txt")
    est = np.loadtxt(out / "est.txt")
    assert len(gt) > 300 and len(est) > 300
    assert np.abs(gt[:, 1:4] - est[:, 1:4]).max() < 1e-9
    assert rep["map_accuracy_cm"]["median"] < 5.0, "GT depth fused with GT pose must be near-exact"
    assert sim.veh.distance > 1.0, "the robot must actually drive"


def test_costmap_picture_draws_every_layer():
    from sim.navgrid import FREE, OCCUPIED, UNKNOWN
    from sim.viz import costmap_png, depth_png

    state = np.full((30, 40), UNKNOWN, np.uint8)
    state[10:20, 10:30] = FREE
    state[10, 10:30] = OCCUPIED
    cost = np.where(state == OCCUPIED, np.inf, 1.0).astype(np.float32)
    fm = np.zeros(state.shape, bool)
    fm[19, 15:18] = True
    img = costmap_png(state, cost, np.zeros(2), 0.1, path=np.array([[1.5, 1.5], [2.5, 1.7]]),
                      est_pose=(1.5, 1.5, 0.3), gt_pose=(1.6, 1.4, 0.2), frontier_mask=fm,
                      target=(2.5, 1.7), scale=4)
    assert img.shape == (30 * 4, 40 * 4, 3) and img.dtype == np.uint8
    colours = {tuple(c) for c in img.reshape(-1, 3)}
    assert (245, 245, 245) in colours and (25, 25, 25) in colours and (150, 150, 150) in colours
    assert (60, 200, 60) in colours, "frontier cells must be drawn"
    d = depth_png(np.linspace(0, 6, 40 * 30).reshape(30, 40).astype(np.float32))
    assert d.shape == (30, 40, 3)


def test_dashboard_state_is_json_serialisable(tmp_path):
    """The stats panel carries numpy scalars; json.dumps must not choke on them mid-run."""
    import argparse
    import json

    from sim.run import Sim

    a = argparse.Namespace(world="apartment", seed=1, pose="gt", depth="gt", duration=0.6,
                           out=str(tmp_path / "d"), imu_noise=1.0, scale_error=1.0, slip=0.0,
                           vio_config="estimator_config.yaml", dashboard=0, headless=True,
                           no_eval=True, dashboard_host="127.0.0.1", width=640, height=360,
                           record_vio_input=False)
    sim = Sim(a)
    sim.run()
    st = sim.stats_dict()
    json.dumps(st, default=float)
    for k in ("t", "rtf", "explored", "collisions", "blocks", "stage_ms", "pose", "depth"):
        assert k in st


def test_divergence_guard_trips_on_a_runaway_pose_and_not_on_a_correction():
    """The guard must catch a filter running away at metres per second without firing on the
    one-off metre-scale snap a healthy filter makes when a delayed update lands."""
    import argparse

    from sim.run import MAX_FROM_START, MAX_SPEED, Sim

    a = argparse.Namespace(world="apartment", seed=1, pose="gt", depth="gt", duration=0.2,
                           out="/tmp/sim_guard_test", imu_noise=1.0, scale_error=1.0, slip=0.0,
                           vio_config="estimator_config.yaml", dashboard=0, headless=True,
                           no_eval=True, dashboard_host="127.0.0.1", width=640, height=360,
                           record_vio_input=False)
    sim = Sim(a)

    def feed(track):
        sim.diverged = False
        sim._c0 = None
        sim._chist = []
        out = []
        for t, c in track:
            sim.t = t
            out.append(sim._check_divergence(np.asarray(c, float)))
        return out

    # a healthy filter: 0.6 m/s of driving with a single 1.5 m snap partway through
    healthy = [(k * 0.05, [min(k, 30) * 0.03 + (1.5 if k > 30 else 0.0), 0.0, 0.5])
               for k in range(120)]
    assert all(feed(healthy)), "a one-off correction must not count as divergence"
    assert not sim.diverged

    # a runaway: 12 m/s sustained
    runaway = [(k * 0.05, [k * 0.6, 0.0, 0.5]) for k in range(120)]
    res = feed(runaway)
    assert not all(res) and sim.diverged
    assert any(e[1] == "vio_diverged" for e in sim.events)
    assert MAX_SPEED > 0.6 and MAX_FROM_START > 12.0
