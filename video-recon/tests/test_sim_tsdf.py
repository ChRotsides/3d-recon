"""The phone's voxel-hash TSDF, compiled unchanged and driven through ctypes."""

import subprocess
from pathlib import Path

import numpy as np
from sim.tsdf import _LIB, Tsdf

SIM = Path(__file__).resolve().parents[1] / "sim"


def _wall_frame(distance=2.0, w=322, h=181):
    """A camera at the origin looking along +x at a wall plane x = distance."""
    fx = fy = 228.5
    cx, cy = (w - 1) / 2, (h - 1) / 2
    K = np.array([fx, fy, cx, cy], np.float32)
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    c = np.array([0.0, 0.0, 1.0])
    mm = np.full((h, w), int(distance * 1000), np.uint16)
    return mm, K, R, c


def test_library_builds():
    if _LIB.exists():
        _LIB.unlink()
    subprocess.run([str(SIM / "build.sh")], check=True, capture_output=True)
    assert _LIB.exists() and _LIB.stat().st_size > 10_000


def test_wall_surface_is_within_one_voxel_and_never_behind():
    mm, K, R, c = _wall_frame(2.0)
    t = Tsdf()
    for _ in range(3):  # minWeight is 2, so one frame extracts nothing
        st = t.integrate(mm, K, R, c, None, 1)
    assert st["blocks"] > 0
    S = t.surface()
    assert len(S) > 1000
    P = S[:, :3]
    assert np.abs(P[:, 0] - 2.0).max() <= t.voxel + 1e-6, "surface further than one voxel from the wall"
    assert (P[:, 0] <= 2.0 + t.voxel + 1e-6).all(), "no surface may sit behind the wall"


def test_colour_is_carried_through():
    mm, K, R, c = _wall_frame(2.0)
    rgb = np.zeros((181, 322, 3), np.uint8)
    rgb[:, :, 0] = 200
    t = Tsdf()
    for _ in range(4):
        t.integrate(mm, K, R, c, rgb, 1)
    S = t.surface()
    assert len(S)
    assert S[:, 3].mean() > 0.4 and S[:, 4].mean() < 0.2


def test_clear_and_stats():
    mm, K, R, c = _wall_frame(1.5)
    t = Tsdf()
    st = t.integrate(mm, K, R, c, None, 2)
    assert st["blocks"] == t.blocks and st["blocks"] > 0
    assert st["integrate_ms"] >= 0.0
    t.clear()
    assert t.blocks == 0
    assert len(t.surface()) == 0


def test_two_distances_land_where_they_should():
    for d in (1.2, 3.5):
        mm, K, R, c = _wall_frame(d)
        t = Tsdf()
        for _ in range(3):
            t.integrate(mm, K, R, c, None, 2)
        P = t.surface()[:, :3]
        assert len(P)
        assert abs(np.median(P[:, 0]) - d) <= t.voxel
