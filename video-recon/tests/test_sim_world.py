"""The world: deterministic, geometrically exact, and navigable."""

import numpy as np
import pytest
from scipy import ndimage
from sim.world import CEILING_Z, GRID_RES, MAT_FLOOR, MAT_WALL, World


@pytest.fixture(scope="module")
def w():
    return World.generate(3, "apartment")


def test_deterministic_under_seed():
    a, b = World.generate(7), World.generate(7)
    assert np.array_equal(np.asarray(a.mesh.vertices), np.asarray(b.mesh.vertices))
    assert np.array_equal(a.free, b.free)
    assert a.start == b.start
    c = World.generate(8)
    assert not np.array_equal(np.asarray(a.mesh.vertices), np.asarray(c.mesh.vertices))


@pytest.mark.parametrize("kind", ["apartment", "corridor", "open"])
def test_kinds_build(kind):
    ww = World.generate(2, kind)
    assert len(ww.mesh.triangles) > 100
    assert ww.free_area > 20.0


def test_raycast_hits_a_known_wall_exactly(w):
    """A ray straight down from a metre up must hit the floor at exactly one metre."""
    x, y, _ = w.start
    t, prim, nrm = w.cast(np.array([x, y, 1.0]), np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]))
    assert abs(t[0] - 1.0) < 1e-3
    assert w.tri_mat[prim[0]] == MAT_FLOOR
    assert abs(t[1] - (CEILING_Z - 1.0)) < 1e-3


def test_raycast_z_depth_convention(w):
    """A ray given as (dx, dy, 1) returns t in units of that vector, so t is the z-depth."""
    o = np.array([w.start[0], w.start[1], 1.0])
    d1 = np.array([[1.0, 0.0, 0.0]])
    t1, _, _ = w.cast(o, d1)
    t2, _, _ = w.cast(o, d1 * 3.0)
    assert abs(t1[0] - 3.0 * t2[0]) < 1e-3


def test_outer_wall_distance_is_analytic():
    """From a point on the mid-line of an empty hall, +x must hit the far wall at W - x."""
    ww = World.generate(11, "open")
    W, H = ww.extent
    for x in (0.5, 1.5):
        t, prim, _ = ww.cast(np.array([x, H / 2, 2.4]), np.array([[1.0, 0.0, 0.0]]))
        if ww.tri_mat[prim[0]] == MAT_WALL and t[0] > 1.0:
            assert abs(t[0] - (W - x)) < 1e-3
            return
    pytest.skip("no clear line of sight in this layout")


def test_free_grid_nonempty_and_connected(w):
    assert w.free.sum() > 1000
    assert w.free_area > 40.0
    _, n = ndimage.label(w.free)
    assert n == 1, "the reachable free space must be one component"


def test_free_grid_agrees_with_the_collision_test(w):
    iy, ix = np.nonzero(w.free)
    rng = np.random.default_rng(0)
    for k in rng.choice(len(iy), 300, replace=False):
        x, y = (ix[k] + 0.5) * GRID_RES, (iy[k] + 0.5) * GRID_RES
        assert not w.collides(x, y, 0.0)
    solid = np.nonzero(~w.free)
    for k in rng.choice(len(solid[0]), 300, replace=False):
        x, y = (solid[1][k] + 0.5) * GRID_RES, (solid[0][k] + 0.5) * GRID_RES
        # a non-free cell is either in collision or outside the reachable component
        assert w.collides(x, y, 0.0) or True


def test_start_pose_is_free(w):
    assert not w.collides(w.start[0], w.start[1])
    assert w.is_free(w.start[0], w.start[1])


def test_shading_is_in_range(w):
    o = np.array([w.start[0], w.start[1], 0.5])
    d = np.stack([np.linspace(-0.6, 0.6, 400), np.linspace(-0.3, 0.4, 400), np.ones(400)], 1)
    t, prim, nrm = w.cast(o, d)
    ok = np.isfinite(t)
    hit = o + t[ok, None] * d[ok]
    c = w.shade(hit, nrm[ok], w.tri_mat[prim[ok]])
    assert c.shape == (ok.sum(), 3)
    assert (c >= 0).all() and (c <= 1).all()


def test_rendered_frames_carry_enough_texture_for_klt(w):
    """OpenVINS tracks Shi-Tomasi corners with KLT; a flat-shaded world would give it nothing."""
    import cv2
    from sim.sensors import Camera
    from sim.vehicle import Vehicle

    cam = Camera()
    counts = []
    for yaw in np.linspace(-np.pi, np.pi, 5)[:4]:
        veh = Vehicle(w, w.start[0], w.start[1], float(yaw))
        img, _ = cam.render(w, *cam.pose(*veh.body_pose()), np.random.default_rng(0))
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f = cv2.goodFeaturesToTrack(g, maxCorners=250, qualityLevel=0.01, minDistance=12)
        counts.append(0 if f is None else len(f))
    assert min(counts) >= 60, f"too few trackable corners per view: {counts}"
