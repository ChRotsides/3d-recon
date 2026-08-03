import numpy as np
import pytest
from recon import geom, synth
from recon.geom import Semantic


def test_polygon_area_and_winding():
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]])
    assert geom.polygon_area(square) == pytest.approx(6.0)
    assert geom.polygon_area(square[::-1]) == pytest.approx(-6.0)
    assert geom.polygon_area(geom.ensure_ccw(square[::-1])) == pytest.approx(6.0)
    assert geom.polygon_perimeter(square) == pytest.approx(10.0)


def test_triangulate_non_convex():
    """Ear clipping must handle the L-shape; a convex-only triangulator
    silently produces the wrong area for every L-shaped room."""
    l_shape = np.array(
        [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0], [2.0, 4.0], [0.0, 4.0]]
    )
    tris = geom.triangulate(l_shape)
    assert len(tris) == len(l_shape) - 2
    total = sum(
        abs(geom.polygon_area(l_shape[t])) for t in tris
    )
    assert total == pytest.approx(abs(geom.polygon_area(l_shape)), rel=1e-9)


def test_point_in_polygon_and_distance():
    poly = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
    pts = np.array([[2.0, 1.5], [-1.0, 1.5], [3.9, 2.9], [5.0, 5.0]])
    assert list(geom.point_in_polygon(pts, poly)) == [True, False, True, False]
    d = geom.distance_to_polygon(np.array([[2.0, 1.0]]), poly)
    assert d[0] == pytest.approx(1.0)


def test_inward_normal_points_into_room():
    poly = geom.ensure_ccw(np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]))
    centre = np.array([2.0, 0.0, 1.5])
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        n = geom.inward_normal_2d(a, b)
        mid = np.array([(a[0] + b[0]) / 2, 0.0, (a[1] + b[1]) / 2])
        assert n @ (centre - mid) > 0, f"wall {i} normal points out of the room"


def test_raycast_depth_is_exact():
    """The synthetic oracle is only worth anything if its depth is exact."""
    spec = synth.preset("empty_rect")
    mesh = synth.build_mesh(spec)
    cfg = synth.CaptureConfig(width=64, height=48)
    K = synth.intrinsics_for(cfg).matrix()
    pos = np.array([2.1, 1.5, 1.675])

    hit = geom.raycast(mesh, K, np.eye(3), pos, 64, 48)
    assert hit.depth[24, 32] == pytest.approx(1.675, abs=1e-9)

    up = geom.raycast(mesh, K, synth._rotation(0.0, np.pi / 2), pos, 64, 48)
    assert up.depth[24, 32] == pytest.approx(spec.ceiling_height_m - 1.5, abs=1e-9)

    down = geom.raycast(mesh, K, synth._rotation(0.0, -np.pi / 2), pos, 64, 48)
    assert down.depth[24, 32] == pytest.approx(1.5, abs=1e-9)


def test_unproject_round_trips():
    spec = synth.preset("empty_rect")
    mesh = synth.build_mesh(spec)
    cfg = synth.CaptureConfig(width=48, height=36)
    K = synth.intrinsics_for(cfg).matrix()
    R = synth._rotation(0.7, -0.2)
    t = np.array([2.0, 1.5, 1.6])
    hit = geom.raycast(mesh, K, R, t, 48, 36)
    pts = geom.unproject(hit.depth, K, R, t)
    expected = hit.point.reshape(-1, 3)[hit.depth.reshape(-1) > 0]
    # Depth is stored float32, so the round trip is exact only to ~1e-7 m.
    assert np.allclose(pts, expected, atol=1e-6)


def test_wall_with_holes_leaves_a_hole():
    a, b = np.array([0.0, 0.0]), np.array([4.0, 0.0])
    n = geom.inward_normal_2d(a, b)
    pieces = geom.wall_with_holes(a, b, 0.0, 2.5, n, [(1.5, 2.5, 0.0, 2.0)],
                                  Semantic.WALL, (1.0, 1.0, 1.0))
    mesh = geom.TriMesh.concat(pieces)
    # A ray aimed through the hole must miss; one aimed beside it must hit.
    t_hole, f_hole = geom.intersect(mesh, np.array([2.0, 1.0, 2.0]), np.array([[0.0, 0.0, -1.0]]))
    t_wall, f_wall = geom.intersect(mesh, np.array([0.5, 1.0, 2.0]), np.array([[0.0, 0.0, -1.0]]))
    assert f_hole[0] == -1
    assert f_wall[0] >= 0 and t_wall[0] == pytest.approx(2.0)
