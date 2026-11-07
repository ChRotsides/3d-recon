"""The occupancy grid and the 2D costmap the robot plans on."""

import numpy as np
from sim.navgrid import OCCUPIED, NavGrid


def _frame(distance=3.0, w=64, h=36):
    fx = fy = 45.0
    K = np.array([fx, fy, (w - 1) / 2, (h - 1) / 2], np.float32)
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # looking along +x
    c = np.array([0.0, 0.0, 0.5])
    return np.full((h, w), int(distance * 1000), np.uint16), K, R, c


def test_one_frame_carves_free_marks_the_hit_and_leaves_the_rest_unknown():
    g = NavGrid(stride=1)
    mm, K, R, c = _frame(3.0)
    g.integrate(mm, K, R, c)
    lin = lambda p: g.log[tuple(((np.asarray(p) - g.origin) / g.res).astype(int)[::-1])]
    for x in (1.0, 1.5, 2.0, 2.5):
        assert lin([x, 0.0, 0.5]) < 0, f"the ray to 3 m must carve {x} m free"
    assert lin([3.0, 0.0, 0.5]) > 0, "the hit cell must be occupied"
    assert lin([-1.0, 0.0, 0.5]) == 0, "space behind the camera stays unknown"
    assert lin([1.5, 2.0, 0.5]) == 0, "space outside the frustum stays unknown"


def test_the_floor_is_free_space_not_an_obstacle():
    """Hits at the estimated floor height are ground; without this the costmap is all lethal."""
    g = NavGrid(stride=1)
    g.floor_z = 0.0
    w, h = 64, 36
    K = np.array([45.0, 45.0, (w - 1) / 2, (h - 1) / 2], np.float32)
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    c = np.array([0.0, 0.0, 0.5])
    # every ray in the bottom row aims down and stops on the floor plane z = 0
    v = np.arange(h)
    yn = (v - K[3]) / K[1]
    z = np.where(yn > 0.05, 0.5 / np.maximum(yn, 1e-6), 5.0)
    mm = np.clip(np.repeat(z[:, None], w, 1) * 1000, 0, 65535).astype(np.uint16)
    for _ in range(3):
        g.integrate(mm, K, R, c)
    state, cost, org = g.costmap(0.0)
    assert (state == OCCUPIED).sum() == 0, "floor hits must not become obstacles"
    # the floor cells themselves carry free evidence, not occupied evidence
    z_floor = int(np.floor((0.0 - g.origin[2]) / g.res))
    assert (g.log[z_floor] < 0).sum() > 20
    assert (g.log[z_floor] > 0).sum() == 0
    # and with the floor unknown to the grid, the very same frame does mark obstacles
    g2 = NavGrid(stride=1)
    for _ in range(3):
        g2.integrate(mm, K, R, c)
    assert (g2.costmap(0.0)[0] == OCCUPIED).any()


def test_costmap_band_only_looks_between_the_given_heights():
    g = NavGrid(stride=1)
    g.floor_z = 0.0
    mm, K, R, c = _frame(3.0)
    for _ in range(2):
        g.integrate(mm, K, R, c)  # a wall hit at z = 0.5, inside the 0.05..0.80 band
    state, cost, org = g.costmap(0.0)
    assert (state == OCCUPIED).any()
    # move the floor down by 2 m: the same hits are now above the band and invisible to it
    state2, _, _ = g.costmap(-2.0)
    assert (state2 == OCCUPIED).sum() == 0


def test_inflation_makes_cells_near_an_obstacle_lethal():
    g = NavGrid(stride=1)
    g.floor_z = 0.0
    mm, K, R, c = _frame(3.0)
    for _ in range(2):
        g.integrate(mm, K, R, c)
    state, cost, org = g.costmap(0.0)
    oy, ox = np.nonzero(state == OCCUPIED)
    k = len(oy) // 2
    for d in range(1, 3):  # inside the lethal radius
        assert not np.isfinite(cost[oy[k], ox[k] - d])
    assert np.isfinite(cost[oy[k], ox[k] - 8])


def test_cost_falls_away_from_obstacles():
    g = NavGrid(stride=1)
    g.floor_z = 0.0
    mm, K, R, c = _frame(4.0)
    for _ in range(2):
        g.integrate(mm, K, R, c)
    _, cost, _ = g.costmap(0.0)
    finite = cost[np.isfinite(cost)]
    assert finite.min() >= 1.0
    assert finite.max() > 1.2, "there must be a gradient, not a flat cost"


def test_surface_marking_and_growth():
    g = NavGrid(stride=1)
    g.floor_z = 0.0
    mm, K, R, c = _frame(2.0)
    g.integrate(mm, K, R, c)
    before = g.log.shape
    P = np.array([[9.03, 9.02, 1.04, 0, 0, 0], [9.13, 9.02, 1.04, 0, 0, 0]], np.float32)
    g.mark_surface(P)
    assert g.log.shape != before, "the grid must grow to cover new points"
    idx = tuple(np.floor((P[0, :3] - g.origin) / g.res).astype(int)[::-1])
    assert g.log[idx] > 0
    # a point on the estimated floor is ground, not an obstacle
    n_before = (g.log > 0).sum()
    g.mark_surface(np.array([[5.03, 5.02, 0.01, 0, 0, 0]], np.float32))
    assert (g.log > 0).sum() == n_before


def test_navigation_modules_never_see_the_world():
    """Structural guarantee: the map and the planner cannot cheat with ground truth."""
    import ast
    from pathlib import Path

    sim = Path(__file__).resolve().parents[1] / "sim"
    for name in ("navgrid.py", "planner.py", "explore.py"):
        tree = ast.parse((sim / name).read_text())
        mods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods += [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods.append(node.module or "")
                mods += [f"{node.module or ''}.{n.name}" for n in node.names]
        bad = [m for m in mods if "world" in m or "vehicle" in m or "sensors" in m]
        assert not bad, f"{name} imports the simulator: {bad}"


def test_flat_indices_survive_a_large_grid():
    """A diverged pose grows the grid until an int32 flat index wraps negative and bincount
    rejects it. Five sweep runs died that way; the indices are int64 now."""
    g = NavGrid(stride=1)
    g.floor_z = 0.0
    g._ensure(np.zeros(3), np.array([60.0, 60.0, 12.0]))
    nz, ny, nx = g.log.shape
    assert nz * ny * nx > 2**21
    pts = np.array([[59.0, 59.0, 11.0], [0.1, 0.1, 0.1]], np.float32)
    idx, ok = g._index(pts)
    assert idx.dtype == np.int64 and (idx >= 0).all()
    assert idx.max() < g.log.size
    g._add(idx, -0.4)   # must not raise
    assert (g.log < 0).sum() == 2


def test_grid_extent_is_capped():
    g = NavGrid(stride=1)
    g._ensure(np.zeros(3), np.ones(3))
    before = g.log.shape
    g._ensure(np.array([-5000.0, -5000.0, -5000.0]), np.array([5000.0, 5000.0, 5000.0]))
    assert g.log.size < 40_000_000, "a runaway pose must not be able to allocate the machine"
    assert g.log.shape >= before
