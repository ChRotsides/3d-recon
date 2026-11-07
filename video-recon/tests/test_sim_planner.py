"""A*, frontier clustering and the pure-pursuit follower, on toy grids."""

import numpy as np
from sim.explore import Explorer, frontiers
from sim.navgrid import FREE, OCCUPIED, UNKNOWN
from sim.planner import PurePursuit, astar, path_blocked, smooth


def test_astar_goes_around_a_wall():
    cost = np.ones((21, 21), np.float32)
    cost[5:16, 10] = np.inf  # a wall with gaps at the top and bottom
    p = astar(cost, (2, 10), (18, 10))
    assert p is not None
    assert p[0] == (2, 10) and p[-1] == (18, 10)
    assert all(np.isfinite(cost[y, x]) for x, y in p)
    assert len(p) > 16, "a straight line would be 16 cells; going round must be longer"


def test_astar_returns_none_when_walled_in():
    cost = np.ones((21, 21), np.float32)
    cost[:, 10] = np.inf
    assert astar(cost, (2, 10), (18, 10)) is None


def test_astar_starts_from_the_nearest_legal_cell_when_the_robot_is_boxed_in():
    cost = np.ones((21, 21), np.float32)
    cost[9:12, 1:4] = np.inf  # the robot stands inside its own inflated footprint
    p = astar(cost, (2, 10), (18, 10))
    assert p is not None and p[-1] == (18, 10)


def test_astar_prefers_the_cheaper_corridor():
    cost = np.ones((21, 41), np.float32)
    cost[0:10, :] = 6.0  # the top half is expensive
    p_cheap = astar(cost, (0, 15), (40, 15))
    assert all(y >= 10 for _, y in p_cheap)


def test_smoothing_keeps_the_path_legal_and_shortens_it():
    cost = np.ones((21, 21), np.float32)
    cost[5:16, 10] = np.inf
    p = astar(cost, (2, 10), (18, 10))
    s = smooth(p, cost)
    assert len(s) < len(p)
    assert s[0] == p[0] and s[-1] == p[-1]
    for a, b in zip(s[:-1], s[1:]):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        xs = np.round(np.linspace(a[0], b[0], 2 * n + 1)).astype(int)
        ys = np.round(np.linspace(a[1], b[1], 2 * n + 1)).astype(int)
        assert np.isfinite(cost[ys, xs]).all()


def test_frontier_clustering_on_a_toy_grid():
    state = np.full((30, 30), UNKNOWN, np.uint8)
    state[10:20, 10:20] = FREE
    cost = np.ones(state.shape, np.float32)
    lab, sizes, cents = frontiers(state, cost)
    assert len(sizes) == 1, "the free square's border is one connected frontier ring"
    assert sizes[0] == 36  # the 10x10 square's boundary cells
    assert 13.5 < cents[0][0] < 15.5 and 13.5 < cents[0][1] < 15.5


def test_tiny_frontier_clusters_are_discarded():
    state = np.full((30, 30), OCCUPIED, np.uint8)
    state[5:15, 5:15] = FREE
    state[9:11, 20] = UNKNOWN
    state[10, 19] = FREE  # a two-cell frontier, under MIN_CLUSTER
    cost = np.ones(state.shape, np.float32)
    _, sizes, _ = frontiers(state, cost)
    assert (sizes >= 5).all()


def test_frontiers_must_be_reachable():
    state = np.full((30, 30), UNKNOWN, np.uint8)
    state[10:20, 10:20] = FREE
    cost = np.full(state.shape, np.inf, np.float32)
    _, sizes, _ = frontiers(state, cost)
    assert len(sizes) == 0, "a frontier the robot cannot stand on is not a frontier"


def test_pure_pursuit_turns_towards_the_path():
    pp = PurePursuit(lookahead=0.6)
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    v, w, done = pp.command((0.0, 0.0, 0.0), path)  # already aligned
    assert v > 0.2 and abs(w) < 1e-6 and not done
    v, w, _ = pp.command((0.0, 0.0, np.deg2rad(20)), path)  # path is to the right
    assert w < 0
    v, w, _ = pp.command((0.0, 0.0, np.deg2rad(-20)), path)
    assert w > 0
    v, w, _ = pp.command((0.0, 0.0, np.pi), path)  # facing away: turn in place
    assert abs(v) < 1e-9 and abs(w) > 0.5


def test_pure_pursuit_reports_done_at_the_end():
    pp = PurePursuit()
    path = np.array([[0.0, 0.0], [1.0, 0.0]])
    _, _, done = pp.command((1.0, 0.0, 0.0), path)
    assert done


def test_pure_pursuit_respects_the_limits():
    pp = PurePursuit(v_max=0.6, w_max=1.2)
    rng = np.random.default_rng(0)
    for _ in range(200):
        path = np.cumsum(rng.normal(0, 0.4, (8, 2)), axis=0)
        v, w, _ = pp.command((rng.normal(), rng.normal(), rng.uniform(-np.pi, np.pi)), path)
        assert abs(v) <= 0.6 + 1e-9 and abs(w) <= 1.2 + 1e-9


def test_path_blocked_detects_a_new_obstacle():
    cost = np.ones((20, 20), np.float32)
    to_cell = lambda xy: (int(xy[0]), int(xy[1]))
    path = np.array([[2.0, 2.0], [5.0, 2.0], [8.0, 2.0]])
    assert not path_blocked(path, cost, to_cell)
    cost[2, 5] = np.inf
    assert path_blocked(path, cost, to_cell)


def test_explorer_picks_the_best_information_per_metre():
    """A big frontier nearby must beat a slightly bigger one far away."""
    state = np.full((40, 80), OCCUPIED, np.uint8)
    state[18:23, 2:40] = FREE
    state[18:23, 41:78] = UNKNOWN
    state[18:23, 40] = FREE
    state[10:18, 3:12] = UNKNOWN
    cost = np.where(state == OCCUPIED, np.inf, 1.0).astype(np.float32)

    class G:
        res = 0.1

        @staticmethod
        def world_to_cell(xy):
            return int(xy[0] / 0.1), int(xy[1] / 0.1)

        @staticmethod
        def cell_to_world(ix, iy):
            return ((np.asarray(ix) + 0.5) * 0.1, (np.asarray(iy) + 0.5) * 0.1)

    e = Explorer()
    tgt = e.replan(0.0, (0.5, 2.0, 0.0), G, state, cost)
    assert tgt is not None
    assert tgt[0] < 2.0, "the near frontier wins on information per metre of path"
    assert e.path is not None and len(e.path) >= 2
