"""Phase-5 stitching tests (PLAN.md §6.3, §1.3).

The end-to-end target is <=15 cm drift across a six-room house. These build up
to it: the yaw convention, then the closed-form doorway solve, then ICP, then
the pose graph, then drift on a real chain.
"""

from __future__ import annotations

import numpy as np
import pytest
from recon import synth_house
from recon.schemas import Doorway, EdgeSource, RoomEdge, SE2Transform
from recon.stitch.align import (
    Alignment,
    align_doorways,
    icp_point_to_plane,
    match_doorways,
    yaw_between,
)
from recon.stitch.posegraph import HousePoseGraph, _relative, _wrap

UP = np.array([0.0, 1.0, 0.0])


def _door(centroid, normal, width=0.83, height=2.03, ident="dw_00") -> Doorway:
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    c = np.asarray(centroid, dtype=np.float64)
    return Doorway(
        id=ident, plane=[*n.tolist(), float(-(n @ c))], centroid=c.tolist(),
        width_m=width, height_m=height,
    )


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# -- yaw convention ---------------------------------------------------------


def test_two_rooms_facing_each_other_need_no_rotation():
    """The rooms see the same door from opposite sides, so anti-parallel inward
    normals mean they are already aligned. If this returns pi, the two rooms get
    stacked on top of each other."""
    n_a = np.array([0.0, 0.0, 1.0])
    n_b = np.array([0.0, 0.0, -1.0])
    assert abs(_wrap(yaw_between(n_a, n_b, UP))) < 1e-9


def test_parallel_normals_require_a_half_turn():
    n = np.array([0.0, 0.0, 1.0])
    assert abs(abs(_wrap(yaw_between(n, n, UP))) - np.pi) < 1e-9


@pytest.mark.parametrize("truth", [0.0, 0.4, -0.9, 2.0, -2.7])
def test_yaw_recovers_an_arbitrary_rotation(truth):
    """`yaw_between` returns the rotation that takes B's frame *into* A's, i.e.
    R(yaw) @ n_b == -n_a. So the test case is built by inverting that."""
    n_a = np.array([0.0, 0.0, 1.0])
    n_b = _yaw_matrix(truth).T @ (-n_a)
    got = yaw_between(n_a, n_b, UP)
    assert abs(_wrap(got - truth)) < 1e-9
    assert np.allclose(_yaw_matrix(got) @ n_b, -n_a, atol=1e-9)


def test_yaw_ignores_any_vertical_component():
    """A doorway normal that picked up a little tilt from noisy plane fitting
    must not change the yaw — gravity already fixed that axis."""
    n_a = np.array([0.0, 0.0, 1.0])
    n_b = np.array([0.0, 0.3, -1.0])
    assert abs(_wrap(yaw_between(n_a, n_b, UP))) < 1e-9


# -- closed-form doorway alignment -----------------------------------------


def test_closed_form_alignment_places_rooms_either_side_of_the_door():
    # Room A's door is at x=2 in A's frame, facing +z (into A).
    dw_a = _door([2.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    # Room B sees the same door at its own origin, facing -z (into B).
    dw_b = _door([0.0, 0.0, 3.0], [0.0, 0.0, -1.0])

    al = align_doorways(dw_a, dw_b, UP, refine=False)
    assert abs(_wrap(al.yaw_rad)) < 1e-9

    moved = _yaw_matrix(al.yaw_rad) @ np.asarray(dw_b.centroid) + np.array(
        [al.tx, 0.0, al.tz]
    )
    assert np.allclose(moved, dw_a.centroid, atol=1e-9)


def test_closed_form_alignment_recovers_a_known_transform():
    truth_yaw, truth_t = 0.7, np.array([3.1, 0.0, -1.4])
    dw_a = _door([1.2, 0.0, 0.4], [0.2, 0.0, 0.98])

    # Build B by inverting a known house-frame placement of the same door.
    r = _yaw_matrix(truth_yaw)
    c_b = r.T @ (np.asarray(dw_a.centroid) - truth_t)
    n_b = r.T @ (-np.asarray(dw_a.plane[:3]))
    dw_b = _door(c_b, n_b)

    al = align_doorways(dw_a, dw_b, UP, refine=False)
    assert abs(_wrap(al.yaw_rad - truth_yaw)) < 1e-8
    assert al.tx == pytest.approx(truth_t[0], abs=1e-8)
    assert al.tz == pytest.approx(truth_t[2], abs=1e-8)


def test_unrefined_alignment_reports_loose_covariance():
    """A doorway-only match must not be trusted like an ICP-verified one."""
    al = align_doorways(
        _door([0, 0, 0], [0, 0, 1]), _door([0, 0, 0], [0, 0, -1]), UP, refine=False
    )
    sig = np.sqrt(np.diag(al.covariance))
    assert sig[0] > np.radians(2.0)
    assert sig[1] > 0.02


# -- ICP --------------------------------------------------------------------


def _wall_cloud(n=900, seed=0):
    """A corner: two perpendicular walls and a floor patch, with normals."""
    rng = np.random.default_rng(seed)
    pts, nrm = [], []
    for _ in range(n):
        which = rng.integers(0, 3)
        if which == 0:  # wall at z = 0, facing +z
            pts.append([rng.uniform(-1.5, 1.5), rng.uniform(0, 2.2), 0.0])
            nrm.append([0.0, 0.0, 1.0])
        elif which == 1:  # wall at x = -1.5, facing +x
            pts.append([-1.5, rng.uniform(0, 2.2), rng.uniform(0, 2.0)])
            nrm.append([1.0, 0.0, 0.0])
        else:  # floor
            pts.append([rng.uniform(-1.5, 1.5), 0.0, rng.uniform(0, 2.0)])
            nrm.append([0.0, 1.0, 0.0])
    return np.asarray(pts), np.asarray(nrm)


def test_icp_recovers_a_small_perturbation():
    a, na = _wall_cloud(seed=1)
    truth_yaw, truth_tx, truth_tz = 0.05, 0.08, -0.06
    r = _yaw_matrix(truth_yaw)
    b = (a - np.array([truth_tx, 0.0, truth_tz])) @ r  # so r @ b + t == a

    init = Alignment(0.0, 0.0, 0.0, float("nan"), 0, np.eye(3), True)
    out = icp_point_to_plane(a, na, b, init, UP, centre=np.array([0.0, 0.5, 0.5]),
                             radius=3.0)
    assert out is not None
    assert abs(_wrap(out.yaw_rad - truth_yaw)) < 0.01
    assert out.tx == pytest.approx(truth_tx, abs=0.02)
    assert out.tz == pytest.approx(truth_tz, abs=0.02)
    assert out.rmse_m < 0.01


def test_icp_tightens_the_covariance_relative_to_the_closed_form():
    a, na = _wall_cloud(seed=2)
    init = Alignment(0.02, 0.03, 0.02, float("nan"), 0, np.eye(3), True)
    out = icp_point_to_plane(a, na, a.copy(), init, UP,
                             centre=np.array([0.0, 0.5, 0.5]), radius=3.0)
    assert out is not None
    loose = np.sqrt(np.diag(align_doorways(
        _door([0, 0, 0], [0, 0, 1]), _door([0, 0, 0], [0, 0, -1]), UP, refine=False
    ).covariance))
    tight = np.sqrt(np.diag(out.covariance))
    assert (tight < loose).all(), f"{tight} vs {loose}"


def test_icp_returns_none_without_overlap():
    a, na = _wall_cloud(seed=3)
    far = a + np.array([50.0, 0.0, 50.0])
    init = Alignment(0.0, 0.0, 0.0, float("nan"), 0, np.eye(3), True)
    assert icp_point_to_plane(a, na, far, init, UP,
                              centre=np.array([0.0, 0.5, 0.5]), radius=2.0) is None


def test_bad_alignment_is_rejected_by_the_rmse_gate():
    """PLAN §6.3 step 4: RMSE > 5 cm means ask the user to re-tag."""
    a, na = _wall_cloud(seed=4)
    # A cloud that cannot be brought into agreement by any 3-DoF transform.
    rng = np.random.default_rng(5)
    b = a + rng.normal(0.0, 0.35, a.shape)
    dw_a = _door([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    dw_b = _door([0.0, 0.0, 0.0], [0.0, 0.0, -1.0])
    al = align_doorways(dw_a, dw_b, UP, points_a=a, points_b=b, normals_a=na)
    assert not al.accepted
    assert "re-tag" in al.reason


# -- doorway matching -------------------------------------------------------


def test_doorway_matching_ranks_by_size_agreement():
    a = [_door([0, 0, 0], [0, 0, 1], width=0.83, ident="a0"),
         _door([1, 0, 0], [0, 0, 1], width=1.20, ident="a1")]
    b = [_door([0, 0, 0], [0, 0, -1], width=1.19, ident="b0"),
         _door([1, 0, 0], [0, 0, -1], width=0.84, ident="b1")]
    pairs = match_doorways(a, b)
    assert (0, 1) in [(i, j) for i, j, _ in pairs]
    assert (1, 0) in [(i, j) for i, j, _ in pairs]
    # A 0.83 door must not be offered as a match for a 1.20 one.
    assert (0, 0) not in [(i, j) for i, j, _ in pairs]


# -- pose graph -------------------------------------------------------------


def _edge(a, b, yaw, tx, tz, sigma=0.01, dw=("d0", "d1")) -> RoomEdge:
    return RoomEdge(
        from_room=a, to_room=b, via_doorway=list(dw),
        transform=SE2Transform(yaw_rad=yaw, tx=tx, tz=tz),
        covariance=np.diag([np.radians(0.5) ** 2, sigma**2, sigma**2]).tolist(),
        source=EdgeSource.USER_TAGGED,
    )


def test_relative_transform_is_self_consistent():
    """`_relative` must invert composition, or every edge residual is wrong."""
    ya, txa, tza = 0.3, 1.0, -2.0
    rel_yaw, rel_tx, rel_tz = 0.4, 0.7, 0.2
    c, s = np.cos(ya), np.sin(ya)
    txb = txa + c * rel_tx + s * rel_tz
    tzb = tza - s * rel_tx + c * rel_tz
    yb = ya + rel_yaw
    got = _relative(ya, txa, tza, yb, txb, tzb)
    assert got[0] == pytest.approx(rel_yaw)
    assert got[1] == pytest.approx(rel_tx)
    assert got[2] == pytest.approx(rel_tz)


def test_compose_inverts_relative():
    """`_compose` builds the spanning-tree initial guess and `_relative` scores
    the residual. If they are not exact inverses the solve starts in the wrong
    basin and Huber cannot save it."""
    from recon.stitch.posegraph import _compose, _invert

    a = (0.3, 1.0, -2.0)
    rel = (0.4, 0.7, 0.2)
    b = _compose(*a, *rel)
    got = _relative(*a, *b)
    assert np.allclose(got, rel, atol=1e-12)

    # And the inverse transform composes back to the identity.
    back = _compose(*b, *_invert(*rel))
    assert np.allclose(back, a, atol=1e-12)


def test_spanning_tree_initialisation_places_a_chain_before_optimising():
    g = HousePoseGraph(solve_scale=False)
    for r in ["r0", "r1", "r2"]:
        g.add_room(r)
    g.add_edge(_edge("r0", "r1", 0.0, 4.0, 0.0))
    g.add_edge(_edge("r1", "r2", 0.0, 4.0, 0.0))
    guess = g.initial_guess("r0")
    assert guess["r1"][1] == pytest.approx(4.0)
    assert guess["r2"][1] == pytest.approx(8.0)


def test_initialisation_follows_edges_in_either_direction():
    """The spanning tree must traverse an edge backwards when the anchor is on
    the `to_room` side, or half the house never gets initialised."""
    g = HousePoseGraph(solve_scale=False)
    for r in ["r0", "r1"]:
        g.add_room(r)
    g.add_edge(_edge("r0", "r1", 0.0, 4.0, 0.0))
    guess = g.initial_guess("r1")
    assert guess["r0"][1] == pytest.approx(-4.0)


def test_chain_of_rooms_is_reconstructed():
    g = HousePoseGraph(solve_scale=False)
    truth = {"r0": (0.0, 0.0, 0.0), "r1": (0.0, 4.0, 0.0), "r2": (0.5, 8.0, 1.0)}
    for r in truth:
        g.add_room(r)
    for a, b in [("r0", "r1"), ("r1", "r2")]:
        ya, txa, tza = truth[a]
        yb, txb, tzb = truth[b]
        g.add_edge(_edge(a, b, *_relative(ya, txa, tza, yb, txb, tzb)))

    res = g.optimise(anchor="r0")
    assert res.converged
    for r, (y, tx, tz) in truth.items():
        n = res.nodes[r]
        assert abs(_wrap(n.yaw_rad - y)) < 1e-3, r
        assert n.tx == pytest.approx(tx, abs=1e-3)
        assert n.tz == pytest.approx(tz, abs=1e-3)


def test_loop_closure_distributes_error_around_the_loop():
    """Four rooms in a square with a deliberately inconsistent final edge. The
    residual must be spread, not dumped entirely on the last room."""
    g = HousePoseGraph(solve_scale=False)
    for r in ["r0", "r1", "r2", "r3"]:
        g.add_room(r)
    g.add_edge(_edge("r0", "r1", 0.0, 4.0, 0.0))
    g.add_edge(_edge("r1", "r2", 0.0, 0.0, 4.0))
    g.add_edge(_edge("r2", "r3", 0.0, -4.0, 0.0))
    # Closing edge is 20 cm inconsistent.
    g.add_edge(_edge("r3", "r0", 0.0, 0.0, -3.8))

    res = g.optimise(anchor="r0")
    assert res.converged
    residuals = list(res.edge_residuals.values())
    # No single edge should absorb the whole error.
    assert max(residuals) < 0.9 * sum(residuals), residuals


_CHAIN = {"r0": (0.0, 0.0, 0.0), "r1": (0.0, 4.0, 0.0), "r2": (0.0, 8.0, 0.0),
          "r3": (0.0, 12.0, 0.0)}


def _chain_graph(outlier_sigma: float | None = None) -> HousePoseGraph:
    g = HousePoseGraph(solve_scale=False)
    for r in _CHAIN:
        g.add_room(r)
    for a, b in [("r0", "r1"), ("r1", "r2"), ("r2", "r3")]:
        ya, txa, tza = _CHAIN[a]
        yb, txb, tzb = _CHAIN[b]
        g.add_edge(_edge(a, b, *_relative(ya, txa, tza, yb, txb, tzb)))
    if outlier_sigma is not None:
        # A mis-tagged doorway claiming r0 and r3 are 1 m apart.
        g.add_edge(_edge("r0", "r3", 0.0, 1.0, 0.0, sigma=outlier_sigma))
    return g


def test_one_bad_doorway_tag_does_not_wreck_the_house():
    """PLAN §7: Huber keeps a single outlier edge from corrupting the graph.

    The mis-tag carries a loose covariance, which is what a bad tag actually
    produces: its geometry does not line up, so ICP reports a poor RMSE and
    `_covariance_from_residuals` widens the edge accordingly.
    """
    clean = _chain_graph().optimise(anchor="r0")
    poisoned = _chain_graph(outlier_sigma=0.15).optimise(anchor="r0")

    drift = abs(poisoned.nodes["r3"].tx - clean.nodes["r3"].tx)
    assert drift < 1.5, f"the outlier moved r3 by {drift:.2f} m"
    assert poisoned.suspected_outliers, "the bad edge should have been flagged"
    assert not poisoned.ambiguous


def test_an_undecidable_topology_is_flagged_rather_than_guessed():
    """A chain plus one short cut, all with identical covariance, is genuinely
    ambiguous: dropping the bad edge and dropping one chain edge give solutions
    with *identical* robust cost, because each violates exactly one constraint
    by the same amount. Huber cannot break that tie — the information is not
    there. The solver must say so instead of picking one and looking confident.
    """
    res = _chain_graph(outlier_sigma=0.01).optimise(anchor="r0")
    assert res.ambiguous, "an undecidable layout was reported as certain"
    assert "re-tag" in res.ambiguity_note


def test_a_well_constrained_house_is_not_flagged_ambiguous():
    """The flag has to be quiet on ordinary input or it is worthless."""
    assert not _chain_graph().optimise(anchor="r0").ambiguous


def test_scale_error_is_distributed_not_piled_up():
    """PLAN §6.2: the whole point of scale-as-a-graph-variable.

    Three rooms share doors. Two agree the shared door is 0.83 m wide; the middle
    room's VIO says 0.87. The correction must be shared, not dumped on one room.
    """
    g = HousePoseGraph(solve_scale=True)
    g.add_room("r0", scale_prior=1.00, scale_sigma=0.03,
               doorway_widths={"d0": 0.83, "d1": 0.83})
    g.add_room("r1", scale_prior=1.00, scale_sigma=0.03,
               doorway_widths={"d0": 0.87, "d1": 0.87})
    g.add_room("r2", scale_prior=1.00, scale_sigma=0.03,
               doorway_widths={"d0": 0.83, "d1": 0.83})
    g.add_edge(_edge("r0", "r1", 0.0, 4.0, 0.0, dw=("d1", "d0")))
    g.add_edge(_edge("r1", "r2", 0.0, 4.0, 0.0, dw=("d1", "d0")))

    res = g.optimise(anchor="r0")
    assert res.converged
    s0, s1, s2 = (res.nodes[r].scale for r in ["r0", "r1", "r2"])
    # The middle room shrinks, its neighbours grow slightly — the residual is
    # shared rather than assigned entirely to r1.
    assert s1 < 1.0 < max(s0, s2), (s0, s1, s2)
    widths = [s0 * 0.83, s1 * 0.87, s2 * 0.83]
    assert max(widths) - min(widths) < 0.01, widths


def test_anchor_pose_is_fixed_but_its_scale_is_not():
    g = HousePoseGraph(solve_scale=True)
    g.add_room("r0", scale_prior=1.05, scale_sigma=0.03, doorway_widths={"d1": 0.90})
    g.add_room("r1", scale_prior=1.00, scale_sigma=0.03, doorway_widths={"d0": 0.83})
    g.add_edge(_edge("r0", "r1", 0.0, 4.0, 0.0, dw=("d1", "d0")))
    res = g.optimise(anchor="r0")
    assert res.nodes["r0"].tx == 0.0 and res.nodes["r0"].yaw_rad == 0.0
    assert res.nodes["r0"].scale != pytest.approx(1.05, abs=1e-6)


def test_house_graph_export_round_trips():
    from recon.schemas import HouseGraph

    g = HousePoseGraph(solve_scale=False)
    g.add_room("r0")
    g.add_room("r1")
    g.add_edge(_edge("r0", "r1", 0.0, 4.0, 0.0))
    hg = g.to_house_graph("prop-1")
    again = HouseGraph.model_validate_json(hg.model_dump_json())
    assert again.property_id == "prop-1"
    assert len(again.floors[0].edges) == 1
    assert again.floors[0].edges[0].via_doorway == ["d0", "d1"]


# -- end to end -------------------------------------------------------------


def test_six_room_house_drift_is_within_target():
    """PLAN §1.3: <=15 cm end-to-end drift across a six-room house.

    Uses analytic doorways so this measures the *stitching* maths, not the
    opening detector. The full-reconstruction version is the slow test below.
    """
    house = synth_house.corridor_house(n_rooms=6, seed=3)
    g = HousePoseGraph(solve_scale=False)
    for p in house.rooms:
        g.add_room(p.room_id)

    # Room i joins room i+1 through i's +x wall (index 1) and i+1's -x wall
    # (index 3) — the walls the two rooms actually share.
    doors = {}
    for a_i, b_i in house.connections:
        pa, pb = house.rooms[a_i], house.rooms[b_i]
        dw_a = synth_house.analytic_doorway(pa.spec, 1, UP)
        dw_b = synth_house.analytic_doorway(pb.spec, 3, UP)
        assert dw_a is not None, f"{pa.room_id} has no door on its +x wall"
        assert dw_b is not None, f"{pb.room_id} has no door on its -x wall"
        doors[(a_i, b_i)] = (dw_a, dw_b)

        al = align_doorways(dw_a, dw_b, UP, refine=False)
        g.add_edge(al.to_edge(pa.room_id, pb.room_id, dw_a.id, dw_b.id,
                              EdgeSource.USER_TAGGED))

    res = g.optimise(anchor=house.rooms[0].room_id)
    assert res.converged
    assert not res.ambiguous

    # Every joint must agree: the doorway each pair shares lands in one place.
    worst = 0.0
    for a_i, b_i in house.connections:
        pa, pb = house.rooms[a_i], house.rooms[b_i]
        dw_a, dw_b = doors[(a_i, b_i)]
        ca = res.apply(pa.room_id, np.asarray(dw_a.centroid)[None, :])[0]
        cb = res.apply(pb.room_id, np.asarray(dw_b.centroid)[None, :])[0]
        worst = max(worst, float(np.linalg.norm(ca - cb)))
    assert worst <= 0.15, f"worst doorway disagreement {worst*100:.1f} cm"

    # ...and the recovered layout must match the true house, not merely be
    # self-consistent. A solution can satisfy every edge and still be a mirror
    # image or a reversed chain.
    for p in house.rooms:
        got = res.apply(p.room_id, np.zeros((1, 3)))[0]
        assert abs(got[0] - p.tx) < 0.15, f"{p.room_id} at x={got[0]:.2f}, true {p.tx:.2f}"
        assert abs(got[2] - p.tz) < 0.15, f"{p.room_id} at z={got[2]:.2f}, true {p.tz:.2f}"
