"""Continuous-capture segmentation tests (PLAN.md §6.3 path A).

Path A is the preferred route when it works, and a genuinely easier problem:
the rooms were never in separate frames, so there is no transform to solve.
What has to be right is the segmentation — which keyframes belong to which room.
"""

from __future__ import annotations

import numpy as np
import pytest
from recon.schemas import Doorway, EdgeSource
from recon.stitch.continuous import (
    detect_crossings,
    edges_from_segmentation,
    segment,
    split_keyframes,
)

UP = np.array([0.0, 1.0, 0.0])


def _doorway(x: float, z: float, normal, width=0.83, height=2.03, ident="dw") -> Doorway:
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    c = np.array([x, 0.0, z])
    return Doorway(id=ident, plane=[*n.tolist(), float(-(n @ c))],
                   centroid=c.tolist(), width_m=width, height_m=height)


def _walk(points, per_leg=12, height=1.5):
    """A trajectory through the given waypoints, at eye height."""
    pts = []
    for a, b in zip(points, points[1:], strict=False):
        for t in np.linspace(0, 1, per_leg, endpoint=False):
            p = np.array(a) + t * (np.array(b) - np.array(a))
            pts.append([p[0], height, p[1]])
    pts.append([points[-1][0], height, points[-1][1]])
    return np.asarray(pts)


# -- crossing detection -----------------------------------------------------


def test_walking_through_a_doorway_is_one_crossing():
    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    path = _walk([(2.0, 1.7), (6.0, 1.7)])
    crossings = detect_crossings(path, [dw], UP)
    assert len(crossings) == 1
    assert crossings[0].doorway_id == "dw"
    assert abs(crossings[0].position[0] - 4.0) < 0.2


def test_walking_past_a_door_along_the_wall_is_not_a_crossing():
    """The plane extends forever; the *opening* does not. Without the lateral
    check, walking down a corridor registers a crossing at every door."""
    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    path = _walk([(2.0, 5.0), (6.0, 5.0)])  # same plane, 3.3 m off centre
    assert detect_crossings(path, [dw], UP) == []


def test_loitering_in_a_doorway_does_not_produce_a_burst():
    """Tracking noise jitters the signed distance across zero. Without the
    clearance requirement each wobble is a room transition."""
    rng = np.random.default_rng(0)
    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    wobble = np.column_stack([
        4.0 + rng.normal(0.0, 0.03, 40),
        np.full(40, 1.5),
        np.full(40, 1.7),
    ])
    assert detect_crossings(wobble, [dw], UP) == []


def test_a_doorway_overhead_is_not_crossed():
    """Everything shares one frame in path A, including other floors."""
    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    dw.centroid = [4.0, 3.2, 1.7]  # a doorway one storey up
    dw.plane = [-1.0, 0.0, 0.0, 4.0]
    path = _walk([(2.0, 1.7), (6.0, 1.7)])
    assert detect_crossings(path, [dw], UP) == []


def test_direction_of_travel_is_recorded():
    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    there = detect_crossings(_walk([(2.0, 1.7), (6.0, 1.7)]), [dw], UP)
    back = detect_crossings(_walk([(6.0, 1.7), (2.0, 1.7)]), [dw], UP)
    assert there[0].direction == -back[0].direction


def test_a_long_step_straddling_a_doorway_is_still_caught():
    """Interpolating to the plane matters: at 2 Hz keyframes and walking pace
    the step can be most of a metre, and neither endpoint is in the opening."""
    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    path = np.array([
        [2.0, 1.5, 1.7], [3.2, 1.5, 1.7], [4.8, 1.5, 1.7], [6.0, 1.5, 1.7],
        [6.8, 1.5, 1.7],
    ])
    assert len(detect_crossings(path, [dw], UP)) == 1


# -- segmentation -----------------------------------------------------------


def test_two_rooms_one_doorway():
    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    path = _walk([(1.0, 1.0), (1.5, 2.5), (3.5, 1.7), (5.0, 1.7), (7.0, 1.0), (6.5, 2.5)])
    seg = segment(path, [dw], UP)
    assert seg.n_rooms == 2
    assert sum(len(v) for v in seg.rooms.values()) == len(path)


def test_a_revisited_room_is_one_room_not_two():
    """Walk to the far room and come back. The first room is seen twice; a
    segmenter that counts visits reports three rooms in a two-room house."""
    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    path = _walk([
        (1.0, 1.0), (1.5, 2.5), (3.5, 1.7),      # room A
        (5.0, 1.7), (7.0, 1.0), (6.5, 2.5),      # room B
        (5.0, 1.7), (3.5, 1.7), (1.2, 1.4),      # back to A
    ])
    seg = segment(path, [dw], UP)
    assert len(seg.crossings) == 2
    assert len(seg.visits) == 3
    assert seg.n_rooms == 2, f"visits {[v.room for v in seg.visits]}"
    assert seg.visits[0].room == seg.visits[2].room


def test_a_capture_that_never_leaves_one_room_is_one_room():
    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    path = _walk([(1.0, 1.0), (2.5, 2.5), (1.5, 2.0), (2.0, 1.0)])
    seg = segment(path, [dw], UP)
    assert seg.n_rooms == 1
    assert not seg.crossings


def test_three_rooms_in_a_row():
    doors = [
        _doorway(4.0, 1.7, [-1.0, 0.0, 0.0], ident="d0"),
        _doorway(8.0, 1.7, [-1.0, 0.0, 0.0], ident="d1"),
    ]
    path = _walk([
        (1.0, 1.0), (2.0, 2.5), (3.5, 1.7),
        (5.0, 1.7), (6.0, 2.5), (7.5, 1.7),
        (9.0, 1.7), (10.0, 2.5), (11.0, 1.0),
    ])
    seg = segment(path, doors, UP)
    assert seg.n_rooms == 3
    assert [c.doorway_id for c in seg.crossings] == ["d0", "d1"]


def test_split_keyframes_partitions_without_loss():
    from recon.schemas import Intrinsics, Keyframe, Pose

    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    path = _walk([(1.0, 1.0), (2.0, 2.0), (3.5, 1.7), (5.0, 1.7), (7.0, 1.5), (6.0, 2.5)])
    kfs = [
        Keyframe(
            id=f"kf_{i:04d}", image=f"images/kf_{i:04d}.png", timestamp_ns=i,
            pose=Pose(R=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], t=p.tolist()),
            intrinsics=Intrinsics(fx=200, fy=200, cx=80, cy=60, width=160, height=120),
        )
        for i, p in enumerate(path)
    ]
    seg = segment(path, [dw], UP)
    split = split_keyframes(kfs, seg)
    assert sum(len(v) for v in split.values()) == len(kfs)
    ids = [k.id for group in split.values() for k in group]
    assert len(set(ids)) == len(kfs), "a keyframe landed in two rooms"


# -- edges ------------------------------------------------------------------


def test_continuous_edges_are_identity_with_tight_covariance():
    """Path A's whole point: the rooms were never in different frames, so there
    is no transform to solve. But it is not *zero* uncertainty either — VIO
    drifts over the walk, and treating it as exact would propagate that drift
    as if it were ground truth."""
    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    path = _walk([(1.0, 1.0), (2.0, 2.0), (3.5, 1.7), (5.0, 1.7), (7.0, 1.5), (6.0, 2.5)])
    seg = segment(path, [dw], UP)
    edges = edges_from_segmentation(seg, {0: "room_a", 1: "room_b"})

    assert len(edges) == 1
    e = edges[0]
    assert e.source is EdgeSource.CONTINUOUS_TRACKING
    assert (e.transform.yaw_rad, e.transform.tx, e.transform.tz) == (0.0, 0.0, 0.0)

    sigma = np.sqrt(np.diag(np.asarray(e.covariance)))
    assert (sigma > 0).all(), "an exact edge would let walk drift pass as truth"
    assert sigma[1] < 0.05, "a continuous edge should beat a doorway ICP match"


def test_a_revisit_does_not_duplicate_an_edge():
    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    path = _walk([
        (1.0, 1.0), (2.0, 2.0), (3.5, 1.7),
        (5.0, 1.7), (7.0, 1.5), (6.0, 2.5),
        (5.0, 1.7), (3.5, 1.7), (1.2, 1.4),
    ])
    seg = segment(path, [dw], UP)
    edges = edges_from_segmentation(seg, {0: "room_a", 1: "room_b"})
    assert len(edges) == 1


def test_continuous_edges_solve_in_the_house_graph():
    """The point of building edges at all: they must drop into the same graph
    the doorway matches use."""
    from recon.stitch.posegraph import HousePoseGraph

    dw = _doorway(4.0, 1.7, [-1.0, 0.0, 0.0])
    path = _walk([(1.0, 1.0), (2.0, 2.0), (3.5, 1.7), (5.0, 1.7), (7.0, 1.5), (6.0, 2.5)])
    seg = segment(path, [dw], UP)

    g = HousePoseGraph(solve_scale=False)
    for rid in ("room_a", "room_b"):
        g.add_room(rid)
    for e in edges_from_segmentation(seg, {0: "room_a", 1: "room_b"}):
        g.add_edge(e)
    res = g.optimise(anchor="room_a")
    assert res.converged
    # Identity edges: both rooms sit in the same frame.
    assert res.nodes["room_b"].tx == pytest.approx(0.0, abs=1e-6)
    assert res.nodes["room_b"].tz == pytest.approx(0.0, abs=1e-6)


# -- end to end -------------------------------------------------------------


@pytest.mark.slow
def test_a_real_walk_through_three_rooms_is_segmented_correctly(tmp_path):
    """PLAN §6.3 path A, end to end: one live session across three rooms.

    Scored against the true room label of every keyframe, which the generator
    records. Uses the analytic doorways so this measures the *segmentation* and
    not whether the opening detector found the doors — two failures that would
    otherwise be indistinguishable.
    """
    import json

    from recon import synth, synth_house
    from recon.io.bundle import BundleDir

    house = synth_house.corridor_house(n_rooms=3, seed=1)
    cfg = synth.CaptureConfig(width=96, height=72)
    labels, out = synth_house.generate_continuous(house, tmp_path, cfg, steps_between=8)
    bundle = BundleDir.load(out).bundle
    assert len(bundle.keyframes) == len(labels)

    # Doorways in the house frame: room i's +x door, which is also room i+1's -x.
    doorways = []
    for i, placement in enumerate(house.rooms[:-1]):
        dw = synth_house.analytic_doorway(placement.spec, 1, UP)
        assert dw is not None
        centroid = placement.to_house(np.asarray(dw.centroid)[None, :])[0]
        n = np.asarray(dw.plane[:3])
        dw.centroid = centroid.tolist()
        dw.plane = [*n.tolist(), float(-(n @ centroid))]
        dw.id = f"door_{i}"
        doorways.append(dw)

    positions = np.array([kf.pose.position for kf in bundle.keyframes])
    seg = segment(positions, doorways, UP)

    assert seg.n_rooms == 3, f"got {seg.n_rooms} rooms from {len(seg.crossings)} crossings"
    assert [c.doorway_id for c in seg.crossings] == ["door_0", "door_1"]

    # Every segment must be pure: one true room per detected room.
    true = json.loads((out / "room_labels.json").read_text())
    for room, idx in seg.rooms.items():
        got = {true[i] for i in idx}
        assert len(got) == 1, f"detected room {room} mixes {got}"
    assert len({next(iter({true[i] for i in idx})) for idx in seg.rooms.values()}) == 3

    edges = edges_from_segmentation(seg, {r: f"room_{r}" for r in seg.rooms})
    assert len(edges) == 2
    assert all(e.source is EdgeSource.CONTINUOUS_TRACKING for e in edges)
