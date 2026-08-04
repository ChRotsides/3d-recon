"""Continuous-capture stitching — PLAN.md §6.3 path A.

    "If the user walks between rooms with the AR session live, everything is
     already in one coordinate frame. Detect room transitions via doorway
     crossing. Add a loop-closure prompt at the end and optimize."

This is the *preferred* path when it works, and it is a fundamentally easier
problem than path B: there is no relative transform to solve, because the rooms
were never in separate frames. What has to be solved instead is **segmentation**
— deciding which keyframes belong to which room — and the plan's answer is to
watch the trajectory cross doorways.

The value is not just convenience. A path-B edge is an ICP fit with centimetres
of uncertainty; a path-A edge is the identity, and its only error is VIO drift
over the walk between the two rooms. That is why the edges built here carry much
tighter covariance, and why the house graph will lean on them over doorway
matches when both exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recon.layout.doorways import doorway_frame
from recon.schemas import Doorway, EdgeSource, Keyframe, RoomEdge, SE2Transform

#: How far off the doorway's centre a crossing may be and still count, as a
#: multiple of the doorway's own half-width. Slightly over 1 because people walk
#: through door frames at an angle, brushing the jamb.
CROSSING_WIDTH_TOLERANCE = 1.3
#: A crossing must clear the threshold by this much on both sides, so that
#: loitering in a doorway does not register as a stream of transitions.
CROSSING_CLEARANCE_M = 0.35
#: Keyframes in a visit shorter than this are treated as pass-through, not a room.
MIN_VISIT_KEYFRAMES = 4


@dataclass
class Crossing:
    """One passage through a doorway."""

    keyframe_index: int
    doorway_id: str
    #: +1 when moving along the doorway's stored normal, -1 against it.
    direction: int
    position: np.ndarray


@dataclass
class Visit:
    """A contiguous run of keyframes between two crossings."""

    index: int
    keyframes: list[int]
    entered_via: str | None = None
    left_via: str | None = None
    #: Assigned once visits are merged into rooms.
    room: int = -1

    @property
    def length(self) -> int:
        return len(self.keyframes)


@dataclass
class ContinuousSegmentation:
    visits: list[Visit] = field(default_factory=list)
    crossings: list[Crossing] = field(default_factory=list)
    #: room index -> keyframe indices, after merging revisits.
    rooms: dict[int, list[int]] = field(default_factory=dict)

    @property
    def n_rooms(self) -> int:
        return len(self.rooms)


def detect_crossings(
    positions: np.ndarray,
    doorways: list[Doorway],
    up: np.ndarray,
    *,
    width_tolerance: float = CROSSING_WIDTH_TOLERANCE,
    clearance_m: float = CROSSING_CLEARANCE_M,
) -> list[Crossing]:
    """Find where a camera trajectory passes through each doorway.

    A crossing is a sign change of the signed distance to the doorway plane,
    with three qualifications, each of which exists because of a way the naive
    version is wrong:

    * the crossing point must be **within the opening**, or walking past a door
      along the wall registers as going through it;
    * the trajectory must **clear the plane on both sides**, or standing in a
      doorway produces a burst of transitions as tracking noise jitters the
      sign;
    * the height is checked too, so a doorway on another floor of the same
      coordinate frame is not crossed by someone walking below it.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    pos = np.asarray(positions, dtype=np.float64)
    out: list[Crossing] = []
    if len(pos) < 2:
        return out

    for dw in doorways:
        centre, normal, along = doorway_frame(dw, up)
        signed = (pos - centre) @ normal
        lateral = np.abs((pos - centre) @ along)
        height = (pos - centre) @ up
        half_width = max(dw.width_m, 0.6) * 0.5 * width_tolerance

        for i in range(len(pos) - 1):
            a, b = signed[i], signed[i + 1]
            if a == 0.0 or np.sign(a) == np.sign(b):
                continue
            # Interpolate to the plane and test the crossing point, not the
            # endpoints — a long step can straddle a doorway it missed.
            t = a / (a - b)
            lat = lateral[i] + t * (lateral[i + 1] - lateral[i])
            hgt = height[i] + t * (height[i + 1] - height[i])
            if lat > half_width:
                continue
            if not (-0.2 <= hgt <= max(dw.height_m, 2.0) + 0.4):
                continue
            if not _clears(signed, i, clearance_m):
                continue
            out.append(
                Crossing(
                    keyframe_index=i + 1,
                    doorway_id=dw.id,
                    direction=int(np.sign(b)),
                    position=pos[i] + t * (pos[i + 1] - pos[i]),
                )
            )

    out.sort(key=lambda c: c.keyframe_index)
    return _dedupe(out)


def _clears(signed: np.ndarray, i: int, clearance_m: float) -> bool:
    """Did the trajectory actually get away from the plane on both sides?"""
    before = signed[max(0, i - 8) : i + 1]
    after = signed[i + 1 : i + 9]
    if not len(before) or not len(after):
        return False
    return bool(np.max(np.abs(before)) >= clearance_m and np.max(np.abs(after)) >= clearance_m)


def _dedupe(crossings: list[Crossing], min_gap: int = 3) -> list[Crossing]:
    """Collapse repeated detections of one physical passage."""
    out: list[Crossing] = []
    for c in crossings:
        if out and c.doorway_id == out[-1].doorway_id and (
            c.keyframe_index - out[-1].keyframe_index
        ) < min_gap:
            continue
        out.append(c)
    return out


def segment(
    positions: np.ndarray,
    doorways: list[Doorway],
    up: np.ndarray,
    *,
    min_visit: int = MIN_VISIT_KEYFRAMES,
    merge_radius_m: float = 2.5,
) -> ContinuousSegmentation:
    """Split a continuous capture into rooms.

    Visits are the runs between crossings; rooms are visits merged. Merging
    matters because a real capture revisits: you walk to the far bedroom and
    come back through the hall, and the hall is one room seen twice, not two.
    Revisits are recognised by their centroids landing close together, which is
    sound precisely because path A never left the shared frame.
    """
    pos = np.asarray(positions, dtype=np.float64)
    crossings = detect_crossings(pos, doorways, up)

    bounds = [0, *[c.keyframe_index for c in crossings], len(pos)]
    visits: list[Visit] = []
    for k in range(len(bounds) - 1):
        idx = list(range(bounds[k], bounds[k + 1]))
        if not idx:
            continue
        visits.append(
            Visit(
                index=len(visits),
                keyframes=idx,
                entered_via=crossings[k - 1].doorway_id if k > 0 else None,
                left_via=crossings[k].doorway_id if k < len(crossings) else None,
            )
        )

    # Short runs are someone stepping through a doorway, not a room.
    real = [v for v in visits if v.length >= min_visit]
    if not real:
        real = visits[:1]

    centroids = [pos[v.keyframes].mean(axis=0) for v in real]
    rooms: dict[int, list[int]] = {}
    assigned: list[np.ndarray] = []
    for v, c in zip(real, centroids, strict=True):
        for r, seen in enumerate(assigned):
            if float(np.linalg.norm(c - seen)) <= merge_radius_m:
                v.room = r
                rooms[r].extend(v.keyframes)
                break
        else:
            v.room = len(assigned)
            assigned.append(c)
            rooms[v.room] = list(v.keyframes)

    for r in rooms:
        rooms[r] = sorted(rooms[r])
    return ContinuousSegmentation(visits=real, crossings=crossings, rooms=rooms)


def split_keyframes(
    keyframes: list[Keyframe], seg: ContinuousSegmentation
) -> dict[int, list[Keyframe]]:
    return {r: [keyframes[i] for i in idx] for r, idx in seg.rooms.items()}


#: Covariance for a continuous-tracking edge.
#:
#: Far tighter than a doorway match, and legitimately so: the two rooms were
#: never in different frames, so the only error is VIO drift accumulated over
#: the walk between them — metres of travel, not a registration problem.
#: Deliberately not zero; treating it as exact would let a long walk's drift
#: propagate as if it were ground truth.
CONTINUOUS_SIGMA_YAW_RAD = np.radians(0.5)
CONTINUOUS_SIGMA_M = 0.02


def edges_from_segmentation(
    seg: ContinuousSegmentation, room_ids: dict[int, str]
) -> list[RoomEdge]:
    """Build house-graph edges from a continuous capture.

    The transform is the **identity**: path A's whole point is that both rooms
    are already in one frame, so there is nothing to solve. What the edge
    carries is the doorway that joins them and a covariance saying how much the
    shared frame can be trusted.
    """
    cov = np.diag([
        CONTINUOUS_SIGMA_YAW_RAD**2, CONTINUOUS_SIGMA_M**2, CONTINUOUS_SIGMA_M**2
    ]).tolist()
    edges: list[RoomEdge] = []
    seen: set[tuple[int, int]] = set()

    ordered = sorted(seg.visits, key=lambda v: v.keyframes[0])
    for prev, nxt in zip(ordered, ordered[1:], strict=False):
        a, b = prev.room, nxt.room
        if a < 0 or b < 0 or a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        doorway = prev.left_via or nxt.entered_via or "unknown"
        edges.append(
            RoomEdge(
                from_room=room_ids[a],
                to_room=room_ids[b],
                via_doorway=[doorway, doorway],
                transform=SE2Transform(),
                covariance=cov,
                source=EdgeSource.CONTINUOUS_TRACKING,
            )
        )
    return edges
