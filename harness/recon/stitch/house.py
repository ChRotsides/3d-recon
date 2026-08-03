"""Whole-house assembly (PLAN.md §6.3, §5).

Ties the pieces together: reconstruct each room, pull its doorways out of the
layout, pair them across rooms, solve the 3-DoF alignments, and hand the lot to
the house pose graph. The output is the canonical `HouseGraph` plus the
transform that puts every room into one frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recon import geom
from recon.io.bundle import BundleDir
from recon.layout.doorways import doorway_frame, extract_doorways
from recon.pipeline import Pipeline, PipelineResult
from recon.schemas import Doorway, EdgeSource, HouseGraph, RoomLayout
from recon.stitch.align import Alignment, align_doorways, match_doorways
from recon.stitch.posegraph import HousePoseGraph, PoseGraphResult

#: Two rooms joined at a doorway sit side by side; they must not occupy the same
#: floor. Above this overlap fraction the pairing is wrong, whatever ICP says.
MAX_FOOTPRINT_OVERLAP = 0.15
#: A loop-closure candidate's two doorways must already land this close together
#: in the current solution. Generous, because the drift the closure is meant to
#: correct is exactly what pushes them apart.
LOOP_CLOSURE_SEARCH_M = 0.60


@dataclass
class RoomReconstruction:
    room_id: str
    layout: RoomLayout
    doorways: list[Doorway]
    scale: float
    scale_sigma: float
    points: np.ndarray
    normals: np.ndarray

    @classmethod
    def from_pipeline(cls, room_id: str, res: PipelineResult, up: np.ndarray, min_weight: float
                      ) -> RoomReconstruction:
        pts, nrm = res.volume.surface_points(min_weight=min_weight)
        return cls(
            room_id=room_id,
            layout=res.layout,
            doorways=extract_doorways(res.layout, up, prefix=f"{room_id}_dw"),
            scale=res.scale.scale,
            scale_sigma=res.scale.sigma,
            points=pts,
            normals=nrm,
        )


@dataclass
class HouseResult:
    graph: HouseGraph
    pose: PoseGraphResult
    rooms: dict[str, RoomReconstruction]
    alignments: list[tuple[str, str, str, str, Alignment]] = field(default_factory=list)
    rejected: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def total_area_m2(self) -> float:
        return float(sum(r.layout.area_m2 for r in self.rooms.values()))

    def summary(self) -> dict:
        return {
            "property_id": self.graph.property_id,
            "rooms": len(self.rooms),
            "edges": len(self.graph.floors[0].edges) if self.graph.floors else 0,
            "rejected_pairs": len(self.rejected),
            "total_area_m2": self.total_area_m2,
            "ambiguous": self.pose.ambiguous,
            "ambiguity_note": self.pose.ambiguity_note,
            "suspected_outliers": self.pose.suspected_outliers,
            "scales": {r: n.scale for r, n in self.pose.nodes.items()},
        }


def reconstruct_rooms(
    bundle_dirs: list[Path],
    pipeline: Pipeline,
    *,
    out_root: Path | None = None,
) -> dict[str, RoomReconstruction]:
    rooms: dict[str, RoomReconstruction] = {}
    for d in bundle_dirs:
        bd = BundleDir.load(d)
        res = pipeline.run(bd, (out_root / d.name) if out_root else None)
        rooms[bd.bundle.room_id] = RoomReconstruction.from_pipeline(
            bd.bundle.room_id, res, bd.bundle.up, pipeline.config.tsdf.min_weight
        )
    return rooms


def stitch(
    rooms: dict[str, RoomReconstruction],
    connections: list[tuple[str, str]],
    up: np.ndarray,
    *,
    property_id: str = "property",
    pairing: dict[tuple[str, str], tuple[str, str]] | None = None,
    solve_scale: bool = True,
    find_loops: bool = True,
) -> HouseResult:
    """Join reconstructed rooms into one house.

    `connections` names which room pairs share a doorway — the user's tagging
    (§6.3 path B step 1). Which *specific* doorway on each side can be given in
    `pairing`; otherwise the best size match is used, because that is the only
    signal available without the tap.
    """
    graph = HousePoseGraph(solve_scale=solve_scale)
    for rid, room in rooms.items():
        graph.add_room(
            rid,
            scale_prior=room.scale,
            scale_sigma=max(room.scale_sigma, 1e-3),
            doorway_widths={d.id: d.width_m for d in room.doorways},
        )

    alignments: list[tuple[str, str, str, str, Alignment]] = []
    rejected: list[tuple[str, str, str]] = []
    # A doorway joins exactly two rooms. Without this, a corridor of
    # same-sized doors lets one doorway be consumed twice — room B gets attached
    # through the door it already reached room A by, which places it on the
    # wrong side as a clean half-turn that neither ICP nor an overlap test
    # objects to, because the result does not intersect anything.
    used_doorways: set[str] = set()

    for a_id, b_id in connections:
        a, b = rooms[a_id], rooms[b_id]
        if not a.doorways or not b.doorways:
            rejected.append((a_id, b_id, "one of the rooms has no detected doorway"))
            continue

        chosen = (pairing or {}).get((a_id, b_id))
        if chosen:
            ia = next(i for i, d in enumerate(a.doorways) if d.id == chosen[0])
            ib = next(i for i, d in enumerate(b.doorways) if d.id == chosen[1])
            candidates = [(ia, ib, 0.0)]
        else:
            candidates = match_doorways(a.doorways, b.doorways)
            if not candidates:
                rejected.append((a_id, b_id, "no doorway pair with compatible dimensions"))
                continue

        best: Alignment | None = None
        best_ids: tuple[str, str] | None = None
        overlap_note = ""
        for ia, ib, _ in candidates[:6]:
            if (a.doorways[ia].id in used_doorways
                    or b.doorways[ib].id in used_doorways):
                continue
            al = align_doorways(
                a.doorways[ia], b.doorways[ib], up,
                points_a=a.points, points_b=b.points, normals_a=a.normals,
            )
            # ICP alone cannot tell a doorway from the one at the other end of a
            # corridor: both give a plausible local fit, and the wrong one comes
            # out as a half-turn that stacks the two rooms on the same floor.
            # Interpenetration is the decisive check, and it is nearly free.
            ov = footprint_overlap(a.layout, b.layout, al, up)
            if ov > MAX_FOOTPRINT_OVERLAP:
                overlap_note = (
                    f"best doorway pairing put {ov*100:.0f}% of {b_id} inside {a_id}"
                )
                continue
            # A verified fit always beats an unverified one, whatever its RMSE:
            # an unverified edge never had the §6.3 gate applied to it at all.
            better = (
                best is None
                or (al.accepted, al.verified) > (best.accepted, best.verified)
                or ((al.accepted, al.verified) == (best.accepted, best.verified)
                    and _better(al, best))
            )
            if better:
                best, best_ids = al, (a.doorways[ia].id, b.doorways[ib].id)

        if best is None:
            rejected.append((
                a_id, b_id,
                overlap_note or "no unused doorway pairing survived the checks",
            ))
            continue
        if not best.accepted:
            rejected.append((a_id, b_id, best.reason))
            continue

        used_doorways.update(best_ids)
        alignments.append((a_id, b_id, best_ids[0], best_ids[1], best))
        graph.add_edge(
            best.to_edge(a_id, b_id, best_ids[0], best_ids[1], EdgeSource.USER_TAGGED)
        )

    pose = graph.optimise()

    if find_loops:
        closures = find_loop_closures(rooms, pose, graph, up)
        if closures:
            for a_id, b_id, dw_a, dw_b, al in closures:
                alignments.append((a_id, b_id, dw_a, dw_b, al))
                graph.add_edge(
                    al.to_edge(a_id, b_id, dw_a, dw_b, EdgeSource.AUTO_REGISTERED)
                )
            # Re-solve with the extra constraints. Plan §6.3 lists loop closures
            # as a graph input precisely because an over-determined graph is what
            # lets accumulated drift be pulled back rather than merely carried.
            pose = graph.optimise()

    disconnected = _disconnected_rooms(graph)
    for rid in disconnected:
        rejected.append((rid, "", "room has no accepted doorway edge; its placement is arbitrary"))
    return HouseResult(
        graph=graph.to_house_graph(property_id),
        pose=pose,
        rooms=rooms,
        alignments=alignments,
        rejected=rejected,
    )


def footprint_overlap(
    layout_a, layout_b, alignment: Alignment, up: np.ndarray
) -> float:
    """Fraction of room B's footprint that lands inside room A's, once aligned.

    Two rooms joined by a doorway share a wall, not a floor. Anything much above
    zero means the pairing is wrong — most often a half-turn that folds one room
    back over the other, which looks entirely reasonable in the ICP residual.
    """
    pa = np.asarray(layout_a.footprint, dtype=np.float64)
    pb = np.asarray(layout_b.footprint, dtype=np.float64)
    if len(pa) < 3 or len(pb) < 3:
        return 0.0

    c, s = np.cos(alignment.yaw_rad), np.sin(alignment.yaw_rad)
    # SE2Transform rotates about +Y as [[c,0,s],[0,1,0],[-s,0,c]]; in (x, z) that
    # is [[c, s], [-s, c]].
    moved = np.column_stack([
        c * pb[:, 0] + s * pb[:, 1] + alignment.tx,
        -s * pb[:, 0] + c * pb[:, 1] + alignment.tz,
    ])

    # Sample B's interior and ask how much of it is inside A. Sampling beats an
    # exact polygon clip here: both footprints are small and possibly non-convex,
    # and a fraction is all the decision needs.
    lo, hi = moved.min(axis=0), moved.max(axis=0)
    n = 40
    gx = np.linspace(lo[0], hi[0], n)
    gz = np.linspace(lo[1], hi[1], n)
    grid = np.stack(np.meshgrid(gx, gz), axis=-1).reshape(-1, 2)
    in_b = geom.point_in_polygon(grid, moved)
    if not in_b.any():
        return 0.0
    in_a = geom.point_in_polygon(grid[in_b], geom.ensure_ccw(pa))
    return float(in_a.mean())


def _better(a: Alignment, b: Alignment) -> bool:
    """Lower ICP RMSE wins; an unrefined match (NaN RMSE) never beats a refined one."""
    if not np.isfinite(a.rmse_m):
        return False
    if not np.isfinite(b.rmse_m):
        return True
    return a.rmse_m < b.rmse_m


def find_loop_closures(
    rooms: dict[str, RoomReconstruction],
    pose: PoseGraphResult,
    graph: HousePoseGraph,
    up: np.ndarray,
    *,
    search_m: float = LOOP_CLOSURE_SEARCH_M,
) -> list[tuple[str, str, str, str, Alignment]]:
    """Find doorways that join two rooms the user did not tag as connected.

    Plan §6.3 lists loop closures among the graph's inputs — "any room pair with
    overlapping geometry" — but nothing was producing them, so the graph was
    always a tree and every doorway was load-bearing. In a house with a loop the
    closing edge is redundant information, and redundancy is the only thing that
    lets accumulated drift be pulled back rather than merely carried to the end.

    The search is cheap because the current solution already places every room
    roughly: a genuine closure has two *unused* doorways that land near each
    other with opposed normals. Each candidate still goes through the full ICP
    and its acceptance gates, so a coincidence of position is not enough.
    """
    used: set[str] = set()
    connected: set[tuple[str, str]] = set()
    for e in graph.edges:
        used.update(e.via_doorway)
        connected.add((e.from_room, e.to_room))
        connected.add((e.to_room, e.from_room))

    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)

    # Every unused doorway, placed in the house frame.
    placed: list[tuple[str, Doorway, np.ndarray, np.ndarray]] = []
    for rid, room in rooms.items():
        for dw in room.doorways:
            if dw.id in used:
                continue
            c, n, _ = doorway_frame(dw, up)
            centre = pose.apply(rid, c[None, :])[0]
            # Rotate the normal by the room's yaw; scale does not affect it.
            yaw = pose.nodes[rid].yaw_rad
            cs, sn = np.cos(yaw), np.sin(yaw)
            rot = np.array([[cs, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, cs]])
            placed.append((rid, dw, centre, rot @ n))

    out: list[tuple[str, str, str, str, Alignment]] = []
    taken: set[str] = set()
    candidates = []
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            rid_a, dw_a, c_a, n_a = placed[i]
            rid_b, dw_b, c_b, n_b = placed[j]
            if rid_a == rid_b or (rid_a, rid_b) in connected:
                continue
            gap = float(np.linalg.norm(c_a - c_b))
            if gap > search_m:
                continue
            # Opposed normals: the two rooms see the same door from either side.
            if float(n_a @ n_b) > -0.5:
                continue
            candidates.append((gap, rid_a, dw_a, rid_b, dw_b))

    for _gap, rid_a, dw_a, rid_b, dw_b in sorted(candidates, key=lambda t: t[0]):
        if dw_a.id in taken or dw_b.id in taken:
            continue
        a, b = rooms[rid_a], rooms[rid_b]
        al = align_doorways(
            dw_a, dw_b, up,
            points_a=a.points, points_b=b.points, normals_a=a.normals,
        )
        if not al.accepted or not al.verified:
            continue
        if footprint_overlap(a.layout, b.layout, al, up) > MAX_FOOTPRINT_OVERLAP:
            continue
        taken.update({dw_a.id, dw_b.id})
        out.append((rid_a, rid_b, dw_a.id, dw_b.id, al))
    return out


def _disconnected_rooms(graph: HousePoseGraph) -> list[str]:
    """Rooms with no path to the rest of the house.

    Their pose is unconstrained, so the optimiser will leave them wherever the
    initial guess put them. Reporting that is essential: a silently floating
    room looks like drift, and someone will spend a day tuning the solver for
    what is really a missing doorway tag.
    """
    if not graph.nodes:
        return []
    adj: dict[str, set[str]] = {r: set() for r in graph.nodes}
    for e in graph.edges:
        adj[e.from_room].add(e.to_room)
        adj[e.to_room].add(e.from_room)
    root = next(iter(graph.nodes))
    seen = {root}
    stack = [root]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return sorted(set(graph.nodes) - seen)


def house_footprints(result: HouseResult) -> dict[str, np.ndarray]:
    """Every room's footprint transformed into the shared house frame."""
    out: dict[str, np.ndarray] = {}
    for rid, room in result.rooms.items():
        poly = np.asarray(room.layout.footprint, dtype=np.float64)
        if len(poly) < 3:
            continue
        pts3 = np.column_stack([poly[:, 0], np.zeros(len(poly)), poly[:, 1]])
        moved = result.pose.apply(rid, pts3)
        out[rid] = np.column_stack([moved[:, 0], moved[:, 2]])
    return out
