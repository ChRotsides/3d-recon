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

from recon.io.bundle import BundleDir
from recon.layout.doorways import extract_doorways
from recon.pipeline import Pipeline, PipelineResult
from recon.schemas import Doorway, EdgeSource, HouseGraph, RoomLayout
from recon.stitch.align import Alignment, align_doorways, match_doorways
from recon.stitch.posegraph import HousePoseGraph, PoseGraphResult


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
        for ia, ib, _ in candidates[:4]:
            al = align_doorways(
                a.doorways[ia], b.doorways[ib], up,
                points_a=a.points, points_b=b.points, normals_a=a.normals,
            )
            if best is None or (al.accepted and not best.accepted) or (
                al.accepted == best.accepted and _better(al, best)
            ):
                best, best_ids = al, (a.doorways[ia].id, b.doorways[ib].id)

        if best is None or not best.accepted:
            rejected.append((a_id, b_id, best.reason if best else "no alignment"))
            continue

        alignments.append((a_id, b_id, best_ids[0], best_ids[1], best))
        graph.add_edge(
            best.to_edge(a_id, b_id, best_ids[0], best_ids[1], EdgeSource.USER_TAGGED)
        )

    pose = graph.optimise()
    return HouseResult(
        graph=graph.to_house_graph(property_id),
        pose=pose,
        rooms=rooms,
        alignments=alignments,
        rejected=rejected,
    )


def _better(a: Alignment, b: Alignment) -> bool:
    """Lower ICP RMSE wins; an unrefined match (NaN RMSE) never beats a refined one."""
    if not np.isfinite(a.rmse_m):
        return False
    if not np.isfinite(b.rmse_m):
        return True
    return a.rmse_m < b.rmse_m


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
