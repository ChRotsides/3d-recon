"""Viewer scene description (PLAN.md §1.1, §3.8).

The v1 deliverable is a "photorealistic walkthrough of the whole property", and
§3.8 chooses an embedded WebView running Babylon.js rather than two native splat
renderers. That needs one thing the pipeline does not otherwise produce: a
manifest tying the per-room artifacts together in the house frame.

This is that manifest. It is deliberately a *description*, not a bundle — it
holds relative paths and transforms, so the same file works served from S3, from
a local directory, or from inside the app.

The coordinate frame is stated explicitly, because Babylon.js is **left-handed
by default** while everything upstream here is right-handed (plan §5). A viewer
that ignores this renders a mirrored house that looks entirely plausible — the
exact failure §7 warns about — so the frame travels with the scene and the
viewer is expected to act on it rather than guess.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from recon.schemas import CANONICAL_FRAME, RoomLayout

#: What the viewer must do to our right-handed, Y-up scene to display it
#: correctly in Babylon's default left-handed frame.
BABYLON_CONVERSION = "negate_z"


@dataclass
class ViewerRoom:
    room_id: str
    #: Relative paths to this room's artifacts, by kind.
    artifacts: dict[str, str]
    #: Room -> house transform. Yaw about +Y, translation in the floor plane.
    yaw_rad: float = 0.0
    tx: float = 0.0
    tz: float = 0.0
    scale: float = 1.0
    #: (x, z) floor polygon in the room's own frame, for the plan overlay.
    footprint: list[list[float]] = field(default_factory=list)
    area_m2: float = 0.0
    ceiling_height_m: float = 0.0
    #: A sensible place to drop the camera when entering this room.
    spawn: list[float] = field(default_factory=lambda: [0.0, 1.6, 0.0])


@dataclass
class ViewerScene:
    property_id: str
    rooms: list[ViewerRoom] = field(default_factory=list)
    #: Doorway centroids in the house frame, so the viewer can offer to walk
    #: through rather than making the user find the gap.
    portals: list[dict[str, Any]] = field(default_factory=list)
    frame: dict[str, Any] = field(default_factory=lambda: {
        **json.loads(CANONICAL_FRAME.model_dump_json()),
        "viewer_conversion": BABYLON_CONVERSION,
    })
    total_area_m2: float = 0.0
    schema_version: int = 1

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> ViewerScene:
        raw = json.loads(text)
        rooms = [ViewerRoom(**r) for r in raw.pop("rooms", [])]
        return cls(rooms=rooms, **raw)


def spawn_point(layout: RoomLayout, eye_height_m: float = 1.6) -> list[float]:
    """Somewhere inside the room to put the camera.

    The centroid of a non-convex footprint can fall outside it — an L-shaped
    room would spawn the viewer inside a wall — so the polygon is sampled and
    the interior point furthest from any wall wins.
    """
    from recon import geom

    poly = np.asarray(layout.footprint, dtype=np.float64)
    if len(poly) < 3:
        return [0.0, eye_height_m, 0.0]

    lo, hi = poly.min(axis=0), poly.max(axis=0)
    gx = np.linspace(lo[0], hi[0], 24)
    gz = np.linspace(lo[1], hi[1], 24)
    grid = np.stack(np.meshgrid(gx, gz), axis=-1).reshape(-1, 2)
    inside = geom.point_in_polygon(grid, geom.ensure_ccw(poly))
    if not inside.any():
        c = poly.mean(axis=0)
        return [float(c[0]), eye_height_m, float(c[1])]
    cand = grid[inside]
    clearance = geom.distance_to_polygon(cand, poly)
    best = cand[int(np.argmax(clearance))]
    return [float(best[0]), float(layout.floor_height_m + eye_height_m), float(best[1])]


def build_scene(
    property_id: str,
    rooms: dict[str, dict[str, Any]],
    *,
    house_result=None,
) -> ViewerScene:
    """Assemble the manifest.

    `rooms` maps room_id -> {"layout": RoomLayout, "artifacts": {kind: path}}.
    `house_result` is an optional `HouseResult`; without it every room sits at
    the origin, which is the correct behaviour for a single-room scan.
    """
    scene = ViewerScene(property_id=property_id)
    for room_id, info in rooms.items():
        layout: RoomLayout = info["layout"]
        node = getattr(house_result, "pose", None)
        placement = node.nodes.get(room_id) if node else None
        scene.rooms.append(
            ViewerRoom(
                room_id=room_id,
                artifacts=dict(info.get("artifacts", {})),
                yaw_rad=float(placement.yaw_rad) if placement else 0.0,
                tx=float(placement.tx) if placement else 0.0,
                tz=float(placement.tz) if placement else 0.0,
                scale=float(placement.scale) if placement else 1.0,
                footprint=[list(map(float, p)) for p in layout.footprint],
                area_m2=float(layout.area_m2),
                ceiling_height_m=float(layout.ceiling_height_m),
                spawn=spawn_point(layout),
            )
        )
    scene.total_area_m2 = float(sum(r.area_m2 for r in scene.rooms))

    if house_result is not None:
        for edge in _edges(house_result):
            a, b = edge.from_room, edge.to_room
            room_a = next((r for r in scene.rooms if r.room_id == a), None)
            if room_a is None:
                continue
            scene.portals.append({
                "from_room": a, "to_room": b,
                "via_doorway": list(edge.via_doorway),
                "source": edge.source.value,
            })
    return scene


def _edges(house_result) -> list:
    graph = getattr(house_result, "graph", None)
    if graph is None or not graph.floors:
        return []
    return list(graph.floors[0].edges)


def write_scene(path: Path, scene: ViewerScene) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scene.to_json())
    return path
