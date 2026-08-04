"""Multi-room synthetic house (PLAN.md §1.3, §6.3).

The whole-house drift target — 15 cm across six rooms — cannot be checked with
single-room scenes. This builds a house as rooms laid out in a shared *true*
frame, then re-expresses each room in its **own** local frame with its own VIO
scale error and yaw offset, exactly as independently scanned rooms arrive.

Ground truth is therefore known twice over: the true placement of every room in
the house frame, and the true transform that stitching has to recover. Drift is
measured against the first, not inferred.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recon import geom, synth
from recon.schemas import Doorway


@dataclass
class RoomPlacement:
    """A room's true position in the house."""

    room_id: str
    spec: synth.RoomSpec
    #: True house-frame transform of this room's local origin.
    yaw_rad: float = 0.0
    tx: float = 0.0
    tz: float = 0.0
    #: The VIO scale error this room was "scanned" with.
    vio_scale_error: float = 1.0
    #: Index of the wall carrying the door that joins it to its parent.
    door_wall: int = 0

    def matrix(self) -> np.ndarray:
        c, s = np.cos(self.yaw_rad), np.sin(self.yaw_rad)
        m = np.eye(4)
        m[0, 0], m[0, 2] = c, s
        m[2, 0], m[2, 2] = -s, c
        m[0, 3], m[2, 3] = self.tx, self.tz
        return m

    def to_house(self, pts: np.ndarray) -> np.ndarray:
        m = self.matrix()
        return np.asarray(pts, dtype=np.float64) @ m[:3, :3].T + m[:3, 3]


@dataclass
class HouseSpec:
    name: str
    rooms: list[RoomPlacement] = field(default_factory=list)
    #: (room_a, room_b, doorway_index_a, doorway_index_b) connections.
    connections: list[tuple[int, int]] = field(default_factory=list)


#: Shared depth so consecutive rooms' connecting walls line up exactly.
CORRIDOR_DEPTH_M = 3.35
#: Where along the shared wall the connecting door sits.
DOOR_Z_M = 1.675


def corridor_room(
    name: str,
    width_m: float,
    *,
    ceiling_m: float = 2.45,
    door_left: bool = False,
    door_right: bool = False,
    window: bool = False,
) -> synth.RoomSpec:
    """A rectangular room with doors on its -x and/or +x walls.

    The connecting doors must sit on the walls the neighbouring rooms are
    actually behind. Putting them on an arbitrary wall produces a "house" whose
    rooms do not connect through the doors that supposedly join them — the
    stitching then has nothing real to solve and any result it produces is
    meaningless.

    Footprint order is [(0,0), (w,0), (w,d), (0,d)], so wall 1 is the +x wall
    (running (w,0) -> (w,d)) and wall 3 is the -x wall (running (0,d) -> (0,0)).
    """
    d = CORRIDOR_DEPTH_M
    openings: list[synth.OpeningSpec] = []
    if door_right:
        # Wall 1 runs +z from (w, 0), so u is measured from z = 0.
        openings.append(
            synth.OpeningSpec(
                1, DOOR_Z_M, synth.DEFAULT_DOOR_WIDTH_M, synth.DEFAULT_DOOR_HEIGHT_M,
                kind="door",
            )
        )
    if door_left:
        # Wall 3 runs -z from (0, d), so the same physical z maps to d - z.
        openings.append(
            synth.OpeningSpec(
                3, d - DOOR_Z_M, synth.DEFAULT_DOOR_WIDTH_M, synth.DEFAULT_DOOR_HEIGHT_M,
                kind="door",
            )
        )
    if window:
        openings.append(
            synth.OpeningSpec(0, width_m * 0.5, 1.30, 1.20, sill_m=0.95, kind="window")
        )
    return synth.RoomSpec(
        name=name,
        footprint=[(0.0, 0.0), (width_m, 0.0), (width_m, d), (0.0, d)],
        ceiling_height_m=ceiling_m,
        openings=openings,
        exterior=window,
    )


#: Which wall index carries which outward direction, for the footprint
#: [(0,0), (w,0), (w,d), (0,d)] that `grid_room` builds.
WALL_FOR_DIRECTION = {"-z": 0, "+x": 1, "+z": 2, "-x": 3}


def grid_room(
    name: str,
    width_m: float,
    depth_m: float,
    doors: set[str],
    *,
    ceiling_m: float = 2.45,
) -> synth.RoomSpec:
    """A rectangular room with doors on named sides ("+x", "-x", "+z", "-z").

    Doors are placed at the midpoint of their wall so that two rooms sharing a
    wall agree on where the opening is. Each wall runs in its own direction
    around the CCW footprint, so `u` is measured differently on each — getting
    that wrong puts the two halves of a shared doorway in different places and
    the rooms stitch into the wrong relative position.
    """
    w, d = width_m, depth_m
    openings: list[synth.OpeningSpec] = []
    dw, dh = synth.DEFAULT_DOOR_WIDTH_M, synth.DEFAULT_DOOR_HEIGHT_M
    # Wall 0 runs +x from (0,0); wall 1 runs +z from (w,0);
    # wall 2 runs -x from (w,d); wall 3 runs -z from (0,d).
    centres = {"-z": w / 2, "+x": d / 2, "+z": w / 2, "-x": d / 2}
    for side in sorted(doors):
        openings.append(
            synth.OpeningSpec(WALL_FOR_DIRECTION[side], centres[side], dw, dh, kind="door")
        )
    return synth.RoomSpec(
        name=name,
        footprint=[(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)],
        ceiling_height_m=ceiling_m,
        openings=openings,
        exterior=False,
    )


def loop_house(width_m: float = 4.0, depth_m: float = 3.4, seed: int = 0) -> HouseSpec:
    """Four rooms in a 2x2 block, every neighbour joined by a doorway.

    A chain has no redundancy: every doorway is load-bearing and an error in one
    propagates to everything downstream. A loop is the case plan §6.3's loop
    closures exist for — the fourth edge is over-determined, and using it is
    what pulls accumulated drift back.

    Layout (x right, z up):  room_3  room_2
                             room_0  room_1
    """
    rng = np.random.default_rng(seed)
    w, d = width_m, depth_m
    spec = [
        ("room_0", 0.0, 0.0, {"+x", "+z"}),
        ("room_1", w, 0.0, {"-x", "+z"}),
        ("room_2", w, d, {"-x", "-z"}),
        ("room_3", 0.0, d, {"+x", "-z"}),
    ]
    rooms = [
        RoomPlacement(
            room_id=rid,
            spec=grid_room(rid, w, d, doors),
            tx=tx, tz=tz,
            vio_scale_error=float(1.0 + rng.normal(0.0, 0.02)),
        )
        for rid, tx, tz, doors in spec
    ]
    return HouseSpec(
        name="loop",
        rooms=rooms,
        connections=[(0, 1), (1, 2), (2, 3), (3, 0)],
    )


def corridor_house(n_rooms: int = 6, seed: int = 0) -> HouseSpec:
    """`n_rooms` rooms in a row, each joined to the previous by one doorway.

    A chain is the worst case for drift — every room's error accumulates through
    every earlier joint with no loop closure to pull it back — which makes it the
    right shape for testing the §1.3 15 cm target.

    Consecutive rooms **share** their connecting wall (no gap), so a scan of one
    room genuinely sees through the doorway into the next. That overlap band is
    the only thing ICP has to work with in §6.3 path B; without it the rooms are
    sealed boxes and the refinement stage is untestable.
    """
    rng = np.random.default_rng(seed)
    rooms: list[RoomPlacement] = []
    connections: list[tuple[int, int]] = []

    widths = [float(rng.uniform(3.6, 4.6)) for _ in range(n_rooms)]
    x_cursor = 0.0
    for i, w in enumerate(widths):
        spec = corridor_room(
            f"room_{i}",
            w,
            ceiling_m=2.45,
            door_left=i > 0,
            door_right=i < n_rooms - 1,
            window=(i % 3 == 1),
        )
        rooms.append(
            RoomPlacement(
                room_id=f"room_{i}",
                spec=spec,
                yaw_rad=0.0,
                tx=x_cursor,
                tz=0.0,
                # A realistic spread of per-room VIO scale error (plan §6.2: ~3%).
                vio_scale_error=float(1.0 + rng.normal(0.0, 0.02)),
                door_wall=3 if i > 0 else 1,
            )
        )
        x_cursor += w  # walls are zero-thickness and shared
        if i > 0:
            connections.append((i - 1, i))

    return HouseSpec(name="corridor", rooms=rooms, connections=connections)


def build_house_mesh(house: HouseSpec) -> geom.TriMesh:
    """Every room's mesh, placed in the shared house frame, plus one exterior.

    The exterior is built once around the whole property. Letting each room
    bring its own would place a slab 4 m beyond that room's bbox — i.e. inside
    the rooms next to it — and every scan of a neighbour would then measure the
    intruding slab rather than its own wall.
    """
    parts = []
    corners = []
    wants_exterior = False
    for p in house.rooms:
        room = _without_exterior(p.spec)
        wants_exterior = wants_exterior or (
            p.spec.exterior and any(o.kind == "window" for o in p.spec.openings)
        )
        m = synth.build_mesh(room)
        w = p.matrix()
        pts = p.spec.polygon
        pts3 = np.column_stack([pts[:, 0], np.zeros(len(pts)), pts[:, 1]])
        moved = pts3 @ w[:3, :3].T + w[:3, 3]
        corners.append(np.column_stack([moved[:, 0], moved[:, 2]]))
        parts.append(
            geom.TriMesh(
                vertices=m.vertices @ w[:3, :3].T + w[:3, 3],
                faces=m.faces,
                face_normals=m.face_normals @ w[:3, :3].T,
                face_labels=m.face_labels,
                face_colors=m.face_colors,
            )
        )

    if wants_exterior and corners:
        allc = np.vstack(corners)
        ceiling = max(p.spec.ceiling_height_m for p in house.rooms)
        parts.append(synth.exterior_box(allc.min(axis=0), allc.max(axis=0), ceiling))
    return geom.TriMesh.concat(parts)


def _without_exterior(spec: synth.RoomSpec) -> synth.RoomSpec:
    return synth.RoomSpec(
        name=spec.name, footprint=list(spec.footprint),
        ceiling_height_m=spec.ceiling_height_m, openings=list(spec.openings),
        panels=list(spec.panels), boxes=list(spec.boxes), exterior=False,
    )


def continuous_path(house: HouseSpec, per_station: int = 1) -> list[tuple[np.ndarray, str]]:
    """A walking route through the whole house, in the house frame.

    Path A's premise is one live AR session across several rooms, so the
    trajectory has to actually pass through the doorways rather than teleport
    between rooms — the segmentation is *defined* by those crossings.
    """
    out: list[tuple[np.ndarray, str]] = []
    for i, placement in enumerate(house.rooms):
        poly = placement.spec.polygon
        stations = synth.plan_stations(placement.spec, max(per_station, 1))
        for st in stations:
            p = placement.to_house(np.array([[st[0], synth.STANDING_EYE_HEIGHT_M, st[1]]]))[0]
            out.append((p, placement.room_id))
        # Then step to the doorway that leads onward, so the walk is continuous.
        if i + 1 < len(house.rooms):
            nxt = house.rooms[i + 1]
            mid = 0.5 * (
                placement.to_house(
                    np.array([[poly[:, 0].mean(), synth.STANDING_EYE_HEIGHT_M,
                               poly[:, 1].mean()]])
                )[0]
                + nxt.to_house(
                    np.array([[nxt.spec.polygon[:, 0].mean(), synth.STANDING_EYE_HEIGHT_M,
                               nxt.spec.polygon[:, 1].mean()]])
                )[0]
            )
            out.append((mid, placement.room_id))
    return out


def room_containing(house: HouseSpec, position: np.ndarray) -> str | None:
    """Which room's footprint contains a house-frame point, if any."""
    p2 = np.array([[position[0], position[2]]])
    for placement in house.rooms:
        poly = placement.spec.polygon
        pts3 = np.column_stack([poly[:, 0], np.zeros(len(poly)), poly[:, 1]])
        moved = placement.to_house(pts3)
        if geom.point_in_polygon(p2, np.column_stack([moved[:, 0], moved[:, 2]]))[0]:
            return placement.room_id
    return None


def generate_continuous(
    house: HouseSpec,
    out_dir: Path,
    cfg: synth.CaptureConfig | None = None,
    *,
    room_id: str = "continuous",
    steps_between: int = 6,
) -> tuple[list[str], Path]:
    """Render one capture session that walks through every room (§6.3 path A).

    Emits a *single* bundle with every pose in the house frame, plus the true
    room label for each keyframe — which is the ground truth the segmentation is
    scored against.
    """
    from PIL import Image

    cfg = cfg or synth.CaptureConfig()
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)

    mesh = build_house_mesh(house)
    intr = synth.intrinsics_for(cfg)
    K = intr.matrix()
    waypoints = continuous_path(house)
    light = np.array([0.0, max(p.spec.ceiling_height_m for p in house.rooms) - 0.15, 0.0])

    keyframes = []
    labels: list[str] = []
    n = 0
    for (a, room_a), (b, _room_b) in zip(waypoints, waypoints[1:], strict=False):
        for k in range(steps_between):
            t = k / steps_between
            pos = a + t * (b - a)
            # Look along the walk, sweeping so walls are actually observed.
            yaw = float(np.arctan2(b[0] - a[0], -(b[2] - a[2]))) + np.sin(n * 0.9) * 1.2
            pitch = np.radians(-12.0 + 18.0 * np.sin(n * 0.6))
            R = synth._rotation(yaw, pitch)
            rgb, depth, sem = synth.render(mesh, K, R, pos, cfg.width, cfg.height, light,
                                           simulate_artifacts=cfg.simulate_artifacts)
            kf_id = f"kf_{n:04d}"
            Image.fromarray(rgb).save(out_dir / "images" / f"{kf_id}.png")
            Image.fromarray(
                np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            ).save(out_dir / "depth" / f"{kf_id}.png")
            Image.fromarray(
                np.clip(sem + 1, 0, 255).astype(np.uint8)
            ).save(out_dir / "masks" / f"{kf_id}.png")
            keyframes.append(
                synth.Keyframe(
                    id=kf_id, image=f"images/{kf_id}.png", depth=f"depth/{kf_id}.png",
                    timestamp_ns=int(n * 5e8),
                    pose=synth.Pose(R=R.tolist(), t=pos.tolist()),
                    intrinsics=intr, blur_score=300.0,
                )
            )
            # Label by which room actually *contains* the camera, not by which
            # waypoint leg it is on. The waypoint between two rooms sits past
            # the shared wall, so leg-based labelling puts keyframes that are
            # physically in the next room under the previous one — and then
            # scores a correct segmentation as wrong.
            labels.append(room_containing(house, pos) or room_a)
            n += 1

    bundle = synth.SessionBundle(
        session_id=str(uuid.uuid4()),
        device=synth.Device(model="synthetic", os="n/a",
                            ar_platform=synth.ARPlatform.SYNTHETIC,
                            ar_version="recon-synth/continuous"),
        room_id=room_id,
        gravity=[0.0, -1.0, 0.0],
        keyframes=keyframes,
        reference_measurements=[
            synth.ReferenceMeasurement(
                type=synth.MeasurementType.VIO_SCALE, measured_value_m=1.0, sigma_m=0.03
            )
        ],
    )
    (out_dir / "bundle.json").write_text(bundle.model_dump_json(indent=2))
    (out_dir / "room_labels.json").write_text(json.dumps(labels, indent=2))
    return labels, out_dir


@dataclass
class HouseGroundTruth:
    property_id: str
    rooms: list[dict]
    connections: list[list[int]]

    def to_json(self) -> str:
        return json.dumps(
            {
                "property_id": self.property_id,
                "rooms": self.rooms,
                "connections": self.connections,
            },
            indent=2,
        )

    def placement(self, room_id: str) -> dict:
        for r in self.rooms:
            if r["room_id"] == room_id:
                return r
        raise KeyError(room_id)


def generate_house(
    house: HouseSpec,
    out_dir: Path,
    cfg: synth.CaptureConfig | None = None,
    *,
    property_id: str | None = None,
) -> HouseGroundTruth:
    """Render every room as an independent capture session.

    Each room is emitted in its *own* local frame — that is the point. Stitching
    has to put them back together from doorway geometry alone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = cfg or synth.CaptureConfig()
    property_id = property_id or str(uuid.uuid4())

    # Rays are cast against the whole house so a scan sees through its doorways
    # into the neighbouring rooms. That overlap is what §6.3's ICP refines on.
    scene = build_house_mesh(house)

    rooms_gt = []
    for placement in house.rooms:
        room_cfg = synth.CaptureConfig(**{**base.__dict__})
        room_cfg.vio_scale_error = placement.vio_scale_error
        synth.generate(
            placement.spec,
            out_dir / placement.room_id,
            room_cfg,
            room_id=placement.room_id,
            scene_mesh=scene,
            world_from_room=placement.matrix(),
        )
        rooms_gt.append(
            {
                "room_id": placement.room_id,
                "preset": placement.spec.name,
                "yaw_rad": placement.yaw_rad,
                "tx": placement.tx,
                "tz": placement.tz,
                "vio_scale_error": placement.vio_scale_error,
                "area_m2": placement.spec.area_m2,
                "footprint": placement.spec.polygon.tolist(),
                "ceiling_height_m": placement.spec.ceiling_height_m,
            }
        )

    gt = HouseGroundTruth(
        property_id=property_id,
        rooms=rooms_gt,
        connections=[list(c) for c in house.connections],
    )
    (out_dir / "house_ground_truth.json").write_text(gt.to_json())
    return gt


# ---------------------------------------------------------------------------
# Analytic doorways, for testing the solver without running reconstruction
# ---------------------------------------------------------------------------


def analytic_doorway(spec: synth.RoomSpec, wall_index: int, up: np.ndarray) -> Doorway | None:
    """The exact doorway a room's spec defines, in the room's local frame.

    Lets the alignment maths be tested against an exact answer, separately from
    whether the opening *detector* found the door — two failures that would
    otherwise be indistinguishable.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    poly = spec.polygon
    for op in spec.openings:
        if op.kind != "door" or op.wall_index != wall_index:
            continue
        a2 = poly[op.wall_index]
        b2 = poly[(op.wall_index + 1) % len(poly)]
        d2 = (b2 - a2) / max(np.linalg.norm(b2 - a2), 1e-12)
        along = np.array([d2[0], 0.0, d2[1]])
        n = geom.inward_normal_2d(a2, b2)
        centroid = np.array([a2[0], 0.0, a2[1]]) + along * op.u_center_m
        return Doorway(
            id=f"dw_{wall_index:02d}",
            plane=[*n.tolist(), float(-(n @ centroid))],
            centroid=centroid.tolist(),
            width_m=op.width_m,
            height_m=op.height_m,
        )
    return None
