"""Synthetic room generator: a physically exact accuracy oracle for Phase 0.

Emits a real `SessionBundle` — poses, intrinsics, gravity, rendered RGB,
ground-truth metric depth and semantic masks — together with the room's *exact*
wall lengths, floor area and ceiling height. That turns every stage downstream
of depth prediction (masking -> fusion -> planes -> Atlanta -> floor plan) into
something measurable to machine precision, with no GPU and no laser measure.

It also reproduces the two failure modes that matter most (PLAN.md §7):

* **Mirrors** reflect the ray, so unmasked depth lands *behind* the wall.
* **Windows** pass the ray through to exterior geometry, producing the classic
  phantom room.

Both are on by default in the `mirror_heavy` preset, so the SAM-3 masking
mitigation can be scored rather than assumed.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recon import geom
from recon.geom import Semantic, TriMesh
from recon.schemas import (
    ARPlatform,
    Device,
    Intrinsics,
    Keyframe,
    MeasurementType,
    Pose,
    ReferenceMeasurement,
    SessionBundle,
    SparsePoint,
    TrackingState,
)

STANDING_EYE_HEIGHT_M = 1.5
DEFAULT_DOOR_HEIGHT_M = 2.03  # PLAN.md §6.2 door prior
DEFAULT_DOOR_WIDTH_M = 0.83


@dataclass
class OpeningSpec:
    """A rectangular hole in wall `wall_index`, positioned along that wall."""

    wall_index: int
    u_center_m: float
    width_m: float
    height_m: float
    sill_m: float = 0.0  # height of the bottom edge above the floor
    kind: str = "door"  # "door" | "window"

    @property
    def u_start(self) -> float:
        return self.u_center_m - self.width_m / 2

    @property
    def u_end(self) -> float:
        return self.u_center_m + self.width_m / 2


@dataclass
class PanelSpec:
    """A flat panel mounted on a wall: mirror, TV, or window glass."""

    wall_index: int
    u_center_m: float
    width_m: float
    height_m: float
    center_height_m: float
    label: Semantic = Semantic.MIRROR


@dataclass
class BoxSpec:
    """Axis-agnostic furniture box, for the cluttered presets."""

    center: tuple[float, float]  # (x, z)
    size: tuple[float, float, float]  # (sx, sy, sz)
    yaw_rad: float = 0.0
    color: tuple[float, float, float] = (0.45, 0.35, 0.28)


@dataclass
class RoomSpec:
    name: str
    #: CCW (x, z) floor polygon in metres.
    footprint: list[tuple[float, float]]
    ceiling_height_m: float = 2.45
    openings: list[OpeningSpec] = field(default_factory=list)
    panels: list[PanelSpec] = field(default_factory=list)
    boxes: list[BoxSpec] = field(default_factory=list)
    #: Adds exterior geometry behind windows so unmasked depth builds a phantom room.
    exterior: bool = True

    @property
    def polygon(self) -> np.ndarray:
        return geom.ensure_ccw(np.asarray(self.footprint, dtype=np.float64))

    @property
    def area_m2(self) -> float:
        return abs(geom.polygon_area(self.polygon))

    @property
    def wall_lengths_m(self) -> np.ndarray:
        return geom.edge_lengths(self.polygon)


# ---------------------------------------------------------------------------
# Presets — the four property types from PLAN.md §8
# ---------------------------------------------------------------------------


def preset(name: str) -> RoomSpec:
    presets = {
        "empty_rect": _empty_rect,
        "l_shaped": _l_shaped,
        "angled": _angled,
        "furnished": _furnished,
        "mirror_heavy": _mirror_heavy,
    }
    if name not in presets:
        raise KeyError(f"unknown preset {name!r}; have {sorted(presets)}")
    return presets[name]()


def _empty_rect() -> RoomSpec:
    return RoomSpec(
        name="empty_rect",
        footprint=[(0.0, 0.0), (4.20, 0.0), (4.20, 3.35), (0.0, 3.35)],
        ceiling_height_m=2.45,
        openings=[OpeningSpec(0, 1.60, DEFAULT_DOOR_WIDTH_M, DEFAULT_DOOR_HEIGHT_M, kind="door")],
    )


def _l_shaped() -> RoomSpec:
    return RoomSpec(
        name="l_shaped",
        footprint=[
            (0.0, 0.0), (5.10, 0.0), (5.10, 2.20),
            (3.00, 2.20), (3.00, 4.40), (0.0, 4.40),
        ],
        ceiling_height_m=2.52,
        openings=[
            OpeningSpec(0, 2.20, DEFAULT_DOOR_WIDTH_M, DEFAULT_DOOR_HEIGHT_M, kind="door"),
            OpeningSpec(5, 2.40, 1.30, 1.20, sill_m=0.95, kind="window"),
        ],
    )


def _angled() -> RoomSpec:
    """Non-Manhattan: one wall at 22.5 deg. Manhattan snapping destroys this room."""
    return RoomSpec(
        name="angled",
        footprint=[(0.0, 0.0), (4.60, 0.0), (4.60, 2.10), (3.20, 3.30), (0.0, 3.30)],
        ceiling_height_m=2.40,
        openings=[
            OpeningSpec(0, 1.10, DEFAULT_DOOR_WIDTH_M, DEFAULT_DOOR_HEIGHT_M, kind="door"),
            OpeningSpec(4, 1.60, 1.40, 1.15, sill_m=0.90, kind="window"),
        ],
    )


def _furnished() -> RoomSpec:
    r = _empty_rect()
    r.name = "furnished"
    r.boxes = [
        BoxSpec((1.10, 1.90), (1.45, 0.55, 2.05), 0.0, (0.32, 0.22, 0.20)),  # bed
        BoxSpec((3.60, 0.70), (0.55, 0.75, 1.10), 0.0, (0.55, 0.42, 0.30)),  # desk
        BoxSpec((3.85, 2.60), (0.60, 1.95, 0.90), 0.0, (0.38, 0.36, 0.34)),  # wardrobe
    ]
    r.openings.append(OpeningSpec(2, 2.00, 1.20, 1.25, sill_m=0.95, kind="window"))
    return r


def _mirror_heavy() -> RoomSpec:
    r = _furnished()
    r.name = "mirror_heavy"
    r.panels = [
        PanelSpec(1, 1.70, 1.10, 1.80, center_height_m=1.25, label=Semantic.MIRROR),
        PanelSpec(3, 1.60, 0.90, 1.40, center_height_m=1.30, label=Semantic.MIRROR),
        PanelSpec(0, 3.20, 1.20, 0.70, center_height_m=1.15, label=Semantic.TV),
    ]
    return r


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

_WALL_RGB = (0.86, 0.85, 0.82)
_FLOOR_RGB = (0.52, 0.40, 0.30)
_CEIL_RGB = (0.93, 0.93, 0.93)


def build_mesh(spec: RoomSpec) -> TriMesh:
    poly = spec.polygon
    y0, y1 = 0.0, spec.ceiling_height_m
    parts: list[TriMesh] = []

    # Floor (normal up, into the room) and ceiling (normal down).
    tris = geom.triangulate(poly)
    fv = np.column_stack([poly[:, 0], np.full(len(poly), y0), poly[:, 1]])
    parts.append(TriMesh(fv, tris, np.tile([0.0, 1.0, 0.0], (len(tris), 1)),
                         np.full(len(tris), int(Semantic.FLOOR), np.int32),
                         np.tile(_FLOOR_RGB, (len(tris), 1))))
    cv = np.column_stack([poly[:, 0], np.full(len(poly), y1), poly[:, 1]])
    parts.append(TriMesh(cv, tris[:, ::-1].copy(), np.tile([0.0, -1.0, 0.0], (len(tris), 1)),
                         np.full(len(tris), int(Semantic.CEILING), np.int32),
                         np.tile(_CEIL_RGB, (len(tris), 1))))

    # Walls, with openings punched out.
    for i in range(len(poly)):
        a2, b2 = poly[i], poly[(i + 1) % len(poly)]
        n = geom.inward_normal_2d(a2, b2)
        holes = [
            (o.u_start, o.u_end, o.sill_m, o.sill_m + o.height_m)
            for o in spec.openings
            if o.wall_index == i
        ]
        parts.extend(geom.wall_with_holes(a2, b2, y0, y1, n, holes, Semantic.WALL, _WALL_RGB))

    # Mirrors / TVs, inset 1 cm off the wall so they occlude cleanly.
    for p in spec.panels:
        parts.append(_panel_mesh(poly, p))

    for b in spec.boxes:
        parts.append(_box_mesh(b))

    if spec.exterior and any(o.kind == "window" for o in spec.openings):
        parts.append(_exterior_mesh(spec))

    return TriMesh.concat(parts)


def _panel_mesh(poly: np.ndarray, p: PanelSpec) -> TriMesh:
    a2, b2 = poly[p.wall_index], poly[(p.wall_index + 1) % len(poly)]
    n = geom.inward_normal_2d(a2, b2)
    d = (b2 - a2) / max(np.linalg.norm(b2 - a2), 1e-9)
    inset = n * 0.01

    def at(u: float, y: float) -> np.ndarray:
        q = a2 + d * u
        return np.array([q[0], y, q[1]]) + inset

    u0, u1 = p.u_center_m - p.width_m / 2, p.u_center_m + p.width_m / 2
    yb, yt = p.center_height_m - p.height_m / 2, p.center_height_m + p.height_m / 2
    color = (0.72, 0.76, 0.80) if p.label == Semantic.MIRROR else (0.05, 0.05, 0.06)
    return geom.quad(at(u0, yb), at(u1, yb), at(u1, yt), at(u0, yt), n, p.label, color)


def _box_mesh(b: BoxSpec) -> TriMesh:
    cx, cz = b.center
    sx, sy, sz = b.size
    c, s = np.cos(b.yaw_rad), np.sin(b.yaw_rad)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    corners = np.array(
        [
            [x, y, z]
            for x in (-sx / 2, sx / 2)
            for y in (0.0, sy)
            for z in (-sz / 2, sz / 2)
        ]
    )
    v = corners @ rot.T + np.array([cx, 0.0, cz])
    # Index layout: bit0=x, bit1=y, bit2=z as ordered above.
    faces = np.array(
        [
            [0, 2, 3], [0, 3, 1],  # -x
            [4, 5, 7], [4, 7, 6],  # +x
            [0, 1, 5], [0, 5, 4],  # -z ... (winding is informational; we shade two-sided)
            [2, 6, 7], [2, 7, 3],  # +z
            [1, 3, 7], [1, 7, 5],  # +y (top)
        ],
        dtype=np.int32,
    )
    tri = v[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return TriMesh(v, faces, nrm, np.full(len(faces), int(Semantic.WALL), np.int32),
                   np.tile(b.color, (len(faces), 1)))


def _exterior_mesh(spec: RoomSpec) -> TriMesh:
    """A slab of 'outside' 4 m beyond the room bbox. This is what a depth model
    hallucinates through an unmasked window."""
    poly = spec.polygon
    lo, hi = poly.min(axis=0) - 4.0, poly.max(axis=0) + 4.0
    y = spec.ceiling_height_m + 3.0
    parts = []
    for x0, z0, x1, z1 in [
        (lo[0], lo[1], hi[0], lo[1]),
        (hi[0], lo[1], hi[0], hi[1]),
        (hi[0], hi[1], lo[0], hi[1]),
        (lo[0], hi[1], lo[0], lo[1]),
    ]:
        n = geom.inward_normal_2d(np.array([x0, z0]), np.array([x1, z1]))
        parts.append(
            geom.quad(
                np.array([x0, -1.0, z0]), np.array([x1, -1.0, z1]),
                np.array([x1, y, z1]), np.array([x0, y, z0]),
                n, Semantic.OUTSIDE, (0.55, 0.62, 0.70),
            )
        )
    return TriMesh.concat(parts)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _procedural_texture(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """View-independent surface detail so images have trackable features.

    Flat white walls are the real-world VIO failure mode (PLAN.md §7); keeping
    some texture here means the synthetic set exercises the normal path, and the
    `texture_strength` knob lets a test dial it toward the pathological case.
    """
    p = points
    t = (
        0.5
        + 0.16 * np.sin(3.1 * p[..., 0] + 1.7)
        * np.sin(2.7 * p[..., 2] - 0.4)
        + 0.10 * np.sin(11.0 * p[..., 1] + 0.9)
        + 0.07 * np.sin(23.0 * (p[..., 0] + p[..., 2]))
    )
    floor = labels == int(Semantic.FLOOR)
    t = np.where(floor, 0.5 + 0.28 * np.sign(np.sin(8.0 * p[..., 0])) * 0.5, t)
    return np.clip(t, 0.0, 1.0)


def render(
    mesh: TriMesh,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int,
    height: int,
    light_pos: np.ndarray,
    texture_strength: float = 1.0,
    simulate_artifacts: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render (rgb uint8, depth float32 metres, semantic label int32).

    With `simulate_artifacts`, mirror rays reflect and window rays continue
    through the glass, so the returned depth contains exactly the phantom
    geometry that masking is supposed to remove.
    """
    hit = geom.raycast(mesh, K, R, t, width, height)
    depth = hit.depth.astype(np.float64).reshape(-1)
    point = hit.point.reshape(-1, 3)
    normal = hit.normal.reshape(-1, 3)
    label = hit.label.reshape(-1)
    face = hit.face_id.reshape(-1)

    dirs_cam = geom.camera_rays(K, width, height).reshape(-1, 3)
    dirs = dirs_cam @ np.asarray(R, dtype=np.float64).T
    ray_len = np.linalg.norm(dirs, axis=1)

    #: The *visible* label, kept before any secondary bounce — this is what a
    #: segmentation model sees and therefore what the mask must be scored against.
    visible_label = label.copy()

    if simulate_artifacts:
        for _bounce in range(2):
            mirror = (visible_label == int(Semantic.MIRROR)) & (face >= 0)
            glass = (visible_label == int(Semantic.WINDOW)) & (face >= 0)
            secondary = mirror | glass
            if not secondary.any():
                break
            d_hat = dirs / ray_len[:, None]
            new_dir = d_hat.copy()
            if mirror.any():
                n = normal[mirror]
                new_dir[mirror] = d_hat[mirror] - 2.0 * np.einsum(
                    "ij,ij->i", d_hat[mirror], n
                )[:, None] * n
            t2, f2 = geom.intersect(mesh, point, new_dir, t_min=1e-4, active=secondary)
            found = secondary & (f2 >= 0)
            if not found.any():
                break
            # Report the folded path length as if the ray had gone straight.
            depth[found] = depth[found] + t2[found] / ray_len[found]
            point[found] = point[found] + t2[found, None] * new_dir[found]
            normal[found] = mesh.face_normals[f2[found]]
            face[found] = f2[found]
            label[found] = mesh.face_labels[f2[found]]
            if not ((label == int(Semantic.MIRROR)) & found).any():
                break

    valid = face >= 0
    base = np.zeros((len(depth), 3))
    base[valid] = mesh.face_colors[face[valid]]

    to_light = light_pos[None, :] - point
    to_light /= np.maximum(np.linalg.norm(to_light, axis=1, keepdims=True), 1e-9)
    lambert = np.abs(np.einsum("ij,ij->i", to_light, normal))
    shade = 0.35 + 0.65 * lambert
    tex = 1.0 + texture_strength * (_procedural_texture(point, label) - 0.5) * 0.55
    rgb = np.clip(base * (shade * tex)[:, None], 0.0, 1.0)
    rgb[~valid] = 0.0

    return (
        (rgb.reshape(height, width, 3) * 255.0).astype(np.uint8),
        depth.reshape(height, width).astype(np.float32),
        visible_label.reshape(height, width).astype(np.int32),
    )


# ---------------------------------------------------------------------------
# Capture trajectory
# ---------------------------------------------------------------------------


def _rotation(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return ry @ rx


def plan_stations(spec: RoomSpec, n_stations: int, clearance_m: float = 0.85) -> np.ndarray:
    """Farthest-point sample of standable positions — a stand-in for the Rust
    next-best-station solver, used only to synthesise a plausible capture."""
    poly = spec.polygon
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    gx = np.arange(lo[0] + 0.2, hi[0], 0.2)
    gz = np.arange(lo[1] + 0.2, hi[1], 0.2)
    grid = np.stack(np.meshgrid(gx, gz), axis=-1).reshape(-1, 2)
    ok = geom.point_in_polygon(grid, poly) & (geom.distance_to_polygon(grid, poly) >= clearance_m)
    cand = grid[ok]
    if len(cand) == 0:  # tiny room: fall back to the centroid
        return poly.mean(axis=0)[None, :]

    chosen = [int(np.argmin(np.linalg.norm(cand - poly.mean(axis=0), axis=1)))]
    while len(chosen) < min(n_stations, len(cand)):
        d = np.min(
            np.linalg.norm(cand[:, None, :] - cand[chosen][None, :, :], axis=2), axis=1
        )
        chosen.append(int(np.argmax(d)))
    return cand[chosen]


@dataclass
class CaptureConfig:
    width: int = 320
    height: int = 240
    hfov_deg: float = 62.0  # typical phone main camera
    n_stations: int = 3
    yaw_step_deg: float = 24.0
    #: A real sweep tilts well down as well as up — the floor is the surface the
    #: floor plan depends on, and a level-only sweep never sees the near floor.
    pitches_deg: tuple[float, ...] = (22.0, 0.0, -28.0)
    max_keyframes: int = 150
    #: VIO scale error, e.g. 1.025 == the reconstruction comes out 2.5% too big.
    vio_scale_error: float = 1.0
    #: Zero-mean per-frame pose noise (metres / radians).
    pose_noise_t_m: float = 0.0
    pose_noise_r_rad: float = 0.0
    #: Low-frequency VIO drift amplitude (metres) accumulated over the capture.
    drift_m: float = 0.0
    texture_strength: float = 1.0
    simulate_artifacts: bool = True
    seed: int = 0


def intrinsics_for(cfg: CaptureConfig) -> Intrinsics:
    fx = (cfg.width / 2.0) / np.tan(np.radians(cfg.hfov_deg) / 2.0)
    return Intrinsics(
        fx=float(fx), fy=float(fx),
        cx=cfg.width / 2.0, cy=cfg.height / 2.0,
        width=cfg.width, height=cfg.height,
    )


# ---------------------------------------------------------------------------
# Bundle generation
# ---------------------------------------------------------------------------


@dataclass
class GroundTruth:
    """What the synthetic scene knows exactly. The oracle for PLAN.md §1.3."""

    room_id: str
    preset: str
    footprint: list[list[float]]
    wall_lengths_m: list[float]
    area_m2: float
    perimeter_m: float
    ceiling_height_m: float
    #: The similarity the reconstruction is expected to sit in. Scale-correcting
    #: it must recover the numbers above.
    applied_scale: float
    openings: list[dict]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def generate(
    spec: RoomSpec,
    out_dir: Path,
    cfg: CaptureConfig | None = None,
    *,
    write_depth: bool = True,
    write_masks: bool = True,
    room_id: str | None = None,
    scene_mesh: TriMesh | None = None,
    world_from_room: np.ndarray | None = None,
) -> tuple[SessionBundle, GroundTruth]:
    """Render a full capture session to `out_dir` and return bundle + ground truth.

    `scene_mesh` and `world_from_room` let a room be captured inside a larger
    scene: stations are still planned in the room's own footprint and the bundle
    still records **room-local** poses, but rays are cast against the whole
    house. That is what lets a scan see through a doorway into the next room,
    which is the overlap the §6.3 ICP refinement needs to exist at all.
    """
    from PIL import Image

    cfg = cfg or CaptureConfig()
    rng = np.random.default_rng(cfg.seed)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    if write_depth:
        (out_dir / "depth").mkdir(exist_ok=True)
    if write_masks:
        (out_dir / "masks").mkdir(exist_ok=True)

    room_mesh = build_mesh(spec)
    mesh = scene_mesh if scene_mesh is not None else room_mesh
    w_from_r = np.eye(4) if world_from_room is None else np.asarray(world_from_room, np.float64)
    intr = intrinsics_for(cfg)
    K = intr.matrix()
    poly = spec.polygon
    light_local = np.array(
        [poly[:, 0].mean(), spec.ceiling_height_m - 0.15, poly[:, 1].mean()]
    )
    light = w_from_r[:3, :3] @ light_local + w_from_r[:3, 3]

    stations = plan_stations(spec, cfg.n_stations)
    yaws = np.radians(np.arange(0.0, 360.0, cfg.yaw_step_deg))
    pitches = np.radians(np.asarray(cfg.pitches_deg))

    room_id = room_id or str(uuid.uuid4())
    keyframes: list[Keyframe] = []
    surface_pts: list[np.ndarray] = []
    n_planned = len(stations) * len(yaws) * len(pitches)
    stride = max(1, int(np.ceil(n_planned / cfg.max_keyframes)))

    counter = 0
    for st in stations:
        for yaw in yaws:
            for pitch in pitches:
                counter += 1
                if (counter - 1) % stride:
                    continue
                if len(keyframes) >= cfg.max_keyframes:
                    break

                progress = len(keyframes) / max(cfg.max_keyframes - 1, 1)
                pos_true = np.array([st[0], STANDING_EYE_HEIGHT_M, st[1]])
                R_true = _rotation(float(yaw), float(pitch))

                # Render in the scene frame, but keep the bundle room-local:
                # an independently scanned room has no idea where the house is.
                pos_world = w_from_r[:3, :3] @ pos_true + w_from_r[:3, 3]
                R_world = w_from_r[:3, :3] @ R_true

                rgb, depth, labels = render(
                    mesh, K, R_world, pos_world, cfg.width, cfg.height, light,
                    texture_strength=cfg.texture_strength,
                    simulate_artifacts=cfg.simulate_artifacts,
                )

                # Corrupt the *reported* pose exactly as VIO would: a global scale
                # error, a slow drift, and small per-frame noise. Depth is scaled
                # to match, so the whole reconstruction is a similarity of truth
                # and the §6.2 solver has a well-posed job.
                s = cfg.vio_scale_error
                drift = cfg.drift_m * progress * np.array([1.0, 0.15, -0.6])
                pos_rep = pos_true * s + drift
                if cfg.pose_noise_t_m > 0:
                    pos_rep = pos_rep + rng.normal(0.0, cfg.pose_noise_t_m, 3)
                R_rep = R_true
                if cfg.pose_noise_r_rad > 0:
                    R_rep = _rotation(
                        float(yaw) + rng.normal(0.0, cfg.pose_noise_r_rad),
                        float(pitch) + rng.normal(0.0, cfg.pose_noise_r_rad),
                    )

                kf_id = f"kf_{len(keyframes):04d}"
                Image.fromarray(rgb).save(out_dir / "images" / f"{kf_id}.png")
                depth_rel = None
                if write_depth:
                    mm = np.clip(depth * s * 1000.0, 0, 65535).astype(np.uint16)
                    Image.fromarray(mm).save(out_dir / "depth" / f"{kf_id}.png")
                    depth_rel = f"depth/{kf_id}.png"
                if write_masks:
                    Image.fromarray(
                        np.clip(labels + 1, 0, 255).astype(np.uint8)
                    ).save(out_dir / "masks" / f"{kf_id}.png")

                keyframes.append(
                    Keyframe(
                        id=kf_id,
                        image=f"images/{kf_id}.png",
                        depth=depth_rel,
                        timestamp_ns=int(counter * 1e8),
                        pose=Pose(R=R_rep.tolist(), t=pos_rep.tolist()),
                        intrinsics=intr,
                        tracking_state=TrackingState.NORMAL,
                        blur_score=float(rng.uniform(180.0, 420.0)),
                        exposure_clip_ratio=float(rng.uniform(0.0, 0.01)),
                    )
                )

                # A sparse-point stand-in for the AR feature cloud.
                keep = (depth > 0) & (labels == int(Semantic.WALL))
                if keep.any():
                    pts = geom.unproject(np.where(keep, depth, 0.0), K, R_true, pos_true)
                    if len(pts):
                        surface_pts.append(pts[rng.choice(len(pts), size=min(12, len(pts)),
                                                          replace=False)] * s + drift)

    sparse = []
    if surface_pts:
        allp = np.vstack(surface_pts)
        sel = rng.choice(len(allp), size=min(600, len(allp)), replace=False)
        sparse = [SparsePoint(xyz=p.tolist()) for p in allp[sel]]

    bundle = SessionBundle(
        session_id=str(uuid.uuid4()),
        device=Device(model="synthetic", os="n/a", ar_platform=ARPlatform.SYNTHETIC,
                      ar_version="recon-synth/1"),
        room_id=room_id,
        gravity=[0.0, -1.0, 0.0],
        keyframes=keyframes,
        sparse_points=sparse,
        reference_measurements=_measurements(spec, cfg),
    )

    gt = GroundTruth(
        room_id=room_id,
        preset=spec.name,
        footprint=poly.tolist(),
        wall_lengths_m=spec.wall_lengths_m.tolist(),
        area_m2=spec.area_m2,
        perimeter_m=geom.polygon_perimeter(poly),
        ceiling_height_m=spec.ceiling_height_m,
        applied_scale=cfg.vio_scale_error,
        openings=[
            {
                "kind": o.kind, "wall_index": o.wall_index,
                "u_start_m": o.u_start, "u_end_m": o.u_end,
                "sill_m": o.sill_m, "height_m": o.height_m, "width_m": o.width_m,
            }
            for o in spec.openings
        ],
    )

    (out_dir / "bundle.json").write_text(bundle.model_dump_json(indent=2))
    (out_dir / "ground_truth.json").write_text(gt.to_json())
    # The room's *own* mesh, not the whole scene: Chamfer must score this room.
    _write_ply(out_dir / "mesh_gt.ply", room_mesh)
    return bundle, gt


def _measurements(spec: RoomSpec, cfg: CaptureConfig) -> list[ReferenceMeasurement]:
    """The constraint set from PLAN.md §6.2, expressed in the *drifted* frame.

    Points are where the reconstruction thinks the door jambs are (i.e. scaled by
    the VIO error); `measured_value_m` is the real tape reading. The solver's job
    is to recover 1/vio_scale_error.
    """
    out = [
        ReferenceMeasurement(
            type=MeasurementType.VIO_SCALE, measured_value_m=1.0, sigma_m=0.03
        ),
        ReferenceMeasurement(
            type=MeasurementType.CEILING_PRIOR,
            point_a=[0.0, 0.0, 0.0],
            point_b=[0.0, spec.ceiling_height_m * cfg.vio_scale_error, 0.0],
            measured_value_m=spec.ceiling_height_m,
            sigma_m=0.10,
        ),
        ReferenceMeasurement(
            type=MeasurementType.CAMERA_HEIGHT,
            point_a=[0.0, 0.0, 0.0],
            point_b=[0.0, STANDING_EYE_HEIGHT_M * cfg.vio_scale_error, 0.0],
            measured_value_m=STANDING_EYE_HEIGHT_M,
            sigma_m=0.08,
        ),
    ]
    poly = spec.polygon
    for o in spec.openings:
        if o.kind != "door":
            continue
        a2, b2 = poly[o.wall_index], poly[(o.wall_index + 1) % len(poly)]
        d = (b2 - a2) / max(np.linalg.norm(b2 - a2), 1e-9)
        s = cfg.vio_scale_error
        pa = np.array([*(a2 + d * o.u_start), 0.0])[[0, 2, 1]] * s
        pb = np.array([*(a2 + d * o.u_end), 0.0])[[0, 2, 1]] * s
        out.append(
            ReferenceMeasurement(
                type=MeasurementType.USER_TAPE,
                point_a=pa.tolist(), point_b=pb.tolist(),
                measured_value_m=o.width_m, sigma_m=0.005,
            )
        )
        out.append(
            ReferenceMeasurement(
                type=MeasurementType.DOOR_PRIOR,
                point_a=[0.0, 0.0, 0.0],
                point_b=[0.0, o.height_m * s, 0.0],
                measured_value_m=DEFAULT_DOOR_HEIGHT_M, sigma_m=0.04,
            )
        )
    return out


def _write_ply(path: Path, mesh: TriMesh) -> None:
    v, f = mesh.vertices, mesh.faces
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(v)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(f)}\n"
        "property list uchar int vertex_indices\n"
        "property uchar label\n"
        "end_header\n"
    )
    lines = [header]
    lines += [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for p in v]
    lines += [f"3 {t[0]} {t[1]} {t[2]} {lbl}\n" for t, lbl in zip(f, mesh.face_labels, strict=True)]
    path.write_text("".join(lines))
