"""Plane extraction and Atlanta-world regularization (PLAN.md §6.4 steps 2-4).

Atlanta, **not** Manhattan. Manhattan assumes three mutually orthogonal
vanishing directions; Atlanta adds a degree of freedom — walls may take any yaw
provided they stay vertical. Real houses have angled walls, bay windows and
non-square rooms, and Manhattan snapping actively destroys them. The `angled`
synthetic preset exists specifically to catch a regression back to Manhattan.

Because gravity is known from the IMU, every wall is vertical by construction,
so wall fitting collapses from a 3D plane problem to a 2D line problem in the
floor plane. That is a large robustness win and it is why gravity is a required
field on the bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recon.schemas import Plane, PlaneKind


@dataclass
class PlaneConfig:
    #: A normal within this angle of +/-up is horizontal; within this of the
    #: floor plane is vertical.
    horizontal_tol_deg: float = 15.0
    vertical_tol_deg: float = 15.0
    #: RANSAC inlier distance for wall lines.
    wall_inlier_m: float = 0.04
    #: Inlier normals must agree with the fitted line normal to this angle.
    wall_normal_tol_deg: float = 25.0
    #: Stop extracting walls below this inlier count.
    min_wall_inliers: int = 200
    #: ...or below this fraction of the largest wall found.
    min_wall_fraction: float = 0.04
    max_walls: int = 24
    ransac_iterations: int = 800
    #: Score hypotheses against a random subset; refit uses every inlier. Keeps
    #: RANSAC roughly linear in room size instead of quadratic.
    ransac_score_subset: int = 20_000
    #: PLAN.md §6.4: snap a wall's yaw to an existing direction only when within
    #: this angle. Larger values start eating genuine angled walls.
    atlanta_snap_deg: float = 3.0
    #: Merge two walls whose direction and offset both agree within these.
    merge_angle_deg: float = 4.0
    merge_offset_m: float = 0.10
    horizontal_bin_m: float = 0.05
    seed: int = 0


@dataclass
class PlaneSet:
    floor: Plane | None
    ceiling: Plane | None
    walls: list[Plane] = field(default_factory=list)
    #: Per-wall inlier points, world frame — needed for openings and extents.
    wall_inliers: list[np.ndarray] = field(default_factory=list)
    floor_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    ceiling_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @property
    def floor_height(self) -> float:
        return -self.floor.offset if self.floor else 0.0

    @property
    def ceiling_height(self) -> float:
        if not (self.floor and self.ceiling):
            return 0.0
        return abs(self.ceiling.offset - self.floor.offset)

    def all_planes(self) -> list[Plane]:
        out = []
        if self.floor:
            out.append(self.floor)
        if self.ceiling:
            out.append(self.ceiling)
        out.extend(self.walls)
        return out


def extract_planes(
    points: np.ndarray,
    normals: np.ndarray,
    up: np.ndarray,
    config: PlaneConfig | None = None,
) -> PlaneSet:
    """Split an oriented point cloud into floor, ceiling and Atlanta walls."""
    cfg = config or PlaneConfig()
    if len(points) == 0:
        return PlaneSet(floor=None, ceiling=None)

    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    cos_up = n @ up

    h_tol = np.cos(np.radians(90.0 - cfg.horizontal_tol_deg))  # |cos| above -> horizontal
    v_tol = np.sin(np.radians(cfg.vertical_tol_deg))  # |cos| below -> vertical

    height = points @ up
    is_horizontal = np.abs(cos_up) >= h_tol
    is_vertical = np.abs(cos_up) <= v_tol

    floor = _fit_horizontal(points, height, is_horizontal & (cos_up > 0), up, cfg, PlaneKind.FLOOR)
    ceiling = _fit_horizontal(
        points, height, is_horizontal & (cos_up < 0), up, cfg, PlaneKind.CEILING
    )

    walls, inliers = _fit_walls(points[is_vertical], n[is_vertical], up, cfg)

    return PlaneSet(
        floor=floor,
        ceiling=ceiling,
        walls=walls,
        wall_inliers=inliers,
        floor_points=points[is_horizontal & (cos_up > 0)],
        ceiling_points=points[is_horizontal & (cos_up < 0)],
    )


def _fit_horizontal(
    points: np.ndarray,
    height: np.ndarray,
    sel: np.ndarray,
    up: np.ndarray,
    cfg: PlaneConfig,
    kind: PlaneKind,
) -> Plane | None:
    """Pick the dominant horizontal level by histogram, then refine by mean.

    The mode is taken over a histogram rather than by RANSAC because furniture
    tops are also horizontal and up-facing; the floor is simply the most
    populated level, and for a ceiling the highest well-supported one.
    """
    if sel.sum() < 50:
        return None
    h = height[sel]
    bins = np.arange(h.min() - cfg.horizontal_bin_m, h.max() + 2 * cfg.horizontal_bin_m,
                     cfg.horizontal_bin_m)
    counts, edges = np.histogram(h, bins=bins)
    if counts.max() == 0:
        return None

    # Floor: the most populated level. Ceiling: the highest level with real
    # support, which beats "most populated" when a wardrobe top is broad.
    if kind is PlaneKind.FLOOR:
        bi = int(np.argmax(counts))
    else:
        strong = np.flatnonzero(counts >= 0.25 * counts.max())
        bi = int(strong[-1])

    level = 0.5 * (edges[bi] + edges[bi + 1])
    near = np.abs(h - level) <= 2.5 * cfg.horizontal_bin_m
    if near.sum() < 50:
        return None
    level = float(h[near].mean())

    normal = up if kind is PlaneKind.FLOOR else -up
    # Hessian form n.x + d = 0 with the normal pointing into the room.
    d = -level if kind is PlaneKind.FLOOR else level
    pts = points[sel][near]
    return Plane(
        kind=kind,
        coeffs=[*normal.tolist(), float(d)],
        inlier_count=int(near.sum()),
        bbox_min=pts.min(axis=0).tolist(),
        bbox_max=pts.max(axis=0).tolist(),
    )


def _basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal axes spanning the floor plane.

    For the canonical Y-up frame this returns exactly (+X, +Z), so floor-plane
    2D coordinates are world (x, z) — which is what `RoomLayout.footprint` is
    documented to contain, and what keeps the shoelace winding consistent with
    the polygons the synthetic generator emits.
    """
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(tmp @ up)) > 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    e0 = tmp - (tmp @ up) * up
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(e0, up)
    e1 /= np.linalg.norm(e1)
    return e0, e1


def _fit_walls(
    pts: np.ndarray, nrm: np.ndarray, up: np.ndarray, cfg: PlaneConfig
) -> tuple[list[Plane], list[np.ndarray]]:
    """Sequential RANSAC over 2D lines in the floor plane."""
    if len(pts) < cfg.min_wall_inliers:
        return [], []
    e0, e1 = _basis(up)
    xy = np.column_stack([pts @ e0, pts @ e1])
    nxy = np.column_stack([nrm @ e0, nrm @ e1])
    nxy /= np.maximum(np.linalg.norm(nxy, axis=1, keepdims=True), 1e-12)

    rng = np.random.default_rng(cfg.seed)
    remaining = np.ones(len(xy), dtype=bool)
    normal_tol = np.cos(np.radians(cfg.wall_normal_tol_deg))

    found: list[tuple[np.ndarray, float, np.ndarray]] = []  # (normal2d, offset, inlier idx)
    for _ in range(cfg.max_walls):
        idx = np.flatnonzero(remaining)
        if len(idx) < cfg.min_wall_inliers:
            break
        score_idx = (
            rng.choice(idx, size=cfg.ransac_score_subset, replace=False)
            if len(idx) > cfg.ransac_score_subset
            else idx
        )
        best_score = -1
        best_model: tuple[np.ndarray, float] | None = None

        # Hypothesise a line from a single oriented point: its normal fixes the
        # direction, its position fixes the offset. One sample, not two, because
        # normals make the direction observable from a single point.
        samples = rng.choice(idx, size=min(cfg.ransac_iterations, len(idx)), replace=False)
        for si in samples:
            n2 = nxy[si]
            if not np.isfinite(n2).all():
                continue
            c = float(n2 @ xy[si])
            dist = np.abs(xy[score_idx] @ n2 - c)
            align = np.abs(nxy[score_idx] @ n2)
            score = int(((dist <= cfg.wall_inlier_m) & (align >= normal_tol)).sum())
            if score > best_score:
                best_score, best_model = score, (n2, c)

        if best_model is None:
            break
        n2, c = best_model
        dist = np.abs(xy[idx] @ n2 - c)
        align = np.abs(nxy[idx] @ n2)
        best_inliers = idx[(dist <= cfg.wall_inlier_m) & (align >= normal_tol)]
        if len(best_inliers) < cfg.min_wall_inliers:
            break
        n2, c = _refit_line(xy[best_inliers], nxy[best_inliers])
        # Re-select inliers against the refit, which pulls in points the seed
        # normal's noise had excluded.
        dist = np.abs(xy @ n2 - c)
        align = np.abs(nxy @ n2)
        hit = remaining & (dist <= cfg.wall_inlier_m) & (align >= normal_tol)
        if hit.sum() < cfg.min_wall_inliers:
            remaining[best_inliers] = False
            continue
        found.append((n2, c, np.flatnonzero(hit)))
        remaining[hit] = False

    if not found:
        return [], []

    biggest = max(len(f[2]) for f in found)
    found = [f for f in found if len(f[2]) >= max(cfg.min_wall_inliers,
                                                  cfg.min_wall_fraction * biggest)]
    found = atlanta_regularize(found, xy, nxy, cfg)
    found = _merge_walls(found, cfg)
    found.sort(key=lambda f: -len(f[2]))

    walls: list[Plane] = []
    inliers: list[np.ndarray] = []
    for n2, c, ids in found:
        n3 = n2[0] * e0 + n2[1] * e1
        # Orient the normal into the room: it must point away from the centroid
        # of all wall points, negated (walls face inward).
        centroid = xy.mean(axis=0)
        if (n2 @ centroid - c) < 0:
            n3, n2, c = -n3, -n2, -c
        p = pts[ids]
        walls.append(
            Plane(
                kind=PlaneKind.WALL,
                coeffs=[*n3.tolist(), float(-c)],
                inlier_count=int(len(ids)),
                bbox_min=p.min(axis=0).tolist(),
                bbox_max=p.max(axis=0).tolist(),
            )
        )
        inliers.append(p)
    return walls, inliers


def _refit_line(xy: np.ndarray, nxy: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares line through inliers; normal disambiguated by votes."""
    mu = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - mu, full_matrices=False)
    n2 = vt[1]  # minor axis == line normal
    n2 /= np.linalg.norm(n2)
    if np.sum(nxy @ n2) < 0:
        n2 = -n2
    return n2, float(n2 @ mu)


def atlanta_regularize(
    found: list[tuple[np.ndarray, float, np.ndarray]],
    xy: np.ndarray,
    nxy: np.ndarray,
    cfg: PlaneConfig,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Snap wall yaw to an already-seen direction, but only within the tolerance.

    Walls are vertical by construction (gravity), so this is the whole of the
    Atlanta constraint. Directions are compared modulo 180 deg: two walls facing
    each other share an orientation.
    """
    if not found:
        return found
    order = sorted(range(len(found)), key=lambda i: -len(found[i][2]))
    directions: list[float] = []
    out: list[tuple[np.ndarray, float, np.ndarray]] = [None] * len(found)  # type: ignore

    for i in order:
        n2, c, ids = found[i]
        ang = np.arctan2(n2[1], n2[0]) % np.pi
        snapped = None
        for d in directions:
            delta = (ang - d + np.pi / 2) % np.pi - np.pi / 2
            if abs(np.degrees(delta)) <= cfg.atlanta_snap_deg:
                snapped = d
                break
        if snapped is None:
            directions.append(ang)
        else:
            new_n = np.array([np.cos(snapped), np.sin(snapped)])
            if new_n @ n2 < 0:
                new_n = -new_n
            # Refit the offset in the snapped direction so the wall stays on its
            # own points rather than sliding to satisfy the angle.
            c = float(np.median(xy[ids] @ new_n))
            n2 = new_n
        out[i] = (n2, c, ids)
    return out


def _merge_walls(
    found: list[tuple[np.ndarray, float, np.ndarray]], cfg: PlaneConfig
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Fuse duplicate detections of one physical wall (common where an opening
    splits a wall into two RANSAC hits)."""
    merged: list[list] = []
    for n2, c, ids in found:
        for m in merged:
            same_dir = abs(float(m[0] @ n2)) >= np.cos(np.radians(cfg.merge_angle_deg))
            flip = float(m[0] @ n2) < 0
            offset = (-c if flip else c) - m[1]
            if same_dir and abs(offset) <= cfg.merge_offset_m:
                w_old, w_new = len(m[2]), len(ids)
                m[1] = (m[1] * w_old + (-c if flip else c) * w_new) / (w_old + w_new)
                m[2] = np.concatenate([m[2], ids])
                break
        else:
            merged.append([n2, c, ids])
    return [(m[0], m[1], m[2]) for m in merged]
