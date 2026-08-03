"""Geometry primitives: triangle meshes, ear-clipping triangulation, ray casting.

Pure NumPy, no Open3D dependency, so the synthetic ground-truth path and its
tests run on any machine. Conventions follow PLAN.md §5: right-handed, Y-up,
camera looks down -Z.

2D floor-plane polygons are given as (x, z) pairs and are treated as CCW when
the shoelace sum 0.5*sum(x_i*z_{i+1} - x_{i+1}*z_i) is positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

EPS = 1e-9


class Semantic(IntEnum):
    """Per-face semantic label. Doubles as the ground-truth mask source."""

    FLOOR = 0
    CEILING = 1
    WALL = 2
    MIRROR = 3
    WINDOW = 4
    TV = 5
    OUTSIDE = 6  # geometry visible through a window; must never enter fusion


#: Labels whose depth returns are physically unreliable (PLAN.md §3.5, §7).
UNRELIABLE = frozenset({Semantic.MIRROR, Semantic.WINDOW, Semantic.TV})


@dataclass
class TriMesh:
    vertices: np.ndarray  # (V, 3) float64
    faces: np.ndarray  # (F, 3) int32
    face_normals: np.ndarray  # (F, 3) float64, oriented toward the room interior
    face_labels: np.ndarray  # (F,) int32, Semantic
    face_colors: np.ndarray  # (F, 3) float64 in [0, 1]

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int32)
        self.face_normals = np.ascontiguousarray(self.face_normals, dtype=np.float64)
        self.face_labels = np.ascontiguousarray(self.face_labels, dtype=np.int32)
        self.face_colors = np.ascontiguousarray(self.face_colors, dtype=np.float64)
        n = len(self.faces)
        if not (len(self.face_normals) == len(self.face_labels) == len(self.face_colors) == n):
            raise ValueError("per-face arrays must all have one entry per face")

    @property
    def triangles(self) -> np.ndarray:
        """(F, 3, 3) vertex positions."""
        return self.vertices[self.faces]

    def area(self) -> np.ndarray:
        t = self.triangles
        return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)

    def sample_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Area-weighted uniform surface samples, for Chamfer distance."""
        w = self.area()
        w = w / w.sum()
        idx = rng.choice(len(self.faces), size=n, p=w)
        t = self.triangles[idx]
        u = rng.random((n, 1))
        v = rng.random((n, 1))
        flip = (u + v) > 1.0
        u[flip] = 1.0 - u[flip]
        v[flip] = 1.0 - v[flip]
        return t[:, 0] + u * (t[:, 1] - t[:, 0]) + v * (t[:, 2] - t[:, 0])

    @classmethod
    def concat(cls, meshes: list[TriMesh]) -> TriMesh:
        verts, faces, normals, labels, colors = [], [], [], [], []
        offset = 0
        for m in meshes:
            verts.append(m.vertices)
            faces.append(m.faces + offset)
            normals.append(m.face_normals)
            labels.append(m.face_labels)
            colors.append(m.face_colors)
            offset += len(m.vertices)
        return cls(
            np.vstack(verts),
            np.vstack(faces),
            np.vstack(normals),
            np.concatenate(labels),
            np.vstack(colors),
        )


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------


def polygon_area(poly: np.ndarray) -> float:
    """Signed shoelace area of an (N, 2) polygon. Positive == CCW."""
    p = np.asarray(poly, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def polygon_perimeter(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=np.float64)
    return float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())


def edge_lengths(poly: np.ndarray) -> np.ndarray:
    p = np.asarray(poly, dtype=np.float64)
    return np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    p = np.asarray(poly, dtype=np.float64)
    return p if polygon_area(p) > 0 else p[::-1].copy()


def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Scalar 2D cross product. NumPy 2 removed `np.cross` for 2-vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    d1 = cross2(b - a, p - a)
    d2 = cross2(c - b, p - b)
    d3 = cross2(a - c, p - c)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def triangulate(poly: np.ndarray) -> np.ndarray:
    """Ear-clipping triangulation of a simple CCW polygon. Returns (T, 3) indices.

    Handles the non-convex footprints we care about (L-shaped rooms, bay
    alcoves). Not robust to self-intersection — rooms aren't.
    """
    p = ensure_ccw(np.asarray(poly, dtype=np.float64))
    n = len(p)
    if n < 3:
        raise ValueError("polygon needs at least 3 vertices")
    idx = list(range(n))
    tris: list[tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3:
        guard += 1
        if guard > 10 * n:
            raise RuntimeError("ear clipping failed to converge; polygon may self-intersect")
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = p[i0], p[i1], p[i2]
            if cross2(b - a, c - b) <= EPS:  # reflex or degenerate
                continue
            if any(
                _point_in_triangle(p[j], a, b, c)
                for j in idx
                if j not in (i0, i1, i2)
            ):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            raise RuntimeError("no ear found; polygon may self-intersect")
    tris.append((idx[0], idx[1], idx[2]))
    return np.asarray(tris, dtype=np.int32)


def point_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd test. `pts` (N, 2), `poly` (M, 2) -> (N,) bool."""
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    poly = np.asarray(poly, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        straddles = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (y - yi) / np.where(yj - yi == 0, np.nan, yj - yi) + xi
        inside ^= straddles & (x < x_cross)
        j = i
    return inside


def distance_to_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Unsigned distance from each point to the polygon boundary."""
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    poly = np.asarray(poly, dtype=np.float64)
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a  # (M, 2)
    ap = pts[:, None, :] - a[None, :, :]  # (N, M, 2)
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom < EPS, EPS, denom)
    t = np.clip(np.einsum("nmj,mj->nm", ap, ab) / denom, 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2).min(axis=1)


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


def quad(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    normal: np.ndarray,
    label: Semantic,
    color: tuple[float, float, float],
) -> TriMesh:
    """Two triangles spanning p0-p1-p2-p3 (in order around the quad)."""
    v = np.array([p0, p1, p2, p3], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    n = np.tile(np.asarray(normal, dtype=np.float64), (2, 1))
    return TriMesh(v, f, n, np.full(2, int(label), np.int32), np.tile(np.asarray(color), (2, 1)))


def wall_with_holes(
    a2: np.ndarray,
    b2: np.ndarray,
    y0: float,
    y1: float,
    normal: np.ndarray,
    holes: list[tuple[float, float, float, float]],
    label: Semantic,
    color: tuple[float, float, float],
) -> list[TriMesh]:
    """A wall panel from floor `y0` to ceiling `y1` between 2D points a2, b2.

    `holes` are (u_start, u_end, y_start, y_end) rectangles in the wall's own
    parameterisation, u measured from `a2` along the edge. Emitted as a 4-way
    split (below / above / left / right) per hole, processed left to right, so
    the result stays a clean triangle soup with exact rectangular openings.
    """
    a2 = np.asarray(a2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)
    length = float(np.linalg.norm(b2 - a2))
    direction = (b2 - a2) / max(length, EPS)

    def at(u: float, y: float) -> np.ndarray:
        p = a2 + direction * u
        return np.array([p[0], y, p[1]], dtype=np.float64)

    pieces: list[TriMesh] = []
    holes = sorted([h for h in holes if h[1] > h[0]], key=lambda h: h[0])
    cursor = 0.0
    for u0, u1, hy0, hy1 in holes:
        u0, u1 = max(0.0, u0), min(length, u1)
        if u1 <= u0:
            continue
        if u0 > cursor:  # full-height panel left of the hole
            pieces.append(quad(at(cursor, y0), at(u0, y0), at(u0, y1), at(cursor, y1),
                               normal, label, color))
        if hy0 > y0:  # under the hole
            pieces.append(quad(at(u0, y0), at(u1, y0), at(u1, hy0), at(u0, hy0),
                               normal, label, color))
        if hy1 < y1:  # over the hole
            pieces.append(quad(at(u0, hy1), at(u1, hy1), at(u1, y1), at(u0, y1),
                               normal, label, color))
        cursor = u1
    if cursor < length:
        pieces.append(quad(at(cursor, y0), at(length, y0), at(length, y1), at(cursor, y1),
                           normal, label, color))
    return pieces


def inward_normal_2d(a2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Inward 3D normal of the wall on CCW edge a2->b2 (see module docstring)."""
    d = np.asarray(b2, dtype=np.float64) - np.asarray(a2, dtype=np.float64)
    n = np.array([-d[1], 0.0, d[0]], dtype=np.float64)
    return n / max(float(np.linalg.norm(n)), EPS)


# ---------------------------------------------------------------------------
# Ray casting
# ---------------------------------------------------------------------------


@dataclass
class RayHit:
    depth: np.ndarray  # (H, W) float32, metric depth along the optical axis; 0 == miss
    face_id: np.ndarray  # (H, W) int32, -1 == miss
    point: np.ndarray  # (H, W, 3) float64 world-space hit position
    normal: np.ndarray  # (H, W, 3) float64 face normal at the hit
    label: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))


def camera_rays(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """(H, W, 3) camera-frame ray directions with z == -1, so t == metric depth."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.arange(width, dtype=np.float64) + 0.5
    v = np.arange(height, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)
    return np.stack([(uu - cx) / fx, -(vv - cy) / fy, -np.ones_like(uu)], axis=-1)


def intersect(
    mesh: TriMesh,
    origins: np.ndarray,
    dirs: np.ndarray,
    t_min: float = 1e-6,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Closest-hit ray/mesh intersection. Möller-Trumbore, batched over rays with
    a Python loop over faces (rooms have O(100) faces).

    `origins` is (3,) or (N, 3); `dirs` is (N, 3), not required to be unit.
    Returns (t, face_id) with t == inf and face_id == -1 where nothing was hit.
    """
    dirs = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
    n_rays = len(dirs)
    origins = np.asarray(origins, dtype=np.float64)
    per_ray_origin = origins.ndim == 2
    if active is None:
        active = np.ones(n_rays, dtype=bool)

    best_t = np.full(n_rays, np.inf)
    best_f = np.full(n_rays, -1, dtype=np.int32)
    tri = mesh.triangles

    for fi in range(len(mesh.faces)):
        v0, v1, v2 = tri[fi]
        e1, e2 = v1 - v0, v2 - v0
        pvec = np.cross(dirs, e2)
        det = pvec @ e1
        ok = active & (np.abs(det) > 1e-12)
        if not ok.any():
            continue
        inv_det = np.zeros(n_rays)
        inv_det[ok] = 1.0 / det[ok]
        tvec = (origins - v0) if per_ray_origin else np.broadcast_to(origins - v0, (n_rays, 3))
        u = np.einsum("ij,ij->i", pvec, tvec) * inv_det
        ok &= (u >= -1e-9) & (u <= 1.0 + 1e-9)
        if not ok.any():
            continue
        qvec = np.cross(tvec, e1)
        v = np.einsum("ij,ij->i", dirs, qvec) * inv_det
        ok &= (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
        if not ok.any():
            continue
        th = np.einsum("j,ij->i", e2, qvec) * inv_det
        ok &= (th > t_min) & (th < best_t)
        if not ok.any():
            continue
        best_t[ok] = th[ok]
        best_f[ok] = fi

    return best_t, best_f


def raycast(
    mesh: TriMesh,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int,
    height: int,
) -> RayHit:
    """Cast one ray per pixel from a world<-camera pose."""
    dirs_cam = camera_rays(K, width, height).reshape(-1, 3)
    dirs = dirs_cam @ np.asarray(R, dtype=np.float64).T  # (N, 3) world
    origin = np.asarray(t, dtype=np.float64)

    best_t, best_f = intersect(mesh, origin, dirs)
    hit = best_f >= 0
    depth = np.where(hit, best_t, 0.0).astype(np.float32)
    point = origin[None, :] + np.where(hit[:, None], best_t[:, None], 0.0) * dirs
    normal = np.zeros((len(dirs), 3))
    normal[hit] = mesh.face_normals[best_f[hit]]
    label = np.full(len(dirs), -1, dtype=np.int32)
    label[hit] = mesh.face_labels[best_f[hit]]

    return RayHit(
        depth=depth.reshape(height, width),
        face_id=best_f.reshape(height, width),
        point=point.reshape(height, width, 3),
        normal=normal.reshape(height, width, 3),
        label=label.reshape(height, width),
    )


def unproject(depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Back-project a metric depth map to world points. Returns (N, 3) for valid pixels."""
    h, w = depth.shape
    dirs = camera_rays(np.asarray(K, dtype=np.float64), w, h).reshape(-1, 3)
    d = depth.reshape(-1)
    valid = d > 0
    pts_cam = dirs[valid] * d[valid, None]
    return pts_cam @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64)
