"""Procedural indoor worlds for the ground-robot simulator.

Everything is axis-aligned boxes, so the mesh, the collision test and the ground-truth free
grid all come from the same list and cannot disagree. World axes are Z up (gravity along -Z),
the same convention as OpenVINS' global frame; the viewer export is the only place Y-up appears.

Open3D's raycasting returns no colour, so the shading is done here in numpy: a per-material
procedural pattern evaluated at the world hit point, Lambertian response to two point lights
plus ambient, then the sensor effects (noise, vignette, 8-bit quantisation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy import ndimage

CEILING_Z = 2.6
DOOR_W = 0.9
DOOR_H = 2.05
ROBOT_RADIUS = 0.25
BODY_Z0, BODY_Z1 = 0.05, 0.60
GRID_RES = 0.05

MAT_FLOOR, MAT_CEIL, MAT_WALL = 0, 1, 2
MAT_OBSTACLE0 = 10  # obstacle k has material MAT_OBSTACLE0 + k

_H1 = np.uint32(0x9E3779B1)
_H2 = np.uint32(0x85EBCA77)
_H3 = np.uint32(0xC2B2AE3D)
_S13, _S16, _S8 = np.uint32(13), np.uint32(16), np.uint32(8)
_INV24 = np.float32(1.0 / 16777216.0)


def _hash01(xi: np.ndarray, yi: np.ndarray, salt: int) -> np.ndarray:
    """Uniform [0,1) hash of an integer lattice point, float32; the noise below interpolates it."""
    h = xi.view(np.uint32) * _H1 + yi.view(np.uint32) * _H2 + np.uint32(salt & 0xFFFFFFFF)
    h ^= h >> _S13
    h *= _H3
    h ^= h >> _S16
    return (h >> _S8).astype(np.float32) * _INV24


def hash_cells(x: np.ndarray, y: np.ndarray, freq: float, salt: int) -> np.ndarray:
    """Uninterpolated per-cell hash: high-frequency speckle that gives KLT something to lock on."""
    return _hash01((x * np.float32(freq) + np.float32(32768.0)).astype(np.int32),
                   (y * np.float32(freq) + np.float32(32768.0)).astype(np.int32), salt)


def vnoise2(x: np.ndarray, y: np.ndarray, salt: int) -> np.ndarray:
    """Smooth value noise on a unit lattice, vectorised."""
    xs = x + np.float32(32768.0)   # the offset makes the values positive, so truncation is floor
    ys = y + np.float32(32768.0)
    xi, yi = xs.astype(np.int32), ys.astype(np.int32)
    fx, fy = xs - xi, ys - yi
    ux = fx * fx * (np.float32(3.0) - np.float32(2.0) * fx)
    uy = fy * fy * (np.float32(3.0) - np.float32(2.0) * fy)
    h00 = _hash01(xi, yi, salt)
    h10 = _hash01(xi + 1, yi, salt)
    h01 = _hash01(xi, yi + 1, salt)
    h11 = _hash01(xi + 1, yi + 1, salt)
    return (h00 + (h10 - h00) * ux) + ((h01 - h00) + (h11 - h10 - h01 + h00) * ux) * uy


@dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray
    mat: int


@dataclass
class World:
    kind: str
    seed: int
    boxes: list[Box]
    rooms: list[tuple[float, float, float, float]]
    doors: list[tuple[float, float]]
    extent: tuple[float, float]
    albedo: np.ndarray  # per-material base colour
    posters: list[tuple]  # (axis, plane, u0, u1, v0, v1, r, g, b)
    lights: np.ndarray  # (2, 3) point-light positions
    mesh: o3d.geometry.TriangleMesh = field(repr=False, default=None)
    tri_mat: np.ndarray = field(repr=False, default=None)
    tri_axis: np.ndarray = field(repr=False, default=None)
    albedo_lum: np.ndarray = field(repr=False, default=None)
    scene: o3d.t.geometry.RaycastingScene = field(repr=False, default=None)
    free: np.ndarray = field(repr=False, default=None)  # GT free grid, (ny, nx) bool
    grid_origin: np.ndarray = field(repr=False, default=None)
    _foot: np.ndarray = field(repr=False, default=None)  # xy rects of boxes in the body band
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ---------------------------------------------------------------- generation

    @staticmethod
    def generate(seed: int = 0, kind: str = "apartment") -> World:
        rng = np.random.default_rng(seed)
        if kind == "apartment":
            W, H = 12.0, 9.0
            rooms, walls, doors = _bsp_rooms(rng, W, H, rng.integers(3, 7))
        elif kind == "corridor":
            W, H = 12.0, 9.0
            rooms, walls, doors = _corridor_layout(rng, W, H)
        elif kind == "open":
            W, H = 12.0, 9.0
            rooms, walls, doors = [(0.0, 0.0, W, H)], [], []
        else:
            raise ValueError(f"unknown world kind {kind!r}")

        boxes: list[Box] = []
        t = 0.15
        boxes.append(Box(np.array([0.0, 0.0, -0.05]), np.array([W, H, 0.0]), MAT_FLOOR))
        boxes.append(Box(np.array([0.0, 0.0, CEILING_Z]), np.array([W, H, CEILING_Z + 0.05]), MAT_CEIL))
        for lo, hi in (
            ((-t, -t, 0.0), (0.0, H + t, CEILING_Z)),
            ((W, -t, 0.0), (W + t, H + t, CEILING_Z)),
            ((-t, -t, 0.0), (W + t, 0.0, CEILING_Z)),
            ((-t, H, 0.0), (W + t, H + t, CEILING_Z)),
        ):
            boxes.append(Box(np.array(lo), np.array(hi), MAT_WALL))
        for lo, hi in walls:
            boxes.append(Box(np.array(lo), np.array(hi), MAT_WALL))

        obstacles = _place_obstacles(rng, rooms, doors, kind)
        for k, obs in enumerate(obstacles):
            for lo, hi in obs:
                boxes.append(Box(np.asarray(lo), np.asarray(hi), MAT_OBSTACLE0 + k))

        n_mat = MAT_OBSTACLE0 + max(1, len(obstacles))
        albedo = np.zeros((n_mat, 3))
        albedo[MAT_FLOOR] = [0.52, 0.40, 0.28]
        albedo[MAT_CEIL] = [0.86, 0.86, 0.84]
        albedo[MAT_WALL] = rng.uniform(0.55, 0.80) * np.array([1.0, 0.98, 0.94])
        hues = rng.permutation(np.linspace(0.0, 1.0, max(1, len(obstacles)), endpoint=False))
        for k in range(len(obstacles)):
            albedo[MAT_OBSTACLE0 + k] = _hsv(hues[k], rng.uniform(0.30, 0.65), rng.uniform(0.35, 0.80))

        posters = _place_posters(rng, W, H, walls, t)
        lights = np.array([[W * 0.30, H * 0.35, CEILING_Z - 0.15], [W * 0.72, H * 0.68, CEILING_Z - 0.15]])

        w = World(kind, seed, boxes, rooms, doors, (W, H), albedo.astype(np.float32), posters, lights)
        w.albedo_lum = (w.albedo @ np.array([0.299, 0.587, 0.114], np.float32)).astype(np.float32)
        w._build_mesh()
        w._build_grid()
        return w

    def _build_mesh(self) -> None:
        verts, tris, mats = [], [], []
        for b in self.boxes:
            v0 = len(verts)
            lo, hi = b.lo, b.hi
            for i in range(8):
                verts.append([hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]])
            # outward-facing triangles of the box (corner index bits: 1=x, 2=y, 4=z)
            faces = [
                (0, 2, 6, 4),  # -x
                (1, 5, 7, 3),  # +x
                (0, 4, 5, 1),  # -y
                (2, 3, 7, 6),  # +y
                (0, 1, 3, 2),  # -z
                (4, 6, 7, 5),  # +z
            ]
            for a, bb, c, d in faces:
                tris.append([v0 + a, v0 + bb, v0 + c])
                tris.append([v0 + a, v0 + c, v0 + d])
                mats += [b.mat, b.mat]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
        mesh.compute_triangle_normals()
        self.mesh = mesh
        self.tri_mat = np.asarray(mats, dtype=np.int32)
        self.tri_axis = np.argmax(np.abs(np.asarray(mesh.triangle_normals)), axis=1).astype(np.int8)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def _build_grid(self) -> None:
        W, H = self.extent
        nx, ny = int(round(W / GRID_RES)), int(round(H / GRID_RES))
        self.grid_origin = np.zeros(2)
        foot = []
        for b in self.boxes:
            if b.mat in (MAT_FLOOR, MAT_CEIL):
                continue
            if b.hi[2] <= BODY_Z0 or b.lo[2] >= BODY_Z1:
                continue
            foot.append([b.lo[0], b.lo[1], b.hi[0], b.hi[1]])
        self._foot = np.asarray(foot, dtype=np.float64).reshape(-1, 4)

        xs = (np.arange(nx) + 0.5) * GRID_RES
        ys = (np.arange(ny) + 0.5) * GRID_RES
        gx, gy = np.meshgrid(xs, ys)
        # exactly the collision test, evaluated on every cell centre: the free grid and
        # World.collides can then never disagree
        occ = np.zeros((ny, nx), dtype=bool)
        for lo_x, lo_y, hi_x, hi_y in self._foot:
            dx = np.clip(gx, lo_x, hi_x) - gx
            dy = np.clip(gy, lo_y, hi_y) - gy
            occ |= dx * dx + dy * dy < ROBOT_RADIUS**2
        free = ~occ
        # the robot never leaves the footprint, and only the reachable component counts
        lab, n = ndimage.label(free)
        if n:
            sizes = ndimage.sum(free, lab, range(1, n + 1))
            keep = int(np.argmax(sizes)) + 1
            free = lab == keep
        self.free = free
        iy, ix = np.nonzero(free)
        dist = ndimage.distance_transform_edt(free)
        k = int(np.argmax(dist[iy, ix]))
        self.start = (float(xs[ix[k]]), float(ys[iy[k]]), float(np.random.default_rng(self.seed).uniform(-np.pi, np.pi)))

    # ---------------------------------------------------------------- queries

    @property
    def free_area(self) -> float:
        return float(self.free.sum()) * GRID_RES**2

    def collides(self, x: float, y: float, yaw: float = 0.0) -> bool:
        """The robot is a disc of ROBOT_RADIUS, so yaw does not matter; kept for the interface."""
        if self._foot.size == 0:
            return False
        cx = np.clip(x, self._foot[:, 0], self._foot[:, 2])
        cy = np.clip(y, self._foot[:, 1], self._foot[:, 3])
        return bool(np.any((cx - x) ** 2 + (cy - y) ** 2 < ROBOT_RADIUS**2))

    def is_free(self, x: float, y: float) -> bool:
        ix, iy = int(x / GRID_RES), int(y / GRID_RES)
        if not (0 <= ix < self.free.shape[1] and 0 <= iy < self.free.shape[0]):
            return False
        return bool(self.free[iy, ix])

    def cast(self, origin: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cast rays from one origin. `dirs` is (N,3) and need not be normalised: t_hit comes
        back in units of the direction vector, which is what makes a (dx,dy,1) ray give z-depth."""
        rays = np.empty((len(dirs), 6), dtype=np.float32)
        rays[:, :3] = origin
        rays[:, 3:] = dirs
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))
        t = ans["t_hit"].numpy()
        prim = ans["primitive_ids"].numpy()
        nrm = ans["primitive_normals"].numpy()
        return t, prim, nrm

    def sample_surface(self, n: int, seed: int = 0) -> np.ndarray:
        """Points on the interior surfaces, for the completeness metric."""
        o3d.utility.random.seed(seed)
        pcd = self.mesh.sample_points_uniformly(number_of_points=n * 3)
        p = np.asarray(pcd.points)
        W, H = self.extent
        keep = (p[:, 0] > -0.02) & (p[:, 0] < W + 0.02) & (p[:, 1] > -0.02) & (p[:, 1] < H + 0.02)
        keep &= (p[:, 2] > 0.0) & (p[:, 2] < CEILING_Z)
        p = p[keep]
        return p[:n] if len(p) > n else p

    # ---------------------------------------------------------------- shading

    def shade(self, hit: np.ndarray, nrm: np.ndarray, mat: np.ndarray, ax: np.ndarray | None = None,
              gray: bool = False) -> np.ndarray:
        """Lambertian colour of surface points with the procedural texture.

        `gray=True` returns a single luminance channel instead of (N, 3). VIO only ever sees the
        grayscale frame, and only every fourth frame is also needed in colour (for the depth
        model and the TSDF), so shading one channel instead of three for the rest cuts the
        renderer - the simulator's dominant cost - by about a third.
        """
        hit = np.ascontiguousarray(hit, dtype=np.float32)
        nrm = np.ascontiguousarray(nrm, dtype=np.float32)
        alb = (self.albedo_lum[mat] if gray else self.albedo[mat])
        ch = alb if gray else alb[:, 0]      # a view used only for shape bookkeeping
        x, y, z = hit[:, 0], hit[:, 1], hit[:, 2]
        if ax is None:
            ax = np.argmax(np.abs(nrm), axis=1)

        def apply(mask, g):
            if gray:
                alb[mask] *= g
            else:
                alb[mask] *= g[:, None]

        is_floor = mat == MAT_FLOOR
        if is_floor.any():
            apply(is_floor, _floor_pattern(x[is_floor], y[is_floor], self.seed))

        is_ceil = mat == MAT_CEIL
        if is_ceil.any():
            xc, yc = x[is_ceil], y[is_ceil]
            g = np.float32(0.92) + np.float32(0.10) * vnoise2(xc * 3.0, yc * 3.0, self.seed + 7)
            g += np.float32(0.05) * hash_cells(xc, yc, 26.0, self.seed + 8)
            apply(is_ceil, g)

        is_wall = mat == MAT_WALL
        if is_wall.any():
            wx, wy, wz = x[is_wall], y[is_wall], z[is_wall]
            axw = ax[is_wall]
            on_x = axw == 0
            u = np.where(on_x, wy, wx)
            v = wz
            pc = np.where(on_x, wx, wy)  # the wall's own plane coordinate
            g = np.float32(0.84) + np.float32(0.20) * vnoise2(u * 3.0, v * 3.0, self.seed + 11)
            g += np.float32(0.09) * hash_cells(u, v, 12.0, self.seed + 12)
            g += np.float32(0.07) * hash_cells(u, v, 34.0, self.seed + 13)
            g[v < 0.10] *= np.float32(0.55)  # skirting board
            wa = alb[is_wall] * (g if gray else g[:, None])
            # cheap reject: only walls whose plane is inside the visible x/y range can carry a poster
            rng_ax = (float(wx.min()), float(wx.max())), (float(wy.min()), float(wy.max()))
            vmin, vmax = float(v.min()), float(v.max())
            for axis, plane, u0, u1, v0, v1, r, gg, b in self.posters:
                lo, hi = rng_ax[axis]
                if plane < lo - 0.2 or plane > hi + 0.2 or v1 < vmin or v0 > vmax:
                    continue
                sel = (on_x if axis == 0 else ~on_x) & (np.abs(pc - plane) < 0.20)
                sel &= (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
                if not sel.any():
                    continue
                us, vs = u[sel], v[sel]
                tex = np.float32(0.55) + np.float32(0.5) * vnoise2(us * 14.0, vs * 14.0, self.seed + 21)
                tex += np.float32(0.14) * hash_cells(us, vs, 40.0, self.seed + 22)
                frame = (us < u0 + 0.04) | (us > u1 - 0.04) | (vs < v0 + 0.04) | (vs > v1 - 0.04)
                tex = np.where(frame, tex * np.float32(0.30), tex)
                if gray:
                    wa[sel] = np.float32(0.299 * r + 0.587 * gg + 0.114 * b) * tex
                else:
                    c = np.empty((len(us), 3), dtype=np.float32)
                    c[:] = (r, gg, b)
                    c *= tex[:, None]
                    wa[sel] = c
            alb[is_wall] = wa

        is_obs = mat >= MAT_OBSTACLE0
        if is_obs.any():
            axo = ax[is_obs]
            uo = np.where(axo == 0, y[is_obs], x[is_obs])
            vo = np.where(axo == 2, y[is_obs], z[is_obs])
            g = np.float32(0.80) + np.float32(0.28) * vnoise2(uo * 7.0, vo * 7.0, self.seed + 31)
            g += np.float32(0.10) * hash_cells(uo, vo, 30.0, self.seed + 32)
            # a darker band near the face edges keeps the silhouette high-contrast
            g *= np.float32(0.92) + np.float32(0.12) * (np.abs(uo * 5.0 % 1.0 - 0.5) + np.abs(vo * 5.0 % 1.0 - 0.5))
            apply(is_obs, g)

        n = nrm * (np.float32(1.0) / np.sqrt(np.einsum("ij,ij->i", nrm, nrm) + np.float32(1e-12)))[:, None]
        lit = np.full(len(hit), np.float32(0.38))
        for lp in self.lights.astype(np.float32):
            d = lp - hit
            d2 = np.einsum("ij,ij->i", d, d)
            ndl = np.abs(np.einsum("ij,ij->i", n, d)) / np.sqrt(d2 + np.float32(1e-12))
            lit += np.float32(0.80) * ndl / (np.float32(1.0) + d2 * np.float32(1.0 / 36.0))
        np.multiply(alb, lit if gray else lit[:, None], out=alb)
        return np.clip(alb, 0.0, 1.0, out=alb)


def _floor_pattern(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 101)
    if rng.random() < 0.5:  # planks along x
        wpl, lpl = 0.18, 1.2
        row = np.floor(y / wpl)
        rowi = (row + np.float32(32768.0)).astype(np.int32)
        off = _hash01(rowi, np.zeros(len(row), dtype=np.int32), seed) * np.float32(lpl)
        plank = np.floor((x + off) / lpl)
        g = np.float32(0.78) + np.float32(0.34) * _hash01(rowi, (plank + np.float32(32768.0)).astype(np.int32), seed + 1)
        g *= np.float32(0.88) + np.float32(0.22) * hash_cells(x, y, 22.0, seed + 2)
        g += np.float32(0.06) * hash_cells(x, y, 40.0, seed + 5)
        grout = (np.abs(y / wpl - row - 0.5) > 0.46) | (np.abs((x + off) / lpl - plank - 0.5) > 0.492)
    else:  # tiles
        sq = 0.40
        gx, gy = np.floor(x / sq), np.floor(y / sq)
        g = np.float32(0.80) + np.float32(0.26) * _hash01((gx + np.float32(32768.0)).astype(np.int32),
                                                          (gy + np.float32(32768.0)).astype(np.int32), seed + 3)
        g *= np.float32(0.90) + np.float32(0.16) * hash_cells(x, y, 14.0, seed + 4)
        g += np.float32(0.07) * hash_cells(x, y, 36.0, seed + 5)
        grout = (np.abs(x / sq - gx - 0.5) > 0.45) | (np.abs(y / sq - gy - 0.5) > 0.45)
    g[grout] *= np.float32(0.55)
    return g


def _hsv(h: float, s: float, v: float) -> np.ndarray:
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return np.array([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i])


# ---------------------------------------------------------------- layouts


def _wall_with_door(axis: int, plane: float, a: float, b: float, door: float, t: float) -> list:
    """Wall along `axis` (0: constant x, 1: constant y) from a to b with a doorway at `door`."""
    half = t / 2
    segs = []
    d0, d1 = door - DOOR_W / 2, door + DOOR_W / 2
    spans = [(a, d0), (d1, b)]
    for s0, s1 in spans:
        if s1 - s0 < 0.02:
            continue
        if axis == 0:
            segs.append(((plane - half, s0, 0.0), (plane + half, s1, CEILING_Z)))
        else:
            segs.append(((s0, plane - half, 0.0), (s1, plane + half, CEILING_Z)))
    if axis == 0:  # lintel over the door
        segs.append(((plane - half, d0, DOOR_H), (plane + half, d1, CEILING_Z)))
    else:
        segs.append(((d0, plane - half, DOOR_H), (d1, plane + half, CEILING_Z)))
    return segs


def _bsp_rooms(rng, W: float, H: float, target: int):
    rooms = [(0.0, 0.0, W, H)]
    walls, doors = [], []
    t = 0.12
    guard = 0
    while len(rooms) < target and guard < 60:
        guard += 1
        areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in rooms]
        i = int(np.argmax(areas))
        x0, y0, x1, y1 = rooms[i]
        along_x = (x1 - x0) >= (y1 - y0)
        span = (x1 - x0) if along_x else (y1 - y0)
        if span < 3.4:
            break
        f = rng.uniform(0.34, 0.66)
        cut = (x0 + f * span) if along_x else (y0 + f * span)
        lo, hi = (y0, y1) if along_x else (x0, x1)
        if hi - lo < DOOR_W + 1.0:
            break
        door = rng.uniform(lo + 0.55, hi - 0.55)
        walls += _wall_with_door(0 if along_x else 1, cut, lo, hi, door, t)
        doors.append((cut, door) if along_x else (door, cut))
        rooms.pop(i)
        if along_x:
            rooms += [(x0, y0, cut, y1), (cut, y0, x1, y1)]
        else:
            rooms += [(x0, y0, x1, cut), (x0, cut, x1, y1)]
    return rooms, walls, doors


def _corridor_layout(rng, W: float, H: float):
    """A loop corridor around a central block with rooms off the outside."""
    t = 0.12
    cw = 1.6  # corridor width
    walls, doors, rooms = [], [], []
    bx0, by0, bx1, by1 = cw + 1.6, cw + 1.4, W - cw - 1.6, H - cw - 1.4
    for lo, hi in (
        ((bx0, by0, 0.0), (bx1, by0 + t, CEILING_Z)),
        ((bx0, by1 - t, 0.0), (bx1, by1, CEILING_Z)),
        ((bx0, by0, 0.0), (bx0 + t, by1, CEILING_Z)),
        ((bx1 - t, by0, 0.0), (bx1, by1, CEILING_Z)),
    ):
        walls.append((lo, hi))
    # side rooms carved out of the corridor ring at the four outer strips
    for k, (axis, plane, a, b) in enumerate(
        ((1, cw, 0.4, W / 2 - 0.4), (1, H - cw, W / 2 + 0.4, W - 0.4))
    ):
        door = rng.uniform(a + 0.6, b - 0.6)
        walls += _wall_with_door(axis, plane, a, b, door, t)
        doors.append((door, plane))
    rooms += [(0.3, 0.3, W / 2 - 0.6, cw - 0.2), (W / 2 + 0.6, H - cw + 0.2, W - 0.3, H - 0.3)]
    rooms += [(0.3, cw + 0.3, bx0 - 0.3, H - 0.3), (bx1 + 0.3, 0.3, W - 0.3, H - cw - 0.3)]
    return rooms, walls, doors


def _place_obstacles(rng, rooms, doors, kind: str):
    """Each obstacle is a list of (lo, hi) boxes; tables are a slab plus four legs."""
    n_target = int(rng.integers(8, 21)) if kind != "open" else int(rng.integers(14, 25))
    placed, foot = [], []
    tries = 0
    while len(placed) < n_target and tries < 800:
        tries += 1
        r = rooms[int(rng.integers(0, len(rooms)))]
        kindr = rng.random()
        if kindr < 0.40:
            sx, sy = rng.uniform(0.3, 1.2), rng.uniform(0.3, 1.2)
            h = rng.uniform(0.3, 1.2)
            shape = "box"
        elif kindr < 0.55:
            sx = sy = rng.uniform(0.22, 0.35)
            h = CEILING_Z
            shape = "pillar"
        elif kindr < 0.78:
            sx, sy = rng.uniform(0.8, 1.4), rng.uniform(0.6, 0.9)
            h = rng.uniform(0.70, 0.80)
            shape = "table"
        else:
            sx, sy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
            h = rng.uniform(0.10, 0.30)
            shape = "low"
        m = 0.40
        if r[2] - r[0] < sx + 2 * m or r[3] - r[1] < sy + 2 * m:
            continue
        cx = rng.uniform(r[0] + m + sx / 2, r[2] - m - sx / 2)
        cy = rng.uniform(r[1] + m + sy / 2, r[3] - m - sy / 2)
        lo = np.array([cx - sx / 2, cy - sy / 2])
        hi = np.array([cx + sx / 2, cy + sy / 2])
        if any(np.hypot(cx - dx, cy - dy) < 1.1 for dx, dy in doors):
            continue
        if any(not (hi[0] + 0.55 < f[0] or lo[0] - 0.55 > f[2] or hi[1] + 0.55 < f[1] or lo[1] - 0.55 > f[3]) for f in foot):
            continue
        foot.append([lo[0], lo[1], hi[0], hi[1]])
        if shape == "table":
            top = h
            parts = [((lo[0], lo[1], top - 0.05), (hi[0], hi[1], top))]
            lw = 0.06
            for ox, oy in ((lo[0] + 0.05, lo[1] + 0.05), (hi[0] - 0.05 - lw, lo[1] + 0.05),
                           (lo[0] + 0.05, hi[1] - 0.05 - lw), (hi[0] - 0.05 - lw, hi[1] - 0.05 - lw)):
                parts.append(((ox, oy, 0.0), (ox + lw, oy + lw, top - 0.05)))
        else:
            parts = [((lo[0], lo[1], 0.0), (hi[0], hi[1], h))]
        placed.append([(np.asarray(a), np.asarray(b)) for a, b in parts])
    return placed


def _place_posters(rng, W: float, H: float, walls, t: float):
    """Contrasting rectangles on the outer walls and on the internal wall planes."""
    out = []
    planes = [(0, 0.0, (0.0, H)), (0, W, (0.0, H)), (1, 0.0, (0.0, W)), (1, H, (0.0, W))]
    for lo, hi in walls:
        if hi[0] - lo[0] < hi[1] - lo[1]:
            planes.append((0, 0.5 * (lo[0] + hi[0]), (lo[1], hi[1])))
        else:
            planes.append((1, 0.5 * (lo[1] + hi[1]), (lo[0], hi[0])))
    for axis, plane, (a, b) in planes:
        for _ in range(int(rng.integers(1, 4))):
            if b - a < 1.6:
                continue
            w = rng.uniform(0.5, min(1.1, b - a - 0.6))
            h = rng.uniform(0.5, 1.0)
            u0 = rng.uniform(a + 0.2, b - w - 0.2)
            v0 = rng.uniform(0.6, 1.9 - h)
            c = _hsv(rng.random(), rng.uniform(0.5, 0.9), rng.uniform(0.35, 0.9))
            out.append((axis, plane, u0, u0 + w, v0, v0 + h, *c))
    return out


_VIGNETTE: dict[tuple[int, int], np.ndarray] = {}
_NOISE_POOL: dict[int, np.ndarray] = {}


def apply_sensor_effects(rgb: np.ndarray, rng: np.random.Generator, noise_sigma: float = 2.0 / 255.0) -> np.ndarray:
    """Float 0..1 HxWx3 -> uint8 with read noise, a mild vignette and 8-bit quantisation."""
    h, w = rgb.shape[:2]
    key = (h, w, rgb.ndim)
    vig = _VIGNETTE.get(key)
    if vig is None:
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) / 0.5
        vig = (255.0 * (1.0 - 0.28 * r2)).astype(np.float32)
        if rgb.ndim == 3:
            vig = vig[:, :, None]
        _VIGNETTE[key] = vig
    out = rgb.astype(np.float32, copy=False) * vig
    if noise_sigma > 0:
        n = out.size
        pool = _NOISE_POOL.get(n)
        if pool is None:
            pool = np.random.default_rng(12345).normal(0.0, 255.0, n * 4).astype(np.float32)
            _NOISE_POOL[n] = pool
        o = int(rng.integers(0, len(pool) - n))
        out += pool[o : o + n].reshape(out.shape) * np.float32(noise_sigma)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
