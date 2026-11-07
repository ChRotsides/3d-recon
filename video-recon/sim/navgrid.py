"""The map the robot navigates on: a 3D log-odds occupancy grid built from the same depth maps
and the same *estimated* poses that go into the TSDF, plus the 2D costmap derived from it.

Nothing here knows the world exists. It sees depth in millimetres, K, and a camera pose in
whatever frame the pose source produces; if that frame drifts, the map drifts with it, which is
the point of the experiment.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

RES = 0.10
MAX_RANGE = 5.0
L_FREE, L_OCC = -0.45, 0.90
L_MIN, L_MAX = -2.5, 3.5
T_FREE, T_OCC = -0.35, 0.35
ROBOT_RADIUS = 0.25
INFLATE_MARGIN = 0.10
# The hard block is the robot's radius plus half a grid cell (the obstacle surface can sit
# anywhere inside the cell the EDT measures to). The specified 0.10 m safety margin is a steep
# cost band on top rather than a second hard block: at 0.10 m resolution a 0.9 m doorway leaves
# 0.45 m of clearance either side, so blocking out to 0.35 m closes every door in the world.
LETHAL_RADIUS = ROBOT_RADIUS + 0.05
BAND_LO, BAND_HI = 0.05, 0.80
FLOOR_TOL = 0.12   # hits this close to the estimated floor are ground, not obstacle

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
UNKNOWN_COST = 2.5
MAX_EXTENT = np.array([40.0, 40.0, 8.0])   # metres; a guard against a diverged pose eating RAM
UNKNOWN_REACH = 2   # cells of unknown the planner may cross to bridge holes in the free carve


class NavGrid:
    def __init__(self, res: float = RES, max_range: float = MAX_RANGE, stride: int = 3):
        self.res = res
        self.max_range = max_range
        self.stride = stride
        self.origin = None  # (3,) world coords of voxel (0,0,0)'s corner
        self.log = None  # (nz, ny, nx) float32
        self._rays = None
        self.floor_z = None   # set by the caller each cycle; without it no hit counts as ground
        self._steps = np.arange(0.05, max_range, res * 0.5, dtype=np.float32)
        self._disc = None

    # ------------------------------------------------------------------ storage

    def _ensure(self, lo: np.ndarray, hi: np.ndarray) -> None:
        pad = 2.0
        lo = np.floor((lo - pad) / self.res) * self.res
        hi = np.ceil((hi + pad) / self.res) * self.res
        # a diverged pose asks for a grid of unbounded size; clamp to something a room-scale
        # robot could plausibly have visited so the process cannot be killed by the allocation
        if self.origin is not None:
            c = self.origin + np.array(self.log.shape[::-1]) * self.res * 0.5
            lo = np.maximum(lo, c - MAX_EXTENT / 2)
            hi = np.minimum(hi, c + MAX_EXTENT / 2)
            if np.any(hi <= lo):
                return
        if self.log is None:
            self.origin = lo
            n = np.maximum(1, np.round((hi - lo) / self.res).astype(int))
            self.log = np.zeros(n[::-1], dtype=np.float32)
            return
        cur_hi = self.origin + np.array(self.log.shape[::-1]) * self.res
        if (lo >= self.origin - 1e-9).all() and (hi <= cur_hi + 1e-9).all():
            return
        nlo = np.minimum(lo, self.origin)
        nhi = np.maximum(hi, cur_hi)
        n = np.maximum(1, np.round((nhi - nlo) / self.res).astype(int))
        new = np.zeros(n[::-1], dtype=np.float32)
        off = np.round((self.origin - nlo) / self.res).astype(int)
        s = self.log.shape
        new[off[2] : off[2] + s[0], off[1] : off[1] + s[1], off[0] : off[0] + s[2]] = self.log
        self.log, self.origin = new, nlo

    def _index(self, pts: np.ndarray):
        """World points -> flat indices into self.log, plus an in-bounds mask.

        int64 throughout: a grid that has grown to a few hundred cells a side has more than
        2^31 flat indices, and an int32 product wraps negative, which np.bincount rejects with
        "'list' argument must have no negative elements". That crashed five sweep runs.
        """
        g = np.floor((pts - self.origin) / self.res).astype(np.int64)
        nz, ny, nx = (np.int64(v) for v in self.log.shape)
        ok = (g[:, 0] >= 0) & (g[:, 0] < nx) & (g[:, 1] >= 0) & (g[:, 1] < ny) & (g[:, 2] >= 0) & (g[:, 2] < nz)
        idx = (g[ok, 2] * ny + g[ok, 1]) * nx + g[ok, 0]
        return idx, ok

    def _add(self, idx: np.ndarray, delta: float) -> None:
        if len(idx) == 0:
            return
        flat = self.log.reshape(-1)
        flat += np.bincount(idx, minlength=flat.size).astype(np.float32) * np.float32(delta)

    # ------------------------------------------------------------------ updates

    def integrate(self, depth_mm: np.ndarray, K, R_cw: np.ndarray, c_w: np.ndarray) -> None:
        """Carve free space from the camera to each hit, then mark the hit cell occupied."""
        st = self.stride
        h, w = depth_mm.shape
        if self._rays is None or self._rays.shape[0] != len(range(0, h, st)) * len(range(0, w, st)):
            fx, fy, cx, cy = [float(v) for v in K]
            u, v = np.meshgrid(np.arange(0, w, st, dtype=np.float32), np.arange(0, h, st, dtype=np.float32))
            self._rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1).reshape(-1, 3)
        d = depth_mm[::st, ::st].astype(np.float32) * np.float32(0.001)
        # flying-pixel rejection, the same 8 % neighbour test tsdf.cpp uses
        pad = np.pad(d, 1, mode="edge")
        jump = np.zeros_like(d, dtype=bool)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            jump |= np.abs(pad[1 + dy : 1 + dy + d.shape[0], 1 + dx : 1 + dx + d.shape[1]] - d) > 0.08 * d
        z = d.reshape(-1)
        good = (z > 0.2) & (z < self.max_range) & ~jump.reshape(-1)
        if not good.any():
            return
        R = np.asarray(R_cw, dtype=np.float32)
        c = np.asarray(c_w, dtype=np.float32)
        dirs = self._rays[good] @ R.T
        zg = z[good]
        hit = c + dirs * zg[:, None]
        self._ensure(np.minimum(hit.min(0), c), np.maximum(hit.max(0), c))
        # free space: sample along each ray in z-depth, stopping one cell short of the hit
        t = self._steps
        m = t[None, :] < (zg[:, None] - self.res)
        if m.any():
            pts = (c[None, None, :] + dirs[:, None, :] * t[None, :, None])[m]
            idx, _ = self._index(pts)
            self._add(idx, L_FREE)
        # the floor is what the robot drives on: its hits are evidence of free space, not of an
        # obstacle. Without this the whole visible floor becomes lethal and nothing is reachable.
        if self.floor_z is None:
            ground = np.zeros(len(hit), bool)
        else:
            ground = np.abs(hit[:, 2] - self.floor_z) < FLOOR_TOL
        idx, _ = self._index(hit[~ground])
        self._add(idx, L_OCC)
        if ground.any():
            idx, _ = self._index(hit[ground])
            self._add(idx, L_FREE)
        np.clip(self.log, L_MIN, L_MAX, out=self.log)

    def mark_surface(self, pts: np.ndarray) -> None:
        """TSDF surface points are evidence of occupancy wherever they land."""
        if len(pts) == 0:
            return
        p = np.ascontiguousarray(pts[:, :3], dtype=np.float32)
        if self.floor_z is not None:
            p = p[np.abs(p[:, 2] - self.floor_z) >= FLOOR_TOL]
            if not len(p):
                return
        self._ensure(p.min(0), p.max(0))
        idx, _ = self._index(p)
        if len(idx):
            u = np.unique(idx)  # one vote per cell per refresh, not one per point
            self._add(u, L_OCC)
            np.clip(self.log, L_MIN, L_MAX, out=self.log)

    def mark_robot(self, xy, floor_z: float, radius: float = ROBOT_RADIUS, z_hi: float = BAND_HI + 0.05) -> None:
        """The robot is standing here, so this column is free.

        A camera 0.5 m up and pitched down still cannot see the ground under its own wheels, so
        without this the cell the robot occupies stays UNKNOWN for ever, A* has no legal start
        cell, and the planner reports every frontier unreachable while the robot spins on the
        spot. Real costmaps clear the footprint for exactly this reason. No ground truth is
        involved: the pose is the estimate, and standing somewhere is evidence it is free.
        """
        if self._disc is None:
            k = int(np.ceil(radius / self.res))
            dy, dx = np.mgrid[-k : k + 1, -k : k + 1]
            m = (dx**2 + dy**2) * self.res**2 <= radius**2
            self._disc = np.stack([dx[m], dy[m]], 1) * self.res
        zs = np.arange(floor_z + 0.02, floor_z + z_hi, self.res, dtype=np.float32)
        d = self._disc
        pts = np.empty((len(d) * len(zs), 3), np.float32)
        pts[:, 0] = np.tile(d[:, 0] + xy[0], len(zs))
        pts[:, 1] = np.tile(d[:, 1] + xy[1], len(zs))
        pts[:, 2] = np.repeat(zs, len(d))
        self._ensure(pts.min(0), pts.max(0))
        idx, _ = self._index(pts)
        self._add(np.unique(idx), L_FREE)
        np.clip(self.log, L_MIN, L_MAX, out=self.log)

    # ------------------------------------------------------------------ costmap

    def costmap(self, floor_z: float):
        """(state, cost, origin_xy) at self.res.

        state is UNKNOWN/FREE/OCCUPIED per 2D cell over the height band above the estimated
        floor; cost is 1.0 in open space, rising near obstacles, and inf where the robot's
        inflated footprint would touch something.
        """
        nz, ny, nx = self.log.shape
        z0 = int(np.floor((floor_z + BAND_LO - self.origin[2]) / self.res))
        z1 = int(np.ceil((floor_z + BAND_HI - self.origin[2]) / self.res))
        z0, z1 = max(0, z0), min(nz, max(1, z1))
        band = self.log[z0:z1]
        occ = (band > T_OCC).any(0)
        free = (band < T_FREE).all(0) & ~occ
        state = np.where(occ, OCCUPIED, np.where(free, FREE, UNKNOWN)).astype(np.uint8)
        dist = ndimage.distance_transform_edt(~occ, sampling=self.res)
        cost = np.where(dist < LETHAL_RADIUS, np.inf,
                        1.0 + 8.0 * np.exp(-(dist - LETHAL_RADIUS) / INFLATE_MARGIN)).astype(np.float32)
        # Planning stays inside known-free space. Unknown cells are crossable only where they
        # touch it, which bridges the speckle holes the ray carving leaves at long range without
        # letting the planner shoot a straight line through an unobserved wall - the failure that
        # otherwise drives the robot into walls it has never seen.
        unk = state == UNKNOWN
        bridge = ndimage.binary_dilation(free, iterations=UNKNOWN_REACH) & unk
        cost[unk & ~bridge] = np.inf
        cost[bridge] *= UNKNOWN_COST
        return state, cost, self.origin[:2].copy()

    def world_to_cell(self, xy) -> tuple[int, int]:
        g = (np.asarray(xy[:2]) - self.origin[:2]) / self.res
        return int(g[0]), int(g[1])

    def cell_to_world(self, ix, iy):
        return (self.origin[0] + (np.asarray(ix) + 0.5) * self.res,
                self.origin[1] + (np.asarray(iy) + 0.5) * self.res)

    def save(self, path) -> None:
        np.savez_compressed(path, log=self.log, origin=self.origin, res=self.res)
