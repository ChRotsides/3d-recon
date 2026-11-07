"""A* on the discovered costmap, string-pulling, and a pure-pursuit follower.

Cells are the navgrid's 2D cells; everything is in the estimate frame. Nothing here knows the
world exists.
"""

from __future__ import annotations

import heapq

import numpy as np

LOOKAHEAD = 0.45   # shorter than the 0.6 m brief value: at 0.6 m the follower cut corners
                   # into walls the costmap had correctly marked
_NBRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]


def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], max_nodes: int = 400_000):
    """8-connected A* over a cost image indexed [y, x]. Returns a list of (ix, iy) or None."""
    ny, nx = cost.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < nx and 0 <= sy < ny and 0 <= gx < nx and 0 <= gy < ny):
        return None
    if not np.isfinite(cost[gy, gx]):
        return None
    if not np.isfinite(cost[sy, sx]):
        # the robot is standing inside its own inflated footprint (a wall it drove close to, or
        # a map that is still forming); start from the nearest cell it can legally occupy
        free = np.argwhere(np.isfinite(cost))
        if not len(free):
            return None
        k = int(np.argmin((free[:, 0] - sy) ** 2 + (free[:, 1] - sx) ** 2))
        if (free[k, 0] - sy) ** 2 + (free[k, 1] - sx) ** 2 > 64:
            return None
        sy, sx = int(free[k, 0]), int(free[k, 1])
    g = np.full(cost.shape, np.inf, dtype=np.float32)
    came = np.full(cost.shape, -1, dtype=np.int64)
    g[sy, sx] = 0.0
    h0 = np.hypot(gx - sx, gy - sy)
    heap = [(h0, 0.0, sy * nx + sx)]
    seen = 0
    while heap:
        f, gc, node = heapq.heappop(heap)
        y, x = divmod(node, nx)
        if gc > g[y, x]:
            continue
        if x == gx and y == gy:
            path = [(x, y)]
            while came[y, x] >= 0:
                y, x = divmod(int(came[y, x]), nx)
                path.append((x, y))
            return path[::-1]
        seen += 1
        if seen > max_nodes:
            return None
        for dx, dy, w in _NBRS:
            nxx, nyy = x + dx, y + dy
            if not (0 <= nxx < nx and 0 <= nyy < ny):
                continue
            c = cost[nyy, nxx]
            if not np.isfinite(c):
                continue
            ng = gc + w * c
            if ng < g[nyy, nxx]:
                g[nyy, nxx] = ng
                came[nyy, nxx] = node
                heapq.heappush(heap, (ng + np.hypot(gx - nxx, gy - nyy), ng, nyy * nx + nxx))
    return None


SHORTCUT_MAX_COST = 3.0   # ~0.44 m of clearance; see NavGrid.costmap for the cost profile


def line_free(cost: np.ndarray, a, b, max_cost: float = np.inf) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    xs = np.round(np.linspace(a[0], b[0], n * 2 + 1)).astype(int)
    ys = np.round(np.linspace(a[1], b[1], n * 2 + 1)).astype(int)
    c = cost[ys, xs]
    return bool(np.isfinite(c).all() and (c <= max_cost).all())


def smooth(path, cost: np.ndarray, max_cost: float = SHORTCUT_MAX_COST):
    """String pulling that respects the clearance gradient.

    Shortcutting on `isfinite` alone undoes the cost gradient A* just paid for: the smoothed
    line hugs the lethal boundary and the follower, which cannot track a line exactly, clips the
    wall. A shortcut is therefore only taken through cells cheaper than `max_cost`; the original
    step to the next cell is always allowed, so a doorway is still passable.
    """
    if not path or len(path) < 3:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_free(cost, path[i], path[j], max_cost):
            j -= 1
        out.append(path[j])
        i = j
    return out


class PurePursuit:
    def __init__(self, lookahead: float = LOOKAHEAD, v_max: float = 0.6, w_max: float = 1.2):
        self.lookahead = lookahead
        self.v_max = v_max
        self.w_max = w_max

    def command(self, pose, path_xy: np.ndarray):
        """pose = (x, y, yaw) in the estimate frame; path_xy = (N, 2) metres. -> (v, omega, done)"""
        if path_xy is None or len(path_xy) == 0:
            return 0.0, 0.0, True
        x, y, yaw = pose
        p = np.asarray(path_xy, dtype=float)
        d = np.linalg.norm(p - np.array([x, y]), axis=1)
        i0 = int(np.argmin(d))
        if i0 >= len(p) - 1 and d[i0] < 0.25:
            return 0.0, 0.0, True
        ahead = np.nonzero(d[i0:] >= self.lookahead)[0]
        tgt = p[i0 + ahead[0]] if len(ahead) else p[-1]
        dx, dy = tgt[0] - x, tgt[1] - y
        # the target in the robot frame; the pure-pursuit curvature is 2*y_r / L^2
        c, s = np.cos(-yaw), np.sin(-yaw)
        xr, yr = c * dx - s * dy, s * dx + c * dy
        ang = np.arctan2(yr, xr)
        L2 = max(1e-3, xr * xr + yr * yr)
        if abs(ang) > np.deg2rad(60) or xr <= 0.02:
            return 0.0, float(np.clip(1.4 * ang, -self.w_max, self.w_max)), False  # turn in place first
        w = float(np.clip(2.0 * yr / L2 * self.v_max, -self.w_max, self.w_max))
        v = float(self.v_max * max(0.25, 1.0 - abs(ang) / np.deg2rad(60)))
        if np.linalg.norm(p[-1] - np.array([x, y])) < 0.5:
            v *= 0.6
        return v, w, False


def path_blocked(path_xy, cost: np.ndarray, to_cell) -> bool:
    if path_xy is None or len(path_xy) == 0:
        return False
    for xy in path_xy:
        ix, iy = to_cell(xy)
        if not (0 <= iy < cost.shape[0] and 0 <= ix < cost.shape[1]) or not np.isfinite(cost[iy, ix]):
            return True
    return False
