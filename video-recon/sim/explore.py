"""Frontier exploration on the discovered costmap.

A frontier is a known-free cell touching unknown space. Clusters are scored by size over the
A* path length to reach them, and then *committed to*: re-scoring every second makes a
differential drive turn towards a new target every second and never translate. A blacklist
holds the frontiers it got stuck on and an escape manoeuvre runs when it stops making progress.
Nothing here knows the world exists.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .navgrid import FREE, UNKNOWN
from .planner import PurePursuit, astar, smooth

MIN_CLUSTER = 5
STUCK_TIME = 8.0
BLACKLIST_TIME = 60.0
BLACKLIST_RADIUS = 1.0
NO_FRONTIER_LIMIT = 25    # replans in a row with no frontier at all before calling it done
STUCK_FRONTIER_LIMIT = 60  # ... or with frontiers that are all unreachable
TARGET_TIMEOUT = 25.0    # give up on a target the robot has not reached in this long
ARRIVE_RADIUS = 0.30
MIN_PATH_CELLS = 4       # a target closer than this is reached the moment it is chosen


def frontiers(state: np.ndarray, cost: np.ndarray, reachable_only: bool = True):
    """(labels, sizes, centroids in cells) of the frontier clusters worth going to."""
    free = state == FREE
    unk = state == UNKNOWN
    nbr = np.zeros_like(unk)
    nbr[1:, :] |= unk[:-1, :]
    nbr[:-1, :] |= unk[1:, :]
    nbr[:, 1:] |= unk[:, :-1]
    nbr[:, :-1] |= unk[:, 1:]
    f = free & nbr
    if reachable_only:
        f &= np.isfinite(cost)
    lab, n = ndimage.label(f, structure=np.ones((3, 3)))
    if n == 0:
        return lab, np.zeros(0), np.zeros((0, 2))
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.nonzero(sizes >= MIN_CLUSTER)[0] + 1
    if not len(keep):
        return lab, np.zeros(0), np.zeros((0, 2))
    cent = np.array(ndimage.center_of_mass(f, lab, keep))  # (k, 2) as (row, col)
    return lab, sizes[keep - 1], cent[:, ::-1]  # -> (col, row) = (ix, iy)


class Explorer:
    def __init__(self, v_max: float = 0.6, w_max: float = 1.2, lookahead: float = 0.6):
        self.follow = PurePursuit(lookahead, v_max, w_max)
        self.target = None  # (x, y) in the estimate frame
        self.target_score = 0.0
        self.path = None
        self.blacklist: list[tuple[float, float, float]] = []  # (x, y, expiry)
        self.last_progress_t = 0.0
        self.last_pos = None
        self.last_yaw = 0.0
        self.escape_until = -1.0
        self.target_deadline = 0.0
        self.done = False
        self.no_frontier = 0
        self.n_frontiers = 0
        self.events: list[tuple] = []

    # ------------------------------------------------------------------ planning

    def replan(self, t: float, pose, grid, state, cost):
        """Pick (or keep) a frontier and lay a path to it.

        Committing matters more than choosing well: re-scoring every second makes the robot turn
        towards a different target every second and never translate at all. A target is therefore
        kept until it is reached, explored away, unreachable, or has taken too long.
        """
        # Two frontier sets. The scoring loop must see the *reachable* ones: filtering by cost
        # splits and re-sizes the clusters, so the biggest twelve of the unfiltered set can all
        # be unreachable while perfectly good reachable ones are never even scored.
        _, sizes, cents = frontiers(state, cost, reachable_only=True)
        self.n_frontiers = int(len(sizes))
        start = grid.world_to_cell(pose[:2])
        self.blacklist = [b for b in self.blacklist if b[2] > t]

        if self.target is not None:
            reached = np.hypot(pose[0] - self.target[0], pose[1] - self.target[1]) < ARRIVE_RADIUS
            if reached or t > self.target_deadline or not self._still_useful(state, grid, self.target):
                self.target, self.path, self.target_score = None, None, 0.0
            else:
                p = self._path_to(cost, grid, start, self.target)
                if p is not None:
                    self.path = p
                    self.no_frontier = 0
                    return self.target
                self.target, self.path, self.target_score = None, None, 0.0

        best, best_score, best_path = None, 0.0, None
        order = np.argsort(-sizes)[:12]  # only the biggest few are worth an A* each
        for k in order:
            cx, cy = cents[k]
            wx, wy = grid.cell_to_world(cx, cy)
            if any((wx - bx) ** 2 + (wy - by) ** 2 < BLACKLIST_RADIUS**2 for bx, by, _ in self.blacklist):
                continue
            goal = self._nearest_reachable(cost, int(round(cx)), int(round(cy)))
            if goal is None:
                continue
            p = astar(cost, start, goal)
            if not p or len(p) < MIN_PATH_CELLS:
                continue   # a goal the robot is already standing on is not worth driving to
            score = float(sizes[k]) / max(0.5, len(p) * grid.res)
            if score > best_score:
                best, best_score, best_path = (float(wx), float(wy)), score, p
        if best is None:
            # No reachable frontier gave a path. Look at every frontier, reachable or not, and
            # head for the cell nearest the biggest one; a doorway that is not fully carved yet
            # looks unreachable until the robot is standing in front of it.
            _, sizes_a, cents_a = frontiers(state, cost, reachable_only=False)
            if not len(sizes_a) and not len(sizes):
                return self._no_frontier(empty=True)
            cents_all = cents_a if len(sizes_a) else cents
            sizes_all = sizes_a if len(sizes_a) else sizes
            k = int(np.argmax(sizes_all))
            near = self._nearest_reachable(cost, int(round(cents_all[k][0])), int(round(cents_all[k][1])), r=16)
            p = astar(cost, start, near) if near else None
            if not (p and len(p) > 3):
                # nothing near the frontier is reachable either: go to the furthest cell the
                # robot *can* stand on. Standing still never makes a frontier reachable; moving
                # to the other side of the known space usually does.
                far = self._furthest_reachable(cost, start)
                p = astar(cost, start, far) if far else None
                near = far
            if p and len(p) > 3:
                self.target = tuple(float(v) for v in grid.cell_to_world(*near))
                self.target_deadline = t + TARGET_TIMEOUT
                self.target_score = 0.0
                self.path = self._to_world(smooth(p, cost), grid)
                self.no_frontier = 0
                return self.target
            return self._no_frontier(empty=False)
        self.no_frontier = 0
        self.target, self.target_score = best, best_score
        self.target_deadline = t + TARGET_TIMEOUT
        self.path = self._to_world(smooth(best_path, cost), grid)
        return best

    def _path_to(self, cost, grid, start, xy):
        goal = self._nearest_reachable(cost, *grid.world_to_cell(xy), r=6)
        if goal is None:
            return None
        p = astar(cost, start, goal)
        if not p or len(p) < 2:
            return None
        return self._to_world(smooth(p, cost), grid)

    @staticmethod
    def _still_useful(state, grid, xy, r=3):
        """A target stops being worth driving to once the space around it is all known."""
        ix, iy = grid.world_to_cell(xy)
        sub = state[max(0, iy - r): iy + r + 1, max(0, ix - r): ix + r + 1]
        return bool(sub.size and (sub == UNKNOWN).any())

    def _no_frontier(self, empty: bool):
        """A map that is still a cone in front of the robot has no reachable frontier yet;
        sweep on the spot and try again. The run is over only after a long run of replans that
        found no frontier at all - unreachable ones get many more chances."""
        self.no_frontier += 1
        self.path = None
        self.target = None
        self.target_score = 0.0
        if self.no_frontier >= (NO_FRONTIER_LIMIT if empty else STUCK_FRONTIER_LIMIT):
            self.done = True
        return None

    @staticmethod
    def _furthest_reachable(cost, start):
        """The finite-cost cell furthest from `start` by 8-connected steps, or None."""
        ny, nx = cost.shape
        sx, sy = start
        ok = np.isfinite(cost)
        if not (0 <= sx < nx and 0 <= sy < ny):
            return None
        if not ok[sy, sx]:
            free = np.argwhere(ok)
            if not len(free):
                return None
            k = int(np.argmin((free[:, 0] - sy) ** 2 + (free[:, 1] - sx) ** 2))
            sy, sx = int(free[k, 0]), int(free[k, 1])
        dist = np.full(cost.shape, -1, np.int32)
        dist[sy, sx] = 0
        frontier = [(sx, sy)]
        last = (sx, sy)
        while frontier:
            nxt = []
            for x, y in frontier:
                d = dist[y, x] + 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    a, b = x + dx, y + dy
                    if 0 <= a < nx and 0 <= b < ny and ok[b, a] and dist[b, a] < 0:
                        dist[b, a] = d
                        nxt.append((a, b))
            if nxt:
                last = nxt[-1]
            frontier = nxt
        return last if dist[last[1], last[0]] > 3 else None

    @staticmethod
    def _nearest_reachable(cost, cx, cy, r=4):
        ny, nx = cost.shape
        if 0 <= cy < ny and 0 <= cx < nx and np.isfinite(cost[cy, cx]):
            return cx, cy
        for rad in range(1, r + 1):
            y0, y1 = max(0, cy - rad), min(ny, cy + rad + 1)
            x0, x1 = max(0, cx - rad), min(nx, cx + rad + 1)
            sub = cost[y0:y1, x0:x1]
            ok = np.nonzero(np.isfinite(sub))
            if len(ok[0]):
                d = (ok[0] + y0 - cy) ** 2 + (ok[1] + x0 - cx) ** 2
                i = int(np.argmin(d))
                return int(ok[1][i] + x0), int(ok[0][i] + y0)
        return None

    @staticmethod
    def _to_world(path_cells, grid):
        if not path_cells:
            return None
        c = np.asarray(path_cells, dtype=float)
        return np.stack(grid.cell_to_world(c[:, 0], c[:, 1]), axis=1)

    # ------------------------------------------------------------------ control

    def command(self, t: float, pose):
        """(v, omega) for the true robot, computed entirely from the estimated pose and map."""
        pos = np.array(pose[:2])
        if self.last_pos is None:
            self.last_pos, self.last_yaw = pos, pose[2]
            self.last_progress_t = t
        # Progress is translation, full stop. Counting rotation as progress lets a robot that is
        # pinned against a wall - which still turns, because the follower keeps steering - reset
        # the timer for ever: one run had it wedged in the same spot for 64 s without the escape
        # ever firing. STUCK_TIME is long enough that a legitimate 180 deg turn in place (about
        # 3 s at the yaw rate limit) never trips it.
        if np.linalg.norm(pos - self.last_pos) > 0.15:
            self.last_pos, self.last_yaw = pos, pose[2]
            self.last_progress_t = t
        if t < self.escape_until:
            return self._escape(t)
        if t - self.last_progress_t > STUCK_TIME:
            self.events.append((t, "stuck", float(pos[0]), float(pos[1])))
            if self.target is not None:
                self.blacklist.append((self.target[0], self.target[1], t + BLACKLIST_TIME))
            self.target = None
            self.target_score = 0.0
            self.path = None
            self.escape_until = t + 4.0
            self.escape_t0 = t
            self.escape_dir = -getattr(self, "escape_dir", -1.0)   # alternate, so it does not retrace
            self.last_progress_t = t
            return self._escape(t)
        if self.path is None:
            return (0.0, 0.55) if self.no_frontier else (0.0, 0.0)   # sweep for new frontiers
        v, w, done = self.follow.command(pose, self.path)
        if done:
            self.path = None   # the target is kept; replan decides whether it is finished
        return v, w

    def _escape(self, t: float):
        """Rotate 90 deg in place, then back up 0.3 m."""
        dt = t - self.escape_t0
        if dt < 1.4:
            return 0.0, self.escape_dir * 1.1
        return -0.35, 0.0
