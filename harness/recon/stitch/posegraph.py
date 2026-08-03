"""House-level pose graph (PLAN.md §6.2, §6.3).

Nodes are one **SE(2) + scale** variable per room — 3 DoF plus 1, because
gravity has already fixed roll, pitch and height. Edges are doorway constraints
with covariance from the ICP residuals. Priors are each room's internal scale
constraint from §6.2. Solved with Levenberg-Marquardt, Huber-robustified so one
bad doorway tag degrades the answer instead of destroying it.

**Why scale is a graph variable and not a per-room multiplier.** Applying each
room's scale independently creates discontinuities at doorways: two rooms that
share a physical door disagree about how wide it is, and the error piles up in
whichever room was scanned last. Making scale a variable with the doorway edges
as constraints spreads the residual across the property instead. That is the
difference the plan draws, and `test_scale_error_is_distributed_not_piled_up`
is what holds us to it.

**Solver.** The plan specifies GTSAM. GTSAM's current wheels pin `numpy<2`,
which conflicts with the rest of this stack (torch, DA3 and MapAnything all want
numpy 2.x) — installing it downgraded numpy and broke SciPy outright. Since the
plan itself lists Ceres as an acceptable alternative, the requirement is really
"LM with a Huber kernel", not "GTSAM specifically". This module implements the
factor graph directly on `scipy.optimize.least_squares` (Trust Region Reflective
with `loss="huber"`), which gives exactly that with no dependency conflict. The
graph is small — a 6-room house is 24 variables — so there is nothing to gain
from a specialised backend. Swapping in GTSAM later means replacing `optimise()`
and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recon.schemas import EdgeSource, Floor, HouseGraph, RoomEdge, SE2Transform

#: Huber transition point, in units of sigma.
HUBER_DELTA = 1.5


@dataclass
class RoomNode:
    room_id: str
    yaw_rad: float = 0.0
    tx: float = 0.0
    tz: float = 0.0
    scale: float = 1.0
    #: Per-room scale estimate from §6.2 and its sigma. Enters as a prior.
    scale_prior: float | None = None
    scale_sigma: float = 0.03
    #: Physical width of each doorway in this room's own (unscaled) frame,
    #: keyed by doorway id. Shared doors give the graph its scale coupling.
    doorway_widths: dict[str, float] = field(default_factory=dict)


@dataclass
class PoseGraphResult:
    nodes: dict[str, RoomNode]
    #: Per-edge final residual norm, keyed by (from_room, to_room, dw_a, dw_b).
    edge_residuals: dict[tuple[str, str, str, str], float]
    iterations: int
    cost: float
    converged: bool
    #: Edges whose Huber weight fell below this are effectively outliers.
    suspected_outliers: list[tuple[str, str]] = field(default_factory=list)
    #: True when a materially different layout fits the constraints just as
    #: well. The pose is then a coin flip and the user must re-tag; see
    #: `HousePoseGraph.optimise`.
    ambiguous: bool = False
    ambiguity_note: str = ""

    def transform(self, room_id: str) -> np.ndarray:
        n = self.nodes[room_id]
        m = SE2Transform(yaw_rad=n.yaw_rad, tx=n.tx, tz=n.tz).matrix()
        m[:3, :3] *= n.scale
        return m

    def apply(self, room_id: str, points: np.ndarray) -> np.ndarray:
        """Take points from a room's own frame into the house frame."""
        n = self.nodes[room_id]
        c, s = np.cos(n.yaw_rad), np.sin(n.yaw_rad)
        r = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return (np.asarray(points, dtype=np.float64) * n.scale) @ r.T + np.array(
            [n.tx, 0.0, n.tz]
        )


class HousePoseGraph:
    """Builds and solves the house graph."""

    def __init__(self, *, solve_scale: bool = True) -> None:
        self.nodes: dict[str, RoomNode] = {}
        self.edges: list[RoomEdge] = []
        self.solve_scale = solve_scale

    def add_room(
        self,
        room_id: str,
        *,
        scale_prior: float | None = None,
        scale_sigma: float = 0.03,
        doorway_widths: dict[str, float] | None = None,
    ) -> RoomNode:
        node = RoomNode(
            room_id=room_id,
            scale=scale_prior if scale_prior else 1.0,
            scale_prior=scale_prior,
            scale_sigma=scale_sigma,
            doorway_widths=doorway_widths or {},
        )
        self.nodes[room_id] = node
        return node

    def add_edge(self, edge: RoomEdge) -> None:
        for r in (edge.from_room, edge.to_room):
            if r not in self.nodes:
                raise KeyError(f"edge references unknown room {r!r}")
        self.edges.append(edge)

    # -- solving ------------------------------------------------------------

    def initial_guess(
        self, anchor: str, *, edge_order: list[int] | None = None
    ) -> dict[str, tuple[float, float, float]]:
        """Propagate poses outward from the anchor along a spanning tree.

        A pose graph **must not** be started from zeros. With every room stacked
        at the origin, a mis-tagged doorway claiming two distant rooms are
        adjacent is *closer* to the initial state than the truth is, and the
        optimiser converges to it — one bad tag then drags a room ten metres,
        which is the failure PLAN §7 says Huber should prevent. Huber bounds an
        outlier's *influence*; it cannot rescue a solve that started in the
        wrong basin.

        `edge_order` selects *and* orders the edges the tree may use. Both halves
        matter, and exclusion matters more: a mis-tagged doorway is typically a
        **short cut** claiming two distant rooms are adjacent, so it sits one hop
        from the anchor and breadth-first will always take it no matter how the
        edges are ordered. The only way to seed a tree that avoids it is to leave
        it out. `optimise` therefore tries leave-one-out trees and lets the
        robust cost pick between them; see its docstring.
        """
        from collections import deque

        order = edge_order if edge_order is not None else list(range(len(self.edges)))
        adj: dict[str, list[tuple[str, RoomEdge, bool]]] = {r: [] for r in self.nodes}
        for i in order:
            e = self.edges[i]
            adj[e.from_room].append((e.to_room, e, True))
            adj[e.to_room].append((e.from_room, e, False))

        guess = {anchor: (0.0, 0.0, 0.0)}
        queue = deque([anchor])
        while queue:
            cur = queue.popleft()
            ya, txa, tza = guess[cur]
            for nxt, e, forward in adj[cur]:
                if nxt in guess:
                    continue
                t = e.transform
                if forward:
                    # cur == from_room: compose the measured A<-B onto A's pose.
                    guess[nxt] = _compose(ya, txa, tza, t.yaw_rad, t.tx, t.tz)
                else:
                    guess[nxt] = _compose(ya, txa, tza, *_invert(t.yaw_rad, t.tx, t.tz))
                queue.append(nxt)

        # Rooms with no path to the anchor keep whatever they were given.
        for r, n in self.nodes.items():
            guess.setdefault(r, (n.yaw_rad, n.tx, n.tz))
        return guess

    def optimise(
        self,
        *,
        anchor: str | None = None,
        max_iterations: int = 200,
        restarts: int = 8,
        seed: int = 0,
    ) -> PoseGraphResult:
        """Levenberg-Marquardt with a Huber kernel over all constraints.

        Multi-start, because the spanning tree that seeds the solve can itself
        be poisoned: a mis-tagged doorway claims two distant rooms are adjacent,
        which is exactly the short cut a breadth-first tree prefers. Seeding
        from that edge puts the solve in a basin where the *correct* edges look
        like the outliers.

        The arbiter is the robust cost. Seeded correctly, one bad edge is
        Huber-capped and everything else fits; seeded from the bad edge, every
        good edge is violated instead and the total cost is far higher. So the
        fix is to try several trees and keep the cheapest solution — a small
        RANSAC over initialisations, which a house-sized graph can afford.
        """
        from scipy.optimize import least_squares

        if not self.nodes:
            raise ValueError("no rooms in the graph")
        ids = list(self.nodes)
        anchor = anchor or ids[0]
        if anchor not in self.nodes:
            raise KeyError(f"anchor room {anchor!r} is not in the graph")

        # Variable layout: pose (3) for every room except the anchor, whose pose
        # is fixed to the origin; scale (1) for *every* room including the
        # anchor. Fixing the anchor's scale too would pin the whole house to
        # whichever room happened to be scanned first — the error pile-up §6.2
        # exists to prevent.
        pose_index: dict[str, int] = {}
        scale_index: dict[str, int] = {}
        cursor = 0
        for r in ids:
            if r != anchor:
                pose_index[r] = cursor
                cursor += 3
        if self.solve_scale:
            for r in ids:
                scale_index[r] = cursor
                cursor += 1

        def pack(guess: dict[str, tuple[float, float, float]]) -> np.ndarray:
            x = np.zeros(cursor)
            for r in ids:
                if r in pose_index:
                    x[pose_index[r]:pose_index[r] + 3] = guess[r]
                if r in scale_index:
                    x[scale_index[r]] = self.nodes[r].scale
            return x

        def unpack(x: np.ndarray, room: str) -> tuple[float, float, float, float]:
            if room in pose_index:
                i = pose_index[room]
                y, tx, tz = float(x[i]), float(x[i + 1]), float(x[i + 2])
            else:
                y, tx, tz = 0.0, 0.0, 0.0
            s = float(x[scale_index[room]]) if room in scale_index else 1.0
            return y, tx, tz, s

        def residuals(x: np.ndarray) -> np.ndarray:
            out: list[float] = []
            for e in self.edges:
                ya, txa, tza, sa = unpack(x, e.from_room)
                yb, txb, tzb, sb = unpack(x, e.to_room)
                sigma = _edge_sigmas(e)

                # Predicted relative transform A<-B versus the measured one.
                rel_yaw, rel_tx, rel_tz = _relative(ya, txa, tza, yb, txb, tzb)
                out.append(_wrap(rel_yaw - e.transform.yaw_rad) / sigma[0])
                out.append((rel_tx - e.transform.tx) / sigma[1])
                out.append((rel_tz - e.transform.tz) / sigma[2])

                if self.solve_scale:
                    # The shared door must come out the same width from both
                    # sides. This is the constraint that couples scale across
                    # the house.
                    wa = self.nodes[e.from_room].doorway_widths.get(e.via_doorway[0])
                    wb = self.nodes[e.to_room].doorway_widths.get(e.via_doorway[1])
                    if wa and wb:
                        out.append((sa * wa - sb * wb) / 0.01)

            if self.solve_scale:
                for r in ids:
                    n = self.nodes[r]
                    if n.scale_prior is None:
                        continue
                    _, _, _, s = unpack(x, r)
                    out.append((s - n.scale_prior) / max(n.scale_sigma, 1e-6))
            return np.asarray(out, dtype=np.float64) if out else np.zeros(1)

        rng = np.random.default_rng(seed)
        n_edges = len(self.edges)
        full = list(range(n_edges))
        # Candidate spanning trees: the full edge set, then each leave-one-out
        # set (which is what actually escapes a poisoned short-cut edge), then a
        # few random orderings as a cheap hedge for larger graphs.
        orders: list[list[int]] = [full]
        if n_edges > 1:
            orders.extend([i for i in full if i != drop] for drop in full)
        for _ in range(max(restarts - 1, 0)):
            perm = list(full)
            rng.shuffle(perm)
            orders.append(perm)

        best = None
        solutions: list[tuple[float, np.ndarray]] = []
        seen: set[tuple] = set()
        for order in orders:
            x0 = pack(self.initial_guess(anchor, edge_order=order))
            key = tuple(np.round(x0, 6))
            if key in seen:
                continue  # this tree is one we have already tried
            seen.add(key)
            sol = least_squares(
                residuals, x0, loss="huber", f_scale=HUBER_DELTA, method="trf",
                max_nfev=max_iterations * max(len(x0), 1) + 100,
            )
            solutions.append((float(sol.cost), sol.x.copy()))
            if best is None or sol.cost < best.cost:
                best = sol
        assert best is not None
        sol = best

        ambiguous, note = _detect_ambiguity(solutions, float(sol.cost), sol.x, pose_index)

        nodes = {}
        for r in ids:
            y, tx, tz, s = unpack(sol.x, r)
            src = self.nodes[r]
            nodes[r] = RoomNode(
                room_id=r, yaw_rad=y, tx=tx, tz=tz, scale=s,
                scale_prior=src.scale_prior, scale_sigma=src.scale_sigma,
                doorway_widths=dict(src.doorway_widths),
            )

        edge_res, outliers = self._score_edges(nodes)
        return PoseGraphResult(
            nodes=nodes,
            edge_residuals=edge_res,
            iterations=int(sol.nfev),
            cost=float(sol.cost),
            converged=bool(sol.success),
            suspected_outliers=outliers,
            ambiguous=ambiguous,
            ambiguity_note=note,
        )

    def _score_edges(
        self, nodes: dict[str, RoomNode]
    ) -> tuple[dict[tuple[str, str, str, str], float], list[tuple[str, str]]]:
        res: dict[tuple[str, str, str, str], float] = {}
        outliers: list[tuple[str, str]] = []
        for e in self.edges:
            a, b = nodes[e.from_room], nodes[e.to_room]
            rel_yaw, rel_tx, rel_tz = _relative(
                a.yaw_rad, a.tx, a.tz, b.yaw_rad, b.tx, b.tz
            )
            sigma = _edge_sigmas(e)
            r = np.array([
                _wrap(rel_yaw - e.transform.yaw_rad) / sigma[0],
                (rel_tx - e.transform.tx) / sigma[1],
                (rel_tz - e.transform.tz) / sigma[2],
            ])
            norm = float(np.linalg.norm(r))
            res[(e.from_room, e.to_room, *e.via_doorway)] = norm
            if norm > 3.0 * HUBER_DELTA:
                outliers.append((e.from_room, e.to_room))
        return res, outliers

    # -- export -------------------------------------------------------------

    def to_house_graph(self, property_id: str, level: int = 0) -> HouseGraph:
        return HouseGraph(
            property_id=property_id,
            floors=[Floor(level=level, rooms=list(self.nodes), edges=list(self.edges))],
        )


#: Two solutions within this relative cost of each other are treated as tied.
AMBIGUITY_COST_TOLERANCE = 0.02
#: ...and count as materially different if any room moves by more than this.
AMBIGUITY_POSITION_M = 0.30


def _detect_ambiguity(
    solutions: list[tuple[float, np.ndarray]],
    best_cost: float,
    best_x: np.ndarray,
    pose_index: dict[str, int],
) -> tuple[bool, str]:
    """Flag the case where a different layout fits the data equally well.

    A chain of rooms plus one mis-tagged short cut is genuinely undecidable when
    every edge carries the same covariance: dropping the bad edge and dropping
    one chain edge produce solutions with *identical* robust cost, because each
    explanation violates exactly one constraint by the same amount. No amount of
    Huber tuning resolves that — the information is not there.

    The plan's real defence is upstream: §6.3 step 4 rejects a bad tag on ICP
    RMSE before it ever reaches the graph, and a mis-tag that survives normally
    arrives with a loose covariance that breaks the tie. When neither applies,
    the honest output is "I cannot tell", so the user can be asked to re-tag —
    which is exactly what §6.3 step 4 says to do.
    """
    if len(solutions) < 2 or not pose_index:
        return False, ""
    scale = max(abs(best_cost), 1e-9)
    for cost, x in solutions:
        if cost > best_cost * (1.0 + AMBIGUITY_COST_TOLERANCE) + 1e-9:
            continue
        if abs(cost - best_cost) / scale > AMBIGUITY_COST_TOLERANCE:
            continue
        for room, i in pose_index.items():
            moved = float(
                np.hypot(x[i + 1] - best_x[i + 1], x[i + 2] - best_x[i + 2])
            )
            if moved > AMBIGUITY_POSITION_M:
                return True, (
                    f"two layouts fit equally well (cost {best_cost:.1f} vs {cost:.1f}); "
                    f"{room} differs by {moved:.2f} m. Ask the user to re-tag the "
                    "doorways connecting it."
                )
    return False, ""


def _compose(
    ya: float, txa: float, tza: float, rel_yaw: float, rel_tx: float, rel_tz: float
) -> tuple[float, float, float]:
    """Apply a relative A<-B transform to A's pose, giving B's pose.

    The exact inverse of `_relative`, which `test_relative_transform_is_self_
    consistent` and `test_compose_inverts_relative` pin down — a sign slip here
    silently mirrors the initial guess and the solve never recovers.
    """
    c, s = np.cos(ya), np.sin(ya)
    return (
        float(ya + rel_yaw),
        float(txa + c * rel_tx + s * rel_tz),
        float(tza - s * rel_tx + c * rel_tz),
    )


def _invert(yaw: float, tx: float, tz: float) -> tuple[float, float, float]:
    """Inverse of an SE(2) transform."""
    c, s = np.cos(yaw), np.sin(yaw)
    return float(-yaw), float(-(c * tx - s * tz)), float(-(s * tx + c * tz))


def _relative(
    ya: float, txa: float, tza: float, yb: float, txb: float, tzb: float
) -> tuple[float, float, float]:
    """Relative transform A<-B for two SE(2) poses expressed in the house frame."""
    d_yaw = yb - ya
    c, s = np.cos(ya), np.sin(ya)
    # Inverse of A's rotation applied to the translation difference. SE2Transform
    # rotates about +Y as [[c,0,s],[0,1,0],[-s,0,c]], so the inverse in the (x, z)
    # plane is [[c,-s],[s,c]].
    dx, dz = txb - txa, tzb - tza
    rel_tx = c * dx - s * dz
    rel_tz = s * dx + c * dz
    return float(d_yaw), float(rel_tx), float(rel_tz)


def _edge_sigmas(e: RoomEdge) -> np.ndarray:
    cov = np.asarray(e.covariance, dtype=np.float64)
    sig = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    # Floor the sigmas: a near-zero covariance from a suspiciously good ICP fit
    # would otherwise let one edge dominate the whole graph.
    return np.maximum(sig, [np.radians(0.1), 0.002, 0.002])


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def build_from_alignments(
    rooms: dict[str, dict],
    alignments: list[tuple[str, str, str, str, object]],
    *,
    solve_scale: bool = True,
) -> HousePoseGraph:
    """Convenience builder.

    `rooms` maps room_id -> {"scale": float, "scale_sigma": float,
    "doorway_widths": {id: width}}. `alignments` is a list of
    (room_a, room_b, doorway_a, doorway_b, Alignment).
    """
    g = HousePoseGraph(solve_scale=solve_scale)
    for rid, info in rooms.items():
        g.add_room(
            rid,
            scale_prior=info.get("scale"),
            scale_sigma=info.get("scale_sigma", 0.03),
            doorway_widths=info.get("doorway_widths", {}),
        )
    for a, b, dw_a, dw_b, al in alignments:
        if not getattr(al, "accepted", True):
            continue
        g.add_edge(al.to_edge(a, b, dw_a, dw_b, EdgeSource.USER_TAGGED))
    return g
