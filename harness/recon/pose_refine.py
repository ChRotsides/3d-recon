"""Depth-backed feature pose refinement.

The phone poses are good enough to initialise reconstruction, but small ARCore
drift makes repeated observations of a wall land on parallel, separated
surfaces.  This module estimates camera-to-camera transforms from independent
RGB feature matches whose endpoints have metric depth, then optimises a small
correction around the captured poses.

All transforms use the repository convention: ``world<-camera`` and a camera
that looks down ``-Z``.  A :class:`PoseEdge` stores ``camera_i<-camera_j``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseEdge:
    """A robust relative-pose measurement and unused validation matches."""

    i: int
    j: int
    relative_ij: np.ndarray
    train_count: int
    inlier_count: int
    train_rmse_m: float
    heldout_i: np.ndarray
    heldout_j: np.ndarray


@dataclass(frozen=True)
class RefinementResult:
    poses: list[np.ndarray]
    accepted: bool
    component: tuple[int, ...]
    heldout_before_m: float
    heldout_after_m: float
    train_before_m: float
    train_after_m: float
    optimizer_cost: float
    optimizer_success: bool


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to row-vector points."""

    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares rigid transform ``target<-source`` (Kabsch)."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if len(source) < 3:
        raise ValueError("at least three points are required")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_mean - rotation @ source_mean
    return transform


def _well_spread(points: np.ndarray) -> bool:
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return len(singular) == 3 and singular[1] > 0.01


def estimate_pose_edge(
    i: int,
    j: int,
    points_i: np.ndarray,
    points_j: np.ndarray,
    *,
    seed: int = 0,
    threshold_m: float = 0.10,
    min_inliers: int = 8,
    min_inlier_ratio: float = 0.35,
    heldout_fraction: float = 0.20,
    iterations: int = 400,
) -> PoseEdge | None:
    """Estimate ``camera_i<-camera_j`` while reserving validation matches.

    The held-out points never participate in RANSAC or the final Kabsch fit.
    This prevents a pose graph from declaring success merely because it fits
    the same correspondences from which its relative poses were estimated.
    """

    points_i = np.asarray(points_i, dtype=np.float64)
    points_j = np.asarray(points_j, dtype=np.float64)
    if points_i.shape != points_j.shape or points_i.ndim != 2 or points_i.shape[1] != 3:
        raise ValueError("point arrays must have equal shape (N, 3)")
    if len(points_i) < min_inliers + 3:
        return None

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points_i))
    heldout_count = max(3, int(round(len(order) * heldout_fraction)))
    heldout_count = min(heldout_count, len(order) - min_inliers)
    heldout = order[:heldout_count]
    train = order[heldout_count:]
    pi, pj = points_i[train], points_j[train]

    best = np.zeros(len(train), dtype=bool)
    best_median = np.inf
    for _ in range(iterations):
        sample = rng.choice(len(train), 3, replace=False)
        if not _well_spread(pj[sample]):
            continue
        candidate = rigid_transform(pj[sample], pi[sample])
        residual = np.linalg.norm(transform_points(candidate, pj) - pi, axis=1)
        inliers = residual < threshold_m
        count = int(inliers.sum())
        median = float(np.median(residual[inliers])) if count else np.inf
        if count > int(best.sum()) or (count == int(best.sum()) and median < best_median):
            best, best_median = inliers, median

    if int(best.sum()) < min_inliers or float(best.mean()) < min_inlier_ratio:
        return None
    transform = rigid_transform(pj[best], pi[best])
    # One refinement pass catches points that fell just outside the first model.
    residual = np.linalg.norm(transform_points(transform, pj) - pi, axis=1)
    inliers = residual < threshold_m
    if int(inliers.sum()) < min_inliers or float(inliers.mean()) < min_inlier_ratio:
        return None
    transform = rigid_transform(pj[inliers], pi[inliers])
    fitted = np.linalg.norm(transform_points(transform, pj[inliers]) - pi[inliers], axis=1)

    return PoseEdge(
        i=i,
        j=j,
        relative_ij=transform,
        train_count=len(train),
        inlier_count=int(inliers.sum()),
        train_rmse_m=float(np.sqrt(np.mean(fitted**2))),
        heldout_i=points_i[heldout],
        heldout_j=points_j[heldout],
    )


def relative_pose_error(measured: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """Translation metres and rotation degrees between two transforms."""

    delta = np.linalg.inv(measured) @ expected
    translation = float(np.linalg.norm(delta[:3, 3]))
    rotation = float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())))
    return translation, rotation


def _connected_components(node_count: int, edges: list[PoseEdge]) -> list[tuple[int, ...]]:
    adjacency = [set() for _ in range(node_count)]
    for edge in edges:
        adjacency[edge.i].add(edge.j)
        adjacency[edge.j].add(edge.i)
    components: list[tuple[int, ...]] = []
    unseen = set(range(node_count))
    while unseen:
        root = min(unseen)
        stack = [root]
        found: set[int] = set()
        while stack:
            node = stack.pop()
            if node in found:
                continue
            found.add(node)
            stack.extend(adjacency[node] - found)
        unseen -= found
        components.append(tuple(sorted(found)))
    return sorted(components, key=lambda c: (-len(c), c[0]))


def correspondence_error(edges: list[PoseEdge], poses: list[np.ndarray]) -> float:
    """Median held-out world-space disagreement in metres."""

    values: list[np.ndarray] = []
    for edge in edges:
        wi = transform_points(poses[edge.i], edge.heldout_i)
        wj = transform_points(poses[edge.j], edge.heldout_j)
        values.append(np.linalg.norm(wi - wj, axis=1))
    return float(np.median(np.concatenate(values))) if values else np.inf


def _edge_transform_error(edges: list[PoseEdge], poses: list[np.ndarray]) -> float:
    values = []
    for edge in edges:
        relative = np.linalg.inv(poses[edge.i]) @ poses[edge.j]
        values.append(relative_pose_error(edge.relative_ij, relative)[0])
    return float(np.median(values)) if values else np.inf


def _correction_matrix(parameters: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    transform[:3, 3] = parameters[3:]
    return transform


def refine_pose_graph(
    initial_poses: list[np.ndarray],
    edges: list[PoseEdge],
    *,
    translation_scale_m: float = 0.05,
    rotation_scale_deg: float = 3.0,
    prior_translation_m: float = 0.35,
    prior_rotation_deg: float = 15.0,
    prior_weight: float = 0.50,
    minimum_improvement: float = 0.03,
    max_nfev: int = 300,
) -> RefinementResult:
    """Optimise corrections for independently anchored feature components.

    The first frame in each component fixes its gauge against the captured
    ARCore world. Components need at least three frames; isolated frames and
    weak two-frame islands are returned bit-for-bit unchanged. The candidate
    is accepted only when its median held-out correspondence error improves by
    ``minimum_improvement``.
    """

    original = [np.asarray(p, dtype=np.float64).copy() for p in initial_poses]
    components = [
        group for group in _connected_components(len(original), edges) if len(group) >= 3
    ]
    component = tuple(sorted(node for group in components for node in group))
    group_by_node = {
        node: group_index for group_index, group in enumerate(components) for node in group
    }
    active_edges = [
        edge
        for edge in edges
        if edge.i in group_by_node and group_by_node[edge.i] == group_by_node.get(edge.j)
    ]
    before = correspondence_error(active_edges, original)
    train_before = _edge_transform_error(active_edges, original)
    if len(component) < 3 or len(active_edges) < 2 or not np.isfinite(before):
        return RefinementResult(
            original, False, component, before, before, train_before, train_before, np.inf, False
        )

    anchors = {group[0] for group in components}
    variable_nodes = [node for node in component if node not in anchors]
    offsets = {node: 6 * index for index, node in enumerate(variable_nodes)}

    def unpack(parameters: np.ndarray) -> list[np.ndarray]:
        poses = [pose.copy() for pose in original]
        for node in variable_nodes:
            start = offsets[node]
            # Right multiplication makes this a correction around the camera,
            # rather than rotating its position about the arbitrary world
            # origin. Its translation norm is therefore the actual position
            # correction and can be bounded meaningfully.
            poses[node] = original[node] @ _correction_matrix(parameters[start : start + 6])
        return poses

    rotation_scale = np.radians(rotation_scale_deg)
    prior_rotation = np.radians(prior_rotation_deg)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        poses = unpack(parameters)
        chunks: list[np.ndarray] = []
        for edge in active_edges:
            predicted = np.linalg.inv(poses[edge.i]) @ poses[edge.j]
            delta = np.linalg.inv(edge.relative_ij) @ predicted
            confidence = np.sqrt(min(edge.inlier_count, 100) / 30.0)
            chunks.append(confidence * delta[:3, 3] / translation_scale_m)
            chunks.append(
                confidence
                * Rotation.from_matrix(delta[:3, :3]).as_rotvec()
                / rotation_scale
            )
        for node in variable_nodes:
            start = offsets[node]
            correction = parameters[start : start + 6]
            chunks.append(prior_weight * correction[:3] / prior_rotation)
            chunks.append(prior_weight * correction[3:] / prior_translation_m)
        return np.concatenate(chunks)

    x0 = np.zeros(6 * len(variable_nodes), dtype=np.float64)
    # Per-axis bounds imply a 13.9 degree / 52 cm maximum vector norm. ARCore
    # drift beyond that is not safe to infer from room textures alone.
    rotation_bound = np.radians(8.0)
    lower = np.tile([-rotation_bound] * 3 + [-0.30] * 3, len(variable_nodes))
    upper = -lower
    solved = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max_nfev,
    )
    candidate = unpack(solved.x)
    after = correspondence_error(active_edges, candidate)
    train_after = _edge_transform_error(active_edges, candidate)
    accepted = bool(
        solved.success
        and np.isfinite(after)
        and after <= before * (1.0 - minimum_improvement)
        and train_after < train_before
    )
    return RefinementResult(
        candidate if accepted else original,
        accepted,
        component,
        before,
        after,
        train_before,
        train_after,
        float(solved.cost),
        bool(solved.success),
    )
