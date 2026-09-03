"""Feature-only spherical panorama stitching with auditable overlap checks.

This module deliberately has no bundle, pose, depth, heading, or tracking
dependencies. Spatially balanced RootSIFT correspondences propose image links;
mutual descriptor checks, MAGSAC, photometric agreement, plausible camera
rotation, and cycle consistency decide which links survive. Camera rotations
and focal length are recovered from those image links alone.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


class StitchError(RuntimeError):
    """Raised when no trustworthy feature panorama can be built."""


@dataclass(frozen=True)
class OverlapEdge:
    """One geometrically and photometrically verified image overlap."""

    first: int
    second: int
    matches: int
    inliers: int
    confidence: float
    gradient_correlation: float = 0.0
    intensity_correlation: float = 0.0
    overlap_fraction: float = 0.0
    rotation_deg: float = 0.0
    first_feature_coverage: float = 0.0
    second_feature_coverage: float = 0.0
    median_reprojection_error: float = 0.0

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.matches if self.matches else 0.0


@dataclass(frozen=True)
class OverlapGraph:
    """Connected components induced by verified pairwise image links."""

    keypoint_counts: tuple[int, ...]
    edges: tuple[OverlapEdge, ...]
    components: tuple[tuple[int, ...], ...]

    @property
    def selected(self) -> tuple[int, ...]:
        return self.components[0] if self.components else ()


@dataclass(frozen=True)
class StitchResult:
    """Panorama pixels plus the image evidence used to produce them."""

    image: np.ndarray
    alpha: np.ndarray
    graph: OverlapGraph
    focal_px: float
    tree_edges: tuple[tuple[int, int], ...]
    cycle_error_deg: float


@dataclass(frozen=True)
class _MatchedEdge:
    public: OverlapEdge
    homography_centered: np.ndarray
    weight: float
    relative_rotation: np.ndarray | None = None


@dataclass(frozen=True)
class _Analysis:
    graph: OverlapGraph
    edges: tuple[_MatchedEdge, ...]
    focal_px: float


@dataclass(frozen=True)
class _FeatureSet:
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray


def _components(node_count: int, edges: list[OverlapEdge]) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(node_count)]
    for edge in edges:
        adjacency[edge.first].add(edge.second)
        adjacency[edge.second].add(edge.first)

    unseen = set(range(node_count))
    found_components: list[tuple[int, ...]] = []
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
        found_components.append(tuple(sorted(found)))

    edge_confidence: dict[int, float] = {node: 0.0 for node in range(node_count)}
    for edge in edges:
        edge_confidence[edge.first] += edge.confidence
        edge_confidence[edge.second] += edge.confidence
    return tuple(
        sorted(
            found_components,
            key=lambda component: (
                -len(component),
                -sum(edge_confidence[node] for node in component),
                component[0],
            ),
        )
    )


def _correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    a = first[mask].astype(np.float64)
    b = second[mask].astype(np.float64)
    if len(a) < 100:
        return -1.0
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-9 else -1.0


def _center_transform(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]], float)


def _photometric_score(
    first: int,
    second: int,
    homography_centered: np.ndarray,
    images: list[np.ndarray],
    gray: list[np.ndarray],
    gradients: list[np.ndarray],
) -> tuple[float, float, float]:
    """Validate a feature homography directly against the overlapping pixels."""

    first_height, first_width = images[first].shape[:2]
    second_to_first = (
        np.linalg.inv(_center_transform(images[first]))
        @ np.linalg.inv(homography_centered)
        @ _center_transform(images[second])
    )
    if abs(second_to_first[2, 2]) < 1e-12:
        return -1.0, -1.0, 0.0
    second_to_first /= second_to_first[2, 2]
    source_mask = np.ones(images[second].shape[:2], np.uint8)
    warped_mask = cv2.warpPerspective(
        source_mask,
        second_to_first,
        (first_width, first_height),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    overlap = float(warped_mask.mean())
    if overlap < 0.01:
        return -1.0, -1.0, overlap

    warped_gray = cv2.warpPerspective(gray[second], second_to_first, (first_width, first_height))
    warped_gradient = cv2.warpPerspective(
        gradients[second], second_to_first, (first_width, first_height)
    )
    textured = warped_mask & ((gradients[first] > 8.0) | (warped_gradient > 8.0))
    return (
        _correlation(gradients[first], warped_gradient, textured),
        _correlation(gray[first], warped_gray, warped_mask),
        overlap,
    )


def _estimate_focal(edges: list[_MatchedEdge], image_size: tuple[int, int]) -> float:
    """Self-calibrate one focal length from rotation homographies."""

    height, width = image_size
    strongest = sorted(edges, key=lambda edge: -edge.weight)[:50]
    if not strongest:
        raise StitchError("no homographies available for focal estimation")

    def loss(focal: float) -> float:
        intrinsic = np.diag([focal, focal, 1.0])
        inverse = np.diag([1.0 / focal, 1.0 / focal, 1.0])
        errors = []
        for edge in strongest:
            candidate = inverse @ edge.homography_centered @ intrinsic
            singular = np.linalg.svd(candidate, compute_uv=False)
            singular /= np.exp(np.mean(np.log(np.maximum(singular, 1e-12))))
            errors.append(np.median(np.abs(np.log(np.maximum(singular, 1e-12)))))
        return float(np.median(errors))

    longest_side = max(height, width)
    solved = minimize_scalar(
        loss,
        bounds=(0.28 * longest_side, 1.6 * longest_side),
        method="bounded",
        options={"xatol": 0.1},
    )
    if not solved.success or not np.isfinite(solved.x):
        raise StitchError("image-only focal estimation failed")
    return float(solved.x)


def _nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def _extract_rootsift(image: np.ndarray) -> _FeatureSet:
    """Extract contrast-normalised, spatially balanced RootSIFT features."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    finder = cv2.SIFT_create(
        nfeatures=10_000,
        contrastThreshold=0.003,
        edgeThreshold=15,
        sigma=1.2,
    )
    keypoints, descriptors = finder.detectAndCompute(enhanced, None)
    if descriptors is None or not keypoints:
        return _FeatureSet((), np.empty((0, 128), dtype=np.float32))

    # Stop highly textured objects from consuming the complete feature budget.
    # Keeping the strongest points per cell gives walls, corners, and furniture
    # independent votes in the subsequent homography fit.
    height, width = gray.shape
    rows, columns, per_cell = 12, 8, 80
    cells: dict[tuple[int, int], list[int]] = {}
    for index, point in enumerate(keypoints):
        column = min(int(point.pt[0] * columns / width), columns - 1)
        row = min(int(point.pt[1] * rows / height), rows - 1)
        cells.setdefault((row, column), []).append(index)
    selected: list[int] = []
    for indices in cells.values():
        indices.sort(key=lambda index: keypoints[index].response, reverse=True)
        selected.extend(indices[:per_cell])
    selected.sort(key=lambda index: keypoints[index].response, reverse=True)
    selected = selected[:6000]

    balanced = np.asarray(descriptors[selected], dtype=np.float32)
    balanced /= np.maximum(balanced.sum(axis=1, keepdims=True), 1e-12)
    np.sqrt(balanced, out=balanced)
    return _FeatureSet(tuple(keypoints[index] for index in selected), balanced)


def _feature_coverage(points: np.ndarray, image_shape: tuple[int, int]) -> tuple[float, int]:
    """Measure whether inliers cover a 2-D image region instead of one tiny patch."""

    if len(points) < 3:
        return 0.0, 0
    height, width = image_shape
    hull = cv2.convexHull(points.astype(np.float32))
    coverage = float(cv2.contourArea(hull) / (height * width))
    columns = np.clip((points[:, 0] * 6 / width).astype(int), 0, 5)
    rows = np.clip((points[:, 1] * 8 / height).astype(int), 0, 7)
    occupied = len(set(zip(rows.tolist(), columns.tolist(), strict=True)))
    return coverage, occupied


def _symmetric_reprojection_error(
    first_points: np.ndarray, second_points: np.ndarray, homography: np.ndarray
) -> float:
    inverse = np.linalg.inv(homography)
    projected_second = cv2.perspectiveTransform(first_points[:, None, :], homography)[:, 0]
    projected_first = cv2.perspectiveTransform(second_points[:, None, :], inverse)[:, 0]
    errors = 0.5 * (
        np.linalg.norm(projected_second - second_points, axis=1)
        + np.linalg.norm(projected_first - first_points, axis=1)
    )
    return float(np.median(errors))


def _match_rootsift_pair(
    first: _FeatureSet,
    second: _FeatureSet,
    image_shape: tuple[int, int],
    *,
    ratio_threshold: float,
) -> tuple[np.ndarray, int, int, float, float, float] | None:
    """Return a mutually matched, spatially supported MAGSAC homography."""

    if len(first.keypoints) < 16 or len(second.keypoints) < 16:
        return None
    matcher = cv2.FlannBasedMatcher(
        {"algorithm": 1, "trees": 5},
        {"checks": 64},
    )
    forward = matcher.knnMatch(first.descriptors, second.descriptors, k=2)
    reverse = matcher.knnMatch(second.descriptors, first.descriptors, k=2)
    reverse_best = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse
        if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance
    }
    matches = [
        pair[0]
        for pair in forward
        if len(pair) == 2
        and pair[0].distance < ratio_threshold * pair[1].distance
        and reverse_best.get(pair[0].trainIdx) == pair[0].queryIdx
    ]
    if len(matches) < 16:
        return None

    first_points = np.float32([first.keypoints[item.queryIdx].pt for item in matches])
    second_points = np.float32([second.keypoints[item.trainIdx].pt for item in matches])
    homography, inlier_mask = cv2.findHomography(
        first_points,
        second_points,
        cv2.USAC_MAGSAC,
        3.0,
        maxIters=10_000,
        confidence=0.999,
    )
    if homography is None or inlier_mask is None:
        return None
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    if inlier_count < 14:
        return None

    first_inliers = first_points[inliers]
    second_inliers = second_points[inliers]
    first_coverage, first_cells = _feature_coverage(first_inliers, image_shape)
    second_coverage, second_cells = _feature_coverage(second_inliers, image_shape)
    if first_cells < 4 or second_cells < 4:
        return None
    if min(first_coverage, second_coverage) < 0.002:
        return None
    try:
        reprojection = _symmetric_reprojection_error(
            first_inliers, second_inliers, homography
        )
    except (cv2.error, np.linalg.LinAlgError):
        return None
    if not np.isfinite(reprojection) or reprojection > 3.0:
        return None
    return (
        homography,
        len(matches),
        inlier_count,
        first_coverage,
        second_coverage,
        reprojection,
    )


def _analyse_images(
    images: list[np.ndarray],
    *,
    confidence_threshold: float,
    match_confidence: float,
    minimum_gradient_correlation: float,
    pair_window: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> _Analysis:
    if len(images) < 2:
        raise ValueError("at least two images are required")
    if any(image is None or image.ndim != 3 for image in images):
        raise ValueError("every input must be a colour image")
    if len({image.shape[:2] for image in images}) != 1:
        raise ValueError("all images must have the same pixel dimensions")

    gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) for image in images]
    gradients = [
        cv2.magnitude(
            cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3),
        )
        for image in gray
    ]
    if progress is not None:
        progress(f"extracting features from {len(images)} images")
    features = [_extract_rootsift(image) for image in images]
    ratio_threshold = 1.0 - match_confidence

    candidates: list[_MatchedEdge] = []
    count = len(images)
    pairs = _capture_order_pairs(count, pair_window)
    if progress is not None:
        progress(f"matching {len(pairs)} candidate image pairs")
    report_every = max(1, len(pairs) // 20)
    for pair_index, (first, second) in enumerate(pairs, start=1):
        if progress is not None and (pair_index == 1 or pair_index % report_every == 0):
            progress(
                f"matched {pair_index}/{len(pairs)} pairs "
                f"({100 * pair_index // len(pairs)}%)"
            )
        matched = _match_rootsift_pair(
            features[first],
            features[second],
            images[first].shape[:2],
            ratio_threshold=ratio_threshold,
        )
        if matched is None:
            continue
        (
            homography_pixels,
            matches,
            inliers,
            first_coverage,
            second_coverage,
            reprojection,
        ) = matched
        confidence = inliers / (8.0 + 0.3 * matches)
        if confidence < confidence_threshold or inliers / matches < 0.30:
            continue
        homography_centered = (
            _center_transform(images[second])
            @ homography_pixels
            @ np.linalg.inv(_center_transform(images[first]))
        )
        try:
            gradient, intensity, overlap = _photometric_score(
                first,
                second,
                homography_centered,
                images,
                gray,
                gradients,
            )
        except (cv2.error, np.linalg.LinAlgError):
            continue
        if gradient < minimum_gradient_correlation or intensity < 0.35 or overlap < 0.08:
            continue
        public = OverlapEdge(
            first=first,
            second=second,
            matches=matches,
            inliers=inliers,
            confidence=confidence,
            gradient_correlation=gradient,
            intensity_correlation=intensity,
            overlap_fraction=overlap,
            first_feature_coverage=first_coverage,
            second_feature_coverage=second_coverage,
            median_reprojection_error=reprojection,
        )
        weight = (
            public.confidence
            * np.log1p(public.inliers)
            * gradient
            * intensity
            * np.sqrt(overlap)
            * np.sqrt(min(first_coverage, second_coverage))
            / (1.0 + reprojection)
        )
        candidates.append(
            _MatchedEdge(
                public=public,
                homography_centered=homography_centered,
                weight=float(weight),
            )
        )

    focal = _estimate_focal(candidates, images[0].shape[:2])
    intrinsic = np.diag([focal, focal, 1.0])
    inverse_intrinsic = np.diag([1.0 / focal, 1.0 / focal, 1.0])
    accepted: list[_MatchedEdge] = []
    for edge in candidates:
        relative = _nearest_rotation(
            inverse_intrinsic @ np.linalg.inv(edge.homography_centered) @ intrinsic
        )
        angle = float(
            np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        )
        if angle > 95.0:
            continue
        public = OverlapEdge(
            first=edge.public.first,
            second=edge.public.second,
            matches=edge.public.matches,
            inliers=edge.public.inliers,
            confidence=edge.public.confidence,
            gradient_correlation=edge.public.gradient_correlation,
            intensity_correlation=edge.public.intensity_correlation,
            overlap_fraction=edge.public.overlap_fraction,
            rotation_deg=angle,
            first_feature_coverage=edge.public.first_feature_coverage,
            second_feature_coverage=edge.public.second_feature_coverage,
            median_reprojection_error=edge.public.median_reprojection_error,
        )
        accepted.append(
            _MatchedEdge(
                public=public,
                homography_centered=edge.homography_centered,
                weight=edge.weight,
                relative_rotation=relative,
            )
        )

    public_edges = [edge.public for edge in accepted]
    graph = OverlapGraph(
        keypoint_counts=tuple(len(feature.keypoints) for feature in features),
        edges=tuple(public_edges),
        components=_components(count, public_edges),
    )
    if progress is not None:
        progress(
            f"accepted {len(accepted)} overlaps; largest component has "
            f"{len(graph.selected)} images"
        )
    return _Analysis(graph=graph, edges=tuple(accepted), focal_px=focal)


def _capture_order_pairs(count: int, window: int | None) -> list[tuple[int, int]]:
    """Candidate pairs near each other in capture order, including loop closure."""

    pairs = list(itertools.combinations(range(count), 2))
    if window is None:
        return pairs
    if window < 1:
        raise ValueError("pair window must be positive")
    return [
        (first, second)
        for first, second in pairs
        if min(second - first, count - (second - first)) <= window
    ]


def find_overlap_graph(
    images: list[np.ndarray],
    *,
    confidence_threshold: float = 0.6,
    match_confidence: float = 0.35,
    minimum_gradient_correlation: float = 0.2,
) -> OverlapGraph:
    """Find the RootSIFT graph accepted by every image-space check."""

    return _analyse_images(
        images,
        confidence_threshold=confidence_threshold,
        match_confidence=match_confidence,
        minimum_gradient_correlation=minimum_gradient_correlation,
        pair_window=None,
        progress=None,
    ).graph


def _maximum_spanning_rotations(
    component: tuple[int, ...], edges: tuple[_MatchedEdge, ...]
) -> tuple[dict[int, np.ndarray], tuple[tuple[int, int], ...], int]:
    nodes = set(component)
    adjacency: dict[int, list[tuple[int, np.ndarray, float]]] = {
        node: [] for node in component
    }
    for edge in edges:
        first, second = edge.public.first, edge.public.second
        if first not in nodes or second not in nodes or edge.relative_rotation is None:
            continue
        adjacency[first].append((second, edge.relative_rotation, edge.weight))
        adjacency[second].append((first, edge.relative_rotation.T, edge.weight))

    root = max(component, key=lambda node: sum(item[2] for item in adjacency[node]))
    rotations = {root: np.eye(3)}
    queue: list[tuple[float, int, int, int, np.ndarray]] = []
    serial = itertools.count()
    for node, relative, weight in adjacency[root]:
        heapq.heappush(queue, (-weight, next(serial), root, node, relative))
    tree: list[tuple[int, int]] = []
    while queue and len(rotations) < len(component):
        _, _, parent, node, relative = heapq.heappop(queue)
        if node in rotations:
            continue
        rotations[node] = _nearest_rotation(rotations[parent] @ relative)
        tree.append((parent, node))
        for neighbor, next_relative, weight in adjacency[node]:
            if neighbor not in rotations:
                heapq.heappush(
                    queue,
                    (-weight, next(serial), node, neighbor, next_relative),
                )
    if len(rotations) != len(component):
        raise StitchError("verified overlap component could not be spanned")
    return rotations, tuple(tree), root


def _refine_rotations(
    initial: dict[int, np.ndarray],
    root: int,
    edges: tuple[_MatchedEdge, ...],
) -> tuple[dict[int, np.ndarray], float]:
    """Reject inconsistent loops, then robustly refine image-only rotations."""

    nodes = sorted(initial)
    variable = [node for node in nodes if node != root]
    offsets = {node: 3 * index for index, node in enumerate(variable)}
    active = [
        edge
        for edge in edges
        if edge.public.first in initial
        and edge.public.second in initial
        and edge.relative_rotation is not None
    ]

    def edge_error(edge: _MatchedEdge, rotations: dict[int, np.ndarray]) -> np.ndarray:
        predicted = rotations[edge.public.first].T @ rotations[edge.public.second]
        return Rotation.from_matrix(edge.relative_rotation.T @ predicted).as_rotvec()

    initial_errors = np.array(
        [np.degrees(np.linalg.norm(edge_error(edge, initial))) for edge in active]
    )
    median = float(np.median(initial_errors)) if len(initial_errors) else 0.0
    cycle_limit = max(10.0, 3.0 * median)
    consistent = [
        edge for edge, error in zip(active, initial_errors, strict=True) if error <= cycle_limit
    ]
    if len(consistent) < len(nodes) - 1:
        consistent = active

    x0 = np.concatenate(
        [Rotation.from_matrix(initial[node]).as_rotvec() for node in variable]
    )

    def unpack(parameters: np.ndarray) -> dict[int, np.ndarray]:
        output = {root: initial[root]}
        for node in variable:
            start = offsets[node]
            output[node] = Rotation.from_rotvec(parameters[start : start + 3]).as_matrix()
        return output

    weight_scale = np.median([edge.weight for edge in consistent]) if consistent else 1.0

    jacobian = lil_matrix((3 * len(consistent), 3 * len(variable)), dtype=np.uint8)
    for edge_index, edge in enumerate(consistent):
        rows = slice(3 * edge_index, 3 * edge_index + 3)
        for node in (edge.public.first, edge.public.second):
            if node in offsets:
                start = offsets[node]
                jacobian[rows, start : start + 3] = 1

    def residuals(parameters: np.ndarray) -> np.ndarray:
        rotations = unpack(parameters)
        chunks = [
            np.sqrt(edge.weight / max(weight_scale, 1e-9)) * edge_error(edge, rotations)
            for edge in consistent
        ]
        return np.concatenate(chunks) if chunks else np.empty(0)

    solved = least_squares(
        residuals,
        x0,
        loss="soft_l1",
        f_scale=np.radians(3.0),
        jac_sparsity=jacobian.tocsr(),
        tr_solver="lsmr",
        max_nfev=80,
    )
    refined = unpack(solved.x) if solved.success else initial
    final_errors = [
        np.degrees(np.linalg.norm(edge_error(edge, refined))) for edge in consistent
    ]
    return refined, float(np.median(final_errors)) if final_errors else 0.0


def _warp_and_blend(
    images: list[np.ndarray],
    rotations: dict[int, np.ndarray],
    focal: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each pixel to one view, then feather only a narrow seam band.

    Broad multiband blending is wrong for translated viewpoints: it produces a
    translucent double of every nearby object.  Ownership instead selects the
    sharpest well-centered source at each output pixel.  A small Gaussian band
    hides exposure steps without averaging whole chairs, monitors, or sofas.
    """

    ids = sorted(rotations)
    corrected = cv2.detail.waveCorrect(
        [rotations[index].astype(np.float32) for index in ids],
        cv2.detail.WAVE_CORRECT_HORIZ,
    )
    height, width = images[0].shape[:2]
    intrinsic = np.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    warper = cv2.PyRotationWarper("spherical", focal)
    corners: list[tuple[int, int]] = []
    warped_images: list[np.ndarray] = []
    warped_masks: list[np.ndarray] = []
    for index, rotation in zip(ids, corrected, strict=True):
        roi = warper.warpRoi((width, height), intrinsic, rotation)
        _, _, roi_width, roi_height = roi
        if (
            roi_width <= 0
            or roi_height <= 0
            or roi_width * roi_height > 8_000_000
        ):
            raise StitchError(
                f"image {index} produced an implausible spherical warp ROI {roi}"
            )
        corner, warped = warper.warp(
            images[index], intrinsic, rotation, cv2.INTER_LINEAR, cv2.BORDER_REFLECT
        )
        _, mask = warper.warp(
            np.full((height, width), 255, np.uint8),
            intrinsic,
            rotation,
            cv2.INTER_NEAREST,
            cv2.BORDER_CONSTANT,
        )
        corners.append(corner)
        warped_images.append(warped)
        warped_masks.append(mask)

    # ExposureCompensator allocates over the union of every corner. Validate
    # that union before handing it the images; otherwise a corrupt/sentinel
    # corner can make OpenCV attempt a hundreds-of-gigabytes allocation.
    minimum_x = min(corner[0] for corner in corners)
    minimum_y = min(corner[1] for corner in corners)
    maximum_x = max(
        corner[0] + warped.shape[1]
        for corner, warped in zip(corners, warped_images, strict=True)
    )
    maximum_y = max(
        corner[1] + warped.shape[0]
        for corner, warped in zip(corners, warped_images, strict=True)
    )
    output_width = maximum_x - minimum_x
    output_height = maximum_y - minimum_y
    if (
        output_width <= 0
        or output_height <= 0
        or output_width * output_height > 30_000_000
    ):
        raise StitchError("feature camera graph produced an implausibly large canvas")

    compensator = cv2.detail.ExposureCompensator_createDefault(
        cv2.detail.ExposureCompensator_GAIN
    )
    compensator.feed(corners, warped_images, warped_masks)
    for source in range(len(warped_images)):
        compensator.apply(
            source,
            corners[source],
            warped_images[source],
            warped_masks[source],
        )

    sharpness = np.array(
        [
            cv2.Laplacian(cv2.cvtColor(images[index], cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
            for index in ids
        ],
        dtype=np.float64,
    )
    sharpness /= max(float(np.median(sharpness)), 1e-9)
    sharpness = np.clip(np.sqrt(sharpness), 0.70, 1.40)

    best = np.full((output_height, output_width), -np.inf, dtype=np.float32)
    owner = np.full((output_height, output_width), -1, dtype=np.int16)
    for source, ((x, y), mask, quality) in enumerate(
        zip(corners, warped_masks, sharpness, strict=True)
    ):
        valid = (mask > 0).astype(np.uint8)
        centered = cv2.distanceTransform(valid, cv2.DIST_L2, 3)
        centered /= max(float(centered.max()), 1e-6)
        score = centered * quality + source * 1e-7
        local_y, local_x = y - minimum_y, x - minimum_x
        local_height, local_width = valid.shape
        best_slice = best[local_y : local_y + local_height, local_x : local_x + local_width]
        owner_slice = owner[local_y : local_y + local_height, local_x : local_x + local_width]
        replace = (valid > 0) & (score > best_slice)
        best_slice[replace] = score[replace]
        owner_slice[replace] = source

    accumulator = np.zeros((output_height, output_width, 3), dtype=np.float64)
    weight_sum = np.zeros((output_height, output_width), dtype=np.float64)
    for source, ((x, y), warped, mask) in enumerate(
        zip(corners, warped_images, warped_masks, strict=True)
    ):
        local_y, local_x = y - minimum_y, x - minimum_x
        local_height, local_width = mask.shape
        owner_slice = owner[local_y : local_y + local_height, local_x : local_x + local_width]
        ownership = ((owner_slice == source) & (mask > 0)).astype(np.float32)
        weight = cv2.GaussianBlur(ownership, (0, 0), sigmaX=2.0)
        weight *= mask > 0
        accumulator[
            local_y : local_y + local_height, local_x : local_x + local_width
        ] += warped * weight[..., None]
        weight_sum[
            local_y : local_y + local_height, local_x : local_x + local_width
        ] += weight

    image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    hit = weight_sum > 1e-6
    image[hit] = np.clip(accumulator[hit] / weight_sum[hit, None], 0, 255).astype(np.uint8)
    alpha = np.where(hit, 255, 0).astype(np.uint8)
    return _crop_to_alpha(image, alpha)


def _crop_to_alpha(image: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = cv2.findNonZero(alpha)
    if points is None:
        raise StitchError("stitcher produced an empty panorama")
    x, y, width, height = cv2.boundingRect(points)
    return image[y : y + height, x : x + width], alpha[y : y + height, x : x + width]


def _external_alpha(image: np.ndarray) -> np.ndarray:
    """Make only border-connected black canvas regions transparent."""

    content = np.any(image > 2, axis=2).astype(np.uint8)
    content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels = cv2.connectedComponents(1 - content)
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    border_labels = border_labels[border_labels != 0]
    external = np.isin(labels, border_labels)
    return np.where(external, 0, 255).astype(np.uint8)


def wrap_equirectangular(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Close a full-width spherical stitch into a pole-safe 2:1 panorama.

    Border transparency from portrait source frames is filled only along image
    columns: ceiling pixels extend toward a common north pole and floor pixels
    extend toward a common south pole. The horizontal boundary is blended in a
    narrow band so a 360 viewer can cross it without a hard exposure step.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be BGR colour")
    if alpha.shape != image.shape[:2]:
        raise ValueError("alpha must match the image dimensions")
    if not np.any(alpha > 0):
        raise StitchError("cannot wrap an empty panorama")

    height, width = alpha.shape
    rows = np.arange(height)
    filled = np.empty_like(image)
    for column in range(width):
        valid_rows = rows[alpha[:, column] > 0]
        if len(valid_rows) == 0:
            # A neighboring longitude is a safer fallback than manufacturing
            # content from an unrelated part of the panorama.
            left = column - 1
            while left >= 0 and not np.any(alpha[:, left] > 0):
                left -= 1
            right = column + 1
            while right < width and not np.any(alpha[:, right] > 0):
                right += 1
            source = left if left >= 0 else right
            valid_rows = rows[alpha[:, source] > 0]
            for channel in range(3):
                filled[:, column, channel] = np.interp(
                    rows,
                    valid_rows,
                    image[valid_rows, source, channel],
                )
            continue
        for channel in range(3):
            filled[:, column, channel] = np.interp(
                rows,
                valid_rows,
                image[valid_rows, column, channel],
            )

    target_height = max(height, int(round(width / 2)))
    top_padding = (target_height - height) // 2
    bottom_padding = target_height - height - top_padding
    wrapped = np.empty((target_height, width, 3), dtype=np.float32)
    wrapped[top_padding : top_padding + height] = filled

    top_edge = cv2.GaussianBlur(filled[0:1].astype(np.float32), (0, 0), sigmaX=24)[0]
    bottom_edge = cv2.GaussianBlur(filled[-1:].astype(np.float32), (0, 0), sigmaX=24)[0]
    north_pole = np.median(top_edge, axis=0)
    south_pole = np.median(bottom_edge, axis=0)
    for row in range(top_padding):
        amount = (row + 1) / max(top_padding, 1)
        amount = amount * amount * (3.0 - 2.0 * amount)
        wrapped[row] = (1.0 - amount) * north_pole + amount * top_edge
    for offset in range(bottom_padding):
        amount = (offset + 1) / max(bottom_padding, 1)
        amount = amount * amount * (3.0 - 2.0 * amount)
        wrapped[-1 - offset] = (1.0 - amount) * bottom_edge + amount * south_pole

    # Longitude collapses at an equirectangular pole. Fade high-frequency edge
    # extensions into one shared colour before that collapse so missing source
    # coverage does not turn into vertical streaks in a sphere viewer.
    pole_band = max(1, int(round(target_height * 0.20)))
    for row in range(pole_band):
        amount = 1.0 - row / pole_band
        amount = amount * amount * (3.0 - 2.0 * amount)
        wrapped[row] = (1.0 - amount) * wrapped[row] + amount * north_pole
        wrapped[-1 - row] = (
            (1.0 - amount) * wrapped[-1 - row] + amount * south_pole
        )

    seam_width = max(1, min(48, width // 20))
    for distance in range(seam_width):
        amount = 1.0 - distance / seam_width
        amount = amount * amount * (3.0 - 2.0 * amount)
        shared = 0.5 * (wrapped[:, distance] + wrapped[:, -1 - distance])
        wrapped[:, distance] = (1.0 - amount) * wrapped[:, distance] + amount * shared
        wrapped[:, -1 - distance] = (
            (1.0 - amount) * wrapped[:, -1 - distance] + amount * shared
        )
    return np.clip(wrapped, 0, 255).astype(np.uint8)


def stitch_largest_component(
    images: list[np.ndarray],
    *,
    confidence_threshold: float = 0.6,
    minimum_gradient_correlation: float = 0.2,
    pair_window: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> StitchResult:
    """Recover and render the largest verified image-only camera graph."""

    analysis = _analyse_images(
        images,
        confidence_threshold=confidence_threshold,
        match_confidence=0.35,
        minimum_gradient_correlation=minimum_gradient_correlation,
        pair_window=pair_window,
        progress=progress,
    )
    selected = analysis.graph.selected
    if len(selected) < 2:
        raise StitchError("no pair of images has trustworthy feature overlap")
    initial, tree, root = _maximum_spanning_rotations(selected, analysis.edges)
    if progress is not None:
        progress("solving and refining camera rotations")
    rotations, cycle_error = _refine_rotations(initial, root, analysis.edges)
    if progress is not None:
        progress("warping and exposure-compensating panorama")
    panorama, alpha = _warp_and_blend(images, rotations, analysis.focal_px)
    return StitchResult(
        image=panorama,
        alpha=alpha,
        graph=analysis.graph,
        focal_px=analysis.focal_px,
        tree_edges=tree,
        cycle_error_deg=cycle_error,
    )
