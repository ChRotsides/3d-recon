"""Persistent, RGB-only incremental overlap graph and capture guidance.

The implementation deliberately reuses the exact feature, pair verification,
focal calibration, and rotation refinement used by the final stitcher.  Only
candidate scheduling and persistence differ: an arriving image is compared to
capture-order neighbours and deterministic loop-closure candidates, and every
expensive result is keyed by content plus configuration checksums.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from recon.image_stitch import (
    OverlapEdge,
    StitchError,
    _FeatureSet,
    _MatchedEdge,
    _center_transform,
    _components,
    _estimate_focal,
    _extract_rootsift,
    _match_rootsift_pair,
    _maximum_spanning_rotations,
    _nearest_rotation,
    _photometric_score,
    _refine_rotations,
)
from recon.live_rgb.pipeline import _prepare_stitch_images

FEATURE_CONFIG = {
    "schema": "rootsift-spatial-v1",
    # Guidance only needs stable overlap, not final-output resolution. Keeping
    # this small makes the next direction arrive while the user is still
    # holding the phone instead of several captures later.
    "max_long_edge": 960,
}
MATCH_CONFIG = {
    "schema": "rgb-neighbour-match-v3",
    "pair_window": 4,
    # The reference capture contains repeated sweeps. Frames 10--25 captures
    # apart are proposed as loop closures; they are still subject to every
    # strict geometric and photometric verifier before becoming graph edges.
    "loop_closure_min_gap": 10,
    "loop_closure_max_gap": 25,
    "match_confidence": 0.35,
    "confidence_threshold": 0.6,
    "minimum_gradient_correlation": 0.2,
    # USAC/MAGSAC consumes OpenCV's process-global RNG. Resetting it before
    # every pair makes a cache miss independent of request and candidate order.
    "opencv_rng_seed": 20,
}
COVERAGE_WIDTH = 24
COVERAGE_HEIGHT = 12
NEIGHBOUR_CANDIDATE_WINDOW = int(MATCH_CONFIG["pair_window"])
GUIDANCE_BAND_ROWS = 4


def _guidance_row_bounds(height: int) -> tuple[int, int]:
    """Return the useful room band; pole/detail shots stay optional."""

    band = min(GUIDANCE_BAND_ROWS, height)
    first = (height - band) // 2
    return first, first + band


def _nearest_coverage_target(
    coverage: dict[str, Any], yaw_degrees: float, pitch_degrees: float, version: int
) -> dict[str, Any] | None:
    """Choose the closest direction that still needs evidence.

    Guidance is relative to the current RGB-localized view. Grid index is only
    a deterministic tie-breaker; it must never decide where the user turns.
    """

    first_row, last_row = _guidance_row_bounds(int(coverage["height"]))
    # One good monocular image is sufficient for depth. Only unphotographed
    # directions are mandatory; amber quality/detail repairs remain optional
    # after the fast scan is complete.
    candidates = [
        cell
        for cell in coverage["cells"]
        if first_row <= int(cell["row"]) < last_row and cell["state"] == "red"
    ]
    if not candidates:
        return None

    def direction(cell: dict[str, Any]) -> tuple[float, float]:
        return (
            (cell["column"] + 0.5) * 360.0 / coverage["width"] - 180.0,
            90.0 - (cell["row"] + 0.5) * 180.0 / coverage["height"],
        )

    current_yaw = math.radians(yaw_degrees)
    current_pitch = math.radians(pitch_degrees)
    current = np.array(
        [
            math.cos(current_pitch) * math.sin(current_yaw),
            math.sin(current_pitch),
            math.cos(current_pitch) * math.cos(current_yaw),
        ]
    )

    def score(cell: dict[str, Any]) -> tuple[float, int]:
        yaw, pitch = map(math.radians, direction(cell))
        candidate = np.array(
            [math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)]
        )
        angle = math.acos(float(np.clip(np.dot(current, candidate), -1.0, 1.0)))
        return angle, int(cell["index"])

    chosen = min(candidates, key=score)
    yaw, pitch = direction(chosen)
    return {
        "version": version,
        "target_id": f"cell-{chosen['row']:02d}-{chosen['column']:02d}",
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "priority": 1.0,
        "reasons": chosen["reason_mask"],
    }


def _digest(value: object) -> str:
    wire = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(wire).hexdigest()


FEATURE_CONFIG_SHA256 = _digest(FEATURE_CONFIG)
MATCH_CONFIG_SHA256 = _digest(MATCH_CONFIG)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.savez_compressed(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _feature_arrays(features: _FeatureSet) -> dict[str, np.ndarray]:
    points = np.asarray(
        [
            (
                point.pt[0],
                point.pt[1],
                point.size,
                point.angle,
                point.response,
                point.octave,
                point.class_id,
            )
            for point in features.keypoints
        ],
        dtype=np.float32,
    ).reshape(-1, 7)
    return {"points": points, "descriptors": features.descriptors.astype(np.float32)}


def _features_from_arrays(points: np.ndarray, descriptors: np.ndarray) -> _FeatureSet:
    keypoints = tuple(
        cv2.KeyPoint(
            float(item[0]),
            float(item[1]),
            float(item[2]),
            float(item[3]),
            float(item[4]),
            int(item[5]),
            int(item[6]),
        )
        for item in points
    )
    return _FeatureSet(keypoints, np.asarray(descriptors, dtype=np.float32))


def classify_frame_quality(image: np.ndarray) -> dict[str, Any]:
    """Return independent image-space quality channels and rejection reasons."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness = float(np.clip(sharpness_raw / 180.0, 0.0, 1.0))
    mean = float(gray.mean())
    dark = float(np.mean(gray <= 8))
    bright = float(np.mean(gray >= 247))
    exposure = float(np.clip(1.0 - abs(mean - 127.5) / 127.5 - 1.5 * (dark + bright), 0.0, 1.0))
    texture = float(np.clip(cv2.Sobel(gray, cv2.CV_32F, 1, 1).std() / 40.0, 0.0, 1.0))
    reasons: list[str] = []
    if sharpness < 0.18:
        reasons.append("blur")
    if exposure < 0.20:
        reasons.append("exposure")
    if texture < 0.08:
        reasons.append("textureless")
    return {
        "sharpness": sharpness,
        "sharpness_raw": sharpness_raw,
        "exposure": exposure,
        "mean_luminance": mean,
        "texture": texture,
        "reasons": reasons,
    }


@dataclass(frozen=True)
class CachedFrame:
    frame_id: str
    checksum: str
    image_path: Path
    quality: dict[str, Any]


class IncrementalGraphEngine:
    """Durable incremental graph for one service storage root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.cache = root / "incremental-cache"
        self.feature_calls = 0
        self.match_calls = 0

    def _feature_path(self, checksum: str) -> Path:
        return self.cache / "features" / FEATURE_CONFIG_SHA256 / f"{checksum}.npz"

    def _pair_path(self, first_sha: str, second_sha: str) -> Path:
        ordered = sorted((first_sha, second_sha))
        key = _digest({"images": ordered, "matching": MATCH_CONFIG_SHA256})
        return self.cache / "pairs" / MATCH_CONFIG_SHA256 / f"{key}.json"

    def _features(self, checksum: str, image: np.ndarray) -> _FeatureSet:
        path = self._feature_path(checksum)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                return _features_from_arrays(data["points"], data["descriptors"])
        self.feature_calls += 1
        features = _extract_rootsift(image)
        _atomic_npz(path, **_feature_arrays(features))
        return features

    def _station_frames(self, station: Path) -> list[CachedFrame]:
        output: list[CachedFrame] = []
        metadata_paths = sorted((station / "frames").glob("*/frame.json"))
        metadata_paths.sort(
            key=lambda path: (
                json.loads(path.read_text())["timestamp_ns"],
                json.loads(path.read_text())["frame_id"],
            )
        )
        for path in metadata_paths:
            metadata = json.loads(path.read_text())
            image_path = path.with_name("frame.jpg")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            quality_path = path.with_name("quality.json")
            if quality_path.exists():
                quality = json.loads(quality_path.read_text())
            else:
                quality = classify_frame_quality(image)
                _atomic_json(quality_path, quality)
            output.append(
                CachedFrame(metadata["frame_id"], metadata["jpeg_sha256"], image_path, quality)
            )
        return output

    @staticmethod
    def _candidate_pairs(count: int) -> list[tuple[int, int]]:
        """Only neighbour and deterministic start-of-scan loop candidates."""

        window = NEIGHBOUR_CANDIDATE_WINDOW
        if count < 2:
            return []
        newest = count - 1
        candidates = {(index, newest) for index in range(max(0, newest - window), newest)}
        minimum = int(MATCH_CONFIG["loop_closure_min_gap"])
        maximum = int(MATCH_CONFIG["loop_closure_max_gap"])
        candidates.update(
            (index, newest)
            for index in range(max(0, newest - maximum), max(0, newest - minimum + 1))
        )
        return sorted(candidates)

    def _match_pair(
        self,
        first_index: int,
        second_index: int,
        frames: list[CachedFrame],
        images: list[np.ndarray],
        features: list[_FeatureSet],
    ) -> dict[str, Any]:
        first_sha = frames[first_index].checksum
        second_sha = frames[second_index].checksum
        path = self._pair_path(first_sha, second_sha)
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["first_checksum"] == first_sha:
                return cached
            # The cache key is order-insensitive. Invert an accepted homography
            # when the same content pair appears in reverse capture order.
            if cached.get("accepted"):
                cached = dict(cached)
                cached["first_checksum"], cached["second_checksum"] = first_sha, second_sha
                cached["homography_centered"] = np.linalg.inv(
                    np.asarray(cached["homography_centered"], dtype=float)
                ).tolist()
            return cached

        self.match_calls += 1
        result: dict[str, Any] = {
            "schema": "live-rgb-pair-cache-v2",
            "matching_config_sha256": MATCH_CONFIG_SHA256,
            "first_checksum": first_sha,
            "second_checksum": second_sha,
            "accepted": False,
            "reason": "false_link",
        }
        cv2.setRNGSeed(int(MATCH_CONFIG["opencv_rng_seed"]))
        matched = _match_rootsift_pair(
            features[first_index],
            features[second_index],
            images[first_index].shape[:2],
            ratio_threshold=1.0 - float(MATCH_CONFIG["match_confidence"]),
        )
        if matched is None:
            _atomic_json(path, result)
            return result
        homography_pixels, matches, inliers, first_coverage, second_coverage, reprojection = matched
        confidence = inliers / (8.0 + 0.3 * matches)
        if confidence < float(MATCH_CONFIG["confidence_threshold"]) or inliers / matches < 0.30:
            result["reason"] = "weak_geometry"
            _atomic_json(path, result)
            return result
        homography_centered = (
            _center_transform(images[second_index])
            @ homography_pixels
            @ np.linalg.inv(_center_transform(images[first_index]))
        )
        gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) for image in images]
        gradients = [
            cv2.magnitude(
                cv2.Sobel(item, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(item, cv2.CV_32F, 0, 1, ksize=3),
            )
            for item in gray
        ]
        try:
            gradient, intensity, overlap = _photometric_score(
                first_index,
                second_index,
                homography_centered,
                images,
                gray,
                gradients,
            )
        except (cv2.error, np.linalg.LinAlgError):
            _atomic_json(path, result)
            return result
        if gradient < float(MATCH_CONFIG["minimum_gradient_correlation"]) or intensity < 0.35 or overlap < 0.08:
            result["reason"] = "photometric_mismatch"
            _atomic_json(path, result)
            return result
        # A projective homography whose singular values are far apart is
        # dominated by translation/non-planarity rather than pure station rotation.
        singular = np.linalg.svd(homography_centered, compute_uv=False)
        parallax_risk = float(np.clip(np.log(max(singular) / max(min(singular), 1e-9)) / 12.0, 0, 1))
        public = OverlapEdge(
            first=first_index,
            second=second_index,
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
            confidence
            * np.log1p(inliers)
            * gradient
            * intensity
            * np.sqrt(overlap)
            * np.sqrt(min(first_coverage, second_coverage))
            / (1.0 + reprojection)
        )
        result.update(
            {
                "accepted": True,
                "reason": "parallax_risk" if parallax_risk >= 0.75 else None,
                "edge": asdict(public),
                "homography_centered": homography_centered.tolist(),
                "weight": float(weight),
                "parallax_risk": parallax_risk,
            }
        )
        _atomic_json(path, result)
        return result

    @staticmethod
    def _matched_edge(cached: dict[str, Any], first: int, second: int) -> _MatchedEdge | None:
        if not cached.get("accepted"):
            return None
        public = OverlapEdge(**{**cached["edge"], "first": first, "second": second})
        return _MatchedEdge(
            public=public,
            homography_centered=np.asarray(cached["homography_centered"], dtype=float),
            weight=float(cached["weight"]),
        )

    def _all_cached_edges(
        self, frames: list[CachedFrame], images: list[np.ndarray]
    ) -> tuple[list[_MatchedEdge], list[dict[str, Any]]]:
        features = [self._features(frame.checksum, image) for frame, image in zip(frames, images)]
        edges: list[_MatchedEdge] = []
        rejected: list[dict[str, Any]] = []
        count = len(frames)
        window = NEIGHBOUR_CANDIDATE_WINDOW
        loop_min = int(MATCH_CONFIG["loop_closure_min_gap"])
        loop_max = int(MATCH_CONFIG["loop_closure_max_gap"])
        for first in range(count):
            for second in range(first + 1, count):
                distance = second - first
                if distance > window and not loop_min <= distance <= loop_max:
                    continue
                cached = self._match_pair(first, second, frames, images, features)
                edge = self._matched_edge(cached, first, second)
                if edge is None:
                    rejected.append({"first": first, "second": second, "reason": cached["reason"]})
                else:
                    edges.append(edge)
        return edges, rejected

    def _solve(self, frames: list[CachedFrame], images: list[np.ndarray]) -> dict[str, Any]:
        if not frames:
            return {
                "selected": [],
                "edges": [],
                "rotations": {},
                "focal_px": None,
                "cycle_error_deg": 0.0,
                "rejections": [],
            }
        if len(frames) == 1:
            return {
                "selected": [0],
                "edges": [],
                "rotations": {"0": np.eye(3).tolist()},
                "focal_px": float(max(images[0].shape[:2])),
                "cycle_error_deg": 0.0,
                "rejections": [],
            }
        edges, rejected = self._all_cached_edges(frames, images)
        public = [edge.public for edge in edges]
        components = _components(len(frames), public)
        selected = components[0] if components else ()
        if len(selected) < 2 or not edges:
            return {
                "selected": list(selected),
                "edges": [asdict(edge) for edge in public],
                "rotations": {str(selected[0]): np.eye(3).tolist()} if selected else {},
                "focal_px": float(max(images[0].shape[:2])),
                "cycle_error_deg": 0.0,
                "rejections": rejected,
            }
        focal = _estimate_focal(edges, images[0].shape[:2])
        intrinsic = np.diag([focal, focal, 1.0])
        inverse = np.diag([1.0 / focal, 1.0 / focal, 1.0])
        accepted: list[_MatchedEdge] = []
        for edge in edges:
            relative = _nearest_rotation(
                inverse @ np.linalg.inv(edge.homography_centered) @ intrinsic
            )
            angle = float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
            if angle > 95.0:
                rejected.append({"first": edge.public.first, "second": edge.public.second, "reason": "false_link"})
                continue
            accepted.append(
                _MatchedEdge(
                    public=OverlapEdge(**{**asdict(edge.public), "rotation_deg": angle}),
                    homography_centered=edge.homography_centered,
                    weight=edge.weight,
                    relative_rotation=relative,
                )
            )
        public = [edge.public for edge in accepted]
        components = _components(len(frames), public)
        selected = components[0] if components else ()
        rotations: dict[int, np.ndarray] = {}
        cycle_error = 0.0
        if len(selected) >= 2:
            relevant = tuple(
                edge
                for edge in accepted
                if edge.public.first in selected and edge.public.second in selected
            )
            try:
                initial, _, root = _maximum_spanning_rotations(selected, relevant)
                rotations, cycle_error = _refine_rotations(initial, root, relevant)
            except StitchError:
                rotations = {}
        return {
            "selected": list(selected),
            "edges": [asdict(edge) for edge in public],
            "rotations": {str(index): rotation.tolist() for index, rotation in rotations.items()},
            "focal_px": focal,
            "cycle_error_deg": cycle_error,
            "rejections": rejected,
        }

    @staticmethod
    def _direction(rotation: np.ndarray) -> tuple[float, float]:
        forward = rotation @ np.array([0.0, 0.0, 1.0])
        yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
        pitch = math.degrees(math.asin(float(np.clip(forward[1], -1.0, 1.0))))
        return yaw, pitch

    def _coverage(
        self, graph: dict[str, Any], frames: list[CachedFrame], version: int
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        observations: list[list[int]] = [[] for _ in range(COVERAGE_WIDTH * COVERAGE_HEIGHT)]
        rotations = {int(key): np.asarray(value) for key, value in graph["rotations"].items()}
        for index, rotation in rotations.items():
            yaw, pitch = self._direction(rotation)
            center_column = int(((yaw + 180.0) % 360.0) / 360.0 * COVERAGE_WIDTH)
            center_row = int(np.clip((90.0 - pitch) / 180.0 * COVERAGE_HEIGHT, 0, COVERAGE_HEIGHT - 1))
            # Approximate the image footprint using a conservative 70x50 degree cone.
            for row in range(max(0, center_row - 2), min(COVERAGE_HEIGHT, center_row + 3)):
                for offset in range(-2, 3):
                    column = (center_column + offset) % COVERAGE_WIDTH
                    observations[row * COVERAGE_WIDTH + column].append(index)
        incident: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(frames))}
        for edge in graph["edges"]:
            incident[edge["first"]].append(edge)
            incident[edge["second"]].append(edge)
        cells = []
        for index, contributors in enumerate(observations):
            row, column = divmod(index, COVERAGE_WIDTH)
            unique = sorted(set(contributors))
            edges = [edge for frame in unique for edge in incident.get(frame, [])]
            independent = len(unique)
            photographed = min(1.0, float(independent))
            edge_strength = float(np.mean([min(1.0, e["confidence"] / 2.0) for e in edges])) if edges else 0.0
            spread = float(np.mean([min(e["first_feature_coverage"], e["second_feature_coverage"]) for e in edges])) if edges else 0.0
            reprojection = float(np.mean([e["median_reprojection_error"] for e in edges])) if edges else 0.0
            sharpness = float(np.mean([frames[item].quality["sharpness"] for item in unique])) if unique else 0.0
            exposure = float(np.mean([frames[item].quality["exposure"] for item in unique])) if unique else 0.0
            parallax = float(max((item.get("parallax_risk", 0.0) for item in graph["rejections"]), default=0.0))
            reasons: list[str] = []
            if independent == 0:
                reasons.append("missing")
            if sharpness and sharpness < 0.35:
                reasons.append("blur")
            if exposure and exposure < 0.35:
                reasons.append("exposure")
            if edges and edge_strength < 0.35:
                reasons.append("weak_edge")
            if parallax >= 0.75:
                reasons.append("parallax_risk")
            state = "red" if independent == 0 else "green" if not reasons else "amber"
            cells.append(
                {
                    "index": index,
                    "row": row,
                    "column": column,
                    "state": state,
                    "channels": {
                        "photographed": photographed,
                        "independent_views": independent,
                        "edge_strength": edge_strength,
                        "feature_spread": spread,
                        "reprojection": reprojection,
                        "cycle_consistency": max(0.0, 1.0 - graph["cycle_error_deg"] / 10.0),
                        "sharpness": sharpness,
                        "exposure": exposure,
                        "parallax_risk": parallax,
                        "seam_support": min(1.0, float(independent)),
                    },
                    "reason_mask": reasons,
                }
            )
        coverage = {
            "version": version,
            "width": COVERAGE_WIDTH,
            "height": COVERAGE_HEIGHT,
            "cells": cells,
        }
        rotations = graph["rotations"]
        if rotations:
            latest = max(rotations, key=int)
            yaw, pitch = self._direction(np.asarray(rotations[latest]))
        else:
            yaw, pitch = 0.0, 0.0
        return coverage, _nearest_coverage_target(coverage, yaw, pitch, version)

    def process_station(self, station: Path) -> dict[str, Any]:
        started = time.perf_counter()
        frames = self._station_frames(station)
        images = [cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR) for frame in frames]
        images = _prepare_stitch_images(images)
        current_path = station / "incremental" / "current.json"
        previous = json.loads(current_path.read_text()) if current_path.exists() else None
        frame_digest = _digest([(frame.frame_id, frame.checksum) for frame in frames])
        if previous and previous.get("frame_digest") == frame_digest:
            return previous
        print(f"[live-rgb guidance] graph: {len(frames)} frames", flush=True)
        graph = self._solve(frames, images)
        version = int(previous["version"] + 1) if previous else 1
        coverage, target = self._coverage(graph, frames, version)
        snapshot = {
            "schema": "live-rgb-guidance-v2",
            "version": version,
            "graph_version": version,
            "frame_digest": frame_digest,
            "frame_count": len(frames),
            "graph": graph,
            "coverage": coverage,
            "localization": {
                "version": version,
                "state": "not_requested",
                "confidence": 0.0,
                "graph_version": version,
            },
            "target": target,
            "runtime_ms": (time.perf_counter() - started) * 1000.0,
            "cache": {
                "feature_config_sha256": FEATURE_CONFIG_SHA256,
                "matching_config_sha256": MATCH_CONFIG_SHA256,
                "feature_extractions_this_process": self.feature_calls,
                "pair_matches_this_process": self.match_calls,
            },
            "uses_arcore": False,
            "inputs": ["rgb_jpeg"],
        }
        version_path = station / "incremental" / "versions" / f"v{version:06d}.json"
        _atomic_json(version_path, snapshot)
        _atomic_json(current_path, snapshot)
        event_path = station / "incremental" / "events" / f"v{version:06d}.json"
        _atomic_json(event_path, {"version": version, "target": target})
        print(
            f"[live-rgb guidance] v{version}: {len(graph['selected'])}/{len(frames)} localized, "
            f"{len(graph['edges'])} edges in {snapshot['runtime_ms']:.0f} ms",
            flush=True,
        )
        return snapshot

    def guidance(self, station: Path) -> dict[str, Any]:
        current = station / "incremental" / "current.json"
        if not current.exists():
            return self.process_station(station)
        return json.loads(current.read_text())

    def localize_preview(self, station: Path, preview_id: str, jpeg: bytes) -> dict[str, Any]:
        """Localize RGB pixels against the current graph without mutating it."""

        snapshot = self.guidance(station)
        graph_version = int(snapshot["graph_version"])
        preview_checksum = hashlib.sha256(jpeg).hexdigest()
        preview_result = station / "incremental" / "previews" / f"{preview_id}.json"
        if preview_result.exists():
            cached = json.loads(preview_result.read_text())
            if cached["jpeg_sha256"] != preview_checksum:
                raise ValueError("preview id already has different content")
            return cached["guidance"]
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("preview body is not a JPEG")
        image = _prepare_stitch_images([image])[0]
        quality = classify_frame_quality(image)
        result_version = int(snapshot["version"]) + 1
        localization: dict[str, Any] = {
            "version": result_version,
            "preview_id": preview_id,
            "state": "unlocalized",
            "confidence": 0.0,
            "graph_version": graph_version,
            "reason_mask": quality["reasons"] or ["ambiguous"],
        }
        if not quality["reasons"] and snapshot["graph"]["rotations"]:
            preview_sha = hashlib.sha256(jpeg).hexdigest()
            preview_features = self._features(preview_sha, image)
            frames = self._station_frames(station)
            best: tuple[float, int, dict[str, Any]] | None = None
            for index_string in sorted(snapshot["graph"]["rotations"], key=int):
                index = int(index_string)
                reference = cv2.imread(str(frames[index].image_path), cv2.IMREAD_COLOR)
                reference = _prepare_stitch_images([reference])[0]
                reference_features = self._features(frames[index].checksum, reference)
                cv2.setRNGSeed(int(MATCH_CONFIG["opencv_rng_seed"]))
                matched = _match_rootsift_pair(
                    preview_features,
                    reference_features,
                    image.shape[:2],
                    ratio_threshold=1.0 - float(MATCH_CONFIG["match_confidence"]),
                )
                if matched is None:
                    continue
                homography, matches, inliers, first_coverage, _, reprojection = matched
                confidence = min(1.0, inliers / max(24.0, matches * 0.55))
                candidate = (
                    confidence,
                    -index,
                    {
                        "reference_index": index,
                        "homography": homography,
                        "inliers": inliers,
                        "feature_spread": first_coverage,
                        "reprojection": reprojection,
                    },
                )
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            if best and best[0] >= 0.35 and best[2]["feature_spread"] >= 0.002:
                reference_index = best[2]["reference_index"]
                reference_rotation = np.asarray(snapshot["graph"]["rotations"][str(reference_index)])
                focal = float(snapshot["graph"]["focal_px"] or max(image.shape[:2]))
                intrinsic = np.diag([focal, focal, 1.0])
                inverse = np.diag([1.0 / focal, 1.0 / focal, 1.0])
                centered = (
                    _center_transform(reference)
                    @ best[2]["homography"]
                    @ np.linalg.inv(_center_transform(image))
                )
                relative = _nearest_rotation(inverse @ np.linalg.inv(centered) @ intrinsic)
                rotation = reference_rotation @ relative.T
                yaw, pitch = self._direction(rotation)
                localization = {
                    "version": result_version,
                    "preview_id": preview_id,
                    "state": "localized",
                    "confidence": best[0],
                    "graph_version": graph_version,
                    "reason_mask": [],
                    "yaw_deg": yaw,
                    "pitch_deg": pitch,
                    "reference_frame_id": frames[reference_index].frame_id,
                }
        response = dict(snapshot)
        response["version"] = result_version
        response["localization"] = localization
        # No arrow is exposed when localization is unknown.
        if localization["state"] != "localized":
            response["target"] = None
        else:
            response["target"] = _nearest_coverage_target(
                response["coverage"],
                float(localization["yaw_deg"]),
                float(localization["pitch_deg"]),
                result_version,
            )
        _atomic_json(
            station / "incremental" / "versions" / f"v{result_version:06d}.json", response
        )
        _atomic_json(station / "incremental" / "current.json", response)
        _atomic_json(
            station / "incremental" / "events" / f"v{result_version:06d}.json",
            {"version": result_version, "localization": localization, "target": response["target"]},
        )
        _atomic_json(
            preview_result,
            {"jpeg_sha256": preview_checksum, "guidance": response},
        )
        return response


def accept_newer(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    """Pure stale-result guard shared by tests and non-Android consumers."""

    if current is None or int(candidate["version"]) > int(current["version"]):
        return candidate
    return current
