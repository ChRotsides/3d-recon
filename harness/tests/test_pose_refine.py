from __future__ import annotations

import numpy as np
from recon.pose_refine import (
    PoseEdge,
    estimate_pose_edge,
    refine_pose_graph,
    relative_pose_error,
    transform_points,
)
from scipy.spatial.transform import Rotation


def transform(rotation_deg=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", rotation_deg, degrees=True).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def test_metric_ransac_recovers_transform_with_outliers():
    rng = np.random.default_rng(4)
    points_j = rng.uniform([-1.0, -0.5, -4.0], [1.0, 0.8, -0.8], size=(120, 3))
    expected = transform((2.0, -4.0, 1.0), (0.18, -0.04, 0.09))
    points_i = transform_points(expected, points_j)
    points_i += rng.normal(0.0, 0.004, size=points_i.shape)
    outliers = rng.choice(len(points_i), 24, replace=False)
    points_i[outliers] = rng.uniform(-3.0, 3.0, size=(len(outliers), 3))

    edge = estimate_pose_edge(0, 1, points_i, points_j, seed=8, threshold_m=0.03)

    assert edge is not None
    translation_m, rotation_deg = relative_pose_error(edge.relative_ij, expected)
    assert translation_m < 0.01
    assert rotation_deg < 0.3
    assert edge.inlier_count >= 70
    assert len(edge.heldout_i) >= 5


def _edge(i, j, true_poses, rng):
    world = rng.uniform([-1.5, -0.8, -4.0], [1.5, 1.2, -0.7], size=(30, 3))
    points_i = transform_points(np.linalg.inv(true_poses[i]), world)
    points_j = transform_points(np.linalg.inv(true_poses[j]), world)
    return PoseEdge(
        i=i,
        j=j,
        relative_ij=np.linalg.inv(true_poses[i]) @ true_poses[j],
        train_count=100,
        inlier_count=90,
        train_rmse_m=0.005,
        heldout_i=points_i,
        heldout_j=points_j,
    )


def test_pose_graph_improves_heldout_points_and_keeps_disconnected_frame():
    rng = np.random.default_rng(11)
    truth = [
        transform(),
        transform((0.0, 8.0, 0.0), (0.7, 0.0, -0.1)),
        transform((0.0, 16.0, 0.0), (1.25, 0.03, -0.25)),
        transform((0.0, 80.0, 0.0), (4.0, 0.2, 1.0)),
    ]
    initial = [matrix.copy() for matrix in truth]
    initial[1] = transform((1.5, -2.0, 0.7), (0.16, -0.06, 0.10)) @ initial[1]
    initial[2] = transform((-1.0, 3.5, -0.6), (-0.13, 0.04, -0.18)) @ initial[2]
    # This isolated pose is intentionally wrong and must never be invented.
    initial[3] = transform((5.0, 0.0, 0.0), (0.5, 0.0, 0.0)) @ initial[3]
    edges = [_edge(0, 1, truth, rng), _edge(1, 2, truth, rng), _edge(0, 2, truth, rng)]

    result = refine_pose_graph(initial, edges, prior_weight=0.05)

    assert result.accepted
    assert result.component == (0, 1, 2)
    assert result.heldout_after_m < 0.15 * result.heldout_before_m
    assert np.array_equal(result.poses[3], initial[3])
    for index in (1, 2):
        translation_m, rotation_deg = relative_pose_error(truth[index], result.poses[index])
        assert translation_m < 0.02
        assert rotation_deg < 0.5


def test_pose_graph_rejects_an_unconstrained_component():
    initial = [transform(), transform(translation=(1.0, 0.0, 0.0))]

    result = refine_pose_graph(initial, [])

    assert not result.accepted
    assert all(np.array_equal(a, b) for a, b in zip(result.poses, initial, strict=True))
