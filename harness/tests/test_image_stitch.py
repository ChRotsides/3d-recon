from __future__ import annotations

import cv2
import numpy as np
from recon.image_stitch import (
    OverlapEdge,
    _components,
    _external_alpha,
    _feature_coverage,
    wrap_equirectangular,
)


def test_components_prefer_the_largest_strict_overlap_island():
    edges = [
        OverlapEdge(0, 1, 20, 15, 1.5),
        OverlapEdge(1, 2, 20, 14, 1.4),
        OverlapEdge(3, 4, 100, 90, 2.5),
    ]

    assert _components(6, edges) == ((0, 1, 2), (3, 4), (5,))


def test_external_alpha_keeps_enclosed_black_objects_opaque():
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    image[5:25, 4:36] = 200
    image[10:20, 15:25] = 0

    alpha = _external_alpha(image)

    assert alpha[0, 0] == 0
    assert alpha[6, 5] == 255
    assert alpha[15, 20] == 255
    assert cv2.countNonZero(alpha) == 20 * 32


def test_feature_coverage_requires_points_to_span_multiple_grid_cells():
    clustered = np.array([[10, 10], [11, 10], [10, 11], [11, 11]], np.float32)
    distributed = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], np.float32)

    clustered_area, clustered_cells = _feature_coverage(clustered, (100, 100))
    distributed_area, distributed_cells = _feature_coverage(distributed, (100, 100))

    assert clustered_area < 0.001
    assert clustered_cells == 1
    assert distributed_area > 0.6
    assert distributed_cells == 4


def test_wrap_equirectangular_fills_poles_and_closes_horizontal_seam():
    image = np.zeros((6, 12, 3), dtype=np.uint8)
    alpha = np.zeros((6, 12), dtype=np.uint8)
    image[1:5, :, 0] = np.arange(12, dtype=np.uint8)
    image[1:5, :, 1] = 100
    alpha[1:5] = 255

    wrapped = wrap_equirectangular(image, alpha)

    assert wrapped.shape == (6, 12, 3)
    assert np.all(wrapped[:, :, 1] == 100)
    assert np.array_equal(wrapped[:, 0], wrapped[:, -1])
