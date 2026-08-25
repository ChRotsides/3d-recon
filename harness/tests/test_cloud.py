"""Voxel downsampling and flying-pixel rejection.

The property that matters for voxelising is *coverage*: it must fill the cells
the input occupies, which is the whole reason it replaced random sampling. The
property that matters for flying pixels is that they are found at edges and
nowhere else — a detector that also fires on a smooth sloping floor would delete
the floor.
"""

from __future__ import annotations

import numpy as np
import pytest
from recon.cloud import (
    choose_voxel_size,
    flying_pixel_mask,
    frame_aware_voxel_filter,
    voxel_downsample,
)


def test_one_point_per_occupied_cell():
    # Two clusters, three cells apart, many points each.
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0, 0], 0.002, size=(500, 3))
    b = rng.normal([0.1, 0, 0], 0.002, size=(500, 3))
    pts = np.concatenate([a, b])
    col = np.full((1000, 3), 128, dtype=np.uint8)

    out_p, out_c = voxel_downsample(pts, col, 0.02)

    assert len(out_p) == 2, "two tight clusters 10 cm apart are two cells at 2 cm"
    assert len(out_c) == 2
    assert np.linalg.norm(out_p[0] - [0, 0, 0]) < 0.01


def test_voxelising_covers_what_sampling_leaves_empty():
    """The measured failure: random sampling fills a fraction of the surface.

    A dense plane, thinned two ways to the same budget. Voxelising must cover
    more distinct cells than a random draw of the same size — that gap is what
    the holes were.
    """
    rng = np.random.default_rng(1)
    n = 200_000
    plane = np.stack(
        [rng.uniform(0, 2, n), rng.uniform(0, 2, n), np.zeros(n)], axis=1
    )
    col = np.full((n, 3), 200, dtype=np.uint8)

    voxelled, _ = voxel_downsample(plane, col, 0.02)
    sample = plane[rng.choice(n, len(voxelled), replace=False)]

    def cells(p):
        q = np.floor(p / 0.02).astype(np.int64)
        return len(np.unique(q, axis=0))

    assert cells(sample) < 0.85 * cells(voxelled)


def test_colour_is_averaged_not_taken_from_one_member():
    pts = np.zeros((2, 3))
    col = np.array([[0, 0, 0], [100, 200, 40]], dtype=np.uint8)

    _, out_c = voxel_downsample(pts, col, 0.05)

    assert len(out_c) == 1
    assert tuple(out_c[0]) == (50, 100, 20)


def test_empty_input_survives():
    p, c = voxel_downsample(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), 0.02)
    assert len(p) == 0 and len(c) == 0


def test_rejects_a_nonpositive_voxel():
    with pytest.raises(ValueError, match="must be positive"):
        voxel_downsample(np.zeros((1, 3)), np.zeros((1, 3), dtype=np.uint8), 0.0)


def test_budget_picks_a_size_that_fits():
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 4, size=(200_000, 3))

    size = choose_voxel_size(pts, 20_000)
    out, _ = voxel_downsample(pts, np.zeros((len(pts), 3), dtype=np.uint8), size)

    assert len(out) <= 20_000
    assert size > 0.01, "a 1 cm grid over this volume cannot fit the budget"


def test_frame_aware_fusion_gives_each_frame_one_vote():
    dense = np.full((100, 3), [0.001, 0.0, 0.0])
    sparse = np.array([[0.019, 0.0, 0.0]])
    points = np.concatenate([dense, sparse])
    colours = np.concatenate(
        [
            np.tile([200, 0, 0], (len(dense), 1)),
            np.array([[0, 0, 200]]),
        ]
    ).astype(np.uint8)
    frames = np.concatenate(
        [np.zeros(len(dense), dtype=np.int16), np.ones(len(sparse), dtype=np.int16)]
    )

    out_p, out_c, stats = frame_aware_voxel_filter(
        points,
        colours,
        frames,
        0.1,
        min_frame_support=1,
        min_neighbours=0,
        huber_delta_m=1.0,
    )

    assert len(out_p) == 1
    assert out_p[0, 0] == pytest.approx(0.010, abs=1e-6)
    assert tuple(out_c[0]) == (100, 0, 100)
    assert stats.frame_voxel_samples == 2


def test_frame_aware_fusion_huber_downweights_one_bad_frame():
    points = np.array([[0.010, 0, 0], [0.012, 0, 0], [0.090, 0, 0]])
    colours = np.full((3, 3), 100, dtype=np.uint8)

    out_p, _, _ = frame_aware_voxel_filter(
        points,
        colours,
        np.arange(3, dtype=np.int16),
        0.1,
        min_frame_support=1,
        min_neighbours=0,
        huber_delta_m=0.01,
    )

    assert out_p[0, 0] < 0.025


def test_single_frame_surface_survives_but_an_isolated_voxel_does_not():
    # A 2x2 patch has three neighbours per cell in the 26-neighbourhood.
    patch = np.array(
        [[0.01, 0.01, 0.0], [0.11, 0.01, 0.0], [0.01, 0.11, 0.0], [0.11, 0.11, 0.0]]
    )
    isolated = np.array([[1.01, 1.01, 1.0]])
    points = np.concatenate([patch, isolated])
    colours = np.full((len(points), 3), 120, dtype=np.uint8)

    out_p, _, stats = frame_aware_voxel_filter(
        points,
        colours,
        np.zeros(len(points), dtype=np.int16),
        0.1,
        min_frame_support=2,
        min_neighbours=3,
    )

    assert len(out_p) == 4
    assert stats.low_support_kept_by_neighbours == 4
    assert stats.dropped_isolated == 1


def test_high_cross_frame_spread_can_be_removed():
    points = np.array([[0.01, 0.01, 0.01], [0.09, 0.09, 0.09]])
    colours = np.full((2, 3), 100, dtype=np.uint8)

    out_p, _, stats = frame_aware_voxel_filter(
        points,
        colours,
        np.array([0, 1], dtype=np.int16),
        0.1,
        min_frame_support=2,
        min_neighbours=0,
        huber_delta_m=1.0,
        max_spread_m=0.05,
    )

    assert len(out_p) == 0
    assert stats.dropped_spread == 1


def test_frame_aware_filter_drops_nonfinite_points_and_reports_them():
    points = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    colours = np.full((2, 3), 100, dtype=np.uint8)

    out_p, _, stats = frame_aware_voxel_filter(
        points,
        colours,
        np.array([0, 0], dtype=np.int16),
        0.1,
        min_frame_support=1,
        min_neighbours=0,
    )

    assert len(out_p) == 1
    assert stats.input_points == 2
    assert stats.finite_points == 1


def test_flying_pixels_are_found_at_the_edge():
    d = np.full((40, 40), 2.0, dtype=np.float32)
    d[:, 20:] = 4.0  # a 2 m step: the far wall seen past a near object

    mask = flying_pixel_mask(d)

    assert mask[:, 19].all() and mask[:, 20].all(), "both sides of the step"
    assert not mask[:, :18].any(), "the flat near surface is untouched"
    assert not mask[:, 22:].any(), "the flat far surface is untouched"


def test_a_smoothly_sloping_surface_is_not_an_edge():
    """A floor receding from 1 m to 5 m must survive — it is the room."""
    yy = np.linspace(1.0, 5.0, 200, dtype=np.float32)
    d = np.tile(yy[:, None], (1, 60))

    assert not flying_pixel_mask(d).any()


def test_threshold_is_relative_to_depth():
    """10 cm at 1 m is an edge; the same 10 cm at 8 m is the sensor's last digit."""
    near = np.full((10, 10), 1.0, dtype=np.float32)
    near[:, 5:] = 1.10  # 10% of its own depth
    far = np.full((10, 10), 8.0, dtype=np.float32)
    far[:, 5:] = 8.10  # 1.25% of its own depth

    assert flying_pixel_mask(near).any()
    assert not flying_pixel_mask(far).any()


def test_no_return_pixels_do_not_manufacture_edges():
    """0 means no data. Treating it as depth would ring every hole with a wall."""
    d = np.full((20, 20), 3.0, dtype=np.float32)
    d[8:12, 8:12] = 0.0

    mask = flying_pixel_mask(d)

    assert not mask.any(), "the border of a hole is not a discontinuity in the surface"


def test_rejects_a_non_2d_depth_map():
    with pytest.raises(ValueError, match="2-D"):
        flying_pixel_mask(np.zeros((4, 4, 3)))
