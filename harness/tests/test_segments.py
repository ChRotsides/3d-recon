"""Compositing SAM's overlapping masks into one label per pixel.

The property that matters is which mask wins an overlap. SAM returns a cushion,
the couch under it and sometimes the wall behind both; alignment needs the
specific object, because the large mask that swallows several objects is the
whole-frame fit we already know is not good enough.
"""

from __future__ import annotations

import numpy as np
import pytest
from recon.segments import (
    BACKGROUND,
    align_segments_to_consensus,
    composite_labels,
    sample_labels_to_grid,
    segment_sizes,
)


def test_the_smaller_mask_wins_the_overlap():
    big = np.zeros((40, 40), bool)
    big[5:35, 5:35] = True          # 900 px
    small = np.zeros((40, 40), bool)
    small[10:20, 10:20] = True      # 100 px, entirely inside big

    labels = composite_labels([big, small], min_area=50)

    assert len(np.unique(labels[labels >= 0])) == 2
    assert len(np.unique(labels[10:20, 10:20])) == 1, "the small mask must be intact"
    # And the big one still owns the part the small one does not cover.
    assert labels[6, 6] != labels[15, 15]


def test_unclaimed_pixels_are_background_not_segment_zero():
    mask = np.zeros((20, 20), bool)
    mask[0:10, 0:10] = True

    labels = composite_labels([mask], min_area=10)

    assert labels[15, 15] == BACKGROUND
    assert labels[15, 15] < 0, "background must not collide with a real segment id"
    assert labels[2, 2] == 0


def test_specks_are_dropped():
    big = np.ones((30, 30), bool)
    speck = np.zeros((30, 30), bool)
    speck[0, 0] = True

    labels = composite_labels([big, speck], min_area=100)

    assert len(np.unique(labels[labels >= 0])) == 1, "a 1 px mask is not an object"


def test_resampling_invents_no_labels():
    labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.int16)

    out = sample_labels_to_grid(labels, 2, 2)

    assert out.shape == (2, 2)
    assert set(np.unique(out)) <= set(np.unique(labels)), "nearest neighbour cannot create ids"


def test_sizes_count_every_segment():
    mask_a = np.zeros((10, 10), bool)
    mask_a[:, :5] = True
    mask_b = np.zeros((10, 10), bool)
    mask_b[:, 5:] = True

    sizes = segment_sizes(composite_labels([mask_a, mask_b], min_area=10))

    assert sizes.tolist() == [50, 50]


def test_rejects_ragged_masks():
    with pytest.raises(ValueError, match="shape"):
        composite_labels([np.ones((10, 10), bool), np.ones((8, 8), bool)], min_area=1)
    with pytest.raises(ValueError, match="no masks"):
        composite_labels([])


def _box(rng, n=4000, size=3.0):
    """Points on the floor and two walls, as a stand-in for a room."""
    floor = np.stack([rng.uniform(0, size, n), np.zeros(n), rng.uniform(0, size, n)], axis=1)
    wall = np.stack([np.zeros(n), rng.uniform(0, 2.0, n), rng.uniform(0, size, n)], axis=1)
    return floor, wall


def test_a_displaced_object_is_brought_back_onto_its_twin():
    """The couch. One frame puts it 12 cm off; segment alignment must close that."""
    rng = np.random.default_rng(0)
    floor, wall = _box(rng)
    couch = rng.uniform([1, 0, 1], [1.8, 0.6, 1.8], size=(3000, 3))

    # Frame 0 sees everything correctly; frame 1 has the couch displaced.
    p0 = np.concatenate([floor, wall, couch])
    s0 = np.concatenate([np.zeros(len(floor)), np.ones(len(wall)), np.full(len(couch), 2)])
    p1 = np.concatenate([floor, wall, couch + [0.12, 0, 0]])
    s1 = s0.copy()

    points = np.concatenate([p0, p1])
    frames = np.concatenate([np.zeros(len(p0), int), np.ones(len(p1), int)])
    seg = np.concatenate([s0, s1]).astype(np.int32)

    out, stats = align_segments_to_consensus(points, frames, seg, rounds=2)

    a = out[len(p0) - len(couch): len(p0)]
    b = out[len(p0) + len(p1) - len(couch):]
    gap = float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))
    assert stats.aligned_segments >= 1
    assert gap < 0.04, f"12 cm of object displacement should close, got {gap * 100:.1f} cm"


def test_a_segment_with_no_partner_is_left_alone():
    """Seen by one frame only: nothing to agree with, so nothing may move it."""
    rng = np.random.default_rng(1)
    floor, wall = _box(rng)
    lonely = rng.uniform([8, 0, 8], [8.5, 0.5, 8.5], size=(2000, 3))

    p0 = np.concatenate([floor, wall, lonely])
    s0 = np.concatenate([np.zeros(len(floor)), np.ones(len(wall)), np.full(len(lonely), 2)])
    p1 = np.concatenate([floor, wall])
    s1 = np.concatenate([np.zeros(len(floor)), np.ones(len(wall))])

    points = np.concatenate([p0, p1])
    frames = np.concatenate([np.zeros(len(p0), int), np.ones(len(p1), int)])
    seg = np.concatenate([s0, s1]).astype(np.int32)

    out, _ = align_segments_to_consensus(points, frames, seg, rounds=1)

    moved = np.abs(out[len(floor) + len(wall): len(p0)] - lonely).max()
    assert moved < 1e-5, f"an unmatched segment moved by {moved * 100:.2f} cm"


def test_background_travels_with_its_frame():
    """Unlabelled points must not tear away from the objects they sit against."""
    rng = np.random.default_rng(2)
    floor, wall = _box(rng)
    loose = rng.uniform([0.2, 0.0, 0.2], [0.6, 0.4, 0.6], size=(500, 3))

    p0 = np.concatenate([floor, wall])
    s0 = np.concatenate([np.zeros(len(floor)), np.ones(len(wall))])
    p1 = np.concatenate([floor + [0.10, 0, 0], wall + [0.10, 0, 0], loose + [0.10, 0, 0]])
    s1 = np.concatenate([np.zeros(len(floor)), np.ones(len(wall)), np.full(len(loose), -1)])

    points = np.concatenate([p0, p1])
    frames = np.concatenate([np.zeros(len(p0), int), np.ones(len(p1), int)])
    seg = np.concatenate([s0, s1]).astype(np.int32)

    out, stats = align_segments_to_consensus(points, frames, seg, rounds=1)

    assert stats.carried_points == len(loose)
    carried = out[len(p0) + len(floor) + len(wall):]
    # It should have moved roughly the same way its frame's segments did.
    assert np.abs(carried.mean(axis=0) - (loose + [0.10, 0, 0]).mean(axis=0)).max() < 0.09


def test_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="segments"):
        align_segments_to_consensus(np.zeros((10, 3)), np.zeros(10, int), np.zeros(9, int))
