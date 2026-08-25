"""The sparse-point depth anchor (recon.depth.da3).

DA3 is aligned to the input poses by fitting its *own* camera estimate onto
them, so the model's pose error arrives as a depth scale error. Nothing
downstream can undo that — plan §6.2's scale stage multiplies poses and depth
together, so it cannot fix depth alone, and every criterion in §1.3 is a length.
The anchor corrects it against ARCore's triangulated points, which are metric
and already in the bundle.

These tests need no model, no GPU and no torch: the anchor is fed a depth map
constructed to be wrong by a known factor, and asserted to recover it.
"""

from __future__ import annotations

import numpy as np
import pytest
from recon.depth.da3 import _MAX_FRAME_DEVIATION, _sparse_point_scales
from recon.schemas import Intrinsics, Keyframe, Pose, SparsePoint

INTR = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


class _Bundle:
    """The two attributes the anchor reads, without a directory on disk."""

    def __init__(self, points):
        self.bundle = type("B", (), {"sparse_points": points})()


def _keyframe(kf_id: str) -> Keyframe:
    """A camera at the origin looking down -Z, per the plan's §5 frame."""
    return Keyframe(
        id=kf_id,
        image=f"images/{kf_id}.jpg",
        timestamp_ns=0,
        pose=Pose(R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], t=[0.0, 0.0, 0.0]),
        intrinsics=INTR,
    )


def _points_in_view(
    n: int, depth_m: float, observations: list[str]
) -> list[SparsePoint]:
    """`n` points spread across the image at a known distance from the camera."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        u = rng.uniform(40, INTR.width - 40)
        v = rng.uniform(40, INTR.height - 40)
        x = (u - INTR.cx) / INTR.fx * depth_m
        y = -(v - INTR.cy) / INTR.fy * depth_m
        out.append(
            SparsePoint(
                xyz=[float(x), float(y), float(-depth_m)],
                observations=observations,
            )
        )
    return out


def _uniform_depth(value: float) -> np.ndarray:
    return np.full((INTR.height, INTR.width), value, dtype=np.float32)


def test_a_uniform_scale_error_is_recovered():
    """The whole point: depth predicted 2x too small comes back as 2x."""
    true_depth = 3.0
    bundle = _Bundle(_points_in_view(300, true_depth, ["kf_0000"]))
    pred = np.stack([_uniform_depth(true_depth / 2.0)])
    scales = _sparse_point_scales(bundle, [_keyframe("kf_0000")], pred)
    assert scales is not None
    assert scales[0] == pytest.approx(2.0, rel=1e-3)


def test_each_frame_gets_its_own_scale():
    """Per frame, not per window.

    One scalar per window leaves frames inside it disagreeing: measured on a
    real capture, per-window scales ran 0.997 to 1.176, so the same wall fused
    up to 18% apart in depth depending on which window saw it.
    """
    bundle = _Bundle(_points_in_view(300, 3.0, ["kf_0000", "kf_0001"]))
    kfs = [_keyframe("kf_0000"), _keyframe("kf_0001")]
    # Same scene, same points; the second frame's prediction is twice as wrong.
    pred = np.stack([_uniform_depth(3.0), _uniform_depth(1.5)])
    scales = _sparse_point_scales(bundle, kfs, pred)
    assert scales[0] == pytest.approx(1.0, rel=1e-3)
    assert scales[1] == pytest.approx(2.0, rel=1e-3)


def test_a_thinly_measured_outlier_frame_falls_back_to_the_window():
    """A handful of points must not move a whole frame.

    Scaling one frame by an outlier puts a step in a continuous surface, which
    reads worse than the uniform error it was trying to remove.
    """
    # Most points sit at 3 m; a couple of stray ones are much closer.
    observed = ["kf_0000", "kf_0001"]
    bundle = _Bundle(
        _points_in_view(300, 3.0, observed) + _points_in_view(3, 0.5, observed)
    )
    kfs = [_keyframe("kf_0000"), _keyframe("kf_0001")]
    pred = np.stack([_uniform_depth(3.0), _uniform_depth(3.0)])
    scales = _sparse_point_scales(bundle, kfs, pred)
    # Both frames see the same points, so nothing should be far from 1.0.
    for s in scales:
        assert abs(s - 1.0) <= _MAX_FRAME_DEVIATION + 1e-6


def test_a_well_measured_frame_is_trusted_over_the_window_median():
    """Evidence, not conformity.

    A frame with hundreds of points that disagrees with its window is
    informative — clamping it re-introduces exactly the per-frame error the
    anchor exists to remove. Measured on a real capture: a frame with 138
    points needing 1.37x was being pinned to its window's 1.00 and left wrong.
    """
    kfs = [_keyframe(f"kf_{i:04d}") for i in range(4)]
    bundle = _Bundle(_points_in_view(400, 3.0, [kf.id for kf in kfs]))
    # Three frames are right; the fourth is 40% off — well past the deviation
    # guard, but backed by the same 400 points.
    pred = np.stack([_uniform_depth(3.0)] * 3 + [_uniform_depth(3.0 / 1.4)])
    scales = _sparse_point_scales(bundle, kfs, pred)
    assert scales[3] == pytest.approx(1.4, rel=1e-2), (
        "a well-measured frame was clamped to its window instead of corrected"
    )


def test_no_points_means_no_scaling_rather_than_a_guess():
    bundle = _Bundle([])
    pred = np.stack([_uniform_depth(3.0)])
    assert _sparse_point_scales(bundle, [_keyframe("kf_0000")], pred) is None


def test_points_behind_the_camera_are_ignored():
    """A point behind the camera projects to a plausible pixel and would
    otherwise contribute a nonsense ratio."""
    behind = [
        SparsePoint(xyz=[0.0, 0.0, 3.0], observations=["kf_0000"])
    ] * 50  # +Z is behind, looking down -Z
    bundle = _Bundle(behind)
    pred = np.stack([_uniform_depth(3.0)])
    assert _sparse_point_scales(bundle, [_keyframe("kf_0000")], pred) is None


def test_points_observed_in_another_frame_cannot_anchor_this_one():
    """Geometric projection is not an observation association.

    This is the regression from the audit: points from unrelated frames could
    satisfy the support threshold and rescale a frame with zero own features.
    """
    bundle = _Bundle(_points_in_view(300, 3.0, ["kf_other"]))
    pred = np.stack([_uniform_depth(1.5)])
    diagnostics = []
    assert (
        _sparse_point_scales(
            bundle, [_keyframe("kf_0000")], pred, diagnostics=diagnostics
        )
        is None
    )
    assert diagnostics[0]["support_count"] == 0
    assert diagnostics[0]["chosen_scale"] is None


def test_anchor_records_support_inliers_dispersion_and_strategy():
    bundle = _Bundle(_points_in_view(300, 3.0, ["kf_0000"]))
    pred = np.stack([_uniform_depth(1.5)])
    diagnostics = []
    scales = _sparse_point_scales(
        bundle, [_keyframe("kf_0000")], pred, diagnostics=diagnostics
    )
    assert scales == pytest.approx([2.0])
    assert diagnostics == [
        {
            "keyframe_id": "kf_0000",
            "support_count": 300,
            "inlier_count": 300,
            "ratio_dispersion": pytest.approx(0.0),
            "raw_scale": pytest.approx(2.0),
            "window_support_count": 300,
            "window_inlier_count": 300,
            "window_ratio_dispersion": pytest.approx(0.0),
            "window_scale": pytest.approx(2.0),
            "chosen_scale": pytest.approx(2.0),
            "strategy": "per_frame",
        }
    ]


def test_an_ambiguous_window_scale_is_rejected():
    """Two equally supported surfaces cannot justify one scalar correction."""
    points = _points_in_view(150, 2.0, ["kf_0000"])
    points += _points_in_view(150, 4.0, ["kf_0000"])
    bundle = _Bundle(points)
    diagnostics = []
    scales = _sparse_point_scales(
        bundle,
        [_keyframe("kf_0000")],
        np.stack([_uniform_depth(2.0)]),
        diagnostics=diagnostics,
    )
    assert scales is None
    assert diagnostics[0]["strategy"] == "window_rejected"
    assert diagnostics[0]["window_ratio_dispersion"] > 0.2
