"""The affine anchor must recover a known correction and refuse a bad one.

These are the two failure modes that matter in the pipeline. Recovering `a` and
`b` from a planted correction is the easy half. The hard half is that the fit
sits between a model that invents depth through windows and a sensor that
returns nothing on dark surfaces — so a correction that *looks* well-supported
by pixel count must still be rejected when it is wrong, because a silently
applied bad anchor moves a whole frame's geometry and nothing downstream can
tell it happened.
"""

from __future__ import annotations

import numpy as np
import pytest
from recon.depth.anchor import A_BOUNDS, MIN_PIXELS, AffineFit, fit_affine_depth


def _scene(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    """A depth map with structure: a slanted floor and a nearer block."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    d = 1.5 + 4.0 * (yy / h) + 0.8 * (xx / w)
    d[h // 3 : 2 * h // 3, w // 4 : w // 2] = 1.2
    return d + rng.normal(0, 0.01, size=d.shape)


def test_recovers_a_planted_affine_correction():
    ref = _scene()
    a_true, b_true = 0.45, 0.30
    # pred is what a model would have produced for this scene under that
    # correction: ref = a*pred + b.
    pred = (ref - b_true) / a_true

    fit = fit_affine_depth(pred, ref)

    assert fit is not None
    assert fit.a == pytest.approx(a_true, rel=0.02)
    assert fit.b == pytest.approx(b_true, abs=0.02)
    assert fit.residual_m < 0.05
    assert fit.inlier_fraction > 0.9


def test_outliers_do_not_drag_the_line():
    """A quarter of the frame confidently wrong — a window the model saw through.

    Plain least squares fails this; that is the whole reason for the reweighting.
    """
    ref = _scene()
    a_true, b_true = 0.5, 0.25
    pred = (ref - b_true) / a_true
    pred[:, :40] = 60.0  # model predicts far distance where the sensor sees wall

    fit = fit_affine_depth(pred, ref)

    assert fit is not None
    assert fit.a == pytest.approx(a_true, rel=0.05)
    assert fit.b == pytest.approx(b_true, abs=0.05)

    ols = np.polyfit(pred.ravel(), ref.ravel(), 1)[0]
    assert abs(ols - a_true) > abs(fit.a - a_true), "robust fit should beat OLS here"


def test_zero_returns_are_not_treated_as_measurements():
    """0 means *no return*, not 'a surface at zero metres'."""
    ref = _scene()
    pred = (ref - 0.2) / 0.6
    holes = np.zeros_like(ref, dtype=bool)
    holes[::3] = True
    ref_holed = ref.copy()
    ref_holed[holes] = 0.0

    fit = fit_affine_depth(pred, ref_holed)

    assert fit is not None
    assert fit.a == pytest.approx(0.6, rel=0.03)
    assert fit.pixels == int((~holes).sum())


def test_refuses_when_there_is_not_enough_evidence():
    ref = _scene()
    pred = ref.copy()
    sparse = np.zeros_like(ref)
    keep = MIN_PIXELS // 2
    flat = sparse.ravel()
    flat[:keep] = ref.ravel()[:keep]

    assert fit_affine_depth(pred, sparse) is None


def test_refuses_a_correction_outside_the_sane_band():
    ref = _scene()
    pred = ref * (A_BOUNDS[1] * 4.0)  # would need a far smaller `a` than allowed

    assert fit_affine_depth(pred, ref) is None


def test_refuses_to_flatten_a_frame_it_does_not_agree_with():
    """The failure that `MIN_CORRELATION` exists for, and it is not hypothetical.

    Six of 27 frames in bundle_e4a5da81 did this: a prediction uncorrelated with
    the measurement, whose best-fitting line is `a -> 0` — discard the depth and
    put the frame at one constant distance. It *lowers* the residual, so a
    residual test endorses it, and it turns a room into a flat wall.
    """
    ref = _scene(seed=1)
    rng = np.random.default_rng(7)
    pred = rng.uniform(1.5, 6.0, size=ref.shape)  # same range, no shared structure

    fit = fit_affine_depth(pred, ref)

    assert fit is None, "a correction that discards the prediction must be refused"

    # And confirm the trap is real: the collapsed fit does look better on residual.
    flat = np.full_like(ref, float(np.median(ref)))
    assert np.median(np.abs(ref - flat)) < np.median(np.abs(ref - pred))


def test_reports_correlation_for_a_frame_it_does_agree_with():
    ref = _scene(seed=2)
    pred = (ref - 0.3) / 0.55

    fit = fit_affine_depth(pred, ref)

    assert fit is not None
    assert fit.correlation > 0.99


def test_apply_clips_negatives_to_no_return():
    fit = AffineFit(a=1.0, b=-2.0, pixels=1000, inlier_fraction=1.0, residual_m=0.0)
    out = fit.apply(np.array([[1.0, 3.0]], dtype=np.float32))

    assert out[0, 0] == 0.0, "a correction into negative depth is no data, not a near surface"
    assert out[0, 1] == pytest.approx(1.0)


def test_held_out_pixels_score_the_same_as_fitted_ones():
    """Two parameters cannot memorise 300k pixels — show it rather than assert it."""
    ref = _scene(seed=3)
    pred = (ref - 0.2) / 0.7

    tiles = ((np.mgrid[0 : ref.shape[0], 0 : ref.shape[1]][0] // 8) % 2).astype(bool)
    fitted = fit_affine_depth(pred, ref, fit_mask=tiles)
    everything = fit_affine_depth(pred, ref)

    assert fitted is not None and everything is not None
    assert fitted.a == pytest.approx(everything.a, rel=0.02)
    assert fitted.residual_m == pytest.approx(everything.residual_m, abs=0.02)


def test_disparity_space_recovers_an_inverse_depth_correction():
    ref = _scene()
    a_true, b_true = 0.8, 0.05
    # ref_disp = a·pred_disp + b, so pred_disp = (ref_disp − b) / a
    pred = 1.0 / ((1.0 / ref - b_true) / a_true)

    fit = fit_affine_depth(pred, ref, space="disparity")

    assert fit is not None
    assert fit.a == pytest.approx(a_true, rel=0.02)
    assert fit.b == pytest.approx(b_true, abs=0.02)
    assert fit.residual_m < 0.05


def test_rejects_mismatched_grids():
    with pytest.raises(ValueError, match="different grids"):
        fit_affine_depth(np.ones((4, 4)), np.ones((4, 5)))
