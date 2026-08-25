"""Fit each frame's predicted depth to the metric depth the device measured.

Monocular depth is affine-invariant: the model recovers relative structure, and
`d_metric ≈ a·d_pred + b` is the family of corrections that can place it in
metres. The sparse anchor in `da3.py` fits only `a`, against the handful of VIO
feature points visible in one frame. Every frame of an `arcore_depth` capture
also carries a *measured* depth map — order 300,000 metric pixels — which is a
far stronger constraint and fits `b` as well.

**Why robust and not plain least squares.** The two sources disagree exactly
where the sensor is weakest: ARCore returns nothing on dark, distant and
specular surfaces, quantises to millimetres, and produces flying pixels at
depth edges, while the model invents distance through windows and mirrors.
Those tails are not rare enough to ignore — a least-squares line is dragged by
them in proportion to their pixel count. The fit here is iteratively reweighted
with a Huber loss around a MAD scale, so a large, confidently-wrong region moves
the line much less than its area suggests.

**Overfitting is not a concern and that is worth stating**, because fitting a
correction against the same measurement you then evaluate against invites the
objection. Two parameters against ~300,000 constraints cannot memorise anything;
what the residual after the fit measures is the part of the disagreement that is
*shape*, which no affine correction can remove. `fit_mask` exists so that claim
can be checked on held-out pixels rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: ARCore depth outside this band is not evidence. Below it are flying pixels
#: and the phone's own body; above it the sensor is extrapolating and its
#: returns are sparser than its confidence suggests.
REF_RANGE_M = (0.15, 12.0)
#: Predicted depth at or below this is either a failed pixel or the model's
#: own zero — never a surface.
MIN_PRED_M = 0.05
#: Fewest valid pixels that may set a frame's correction. A frame pointed at a
#: window can fall below this, and its neighbours' fit is worth more than a
#: line through fifty pixels of doorframe.
MIN_PIXELS = 2000
#: Huber cutoff in robust standard deviations. 1.345 is the usual choice: 95%
#: efficiency against clean Gaussian noise, while bounding a single outlier's
#: pull.
HUBER_K = 1.345
#: A correction outside this is not a scale error, it is a failed prediction —
#: reject rather than apply, and let the caller fall back. Wide on purpose: the
#: real guard is `MIN_CORRELATION`, and this only catches numerical nonsense.
A_BOUNDS = (0.05, 20.0)
#: How much the prediction and the measurement must agree about a frame's
#: *shape* before a correction is worth applying.
#:
#: This is the guard that matters, and it exists because of a specific failure.
#: When a frame's prediction is uncorrelated with what the sensor measured, the
#: best-fitting line is `a → 0`: throw the prediction away and put the whole
#: frame at one constant depth. Measured on bundle_e4a5da81, six of 27 frames
#: did exactly that — `a` between 0.012 and 0.16 with `b` near 2.7 m — and each
#: one *lowered* its residual by doing so (kf_0007: 0.79 m → 0.15 m). Scored on
#: agreement with the sensor it is an improvement; as geometry it is a flat wall
#: where a room was. A residual test cannot see the difference. This can.
MIN_CORRELATION = 0.3


@dataclass(frozen=True)
class AffineFit:
    """`d_metric ≈ a·d_pred + b`, and how well that held."""

    a: float
    b: float
    #: Valid pixels the fit saw.
    pixels: int
    #: Fraction of them inside the Huber band at convergence — how much of the
    #: frame the line actually explains.
    inlier_fraction: float
    #: Robust (MAD) residual after the fit, in metres. This is the shape
    #: disagreement that survives, and the number the fit cannot improve.
    residual_m: float
    #: Weighted correlation between prediction and measurement over the pixels
    #: the fit trusted — how much the two agree about the frame's shape, before
    #: any question of scale. Low means no affine correction is meaningful.
    correlation: float = 1.0
    #: "depth" or "disparity" — the space the line was fitted in.
    space: str = "depth"

    def apply(self, pred: np.ndarray) -> np.ndarray:
        """Correct a prediction. Non-positive results become 0 — *no return*,
        the same convention the depth maps use — because a negative depth is
        not a near surface, it is the correction leaving its valid range."""
        out = _apply(pred.astype(np.float64), self.a, self.b, self.space)
        return np.where(np.isfinite(out) & (out > 0.0), out, 0.0).astype(np.float32)


def _to_space(d: np.ndarray, space: str) -> np.ndarray:
    if space == "depth":
        return d
    # Disparity. Monocular models are affine-invariant in inverse depth more
    # often than in depth, so which space this line lives in is a question to
    # answer by measurement, not by assumption.
    return 1.0 / np.maximum(d, MIN_PRED_M)


def _apply(pred: np.ndarray, a: float, b: float, space: str) -> np.ndarray:
    if space == "depth":
        return a * pred + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / (a / np.maximum(pred, MIN_PRED_M) + b)


#: Points drawn for the Theil–Sen start. 192 gives ~18k pairwise slopes, which
#: costs microseconds and is far past the point of diminishing returns.
_START_SAMPLES = 192


def _robust_start(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """A first line that outliers cannot choose, via the median pairwise slope.

    Iterative reweighting only removes outliers it can already see as residuals,
    so it inherits whatever line it starts from. Started at the unweighted fit,
    a quarter of the frame predicting through a window drags the line so far
    that every *inlier* looks like the outlier, and the reweighting converges
    onto the contamination and stays there — the failure this exists to prevent.
    Theil–Sen tolerates just under a third of the points being arbitrary, which
    covers a window wall or a mirror.
    """
    n = x.size
    if n > _START_SAMPLES:
        # Stratified by depth: even spacing through the x-sorted values keeps
        # near and far surfaces both represented, where a head slice would take
        # one wall.
        order = np.argsort(x, kind="stable")
        idx = np.linspace(0, n - 1, _START_SAMPLES).astype(np.intp)
        xs, ys = x[order][idx], y[order][idx]
    else:
        xs, ys = x, y

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    # Pairs at the same depth carry no slope information, and a saturated
    # region is entirely such pairs.
    usable = np.abs(dx) > 1e-9
    if not usable.any():
        return None
    a0 = float(np.median(dy[usable] / dx[usable]))
    if not np.isfinite(a0):
        return None
    b0 = float(np.median(y - a0 * x))
    if not np.isfinite(b0):
        return None
    return a0, b0


def fit_affine_depth(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    space: str = "depth",
    fit_mask: np.ndarray | None = None,
    min_pixels: int = MIN_PIXELS,
    iterations: int = 6,
) -> AffineFit | None:
    """Robustly fit one frame's prediction to its measured depth.

    `ref` is metric metres with 0 meaning no return, on the same pixel grid as
    `pred`. Returns None when there is not enough evidence or the fit lands
    outside `A_BOUNDS`, which the caller should treat as "leave this frame
    alone" rather than as an error.

    `fit_mask` restricts which pixels the line is fitted on without restricting
    what it is scored against — the hook for a held-out check.
    """
    if space not in ("depth", "disparity"):
        raise ValueError(f"space must be 'depth' or 'disparity', not {space!r}")
    if pred.shape != ref.shape:
        raise ValueError(f"pred {pred.shape} and ref {ref.shape} are different grids")

    lo, hi = REF_RANGE_M
    valid = (
        np.isfinite(pred) & np.isfinite(ref) & (ref > lo) & (ref < hi) & (pred > MIN_PRED_M)
    )
    if fit_mask is not None:
        valid = valid & fit_mask
    n = int(valid.sum())
    if n < min_pixels:
        return None

    x = _to_space(pred[valid].astype(np.float64), space)
    y = _to_space(ref[valid].astype(np.float64), space)

    start = _robust_start(x, y)
    if start is None:
        return None
    a, b = start

    w = np.ones_like(x)
    for _ in range(iterations):
        resid = y - (a * x + b)
        scale = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        if not np.isfinite(scale) or scale <= 1e-9:
            # A perfect (or degenerate) fit — reweighting has nothing left to do.
            break
        t = np.abs(resid) / (HUBER_K * scale)
        w = np.where(t <= 1.0, 1.0, 1.0 / np.maximum(t, 1e-12))

        sw = w.sum()
        if sw <= 0:
            return None
        mx = float((w * x).sum() / sw)
        my = float((w * y).sum() / sw)
        dx = x - mx
        vxx = float((w * dx * dx).sum())
        if not np.isfinite(vxx) or vxx <= 0:
            return None
        a = float((w * dx * (y - my)).sum() / vxx)
        b = float(my - a * mx)
        if not (np.isfinite(a) and np.isfinite(b)):
            return None

    if not (A_BOUNDS[0] <= a <= A_BOUNDS[1]):
        return None

    # Do the two agree about this frame's shape at all? Measured on the pixels
    # the fit ended up trusting, and on those only. Carrying the Huber weights
    # into this would not do: an outlier sits far from the mean in x, so even
    # at a small weight it inflates the x variance and drags the correlation
    # down — the rejected region getting a vote on whether the rest is sound.
    final = y - (a * x + b)
    final_scale = 1.4826 * float(np.median(np.abs(final - np.median(final))))
    if not np.isfinite(final_scale):
        return None
    # Floor rather than special-case: when the line fits its inliers exactly the
    # MAD is 0, and a zero band would either admit everything or nothing
    # depending on how the comparison is written. Neither is what "the pixels
    # the fit trusted" means, and admitting everything hands the outliers the
    # vote this measurement exists to deny them.
    band = HUBER_K * max(final_scale, 1e-6 * max(1.0, abs(float(np.median(y)))))
    inlier = np.abs(final) <= band
    if inlier.sum() < 2:
        return None
    xi, yi = x[inlier], y[inlier]
    cxx = float(np.var(xi))
    cyy = float(np.var(yi))
    correlation = (
        float(np.mean((xi - xi.mean()) * (yi - yi.mean())) / np.sqrt(cxx * cyy))
        if cxx > 0 and cyy > 0
        else 0.0
    )
    if not np.isfinite(correlation) or correlation < MIN_CORRELATION:
        return None

    # Score in metres regardless of the space fitted in, so the two are
    # comparable, and on every valid pixel rather than only the fitted ones.
    scored = (
        np.isfinite(pred) & np.isfinite(ref) & (ref > lo) & (ref < hi) & (pred > MIN_PRED_M)
    )
    corrected = _apply(pred[scored].astype(np.float64), a, b, space)
    metric_resid = ref[scored].astype(np.float64) - corrected
    ok = np.isfinite(metric_resid)
    if not ok.any():
        return None
    metric_resid = metric_resid[ok]
    residual_m = 1.4826 * float(
        np.median(np.abs(metric_resid - np.median(metric_resid)))
    )
    inliers = float(
        np.mean(np.abs(metric_resid) <= HUBER_K * max(residual_m, 1e-9))
    )

    return AffineFit(
        a=a,
        b=b,
        pixels=n,
        inlier_fraction=inliers,
        residual_m=residual_m,
        correlation=correlation,
        space=space,
    )
