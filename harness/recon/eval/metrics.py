"""Evaluation metrics (PLAN.md §8).

Every number here maps to a row of the §1.3 success criteria. Wall-length error
is the one the build gates on: a regression of more than 0.5 percentage points
fails CI.

Wall matching is **geometric, not index-based**. The contour tracer picks its
own starting vertex and winding, so a predicted footprint is only ever equal to
ground truth up to a cyclic rotation. Comparing wall 0 to wall 0 would report
enormous errors for a perfect reconstruction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from recon import geom
from recon.schemas import RoomLayout


@dataclass
class GeometryMetrics:
    chamfer_m: float
    accuracy_m: float  # pred -> gt, median
    completeness_m: float  # gt -> pred, median
    accuracy_p95_m: float
    completeness_p95_m: float
    #: Fraction of ground-truth surface within 5 cm of the reconstruction.
    recall_5cm: float
    precision_5cm: float


@dataclass
class LayoutMetrics:
    area_error_pct: float
    perimeter_error_pct: float
    ceiling_error_m: float
    #: Signed relative error per matched wall, percent.
    wall_errors_pct: list[float] = field(default_factory=list)
    wall_error_mean_abs_pct: float = 0.0
    wall_error_max_abs_pct: float = 0.0
    #: Largest absolute wall error in metres — the §1.3 target is the *larger*
    #: of 1.5% and 3 cm, so both forms are reported.
    wall_error_max_abs_m: float = 0.0
    n_walls_pred: int = 0
    n_walls_gt: int = 0
    matched_walls: int = 0
    corner_rmse_m: float = 0.0
    openings_pred: int = 0
    openings_gt: int = 0

    def passes(self) -> bool:
        """PLAN.md §1.3: wall length <=1.5% or <=3 cm, area <=3%."""
        wall_ok = self.wall_error_max_abs_pct <= 1.5 or self.wall_error_max_abs_m <= 0.03
        return wall_ok and abs(self.area_error_pct) <= 3.0 and self.n_walls_pred > 0


@dataclass
class PhotometricMetrics:
    psnr_db: float
    ssim: float
    n_views: int


def chamfer(
    pred: np.ndarray, gt: np.ndarray, threshold_m: float = 0.05
) -> GeometryMetrics:
    """Bidirectional point-cloud distances. `pred`/`gt` are (N, 3) samples."""
    from scipy.spatial import cKDTree

    if len(pred) == 0 or len(gt) == 0:
        inf = float("inf")
        return GeometryMetrics(inf, inf, inf, inf, inf, 0.0, 0.0)

    d_pred = cKDTree(gt).query(pred, k=1)[0]  # accuracy
    d_gt = cKDTree(pred).query(gt, k=1)[0]  # completeness
    return GeometryMetrics(
        chamfer_m=float(d_pred.mean() + d_gt.mean()) / 2.0,
        accuracy_m=float(np.median(d_pred)),
        completeness_m=float(np.median(d_gt)),
        accuracy_p95_m=float(np.percentile(d_pred, 95)),
        completeness_p95_m=float(np.percentile(d_gt, 95)),
        recall_5cm=float((d_gt < threshold_m).mean()),
        precision_5cm=float((d_pred < threshold_m).mean()),
    )


def match_walls(
    pred_poly: np.ndarray, gt_poly: np.ndarray
) -> list[tuple[int, int]]:
    """Match predicted footprint edges to ground-truth edges.

    Greedy on midpoint distance with a direction-agreement gate — robust to the
    cyclic rotation and winding flip the contour tracer can introduce, and it
    degrades gracefully when the two polygons have different vertex counts.
    """
    if len(pred_poly) < 3 or len(gt_poly) < 3:
        return []

    def edges(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = p
        b = np.roll(p, -1, axis=0)
        mid = 0.5 * (a + b)
        d = b - a
        n = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        return mid, d / n

    pm, pd = edges(pred_poly)
    gm, gd = edges(gt_poly)
    cost = np.linalg.norm(pm[:, None, :] - gm[None, :, :], axis=2)
    align = np.abs(np.einsum("ik,jk->ij", pd, gd))
    cost = np.where(align > np.cos(np.radians(25.0)), cost, np.inf)

    pairs: list[tuple[int, int]] = []
    used_p, used_g = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for i, j in order:
        if not np.isfinite(cost[i, j]):
            break
        if i in used_p or j in used_g:
            continue
        pairs.append((int(i), int(j)))
        used_p.add(int(i))
        used_g.add(int(j))
    return pairs


def layout_metrics(layout: RoomLayout, gt: dict) -> LayoutMetrics:
    """Compare a reconstructed layout with a synthetic scene's ground truth."""
    pred_poly = np.asarray(layout.footprint, dtype=np.float64)
    gt_poly = geom.ensure_ccw(np.asarray(gt["footprint"], dtype=np.float64))

    gt_area = float(gt["area_m2"])
    gt_perim = float(gt["perimeter_m"])
    gt_ceiling = float(gt["ceiling_height_m"])

    m = LayoutMetrics(
        area_error_pct=(layout.area_m2 / gt_area - 1.0) * 100.0 if gt_area else float("inf"),
        perimeter_error_pct=(layout.perimeter_m / gt_perim - 1.0) * 100.0
        if gt_perim
        else float("inf"),
        ceiling_error_m=layout.ceiling_height_m - gt_ceiling,
        n_walls_pred=len(pred_poly),
        n_walls_gt=len(gt_poly),
        openings_pred=len(layout.openings),
        openings_gt=len(gt.get("openings", [])),
    )
    if len(pred_poly) < 3:
        m.wall_error_max_abs_pct = float("inf")
        m.wall_error_max_abs_m = float("inf")
        return m

    pred_len = geom.edge_lengths(pred_poly)
    gt_len = geom.edge_lengths(gt_poly)
    pairs = match_walls(pred_poly, gt_poly)
    errs_pct, errs_m = [], []
    for i, j in pairs:
        if gt_len[j] < 1e-6:
            continue
        errs_pct.append((pred_len[i] / gt_len[j] - 1.0) * 100.0)
        errs_m.append(pred_len[i] - gt_len[j])

    m.matched_walls = len(errs_pct)
    m.wall_errors_pct = [float(e) for e in errs_pct]
    if errs_pct:
        m.wall_error_mean_abs_pct = float(np.mean(np.abs(errs_pct)))
        m.wall_error_max_abs_pct = float(np.max(np.abs(errs_pct)))
        m.wall_error_max_abs_m = float(np.max(np.abs(errs_m)))
    else:
        m.wall_error_max_abs_pct = float("inf")
        m.wall_error_max_abs_m = float("inf")

    # Corner RMSE, over the matched-edge start vertices.
    if pairs:
        d = [
            np.linalg.norm(pred_poly[i] - gt_poly[j])
            for i, j in pairs
        ]
        m.corner_rmse_m = float(np.sqrt(np.mean(np.square(d))))
    return m


# ---------------------------------------------------------------------------
# Photometric (held-out novel views, PLAN.md §1.3 >= 26 dB)
# ---------------------------------------------------------------------------


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(max_val * max_val / mse))


def ssim(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    try:
        from skimage.metrics import structural_similarity

        a = np.asarray(a)
        b = np.asarray(b)
        channel_axis = 2 if a.ndim == 3 else None
        return float(
            structural_similarity(a, b, data_range=max_val, channel_axis=channel_axis)
        )
    except ImportError:  # pragma: no cover
        return float("nan")


def photometric_metrics(
    rendered: list[np.ndarray], reference: list[np.ndarray]
) -> PhotometricMetrics:
    if not rendered:
        return PhotometricMetrics(psnr_db=float("nan"), ssim=float("nan"), n_views=0)
    ps = [psnr(r, g) for r, g in zip(rendered, reference, strict=True)]
    ss = [ssim(r, g) for r, g in zip(rendered, reference, strict=True)]
    return PhotometricMetrics(
        psnr_db=float(np.mean([p for p in ps if np.isfinite(p)])),
        ssim=float(np.nanmean(ss)),
        n_views=len(rendered),
    )


def as_dict(obj) -> dict:
    return asdict(obj)
