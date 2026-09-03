"""Put the SfM result into the sensors' world: orientation, gravity and scale.

COLMAP returns poses in an arbitrary frame at an arbitrary scale. The rotation
vector sensor knows, for every keyframe, how the camera was oriented relative
to gravity (and to north, with the magnetometer). One global rotation aligns
the two — Wahba's problem, closed form by SVD — and the residual per frame is
a direct check of how much the images and the sensors agree.

Scale comes from the accelerometer. Over a short interval the IMU predicts a
displacement up to an unknown initial velocity; over three keyframes the
velocity term cancels in the second difference of positions, leaving a linear
relation between what SfM measured and what the IMU integrated. One robust
least squares over every triplet gives metres per COLMAP unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from videorecon.imu import CV_TO_GL, ENU_TO_Y_UP, double_integrate


@dataclass(frozen=True)
class RotationAlignment:
    #: Maps SfM world to the sensors' ENU world: R_enu = rotation @ R_sfm.
    rotation: np.ndarray
    residual_deg: np.ndarray
    median_deg: float
    p95_deg: float


def solve_rotation(target: np.ndarray, source: np.ndarray) -> RotationAlignment:
    """Rotation `R` minimising Σ ‖target_i − R source_i‖² over rotation matrices."""
    target = np.asarray(target, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if target.shape != source.shape or target.ndim != 3:
        raise ValueError("expected two (N, 3, 3) stacks")
    b = np.einsum("nij,nkj->ik", target, source)  # Σ T_i S_iᵀ
    u, _, vt = np.linalg.svd(b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    aligned = np.einsum("ij,njk->nik", r, source)
    rel = np.einsum("nji,njk->nik", target, aligned)  # T_iᵀ (R S_i)
    residual = np.rad2deg(Rotation.from_matrix(rel).magnitude())
    return RotationAlignment(
        r, residual, float(np.median(residual)), float(np.percentile(residual, 95))
    )


@dataclass(frozen=True)
class ScaleEstimate:
    metres_per_unit: float
    method: str
    triplets: int
    inliers: int
    #: Median absolute deviation of the inlier ratios, relative to the scale.
    dispersion: float
    note: str = ""

    @property
    def usable(self) -> bool:
        return (
            np.isfinite(self.metres_per_unit)
            and self.metres_per_unit > 0
            and self.inliers >= 8
            and np.isfinite(self.dispersion)
            and self.dispersion <= MAX_SCALE_DISPERSION
        )


#: Per-triplet ratios spreading more than this around the fit mean the IMU is
#: not measuring what the keyframes sampled. Measured on a real walk: 28.9 (a
#: 50x wrong answer); the synthetic smooth walk gives 0.05.
MAX_SCALE_DISPERSION = 0.35

#: Where a person holds a phone while filming a room, metres above the floor.
#: Measured: a 177 cm person at chest height is 1.30 m.
DEFAULT_CAMERA_HEIGHT_M = 1.3


def scale_from_floor(
    points_y_up: np.ndarray,
    camera_positions_y_up: np.ndarray,
    *,
    camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M,
    min_slab_fraction: float = 0.4,
) -> ScaleEstimate:
    """Metres per unit from the floor's distance below the cameras.

    The sensors give gravity, so 'down' is known exactly. Below the cameras the
    sparse points pile up in horizontal slabs: table tops, counters, seats, and
    lowest of all the floor. Tables are often *denser* than the floor, so the
    floor is the **lowest** slab that still holds at least `min_slab_fraction`
    of the densest slab's points; only outliers lie below it. The one
    assumption is the height the phone was held at: a 10 cm error is 7 % of
    scale, against an accelerometer that cannot tell walking steps from the path.
    """
    pts = np.asarray(points_y_up, dtype=np.float64).reshape(-1, 3)
    cams = np.asarray(camera_positions_y_up, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 200 or len(cams) < 3:
        return ScaleEstimate(float("nan"), "floor", 0, 0, float("nan"), "too few points")
    cam_y = float(np.median(cams[:, 1]))
    rel = pts[:, 1] - cam_y
    below = rel[rel < 0]
    if len(below) < 100:
        return ScaleEstimate(
            float("nan"), "floor", 0, 0, float("nan"), "no points below the cameras"
        )
    # Scale-free bins: 2 % of the distance from the cameras to the lowest real points.
    depth_below = float(-np.percentile(below, 2))
    width = 0.02 * depth_below
    edges = np.arange(-depth_below - width, 0.0, width)
    hist, edges = np.histogram(below, edges)
    # Smooth over three bins so a slab split across an edge still counts once.
    smooth = np.convolve(hist, np.ones(3), mode="same")
    threshold = min_slab_fraction * smooth.max()
    candidates = np.flatnonzero(smooth >= threshold)
    k = int(candidates.min())
    centre = 0.5 * (edges[k] + edges[k + 1])
    slab = (below >= centre - 1.5 * width) & (below <= centre + 1.5 * width)
    floor_rel = float(np.median(below[slab]))
    height_units = -floor_rel
    if height_units <= 0:
        return ScaleEstimate(float("nan"), "floor", 0, 0, float("nan"), "floor above the cameras")
    scale = camera_height_m / height_units
    cam_spread = float(np.ptp(cams[:, 1]) / height_units)
    outliers_below = float((below < floor_rel - 3 * width).mean())
    return ScaleEstimate(
        scale,
        "floor",
        int(len(below)),
        int(slab.sum()),
        cam_spread,
        note=f"camera height assumed {camera_height_m} m; floor {height_units:.2f} units below "
        f"the cameras; {outliers_below * 100:.1f}% of points lie below it; camera height "
        f"varies {cam_spread * 100:.0f}% along the walk",
    )


def scale_from_imu(
    positions: np.ndarray,
    times_ns: np.ndarray,
    imu_times_ns: np.ndarray,
    accel_world: np.ndarray,
    *,
    max_gap_s: float = 0.8,
    min_motion_units: float = 0.0,
) -> ScaleEstimate:
    """Metres per SfM unit from keyframe positions and world-frame acceleration.

    `positions` are camera centres in the *sensor-aligned* frame (same axes as
    `accel_world`), in SfM units; `times_ns` their capture times. For each run
    of three keyframes closer than `max_gap_s`:

        (p_k − p_i)/Δt_ik − (p_j − p_i)/Δt_ij  =  D_ik/Δt_ik − D_ij/Δt_ij  /  s

    where D are double integrals of acceleration from t_i. The left side is in
    SfM units, the right in metres, and `s` is what is wanted.

    Measured on a real hand-held walk this is unreliable: walking puts ~2 Hz
    step oscillations into the accelerometer that 4 Hz keyframes alias, and the
    per-triplet ratios spread by 30x. `ScaleEstimate.usable` rejects that
    spread, and the pipeline falls back to `scale_from_floor`.
    """
    positions = np.asarray(positions, dtype=np.float64)
    times_ns = np.asarray(times_ns, dtype=np.int64)
    n = len(positions)
    lhs, rhs = [], []
    for i in range(n - 2):
        j, k = i + 1, i + 2
        dt_ij = (times_ns[j] - times_ns[i]) * 1e-9
        dt_ik = (times_ns[k] - times_ns[i]) * 1e-9
        if dt_ij <= 0 or dt_ik <= dt_ij or dt_ik > max_gap_s:
            continue
        d_ij = double_integrate(imu_times_ns, accel_world, times_ns[i], times_ns[j])
        d_ik = double_integrate(imu_times_ns, accel_world, times_ns[i], times_ns[k])
        if not (np.all(np.isfinite(d_ij)) and np.all(np.isfinite(d_ik))):
            continue
        left = (positions[k] - positions[i]) / dt_ik - (positions[j] - positions[i]) / dt_ij
        right = d_ik / dt_ik - d_ij / dt_ij
        if np.linalg.norm(left) <= min_motion_units:
            continue
        lhs.append(left)
        rhs.append(right)
    if len(lhs) < 3:
        return ScaleEstimate(
            float("nan"), "imu", len(lhs), 0, float("nan"), "too few keyframe triplets"
        )
    lhs_arr = np.asarray(lhs)
    rhs_arr = np.asarray(rhs)
    num = np.einsum("ni,ni->n", lhs_arr, rhs_arr)
    den = np.einsum("ni,ni->n", lhs_arr, lhs_arr)
    strength = np.sqrt(den)
    strong = strength >= np.percentile(strength, 30)
    ratios = num[strong] / den[strong]
    positive = ratios > 0
    if positive.sum() < 3:
        return ScaleEstimate(
            float("nan"), "imu", len(lhs), 0, float("nan"), "IMU and SfM disagree on direction"
        )
    ratios = ratios[positive]
    median = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - median))) + 1e-12
    band = np.abs(ratios - median) <= 3.0 * 1.4826 * mad
    idx = np.flatnonzero(strong)[positive][band]
    s = float(num[idx].sum() / den[idx].sum())
    dispersion = float(np.median(np.abs(ratios[band] - s)) / s) if s > 0 else float("nan")
    return ScaleEstimate(s, "imu", len(lhs), int(band.sum()), dispersion)


@dataclass
class MetricPoses:
    """Poses and points in the repo frame: Y up, world<-camera, camera looks -Z."""

    #: image name -> (R_wc 3x3, t 3) in the repo (OpenGL, Y-up) convention.
    poses: dict[str, tuple[np.ndarray, np.ndarray]]
    points: np.ndarray
    rotation_to_enu: np.ndarray
    metres_per_unit: float
    diagnostics: dict = field(default_factory=dict)


def to_metric_frame(
    cam_from_world: dict[str, np.ndarray],
    points: np.ndarray,
    rotation_to_enu: np.ndarray,
    metres_per_unit: float,
    *,
    heading_deg: float = 0.0,
) -> MetricPoses:
    """Apply the alignment rotation and scale, then convert to the repo frame.

    `heading_deg` rotates the ENU world about up so that a chosen direction
    reads as north; with the game rotation vector the yaw is arbitrary, and the
    magnetometer track supplies this angle when it is trusted.
    """
    yaw = Rotation.from_euler("z", heading_deg, degrees=True).as_matrix()
    r_enu = yaw @ rotation_to_enu
    poses = {}
    for name, w2c in cam_from_world.items():
        r_wc_cv = w2c[:3, :3].T
        centre = -w2c[:3, :3].T @ w2c[:3, 3]
        r_wc_enu = r_enu @ r_wc_cv
        c_enu = metres_per_unit * (r_enu @ centre)
        r_gl = ENU_TO_Y_UP @ r_wc_enu @ CV_TO_GL
        t_gl = ENU_TO_Y_UP @ c_enu
        poses[name] = (r_gl, t_gl)
    pts = (
        (metres_per_unit * (r_enu @ np.asarray(points, dtype=np.float64).T)).T
        if len(points)
        else np.zeros((0, 3))
    )
    pts = (ENU_TO_Y_UP @ pts.T).T if len(pts) else pts
    return MetricPoses(poses, pts, r_enu, metres_per_unit)


def heading_offset_deg(r_game: np.ndarray, r_magnetic: np.ndarray) -> tuple[float, float]:
    """Yaw that turns the game-rotation-vector world into the magnetic one.

    Both tracks share gravity; they differ by a slowly drifting rotation about
    up. Returns the median yaw and the spread (p95 − p5) across samples.
    """
    rel = np.einsum("nij,nkj->nik", r_magnetic, r_game)  # R_mag R_gameᵀ
    yaw = np.rad2deg(np.arctan2(rel[:, 1, 0], rel[:, 0, 0]))
    yaw = np.unwrap(np.deg2rad(yaw))
    yaw = np.rad2deg(yaw)
    return float(np.median(yaw)), float(np.percentile(yaw, 95) - np.percentile(yaw, 5))
