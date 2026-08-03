"""Room-to-room alignment (PLAN.md §6.3, path B).

Because gravity is known from the IMU, joining two independently scanned rooms
is only **3 DoF**: yaw about the vertical, plus a translation in the floor
plane. That is the single biggest simplification in the whole stitching problem
and it is why `gravity` is a required field on the bundle.

    1. Match the same doorway in both rooms (user-tapped, or auto-detected).
    2. Yaw from the door-plane normals.
    3. Translation from the threshold centroids and the floor planes.
    4. Point-to-plane ICP within 2 m of the doorway, initialised from 2-3.
       Reject if RMSE > 5 cm and ask the user to re-tag.

Step 2 has a sign subtlety worth stating: the two rooms see the *same* physical
door from *opposite* sides, so their inward normals are anti-parallel when the
rooms are correctly aligned. The rotation we want takes `n_b` to `-n_a`, not to
`n_a`. Getting this wrong stacks the two rooms on top of each other instead of
placing them either side of the door, and it looks plausible enough in a top-down
view to survive a casual check — so it is asserted in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recon.layout.doorways import doorway_frame
from recon.schemas import Doorway, EdgeSource, RoomEdge, SE2Transform

#: PLAN.md §6.3 step 4.
ICP_REJECT_RMSE_M = 0.05
ICP_NEIGHBOURHOOD_M = 2.0


@dataclass
class Alignment:
    """The transform taking room B's frame into room A's frame."""

    yaw_rad: float
    tx: float
    tz: float
    rmse_m: float
    inliers: int
    #: 3x3 covariance over (yaw, tx, tz), derived from the ICP residuals.
    covariance: np.ndarray
    accepted: bool
    reason: str = ""

    def matrix(self, up: np.ndarray | None = None) -> np.ndarray:
        return SE2Transform(yaw_rad=self.yaw_rad, tx=self.tx, tz=self.tz).matrix()

    def to_edge(
        self, from_room: str, to_room: str, dw_a: str, dw_b: str, source: EdgeSource
    ) -> RoomEdge:
        return RoomEdge(
            from_room=from_room,
            to_room=to_room,
            via_doorway=[dw_a, dw_b],
            transform=SE2Transform(yaw_rad=self.yaw_rad, tx=self.tx, tz=self.tz),
            covariance=self.covariance.tolist(),
            source=source,
            icp_rmse_m=self.rmse_m,
        )


def yaw_between(
    n_a: np.ndarray, n_b: np.ndarray, up: np.ndarray, *, opposing: bool = True
) -> float:
    """Yaw that rotates `n_b` onto `-n_a` (or onto `n_a` if `opposing=False`).

    PLAN.md §6.3 writes this as `psi = atan2((n_a x n_b) . g, n_a . n_b)`. Two
    corrections are folded in here:

    * The plan's expression, with gravity (down) rather than up, gives the yaw
      of the *opposite* sense. Using `up` keeps it consistent with SE2Transform,
      which rotates about +Y.
    * The two rooms see the door from opposite sides, so the target is `-n_a`.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    target = -np.asarray(n_a, dtype=np.float64) if opposing else np.asarray(n_a, np.float64)
    a = _project(target, up)
    b = _project(np.asarray(n_b, dtype=np.float64), up)
    return float(np.arctan2(float(np.cross(b, a) @ up), float(b @ a)))


def _project(v: np.ndarray, up: np.ndarray) -> np.ndarray:
    w = v - up * float(v @ up)
    n = float(np.linalg.norm(w))
    return w / n if n > 1e-12 else w


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def align_doorways(
    dw_a: Doorway,
    dw_b: Doorway,
    up: np.ndarray,
    points_a: np.ndarray | None = None,
    points_b: np.ndarray | None = None,
    normals_a: np.ndarray | None = None,
    *,
    refine: bool = True,
    reject_rmse_m: float = ICP_REJECT_RMSE_M,
) -> Alignment:
    """Solve the 3-DoF transform taking room B into room A's frame.

    `points_*` are the rooms' fused surface clouds. When supplied, ICP refines
    the closed-form solution using geometry within `ICP_NEIGHBOURHOOD_M` of the
    doorway — near the door the two rooms genuinely overlap (the jamb, the
    threshold, the floor), which is exactly where ICP has signal.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    c_a, n_a, _ = doorway_frame(dw_a, up)
    c_b, n_b, _ = doorway_frame(dw_b, up)

    yaw = yaw_between(n_a, n_b, up)
    r = _yaw_matrix(yaw)
    # Translation places B's threshold centroid onto A's.
    t = c_a - r @ c_b

    align = Alignment(
        yaw_rad=yaw,
        tx=float(t[0]),
        tz=float(t[2]),
        rmse_m=float("nan"),
        inliers=0,
        covariance=_default_covariance(),
        accepted=True,
        reason="closed-form doorway match",
    )

    if not refine or points_a is None or points_b is None or len(points_a) == 0:
        return align

    refined = icp_point_to_plane(
        points_a, normals_a, points_b, align, up, centre=c_a,
        radius=ICP_NEIGHBOURHOOD_M,
    )
    if refined is None:
        align.reason = "no overlap near the doorway; kept the closed-form solution"
        return align

    refined.accepted = refined.rmse_m <= reject_rmse_m
    if not refined.accepted:
        refined.reason = (
            f"ICP RMSE {refined.rmse_m*100:.1f} cm exceeds the "
            f"{reject_rmse_m*100:.0f} cm gate; ask the user to re-tag the doorway"
        )
    else:
        refined.reason = "ICP refined"
    return refined


def icp_point_to_plane(
    points_a: np.ndarray,
    normals_a: np.ndarray | None,
    points_b: np.ndarray,
    init: Alignment,
    up: np.ndarray,
    *,
    centre: np.ndarray,
    radius: float = ICP_NEIGHBOURHOOD_M,
    iterations: int = 30,
    max_correspondence_m: float = 0.30,
    min_points: int = 50,
) -> Alignment | None:
    """3-DoF point-to-plane ICP, restricted to a neighbourhood of the doorway.

    Solving in 3 DoF rather than 6 is not just cheaper — it is what stops ICP
    from tilting one room relative to the other to explain depth noise. Gravity
    already fixed the other three degrees of freedom and they should stay fixed.
    """
    from scipy.spatial import cKDTree

    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)

    near_a = np.linalg.norm(a - centre, axis=1) <= radius
    if near_a.sum() < min_points:
        return None
    a_near = a[near_a]
    n_near = None
    if normals_a is not None:
        n_near = np.asarray(normals_a, dtype=np.float64)[near_a]
        n_len = np.linalg.norm(n_near, axis=1, keepdims=True)
        n_near = n_near / np.maximum(n_len, 1e-12)

    tree = cKDTree(a_near)
    yaw, tx, tz = init.yaw_rad, init.tx, init.tz
    rmse = float("nan")
    used = 0

    for _ in range(iterations):
        r = _yaw_matrix(yaw)
        moved = b @ r.T + np.array([tx, 0.0, tz])
        # Only B points that landed near the doorway can correspond.
        near_b = np.linalg.norm(moved - centre, axis=1) <= radius
        if near_b.sum() < min_points:
            return None
        src = moved[near_b]

        dist, idx = tree.query(src, k=1)
        keep = dist <= max_correspondence_m
        if keep.sum() < min_points:
            return None
        src = src[keep]
        dst = a_near[idx[keep]]
        nrm = n_near[idx[keep]] if n_near is not None else None
        used = int(keep.sum())

        delta = _solve_3dof_step(src, dst, nrm)
        if delta is None:
            break
        d_yaw, d_tx, d_tz = delta

        # Compose the incremental transform on the left of the current one.
        dr = _yaw_matrix(d_yaw)
        t_new = dr @ np.array([tx, 0.0, tz]) + np.array([d_tx, 0.0, d_tz])
        yaw, tx, tz = yaw + d_yaw, float(t_new[0]), float(t_new[2])

        residual = _residual(src, dst, nrm, d_yaw, d_tx, d_tz)
        rmse = float(np.sqrt(np.mean(residual**2)))
        if abs(d_yaw) < 1e-7 and abs(d_tx) < 1e-6 and abs(d_tz) < 1e-6:
            break

    return Alignment(
        yaw_rad=float(yaw),
        tx=float(tx),
        tz=float(tz),
        rmse_m=rmse,
        inliers=used,
        covariance=_covariance_from_residuals(rmse, used),
        accepted=True,
    )


def _solve_3dof_step(
    src: np.ndarray, dst: np.ndarray, nrm: np.ndarray | None
) -> tuple[float, float, float] | None:
    """One linearised point-to-plane step over (d_yaw, d_tx, d_tz).

    For a small yaw about +Y, a point p moves by d_yaw * (up x p) = d_yaw *
    (p_z, 0, -p_x). Projecting onto the target normal gives a 3-column linear
    system; with no normals available it degrades to point-to-point.
    """
    if nrm is None:
        # Point-to-point: solve x and z translation and yaw by least squares on
        # the full residual vector.
        jac = np.zeros((len(src) * 2, 3))
        res = np.zeros(len(src) * 2)
        jac[0::2, 0] = src[:, 2]
        jac[0::2, 1] = 1.0
        jac[1::2, 0] = -src[:, 0]
        jac[1::2, 2] = 1.0
        res[0::2] = dst[:, 0] - src[:, 0]
        res[1::2] = dst[:, 2] - src[:, 2]
    else:
        rot_dir = np.column_stack([src[:, 2], np.zeros(len(src)), -src[:, 0]])
        jac = np.column_stack([
            np.einsum("ij,ij->i", rot_dir, nrm),
            nrm[:, 0],
            nrm[:, 2],
        ])
        res = np.einsum("ij,ij->i", dst - src, nrm)

    try:
        sol, *_ = np.linalg.lstsq(jac, res, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(sol).all():
        return None
    # Cap the step so a bad correspondence set cannot throw the solution.
    d_yaw = float(np.clip(sol[0], -0.2, 0.2))
    d_tx = float(np.clip(sol[1], -0.5, 0.5))
    d_tz = float(np.clip(sol[2], -0.5, 0.5))
    return d_yaw, d_tx, d_tz


def _residual(
    src: np.ndarray,
    dst: np.ndarray,
    nrm: np.ndarray | None,
    d_yaw: float,
    d_tx: float,
    d_tz: float,
) -> np.ndarray:
    moved = src @ _yaw_matrix(d_yaw).T + np.array([d_tx, 0.0, d_tz])
    if nrm is None:
        return np.linalg.norm(dst - moved, axis=1)
    return np.einsum("ij,ij->i", dst - moved, nrm)


def _default_covariance() -> np.ndarray:
    """Covariance for an unrefined, doorway-only match.

    Deliberately loose: a closed-form match from two detected door planes is
    worth much less than an ICP-verified one, and the pose graph must be told
    so or it will trust them equally.
    """
    return np.diag([np.radians(4.0) ** 2, 0.05**2, 0.05**2])


def _covariance_from_residuals(rmse: float, n: int) -> np.ndarray:
    if not np.isfinite(rmse) or n <= 0:
        return _default_covariance()
    sigma = max(rmse, 0.005)
    # Angular uncertainty scales with the residual over the neighbourhood
    # radius: the same positional error over a longer lever arm is less yaw.
    sigma_yaw = max(sigma / ICP_NEIGHBOURHOOD_M, np.radians(0.2))
    scale = 1.0 / max(np.sqrt(n), 1.0)
    return np.diag([(sigma_yaw * scale) ** 2, (sigma * scale) ** 2, (sigma * scale) ** 2])


def match_doorways(
    doorways_a: list[Doorway],
    doorways_b: list[Doorway],
    *,
    width_tolerance_m: float = 0.12,
    height_tolerance_m: float = 0.20,
) -> list[tuple[int, int, float]]:
    """Auto-pair doorways between two rooms by physical size.

    A weak signal on its own — a house has many 0.83 m doors — so this returns
    *ranked candidates*, not a decision. §6.3 resolves the ambiguity with the
    user's tap or with MapAnything registration; this narrows the field.
    """
    out: list[tuple[int, int, float]] = []
    for i, a in enumerate(doorways_a):
        for j, b in enumerate(doorways_b):
            dw = abs(a.width_m - b.width_m)
            dh = abs(a.height_m - b.height_m)
            if dw > width_tolerance_m or dh > height_tolerance_m:
                continue
            out.append((i, j, float(dw + 0.5 * dh)))
    out.sort(key=lambda t: t[2])
    return out
