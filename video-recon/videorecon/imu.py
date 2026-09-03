"""Orientation from the phone's own sensors, and the conventions that make it usable.

Android's rotation-vector sensors (`ROTATION_VECTOR` fuses gyro + accelerometer +
magnetometer; `GAME_ROTATION_VECTOR` leaves the magnetometer out) report a unit
quaternion whose rotation matrix maps *device* coordinates to an *East-North-Up*
world: `v_world = R_wd @ v_device`. That is the documented meaning of
`SensorManager.getRotationMatrixFromVector`, and the one property every other
convention here hangs off.

Device frame (Android): +X right, +Y up along the screen in portrait, +Z out of
the screen towards the user. The back camera looks along -Z.

Camera frame (OpenCV, what COLMAP and DA3 want): +x right in the image, +y down,
+z forward along the optical axis.

Repo frame (`recon_world_v1`, see harness/recon/frames.py): right-handed, +Y up,
world<-camera poses with the camera looking down -Z. Every pose that leaves this
package is in that frame so the existing depth adapter and fusion code apply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

#: ENU (x east, y north, z up) -> repo world (X east, Y up, Z south). Right-handed.
ENU_TO_Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

#: OpenCV camera axes -> OpenGL camera axes (and back; it is its own inverse).
CV_TO_GL = np.diag([1.0, -1.0, -1.0])

#: Standard gravity used to strip gravity from raw accelerometer samples.
GRAVITY_M_S2 = 9.80665


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of an Android rotation vector `(x, y, z, w)`.

    Reproduces `SensorManager.getRotationMatrixFromVector` exactly, so the result
    is `R_wd`: device -> East-North-Up world.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError("expected quaternions as (..., 4) in (x, y, z, w) order")
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-9):
        raise ValueError("zero-norm quaternion")
    return Rotation.from_quat(q / n).as_matrix()


def device_to_camera(sensor_orientation_deg: int) -> np.ndarray:
    """`R_dc`: OpenCV camera axes expressed in device axes, for a back camera.

    `sensor_orientation_deg` is `CameraCharacteristics.SENSOR_ORIENTATION`: the
    clockwise rotation that makes the sensor's native image upright in the
    device's natural (portrait) orientation. Video is recorded in the sensor's
    native orientation, so image +u/+v are the sensor's axes, not the screen's.

    Derivation: rotating an image clockwise by θ sends the sensor's +u axis to
    `(cos θ, sin θ)` and +v to `(-sin θ, cos θ)` in upright image coordinates;
    upright +u is device +X and upright +v is device -Y; the optical axis is -Z.
    """
    theta = np.deg2rad(sensor_orientation_deg % 360)
    c, s = np.cos(theta), np.sin(theta)
    x_cam = np.array([c, -s, 0.0])
    y_cam = np.array([-s, -c, 0.0])
    z_cam = np.array([0.0, 0.0, -1.0])
    r = np.stack([x_cam, y_cam, z_cam], axis=1)
    assert np.isclose(np.linalg.det(r), 1.0), "camera axes must be right-handed"
    return r


def camera_to_world_gl(r_wd: np.ndarray, sensor_orientation_deg: int) -> np.ndarray:
    """Repo-frame rotation of the camera (world<-camera, camera looks -Z, Y up)."""
    r_wc_cv = ENU_TO_Y_UP @ r_wd @ device_to_camera(sensor_orientation_deg)
    return r_wc_cv @ CV_TO_GL


def gravity_in_device(r_wd: np.ndarray) -> np.ndarray:
    """Unit vector pointing *down* in device axes, from the orientation."""
    return r_wd.T @ np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class OrientationTrack:
    """Timestamped device orientations, interpolable by slerp."""

    timestamps_ns: np.ndarray
    quaternions_xyzw: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.timestamps_ns)
        if t.ndim != 1 or len(t) < 2:
            raise ValueError("an orientation track needs at least two samples")
        if np.any(np.diff(t) <= 0):
            raise ValueError("orientation timestamps must be strictly increasing")
        if self.quaternions_xyzw.shape != (len(t), 4):
            raise ValueError("quaternions must be (N, 4)")

    @classmethod
    def from_samples(cls, timestamps_ns: np.ndarray, quats: np.ndarray) -> OrientationTrack:
        t = np.asarray(timestamps_ns, dtype=np.int64)
        q = np.asarray(quats, dtype=np.float64)
        order = np.argsort(t, kind="stable")
        t, q = t[order], q[order]
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, q = t[keep], q[keep]
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        # Keep consecutive quaternions in the same hemisphere so slerp takes the
        # short way round; the sensor is free to flip sign between samples.
        for i in range(1, len(q)):
            if np.dot(q[i - 1], q[i]) < 0:
                q[i] = -q[i]
        return cls(t, q)

    @property
    def span_ns(self) -> tuple[int, int]:
        return int(self.timestamps_ns[0]), int(self.timestamps_ns[-1])

    def rotation_at(self, timestamps_ns: np.ndarray) -> np.ndarray:
        """`R_wd` at each query time, slerped; clamped at the ends of the track."""
        t = np.asarray(timestamps_ns, dtype=np.float64)
        lo, hi = self.span_ns
        t = np.clip(t, lo, hi)
        slerp = Slerp(
            self.timestamps_ns.astype(np.float64), Rotation.from_quat(self.quaternions_xyzw)
        )
        return slerp(t).as_matrix()

    def angular_rate_deg_s(self) -> np.ndarray:
        """Angular speed between consecutive samples, one value per interval."""
        rot = Rotation.from_quat(self.quaternions_xyzw)
        rel = rot[:-1].inv() * rot[1:]
        dt = np.diff(self.timestamps_ns) * 1e-9
        return np.rad2deg(rel.magnitude()) / dt


def world_linear_acceleration(
    r_wd: np.ndarray, accel_device: np.ndarray, gravity: float = GRAVITY_M_S2
) -> np.ndarray:
    """Accelerometer samples rotated into ENU with gravity removed.

    The raw accelerometer measures specific force: at rest it reads +g *up* in
    device axes (the reaction to gravity). Rotating to world and subtracting
    `(0, 0, g)` leaves the true linear acceleration.
    """
    a_world = np.einsum("nij,nj->ni", r_wd, accel_device)
    a_world[:, 2] -= gravity
    return a_world


def double_integrate(
    timestamps_ns: np.ndarray, accel_world: np.ndarray, t0_ns: int, t1_ns: int
) -> np.ndarray:
    """`∬ a dt²` from `t0` to `t1`, trapezoidal, zero initial velocity.

    Used by the scale estimator: the displacement the IMU predicts over a short
    interval, up to the unknown initial velocity term that the caller cancels.
    """
    t = timestamps_ns.astype(np.float64) * 1e-9
    mask = (timestamps_ns >= t0_ns) & (timestamps_ns <= t1_ns)
    if mask.sum() < 3:
        return np.full(3, np.nan)
    ts = t[mask]
    a = accel_world[mask]
    dt = np.diff(ts)
    v = np.concatenate([[np.zeros(3)], np.cumsum(0.5 * (a[1:] + a[:-1]) * dt[:, None], axis=0)])
    p = np.sum(0.5 * (v[1:] + v[:-1]) * dt[:, None], axis=0)
    return p
