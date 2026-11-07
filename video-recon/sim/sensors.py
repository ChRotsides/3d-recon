"""Camera and IMU simulation.

Camera axes are OpenCV (x right, y down, z forward); the body/IMU frame is x forward, y left,
z up. The mount is fixed: 0.50 m up on the forward axis, pitched 8 deg down, no offset forward.

The GT depth is z-depth, not ray length: the rays are cast with camera-frame direction
(x, y, 1), and Open3D returns t in units of that direction vector, so t *is* the z-depth.

The IMU is differentiated from the true pose sequence at the IMU rate (central differences,
so one sample of latency) rather than from an analytic model, which keeps it consistent with
whatever the vehicle and its suspension actually did.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .world import apply_sensor_effects

GRAVITY = 9.81
CAM_HEIGHT = 0.50
CAM_PITCH_DEG = 8.0
IMG_W, IMG_H = 640, 360
HFOV_DEG = 70.0

# noise densities: video-recon/vio/config/kalibr_imu_chain.yaml
ACC_N, GYR_N = 2.0e-2, 1.7e-3
ACC_W, GYR_W = 3.0e-3, 2.0e-4


def intrinsics(w: int = IMG_W, h: int = IMG_H, hfov_deg: float = HFOV_DEG) -> np.ndarray:
    fx = (w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return np.array([fx, fx, w / 2.0 - 0.5, h / 2.0 - 0.5])


def imu_cam_extrinsics(pitch_deg: float = CAM_PITCH_DEG, height: float = CAM_HEIGHT):
    """`R_bc` (OpenCV camera axes expressed in body axes) and the camera centre in body axes.

    With no pitch the camera's +z (forward) is body +x, its +x (right) is body -y and its +y
    (down) is body -z. Pitching the nose down is a rotation about body +y (left)."""
    r_bc0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    t = np.deg2rad(pitch_deg)
    ry = np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])
    return ry @ r_bc0, np.array([0.0, 0.0, height])


class Camera:
    def __init__(self, width: int = IMG_W, height: int = IMG_H, hfov_deg: float = HFOV_DEG,
                 pitch_deg: float = CAM_PITCH_DEG, mount_height: float = CAM_HEIGHT):
        self.width, self.height = width, height
        self.K = intrinsics(width, height, hfov_deg)
        self.R_bc, self.p_bc = imu_cam_extrinsics(pitch_deg, mount_height)
        fx, fy, cx, cy = self.K
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        self.rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1).reshape(-1, 3)

    def pose(self, R_wb: np.ndarray, p_wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return R_wb @ self.R_bc, p_wb + R_wb @ self.p_bc

    def base_from_camera(self, R_wc: np.ndarray, c_w: np.ndarray) -> tuple[float, float, float]:
        """The robot base pose implied by an estimated camera pose and the known mount."""
        R_wb = R_wc @ self.R_bc.T
        p = c_w - R_wb @ self.p_bc
        yaw = float(np.arctan2(R_wb[1, 0], R_wb[0, 0]))
        return float(p[0]), float(p[1]), yaw

    def render(self, world, R_wc: np.ndarray, c_w: np.ndarray, rng: np.random.Generator | None = None,
               noise: bool = True, gray: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(image, z-depth). `gray=True` returns an (H, W) luminance image instead of (H, W, 3);
        VIO only needs that, and it is a third of the shading work."""
        dirs = (self.rays @ R_wc.T).astype(np.float32)
        t, prim, nrm = world.cast(np.asarray(c_w, dtype=np.float32), dirs)
        ok = np.isfinite(t)
        shape = (self.height, self.width) if gray else (self.height, self.width, 3)
        if ok.all():   # a closed room: no misses, so the masking can be skipped
            depth = t
            hit = np.asarray(c_w, dtype=np.float32) + t[:, None] * dirs
            img = world.shade(hit, nrm, world.tri_mat[prim], world.tri_axis[prim], gray)
        else:
            depth = np.where(ok, t, 0.0).astype(np.float32)
            hit = np.asarray(c_w, dtype=np.float32) + t[ok, None] * dirs[ok]
            img = np.zeros((len(t),) if gray else (len(t), 3), dtype=np.float32)
            img[ok] = world.shade(hit, nrm[ok], world.tri_mat[prim[ok]], world.tri_axis[prim[ok]], gray)
        rng = rng or np.random.default_rng(0)
        out = apply_sensor_effects(img.reshape(shape), rng, 2.0 / 255.0 if noise else 0.0)
        return out, depth.reshape(self.height, self.width)


class ImuSim:
    """Synthesises IMU samples from the true body poses at `rate` Hz.

    Feed one pose per tick; a sample comes back one tick late (central differences need the
    next pose). `noise=0` gives a perfect IMU.
    """

    def __init__(self, rate: float = 200.0, noise: float = 1.0, seed: int = 0):
        self.dt = 1.0 / rate
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed + 4242)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self._h: list[tuple[float, np.ndarray, np.ndarray]] = []
        self.sigma_a = ACC_N * np.sqrt(rate)
        self.sigma_g = GYR_N * np.sqrt(rate)

    def sample(self, t: float, R_wb: np.ndarray, p_w: np.ndarray):
        """Returns (t, gyro(3), accel(3)) for the previous tick, or None while warming up."""
        self._h.append((t, R_wb.copy(), np.asarray(p_w, dtype=float).copy()))
        if len(self._h) < 3:
            return None
        (t0, R0, p0), (t1, R1, p1), (t2, R2, p2) = self._h[-3:]
        dt = self.dt
        a_w = (p2 - 2.0 * p1 + p0) / (dt * dt)
        gyro = Rotation.from_matrix(R0.T @ R2).as_rotvec() / (2.0 * dt)
        accel = R1.T @ (a_w + np.array([0.0, 0.0, GRAVITY]))
        if self.noise > 0:
            gyro = gyro + self.bg + self.rng.normal(0, self.sigma_g * self.noise, 3)
            accel = accel + self.ba + self.rng.normal(0, self.sigma_a * self.noise, 3)
            self.bg += self.rng.normal(0, GYR_W * np.sqrt(dt) * self.noise, 3)
            self.ba += self.rng.normal(0, ACC_W * np.sqrt(dt) * self.noise, 3)
        if len(self._h) > 4:
            self._h.pop(0)
        return t1, gyro, accel


def integrate_imu(samples, R0: np.ndarray, p0: np.ndarray, v0: np.ndarray, dt: float):
    """Leapfrog strapdown, the exact inverse of the central-difference synthesis above.
    Used by the tests to check the frames, signs and gravity of the synthesised IMU."""
    g = np.array([0.0, 0.0, -GRAVITY])
    Rs = [R0, R0 @ Rotation.from_rotvec(samples[0][1] * dt).as_matrix()]
    ps = [p0, p0 + v0 * dt + 0.5 * (Rs[0] @ samples[0][2] + g) * dt * dt]
    for k in range(1, len(samples)):
        _, w, a = samples[k]
        Rs.append(Rs[k - 1] @ Rotation.from_rotvec(w * 2.0 * dt).as_matrix())
        ps.append(2.0 * ps[k] - ps[k - 1] + (Rs[k] @ a + g) * dt * dt)
    return np.asarray(Rs), np.asarray(ps)
