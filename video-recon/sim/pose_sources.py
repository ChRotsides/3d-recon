"""Where the robot's pose estimate comes from. Every source returns the same thing:

    (t, R_cw, c_w, ok)   camera->world rotation (OpenCV axes) and camera centre, in the
                         source's own frame, which for `vio` and `gt_noisy` drifts away from
                         the world frame. The navigation stack only ever sees this.

Ground truth is passed in so the sources can *derive* an estimate from it, but only `gt`
copies it through; the map and the planner never see the true pose.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_DIR = Path(__file__).resolve().parent
IMU_LEAD = 0.05   # s of IMU the estimator must already hold beyond a frame's timestamp


class GtPose:
    name = "gt"

    def __init__(self, **_):
        self.initialised = True
        self.init_time = 0.0

    def update(self, t, R_cw, c_w, img_gray=None, imu=()):
        return t, R_cw, np.asarray(c_w, dtype=float), True

    def close(self):
        pass


class GtNoisyPose:
    """Odometry-like drift: the true frame-to-frame motion, corrupted, then integrated.

    Translation noise is 2 % of the step plus 3 mm; rotation noise is 0.3 deg/s plus 1 % of the
    angle turned; `scale_error` multiplies every translation, which is the error a monocular
    system makes when its scale is wrong.
    """

    name = "gt_noisy"

    def __init__(self, seed: int = 0, scale_error: float = 1.0, **_):
        self.rng = np.random.default_rng(seed + 31337)
        self.scale = float(scale_error)
        self.initialised = True
        self.init_time = 0.0
        self._prev = None
        self._R = None
        self._c = None

    def update(self, t, R_cw, c_w, img_gray=None, imu=()):
        c_w = np.asarray(c_w, dtype=float)
        if self._prev is None:
            self._prev = (t, R_cw.copy(), c_w.copy())
            self._R, self._c = R_cw.copy(), c_w.copy()
            return t, self._R, self._c, True
        t0, R0, c0 = self._prev
        dt = max(1e-6, t - t0)
        d_c = R0.T @ (c_w - c0)  # step in the previous camera frame
        d_R = R0.T @ R_cw
        rv = Rotation.from_matrix(d_R).as_rotvec()
        step = float(np.linalg.norm(d_c))
        ang = float(np.linalg.norm(rv))
        d_c = d_c * self.scale + self.rng.normal(0.0, 0.02 * step + 0.003, 3)
        rv = rv + self.rng.normal(0.0, np.deg2rad(0.3) * dt + 0.01 * ang, 3)
        self._c = self._c + self._R @ d_c
        self._R = self._R @ Rotation.from_rotvec(rv).as_matrix()
        self._prev = (t, R_cw.copy(), c_w.copy())
        return t, self._R, self._c, True

    def close(self):
        pass


class VioPose:
    """Real OpenVINS through the streaming runner (video-recon/vio/vio_stream.cpp).

    IMU samples are pushed as they are produced; each camera frame is written to a PNG in the
    run directory and *held* until the estimator already has IMU past its timestamp, then
    handed over and exactly one reply (P+L or N) read back. Poses come out as the IMU pose in
    OpenVINS' gravity-aligned global frame; the nominal mount converts them to the camera. The
    frame's PNG is deleted once consumed.
    """

    name = "vio"

    @staticmethod
    def write_config(out_dir, config: str, R_bc, p_bc, K, width: int, height: int) -> Path:
        """Materialise the estimator config for *this* camera into the run directory.

        The checked-in yaml is written for 640x360; deriving the chain here from sensors.Camera
        means the resolution, the intrinsics and the mount can never drift out of step with the
        simulator that produced the frames, and any --width/--height works unchanged.
        """
        d = Path(out_dir) / "vio_config"
        d.mkdir(parents=True, exist_ok=True)
        src = _DIR / "config"
        for n in (config, "kalibr_imu_chain.yaml"):
            shutil.copy(src / n, d / n)
        rows = "\n".join("    - [%s, %.9f]" % (", ".join(f"{v:.9f}" for v in R_bc[i]), p_bc[i])
                         for i in range(3))
        (d / "kalibr_imucam_chain.yaml").write_text(
            "%YAML:1.0\n\n"
            "# Written by pose_sources.VioPose.write_config from sensors.Camera.\n"
            "# OpenVINS reads `T_imu_cam` as T_CtoI: the 3x3 block is R_bc (OpenCV camera axes\n"
            "# expressed in body/IMU axes) and the translation is the camera centre in body axes.\n"
            "cam0:\n  T_imu_cam:\n" + rows + "\n    - [0.0, 0.0, 0.0, 1.0]\n"
            "  cam_overlaps: []\n  camera_model: pinhole\n"
            "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n  distortion_model: radtan\n"
            f"  intrinsics: [{K[0]:.4f}, {K[1]:.4f}, {K[2]:.4f}, {K[3]:.4f}]\n"
            f"  resolution: [{width}, {height}]\n  rostopic: /cam0/image_raw\n")
        return d / config

    def __init__(self, out_dir, R_bc, p_bc, config: str = "estimator_config.yaml", seed: int = 0,
                 record: bool = False, K=None, width: int = 640, height: int = 360, **_):
        import cv2

        self._cv2 = cv2
        self.R_bc = np.asarray(R_bc, dtype=float)
        self.p_bc = np.asarray(p_bc, dtype=float)
        self.frames_dir = Path(out_dir) / "vio_frames"
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        cfg = (self.write_config(out_dir, config, self.R_bc, self.p_bc, K, width, height)
               if K is not None else _DIR / "config" / config)
        self.proc = subprocess.Popen(
            [str(_DIR / "vio_bin.sh"), str(cfg)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=open(Path(out_dir) / "vio.log", "wb"), text=True, bufsize=1,
        )
        self.initialised = False
        self.init_time = None
        self.landmarks = np.zeros((0, 3))
        self.calib = None
        self.frames = 0
        self._imu_t = -1e9
        self._pending: list = []
        self._R = np.eye(3)
        self._c = np.zeros(3)
        self._t = 0.0
        # optional: keep the exact stream in run_offline's on-disk format, so the same input can
        # be replayed offline and the streaming plumbing ruled in or out
        self.rec = Path(out_dir) / "vio_input" if record else None
        if self.rec:
            (self.rec / "cam0").mkdir(parents=True, exist_ok=True)
            self.rec_imu = open(self.rec / "imu.csv", "w")
            self.rec_cam = open(self.rec / "cam0.csv", "w")

    def feed_imu(self, t, gyro, accel):
        self._imu_t = max(self._imu_t, t)
        if self.rec:
            self.rec_imu.write("%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n"
                               % (int(round(t * 1e9)), gyro[0], gyro[1], gyro[2],
                                  accel[0], accel[1], accel[2]))
        self.proc.stdin.write(
            "I %d %.9f %.9f %.9f %.9f %.9f %.9f\n"
            % (int(round(t * 1e9)), gyro[0], gyro[1], gyro[2], accel[0], accel[1], accel[2])
        )

    def update(self, t, R_cw_gt, c_w_gt, img_gray=None, imu=()):
        for s in imu:
            self.feed_imu(*s)
        path = self.frames_dir / f"{int(round(t * 1e9))}.png"
        self._cv2.imwrite(str(path), img_gray, [self._cv2.IMWRITE_PNG_COMPRESSION, 1])
        if self.rec:
            self._cv2.imwrite(str(self.rec / "cam0" / path.name), img_gray)
            self.rec_cam.write("%d\n" % int(round(t * 1e9)))
        self._pending.append((t, path))
        # Hold each frame until the estimator already holds IMU past it, exactly as
        # run_offline.cpp does (`imu[i].timestamp <= t + 0.05`). Submitting a frame with IMU
        # only up to t - dt made OpenVINS diverge within ten seconds; this queue is the
        # difference between a 49 m runaway and a filter that tracks.
        ok = False
        while self._pending and self._pending[0][0] <= self._imu_t - IMU_LEAD:
            ft, fpath = self._pending.pop(0)
            ok = self._submit(ft, fpath) or ok
        self.frames += 1
        return (self._t if self.initialised else t), self._R, self._c, (self.initialised and ok)

    def _submit(self, t, path) -> bool:
        self.proc.stdin.write("C %d %s\n" % (int(round(t * 1e9)), path))
        self.proc.stdin.flush()
        ok = False
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            v = line.split()
            if v[0] == "N" and len(v) == 2:
                break
            if v[0] == "P" and len(v) == 9:
                p = np.array([float(x) for x in v[2:5]])
                q = np.array([float(x) for x in v[5:9]])
                R_ItoG = Rotation.from_quat(q).as_matrix()
                self._t = float(v[1])
                self._R = R_ItoG @ self.R_bc
                self._c = p + R_ItoG @ self.p_bc
                ok = True
                if not self.initialised:
                    self.initialised = True
                    self.init_time = t
                continue
            if v[0] == "L" and ok:
                n = int(v[2])
                self.landmarks = (np.array([float(x) for x in v[3: 3 + 3 * n]]).reshape(n, 3)
                                  if n else np.zeros((0, 3)))
                break
        try:
            os.remove(path)
        except OSError:
            pass
        return ok

    def close(self):
        if self.rec:
            self.rec_imu.close()
            self.rec_cam.close()
            self.rec = None
        try:
            self.proc.stdin.write("Q\n")
            self.proc.stdin.flush()
            deadline = time.time() + 5.0
            while time.time() < deadline:
                line = self.proc.stdout.readline()
                if not line:
                    break
                if line.startswith("K "):
                    v = [float(x) for x in line.split()[1:]]
                    self.calib = {"R_ItoC": v[0:9], "p_IinC": v[9:12],
                                  "intrinsics": v[12:20], "dt_CAMtoIMU": v[20]}
                    break
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.frames_dir, ignore_errors=True)


def make(kind: str, **kw):
    return {"gt": GtPose, "gt_noisy": GtNoisyPose, "vio": VioPose}[kind](**kw)
