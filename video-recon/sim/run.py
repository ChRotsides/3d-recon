"""The closed loop: the robot explores an unknown world using only the pose and map it builds.

    python video-recon/sim/run.py --world apartment --seed 3 --pose vio --depth metric3d \
        --duration 300 --out out/sim/vio_metric3d --dashboard 8779

At 200 Hz the vehicle steps and the IMU is sampled; every 50 ms the camera renders and the pose
source consumes the frame; every 250 ms depth is computed and fused into the phone's TSDF and
into the navigation grid; every second the frontier explorer replans. The commands go to the
true robot, but everything that produced them came from the estimate.

Ground truth is written to gt.txt and used for the progress line and the report. The navigation
modules (navgrid, planner, explore) never import world.py and never receive a true pose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim import depth_sources, pose_sources
from sim.eval import explored_fraction, match_times, umeyama
from sim.explore import Explorer, frontiers
from sim.navgrid import NavGrid
from sim.planner import path_blocked
from sim.sensors import Camera, ImuSim
from sim.tsdf import Tsdf
from sim.vehicle import Vehicle
from sim.viz import costmap_png, depth_png
from sim.world import World

IMU_RATE = 200.0
CAM_PERIOD = 0.05
DEPTH_PERIOD = 0.25
PLAN_PERIOD = 1.0
COSTMAP_PERIOD = 0.5
CAM_HEIGHT = 0.50
MAX_SPEED = 8.0        # m/s of estimated motion averaged over MAX_SPEED_WINDOW. The robot does
                       # 0.6 m/s; a filter correcting itself peaks near 4, a runaway sustains
                       # 13+, so this catches the runaway and leaves healthy corrections alone
MAX_SPEED_WINDOW = 1.0  # s
MAX_FROM_START = 30.0   # m from the first estimated pose
INIT_SETTLE = 5.0       # s of manoeuvre after the first pose before the map is trusted


def init_manoeuvre(dt: float):
    """Forward 0.5 m, 30 deg left, 30 deg back, a short forward-back, then return; looped.
    Planar constant-velocity motion leaves monocular scale unobservable, so the estimator is
    given starts, stops and turns before the robot is allowed to trust it."""
    p = dt if dt < 6.4 else 0.8 + (dt - 0.8) % 5.6
    for lim, v, w in ((0.8, 0.0, 0.0), (2.2, 0.35, 0.0), (3.0, 0.0, 0.65), (3.8, 0.0, -0.65),
                      (4.4, 0.3, 0.0), (5.0, -0.3, 0.0), (6.4, -0.35, 0.0)):
        if p < lim:
            return v, w
    return 0.0, 0.0


class Sim:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        (self.out / "frames").mkdir(parents=True, exist_ok=True)
        (self.out / "costmaps").mkdir(parents=True, exist_ok=True)
        self.world = World.generate(a.seed, a.world)
        self.cam = Camera(a.width, a.height)
        self.veh = Vehicle(self.world, *self.world.start, rate=IMU_RATE, seed=a.seed, slip=a.slip)
        self.imu = ImuSim(IMU_RATE, noise=a.imu_noise, seed=a.seed)
        self.pose = pose_sources.make(
            a.pose, seed=a.seed, scale_error=a.scale_error, out_dir=self.out,
            R_bc=self.cam.R_bc, p_bc=self.cam.p_bc, config=a.vio_config,
            record=getattr(a, "record_vio_input", False),
            K=self.cam.K, width=self.cam.width, height=self.cam.height)
        self.depth = depth_sources.make(a.depth, self.cam.K, self.cam.width, self.cam.height, seed=a.seed)
        self.tsdf = Tsdf()
        self.grid = NavGrid()
        self.explorer = Explorer()
        self.rng = np.random.default_rng(a.seed + 77)

        self.t = 0.0
        self.floor_z = 0.0
        self.state = self.cost = None
        self.origin_xy = np.zeros(2)
        self.surface = np.zeros((0, 6), np.float32)
        self.gt_log, self.est_log, self.cmd_log, self.events, self.explored_log = [], [], [], [], []
        self.stage = {k: [] for k in ("render", "pose", "depth", "tsdf", "navgrid", "plan")}
        self.tsdf_stats = {}
        self.depth_info = {}
        self.n_depth = 0
        self.align = (1.0, np.eye(3), np.zeros(3))
        self.explored = 0.0
        self.mapping = False
        self._init_t = None
        self.last_rgb = None
        self.diverged = False
        self._c0 = None
        self._chist: list = []
        self._was_contact = False
        self.wall0 = time.time()
        self.lock = threading.Lock()
        self.snap = {}
        self.stop = False

    # ------------------------------------------------------------------ helpers

    def _log_event(self, kind, detail=""):
        self.events.append((round(self.t, 3), kind, str(detail)))

    def _refresh_costmap(self):
        t0 = time.time()
        self.state, self.cost, self.origin_xy = self.grid.costmap(self.floor_z)
        self.stage["plan"].append(1000 * (time.time() - t0))

    def _update_alignment(self):
        if len(self.est_log) < 20:
            return
        tg = np.array([r[0] for r in self.gt_log])
        pg = np.array([r[1:4] for r in self.gt_log])
        te = np.array([r[0] for r in self.est_log])
        pe = np.array([r[1:4] for r in self.est_log])
        j, ok = match_times(tg, te)
        if ok.sum() >= 10:
            self.align = umeyama(pe[ok], pg[j[ok]], with_scale=False)

    def _explored_now(self):
        if self.state is None:
            return 0.0
        s, R, t = self.align
        f, _ = explored_fraction(self.world, self.state, self.origin_xy, self.grid.res, s, R, t)
        return f

    def _gt_in_est_frame(self):
        s, R, t = self.align
        x, y, yaw = self.veh.base
        p = (np.array([x, y, 0.0]) - t) @ R / s
        yaw_e = yaw - np.arctan2(R[1, 0], R[0, 0])
        return float(p[0]), float(p[1]), float(yaw_e)

    def _check_divergence(self, c_est) -> bool:
        """A monocular VIO that loses its scale runs away at metres per second. Mapping that
        pose fills the grid with garbage and, before this guard existed, overflowed the grid
        index and killed the process. Stop mapping and stop the robot instead, and say so.

        The test is a *sustained* speed, not a single frame-to-frame jump: a filter legitimately
        snaps by a metre when a delayed update lands, and calling that divergence threw away
        runs that went on to track perfectly well.
        """
        c = np.asarray(c_est, dtype=float)
        if self._c0 is None:
            self._c0 = c.copy()
        self._chist.append((self.t, c.copy()))
        while len(self._chist) > 2 and self._chist[0][0] < self.t - MAX_SPEED_WINDOW:
            self._chist.pop(0)
        far = float(np.linalg.norm(c - self._c0))
        t0, c0 = self._chist[0]
        dt = self.t - t0
        speed = float(np.linalg.norm(c - c0)) / dt if dt > 0.4 else 0.0
        if speed > MAX_SPEED or far > MAX_FROM_START:
            self.diverged = True
            self._div_t = self.t
            self._log_event("vio_diverged", f"{speed:.1f} m/s sustained, {far:.1f} m from start")
            print(f"[!] pose source diverged at t={self.t:.1f}s ({speed:.1f} m/s sustained, "
                  f"{far:.1f} m from start); mapping stopped", flush=True)
            return False
        return True

    # ------------------------------------------------------------------ the loop

    def run(self):
        a = self.a
        dt = 1.0 / IMU_RATE
        n = 0
        next_cam = 0.0
        next_depth = 1e9
        next_plan = 1e9
        next_costcheck = 1e9
        next_report = 0.0
        next_frame_dump = 0.0
        next_costmap = 0.0
        next_explored = 0.0
        imu_buf = []
        v_cmd = w_cmd = 0.0
        init_t0 = None

        while self.t < a.duration and not self.stop:
            self.veh.step(v_cmd, w_cmd)
            if self.veh.in_contact and not self._was_contact:
                self._log_event("collision", f"{self.veh.x:.2f},{self.veh.y:.2f}")
            self._was_contact = self.veh.in_contact
            R_wb, p_wb = self.veh.body_pose()
            s = self.imu.sample(self.t, R_wb, p_wb)
            if s:
                imu_buf.append(s)

            if self.t >= next_cam:
                next_cam += CAM_PERIOD
                t0 = time.time()
                R_wc, c_w = self.cam.pose(R_wb, p_wb)
                # VIO only ever sees grayscale; colour is needed on depth frames (the model input
                # and the TSDF's colour) and for the dashboard, which is 4 of every 20 frames
                want_colour = (self.t + 1e-9 >= next_depth) or (self.a.dashboard and self.t >= next_report)
                img = self.cam.render(self.world, R_wc, c_w, self.rng, gray=not want_colour)
                img, gt_depth = img
                self.stage["render"].append(1000 * (time.time() - t0))
                gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                t0 = time.time()
                te, R_est, c_est, ok = self.pose.update(self.t, R_wc, c_w, gray, imu_buf)
                imu_buf = []
                self.stage["pose"].append(1000 * (time.time() - t0))
                self.gt_log.append([self.t, *c_w, *Rotation.from_matrix(R_wc).as_quat()])
                if ok and not self.diverged:
                    ok = self._check_divergence(c_est)
                if ok:
                    self.est_log.append([self.t, *c_est, *Rotation.from_matrix(R_est).as_quat()])
                    self.floor_z = 0.9 * self.floor_z + 0.1 * (c_est[2] - CAM_HEIGHT) if self.mapping \
                        else c_est[2] - CAM_HEIGHT
                    if self._init_t is None:
                        self._init_t = self.t
                        self._log_event("init", f"pose source initialised after {self.t:.2f} s")
                    # Keep doing the manoeuvre for a few seconds after the first pose. A
                    # monocular filter's scale is still settling then, and mapping on it writes
                    # geometry that can never be taken back.
                    if not self.mapping and self.t >= self._init_t + INIT_SETTLE:
                        self.mapping = True
                        self._log_event("mapping", f"trusting the estimate from t={self.t:.2f} s")
                        next_depth = self.t + CAM_PERIOD   # the next frame renders in colour
                        next_plan = self.t + 0.3
                        next_costcheck = self.t + 0.3
                if img.ndim == 3:
                    self.last_rgb = img
                self.last = (img, gt_depth, R_est, c_est, ok)

            if self.mapping and not self.diverged and self.t >= next_depth:
                next_depth += DEPTH_PERIOD
                _, gt_depth, R_est, c_est, ok = self.last
                img = self.last_rgb
                if ok and img is not None:
                    t0 = time.time()
                    mm, info = self.depth.compute(img, gt_depth, R_est)
                    self.stage["depth"].append(1000 * (time.time() - t0))
                    self.depth_info = info
                    if mm is not None:
                        small = cv2.resize(img, (mm.shape[1], mm.shape[0]), interpolation=cv2.INTER_AREA)
                        t0 = time.time()
                        self.tsdf_stats = self.tsdf.integrate(mm, self.depth.K, R_est, c_est, small, 2)
                        self.surface = self.tsdf.surface()
                        self.stage["tsdf"].append(1000 * (time.time() - t0))
                        t0 = time.time()
                        self.grid.floor_z = self.floor_z
                        self.grid.integrate(mm, self.depth.K, R_est, c_est)
                        self.grid.mark_robot(self.cam.base_from_camera(R_est, c_est)[:2], self.floor_z)
                        if self.n_depth % 4 == 0:
                            self.grid.mark_surface(self.surface)
                        self.stage["navgrid"].append(1000 * (time.time() - t0))
                        self.n_depth += 1
                        self.last_depth_mm = mm

            if self.mapping and not self.diverged and self.t >= next_costcheck:
                next_costcheck += COSTMAP_PERIOD
                self._refresh_costmap()
                if self.explorer.path is not None and path_blocked(
                        self.explorer.path, self.cost, self.grid.world_to_cell):
                    self._log_event("replan", "path crosses newly occupied cells")
                    next_plan = self.t
            if self.mapping and not self.diverged and self.t >= next_plan:
                next_plan = self.t + PLAN_PERIOD
                if self.state is None:      # an early re-path can outrun the costmap schedule
                    self._refresh_costmap()
                _, _, R_est, c_est, ok = self.last
                if ok:
                    pose_e = self.cam.base_from_camera(R_est, c_est)
                    self.explorer.replan(self.t, pose_e, self.grid, self.state, self.cost)
                    if self.explorer.done:
                        self._log_event("done", "no frontiers left")
                        break

            if self.diverged:
                # Do not stop dead: standing still is precisely what makes a monocular scale
                # unobservable, so a stopped robot can never recover. Fall back to the bounded
                # initialisation manoeuvre, which translates and turns and stays near its start.
                v_cmd, w_cmd = init_manoeuvre(self.t - self._div_t)
            elif self.mapping:
                _, _, R_est, c_est, ok = self.last
                if ok:
                    pose_e = self.cam.base_from_camera(R_est, c_est)
                    nev = len(self.explorer.events)
                    v_cmd, w_cmd = self.explorer.command(self.t, pose_e)
                    for e in self.explorer.events[nev:]:
                        self._log_event("stuck", f"{e[2]:.2f},{e[3]:.2f}")
                    # The follower ran out of path but the target stands: re-path now rather than
                    # idle until the next scheduled replan, which costs up to a second per
                    # waypoint. Only with a live target - the explorer counts consecutive empty
                    # replans to decide the run is over, and those must stay one per second.
                    if self.explorer.path is None and self.explorer.target is not None \
                            and self.t < next_plan - 0.1:
                        next_plan = self.t + 0.1
            else:
                if init_t0 is None:
                    init_t0 = self.t
                v_cmd, w_cmd = init_manoeuvre(self.t - init_t0)

            if n % 20 == 0:
                self.cmd_log.append([round(self.t, 3), round(v_cmd, 4), round(w_cmd, 4),
                                     round(self.veh.v, 4), round(self.veh.omega, 4)])
            if self.t >= next_explored:
                next_explored = self.t + 5.0
                self._update_alignment()
                self.explored = self._explored_now()
                self.explored_log.append([round(self.t, 2), round(self.explored, 4)])
                el = time.time() - self.wall0
                print(f"[{100 * self.t / self.a.duration:5.1f}%] t={self.t:6.1f}s  explored {100 * self.explored:5.1f}%  "
                      f"blocks {self.tsdf_stats.get('blocks', 0):5d}  pts {self.tsdf_stats.get('points', 0):7d}  "
                      f"coll {self.veh.collisions:3d}  wall {el:6.1f}s  RTF {self.t / max(1e-6, el):.2f}", flush=True)
            if self.t >= next_frame_dump and self.mapping:
                next_frame_dump = self.t + 2.0
                self._dump_frame()
            if self.t >= next_costmap and self.state is not None:
                next_costmap = self.t + 10.0
                self._dump_costmap()
            if self.a.dashboard and self.t >= next_report:
                next_report = self.t + 0.5
                self._snapshot()

            self.t += dt
            n += 1

        self.finish()

    # ------------------------------------------------------------------ outputs

    def _dump_frame(self):
        img = self.last_rgb
        if img is None:
            return
        k = int(self.t)
        cv2.imwrite(str(self.out / "frames" / f"rgb_{k:04d}.jpg"), img[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        if hasattr(self, "last_depth_mm"):
            cv2.imwrite(str(self.out / "frames" / f"depth_{k:04d}.png"), self.last_depth_mm)

    def _dump_costmap(self):
        _, sizes, cents = frontiers(self.state, self.cost)
        fm = None
        if len(sizes):
            fm = np.zeros(self.state.shape, bool)
            for cx, cy in cents.astype(int):
                fm[max(0, cy - 1): cy + 2, max(0, cx - 1): cx + 2] = True
        _, _, R_est, c_est, ok = self.last
        pose_e = self.cam.base_from_camera(R_est, c_est) if ok else None
        img = costmap_png(self.state, self.cost, self.origin_xy, self.grid.res,
                          path=self.explorer.path, est_pose=pose_e, gt_pose=self._gt_in_est_frame(),
                          frontier_mask=fm, target=self.explorer.target)
        cv2.imwrite(str(self.out / "costmaps" / f"t{int(self.t):04d}.png"), img)
        return img

    def _snapshot(self):
        _, gt_depth, R_est, c_est, ok = self.last
        img = self.last_rgb
        if img is None:
            return
        enc = lambda a: cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 72])[1].tobytes()
        est_d = getattr(self, "last_depth_mm", None)
        with self.lock:
            self.snap["rgb"] = enc(img[:, :, ::-1])
            self.snap["gtdepth"] = enc(depth_png(cv2.resize(gt_depth, (322, 181), interpolation=cv2.INTER_NEAREST)))
            if est_d is not None:
                self.snap["depth"] = enc(depth_png(est_d.astype(np.float32) / 1000.0))
            if self.state is not None:
                _, sizes, cents = frontiers(self.state, self.cost)
                fm = None
                if len(sizes):
                    fm = np.zeros(self.state.shape, bool)
                    for cx, cy in cents.astype(int):
                        fm[max(0, cy - 1): cy + 2, max(0, cx - 1): cx + 2] = True
                pose_e = self.cam.base_from_camera(R_est, c_est) if ok else None
                self.snap["costmap"] = cv2.imencode(".png", costmap_png(
                    self.state, self.cost, self.origin_xy, self.grid.res, path=self.explorer.path,
                    est_pose=pose_e, gt_pose=self._gt_in_est_frame(), frontier_mask=fm,
                    target=self.explorer.target))[1].tobytes()
            self.snap["stats"] = self.stats_dict()
            self.snap["surface"] = self.surface

    def stats_dict(self):
        el = time.time() - self.wall0
        ms = {k: round(float(np.mean(v[-40:])), 2) for k, v in self.stage.items() if v}
        return {"t": round(self.t, 2), "duration": self.a.duration, "wall": round(el, 1),
                "rtf": round(self.t / max(1e-6, el), 3), "explored": round(self.explored, 4),
                "collisions": self.veh.collisions, "distance_m": round(self.veh.distance, 2),
                "blocks": self.tsdf_stats.get("blocks", 0), "points": self.tsdf_stats.get("points", 0),
                "stage_ms": ms, "depth_ms": round(float(self.depth_info.get("ms", 0)), 1),
                "depth_scale": round(float(self.depth_info.get("scale", 0)), 4),
                "depth_inliers": round(float(self.depth_info.get("inliers", 0)), 3),
                "mapping": self.mapping, "frontiers": getattr(self.explorer, "n_frontiers", 0),
                "pose": self.a.pose, "depth": self.a.depth, "world": self.a.world, "seed": self.a.seed,
                "vio_init_s": getattr(self.pose, "init_time", None),
                "target": None if self.explorer.target is None else [round(v, 2) for v in self.explorer.target]}

    def finish(self):
        o = self.out
        np.savetxt(o / "gt.txt", np.asarray(self.gt_log), fmt="%.6f")
        np.savetxt(o / "est.txt", np.asarray(self.est_log) if self.est_log else np.zeros((0, 8)), fmt="%.6f")
        with open(o / "commands.csv", "w") as f:
            f.write("t,v_cmd,w_cmd,v_true,omega_true\n")
            f.writelines(",".join(map(str, r)) + "\n" for r in self.cmd_log)
        with open(o / "events.csv", "w") as f:
            f.write("t,kind,detail\n")
            f.writelines(",".join(map(str, r)) + "\n" for r in self.events)
        np.savetxt(o / "explored.csv", np.asarray(self.explored_log or [[0, 0]]), delimiter=",", fmt="%.4f")
        if self.state is None and self.grid.log is not None:
            self._refresh_costmap()
        if self.grid.log is not None:
            self.grid.save(o / "navgrid.npz")
        if len(self.surface):
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.surface[:, :3].astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(np.clip(self.surface[:, 3:], 0, 1).astype(np.float64))
            o3d.io.write_point_cloud(str(o / "tsdf_surface.ply"), pcd)
        if self.state is not None:
            self._dump_costmap()
        self.pose.close()
        self._update_alignment()
        stats = {"config": {"world": self.a.world, "seed": self.a.seed, "pose": self.a.pose,
                            "depth": self.a.depth, "duration": self.a.duration,
                            "imu_noise": self.a.imu_noise, "scale_error": self.a.scale_error,
                            "vio_config": self.a.vio_config, "slip": self.a.slip},
                 "world": self.a.world, "seed": self.a.seed, "floor_z": float(self.floor_z),
                 "runtime": {"sim_s": round(self.t, 2), "wall_s": round(time.time() - self.wall0, 1),
                             "rtf": round(self.t / max(1e-6, time.time() - self.wall0), 3)},
                 "stage_ms": {k: {"mean": round(float(np.mean(v)), 2), "p90": round(float(np.percentile(v, 90)), 2),
                                  "n": len(v)} for k, v in self.stage.items() if v},
                 "diverged": self.diverged,
                 "collisions": self.veh.collisions, "blocked_ticks": self.veh.blocked_ticks,
                 "distance_m": round(self.veh.distance, 2),
                 "stuck_events": sum(1 for e in self.events if e[1] == "stuck"),
                 "vio_init_s": getattr(self.pose, "init_time", None),
                 "vio_calib": getattr(self.pose, "calib", None),
                 "depth_frames": self.n_depth,
                 "depth_scale_mean": (round(float(np.mean(self.depth.scale_log)), 4)
                                      if getattr(self.depth, "scale_log", None) else None),
                 "depth_scale_jitter": _jitter(getattr(self.depth, "scale_log", [])),
                 "tsdf": self.tsdf_stats, "explored_final": round(self.explored, 4)}
        (o / "stats.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps(stats["runtime"] | {"explored": stats["explored_final"],
                                             "collisions": stats["collisions"]}, indent=1), flush=True)


def _jitter(log):
    a = np.asarray([v for v in log if v > 0], dtype=float)
    return round(float(np.std(np.diff(np.log(a)))), 4) if len(a) > 3 else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="apartment", choices=("apartment", "corridor", "open"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pose", default="gt", choices=("gt", "gt_noisy", "vio"))
    ap.add_argument("--depth", default="gt", choices=("gt", "gt_noisy", "da3", "dav2", "metric3d", "unidepth"))
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--imu-noise", type=float, default=1.0)
    ap.add_argument("--scale-error", type=float, default=1.0)
    ap.add_argument("--slip", type=float, default=0.0)
    ap.add_argument("--vio-config", default="estimator_config.yaml")
    ap.add_argument("--record-vio-input", action="store_true",
                    help="write imu.csv / cam0.csv / cam0/ in run_offline format under the out dir")
    ap.add_argument("--dashboard", type=int, default=0)
    ap.add_argument("--dashboard-host", default="0.0.0.0")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args(argv)
    if a.headless:
        a.dashboard = 0
    sim = Sim(a)
    if a.dashboard:
        from sim.dashboard import serve

        serve(sim, a.dashboard, a.dashboard_host)
        import socket

        lan = []
        try:   # the address a machine on the LAN would reach; no packet is actually sent
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
                sk.connect(("192.168.0.1", 1))
                lan = [sk.getsockname()[0]]
        except OSError:
            try:
                lan = [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2]
                       if not ip.startswith("127.")]
            except OSError:
                lan = []
        print(f"dashboard: http://localhost:{a.dashboard}/", flush=True)
        for ip in lan:
            print(f"           http://{ip}:{a.dashboard}/", flush=True)
    try:
        sim.run()
    except KeyboardInterrupt:
        sim.stop = True
        sim.finish()
    if not a.no_eval:
        from sim.eval import evaluate

        rep = evaluate(sim.out)
        print(json.dumps({k: rep[k] for k in ("explored_final", "ate_se3", "scale_error",
                                              "map_accuracy_cm", "completeness_5cm", "collisions")
                          if k in rep}, indent=1), flush=True)
    return sim


if __name__ == "__main__":
    main()
    # Open3D's atexit leak check reports the last raycasting buffer and exits non-zero, which
    # makes an otherwise successful run look failed. Everything is written and flushed by here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
