"""Export several VIO runs of one recording for the walk viewer.

usage: export_walk2.py REC_DIR OUT_JSON [--ref RUN_DIR] name=traj.txt [name=traj.txt ...]
With --ref (a videorecon output dir with bundle/ and mesh.ply) every run is Sim(3)-aligned
to the COLMAP reference. Without it, runs are metric in the first run's gravity-aligned
frame (Y up), the others SE(3)-aligned to the first over common timestamps.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, "" + str(Path(__file__).resolve().parents[1]) + "")
from videorecon.recording import Recording, video_frame_pts_ns, match_video_to_capture_results
from videorecon.imu import device_to_camera

ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("out"); ap.add_argument("--ref"); ap.add_argument("runs", nargs="+")
a = ap.parse_args()
CV_TO_GL = np.diag([1.0, -1.0, -1.0]); ZUP_TO_YUP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
rec = Recording.load(a.rec)
pts = video_frame_pts_ns(rec.video_path); timing = match_video_to_capture_results(pts, rec.frames[:, 1])
frame_ts = timing.timestamps_ns * 1e-9; video_t = (pts - pts[0]) * 1e-9
fx = rec.intrinsics().fx

def load_run(path):
    lines = Path(path + ".calib.txt").read_text().split("\n")
    R_ItoC = np.array([[float(x) for x in lines[i].split()] for i in (1, 2, 3)]); p_IinC = np.array([float(x) for x in lines[5].split()])
    v = np.loadtxt(path); t = v[:, 0]; R_ItoG = Rotation.from_quat(v[:, 4:8]).as_matrix()
    R_CtoI = R_ItoC.T; p_CinI = -R_CtoI @ p_IinC
    return t, v[:, 1:4] + R_ItoG @ p_CinI, R_ItoG @ R_CtoI   # camera centre, world<-camera (OpenCV axes)

def umeyama(src, dst, with_scale):
    mu_s, mu_d = src.mean(0), dst.mean(0); S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src)); d = np.eye(3); d[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ d @ Vt; s = (np.trace(np.diag(sig) @ d) / (S ** 2).sum() * len(src)) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s

def match(t_a, t_b, tol=0.005):
    j = np.searchsorted(t_a, t_b); j = np.clip(j, 1, len(t_a) - 1)
    j = np.where(np.abs(t_a[j - 1] - t_b) < np.abs(t_a[j] - t_b), j - 1, j)
    ok = np.abs(t_a[j] - t_b) < tol; return j, ok

def to_poses(t, c, R):
    fi = np.searchsorted(frame_ts, t); fi = np.clip(fi, 0, len(frame_ts) - 1); out = []
    for k in range(len(t)):
        if abs(frame_ts[fi[k]] - t[k]) > 0.002: continue
        q = Rotation.from_matrix(R[k] @ CV_TO_GL).as_quat()
        out.append([round(float(video_t[fi[k]]), 4), *np.round(c[k], 4).tolist(), *np.round(q, 5).tolist()])
    return out

runs = {}; ref = None; mesh = None; notes = {}
for spec in a.runs:
    name, path = spec.split("="); runs[name] = load_run(path)
if a.ref:
    run_dir = Path(a.ref); manifest = json.loads((run_dir / "manifest.json").read_text()); bundle = json.loads((run_dir / "bundle" / "bundle.json").read_text())
    rot = float(manifest["stages"].get("bundle_rotation_deg", 0)); Rz = Rotation.from_euler("z", rot, degrees=True).as_matrix()
    rc, rR, rt = [], [], []
    for k in bundle["keyframes"]:
        rc.append(np.asarray(k["pose"]["t"])); rR.append(np.asarray(k["pose"]["R"]) @ CV_TO_GL @ Rz); rt.append(k["timestamp_ns"] * 1e-9)
    rc, rR, rt = map(np.array, (rc, rR, rt)); ref = to_poses(rt, rc, rR)
    mj = json.loads((run_dir / "mesh.json").read_text()); mesh = {"min": mj["min"], "max": mj["max"]}
    base_t, base_c = rt, rc
else:
    first = next(iter(runs)); t0, c0, R0 = runs[first]
    base_t, base_c = t0, (ZUP_TO_YUP @ c0.T).T
out_runs = {}
for name, (t, c, R) in runs.items():
    j, ok = match(t, base_t)
    if a.ref or name != next(iter(runs)):
        s, Rg, tg = umeyama(c[j[ok]], base_c[ok], with_scale=bool(a.ref))
    else:
        s, Rg, tg = 1.0, ZUP_TO_YUP, np.zeros(3)
    c_al = (s * (Rg @ c.T)).T + tg; R_al = np.einsum("ij,njk->nik", Rg, R)
    err = np.linalg.norm(c_al[j[ok]] - base_c[ok], axis=1) if ok.sum() else np.zeros(1)
    end = c_al[t > t[-1] - 1.0].mean(0)
    notes[name] = {"scale_applied": round(float(s), 4), "vs_base_median_cm": round(100 * float(np.median(err)), 1), "vs_base_rmse_cm": round(100 * float(np.sqrt((err ** 2).mean())), 1),
                   "path_m": round(float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()), 1), "start_end_drift_cm": round(100 * float(np.linalg.norm(end - c[0])), 0),
                   "extent_m": np.round(c.max(0) - c.min(0), 2).tolist(), "first_pose_s": round(float(video_t[min(np.searchsorted(frame_ts, t[0]), len(video_t) - 1)]), 1)}
    out_runs[name] = to_poses(t, c_al, R_al)
Path(a.out).write_text(json.dumps({"runs": out_runs, "ref": ref, "mesh": mesh, "notes": notes, "duration": float(video_t[-1]),
                                   "fov_deg": 2 * np.degrees(np.arctan(0.5 * 1080 / fx)), "base": "COLMAP reference" if a.ref else next(iter(runs))}))
print(json.dumps(notes, indent=1))
