"""Export per-frame VIO poses (aligned to the reference frame) + the COLMAP keyframe path
as JSON for the side-by-side walk viewer."""
import sys, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, "" + str(Path(__file__).resolve().parents[1]) + "")
from videorecon.recording import Recording, video_frame_pts_ns, match_video_to_capture_results
from videorecon.imu import device_to_camera

traj_path, run_dir, rec_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
CV_TO_GL = np.diag([1.0, -1.0, -1.0])
lines = Path(str(traj_path) + ".calib.txt").read_text().split("\n")
R_ItoC = np.array([[float(x) for x in lines[i].split()] for i in (1, 2, 3)])
p_IinC = np.array([float(x) for x in lines[5].split()])
vio = np.loadtxt(traj_path)
t_vio = vio[:, 0]; p_IinG = vio[:, 1:4]; R_ItoG = Rotation.from_quat(vio[:, 4:8]).as_matrix()
R_CtoI = R_ItoC.T; p_CinI = -R_CtoI @ p_IinC
R_CtoG = R_ItoG @ R_CtoI
c_vio = p_IinG + R_ItoG @ p_CinI

manifest = json.loads((run_dir / "manifest.json").read_text())
bundle = json.loads((run_dir / "bundle" / "bundle.json").read_text())
rot = float(manifest["stages"].get("bundle_rotation_deg", 0))
Rz = Rotation.from_euler("z", rot, degrees=True).as_matrix()
ref_c, ref_R, ref_t = [], [], []
for k in bundle["keyframes"]:
    R_gl = np.asarray(k["pose"]["R"]); t = np.asarray(k["pose"]["t"])
    ref_c.append(t); ref_R.append(R_gl @ CV_TO_GL @ Rz); ref_t.append(k["timestamp_ns"] * 1e-9)
ref_c, ref_R, ref_t = map(np.array, (ref_c, ref_R, ref_t))

j = np.searchsorted(t_vio, ref_t); j = np.clip(j, 1, len(t_vio) - 1)
j = np.where(np.abs(t_vio[j - 1] - ref_t) < np.abs(t_vio[j] - ref_t), j - 1, j)
ok = np.abs(t_vio[j] - ref_t) < 0.005
A, B = c_vio[j[ok]], ref_c[ok]
mu_s, mu_d = A.mean(0), B.mean(0); S, D = A - mu_s, B - mu_d
U, sig, Vt = np.linalg.svd(D.T @ S / len(A)); d = np.eye(3); d[2, 2] = np.sign(np.linalg.det(U @ Vt))
R = U @ d @ Vt; s = np.trace(np.diag(sig) @ d) / (S ** 2).sum() * len(A); t = mu_d - s * R @ mu_s
print(f"Sim(3): scale {s:.4f}")
c_al = (s * (R @ c_vio.T)).T + t
R_al = np.einsum("ij,njk->nik", R, R_CtoG)   # world<-camera (OpenCV axes) in the reference frame

rec = Recording.load(rec_dir)
pts = video_frame_pts_ns(rec.video_path)
timing = match_video_to_capture_results(pts, rec.frames[:, 1])
frame_ts = timing.timestamps_ns * 1e-9
video_t = (pts - pts[0]) * 1e-9
# VIO pose k has timestamp frame_ts[i] for some frame i -> video time video_t[i]
fi = np.searchsorted(frame_ts, t_vio); fi = np.clip(fi, 0, len(frame_ts) - 1)
poses = []
for k in range(len(t_vio)):
    i = fi[k]
    if abs(frame_ts[i] - t_vio[k]) > 0.002: continue
    q = Rotation.from_matrix(R_al[k] @ CV_TO_GL).as_quat()   # GL camera (looks -Z), world<-camera
    poses.append([round(float(video_t[i]), 4), *np.round(c_al[k], 4).tolist(), *np.round(q, 5).tolist()])
kf = []
for i, (c, Rm, tt) in enumerate(zip(ref_c, ref_R, ref_t)):
    vi = np.argmin(np.abs(frame_ts - tt))
    q = Rotation.from_matrix(Rm @ CV_TO_GL).as_quat()
    kf.append([round(float(video_t[vi]), 4), *np.round(c, 4).tolist(), *np.round(q, 5).tolist()])
out.write_text(json.dumps({"vio": poses, "ref": kf, "scale": s, "duration": float(video_t[-1]),
                           "fov_deg": 2 * np.degrees(np.arctan(0.5 * 1080 / 1278.5)),
                           "mesh_min": json.loads((run_dir / "mesh.json").read_text())["min"],
                           "mesh_max": json.loads((run_dir / "mesh.json").read_text())["max"]}))
print(f"wrote {len(poses)} VIO poses, {len(kf)} keyframes, duration {video_t[-1]:.1f}s")
