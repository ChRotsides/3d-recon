"""Compare an OpenVINS trajectory against the COLMAP+IMU reference run.

Reports: Sim(3) and SE(3) trajectory errors of the camera centres, the scale the
VIO recovered relative to the reference's floor-based scale, and the rotation
residual against the phone's own game-rotation-vector (the same statistic the
pipeline's align stage reports for COLMAP).
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, "/home/charalambos/Documents/Projects/3d-recon/video-recon")
from videorecon.recording import Recording, video_frame_pts_ns, match_video_to_capture_results
from videorecon.imu import device_to_camera

traj_path, run_dir, rec_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
calib = {}
cp = Path(str(traj_path) + ".calib.txt")
if cp.exists():
    lines = cp.read_text().split("\n")
    R = np.array([[float(x) for x in lines[i].split()] for i in (1, 2, 3)])
    p = np.array([float(x) for x in lines[5].split()])
    calib = {"R_ItoC": R, "p_IinC": p, "intr": [float(x) for x in lines[7].split()], "dt": float(lines[9])}
    print("online calibration:")
    print("  R_ItoC angle from prior:", f"{np.degrees(np.linalg.norm(Rotation.from_matrix(R @ device_to_camera(90)).as_rotvec())):.2f}°")
    print("  p_IinC (m):", np.round(p, 4).tolist())
    print("  intrinsics:", np.round(calib["intr"], 2).tolist())
    print("  dt cam->imu (ms):", f"{calib['dt']*1e3:.2f}")

vio = np.loadtxt(traj_path)
t_vio = vio[:, 0]
p_IinG = vio[:, 1:4]
R_ItoG = Rotation.from_quat(vio[:, 4:8]).as_matrix()
R_ItoC = calib.get("R_ItoC", device_to_camera(90).T)
p_IinC = calib.get("p_IinC", np.zeros(3))
R_CtoI = R_ItoC.T
p_CinI = -R_CtoI @ p_IinC
R_CtoG = R_ItoG @ R_CtoI
c_vio = p_IinG + R_ItoG @ p_CinI

# reference: the aligned, metric poses the depth stage consumed (repo frame, Y up,
# world<-camera with the camera looking -Z), from bundle/bundle.json.
manifest = json.loads((run_dir / "manifest.json").read_text())
bundle = json.loads((run_dir / "bundle" / "bundle.json").read_text())
CV_TO_GL = np.diag([1.0, -1.0, -1.0])
ref_c, ref_R, ref_t = [], [], []
for k in bundle["keyframes"]:
    R_gl = np.asarray(k["pose"]["R"]); t = np.asarray(k["pose"]["t"])
    ref_c.append(t); ref_R.append(R_gl @ CV_TO_GL); ref_t.append(k["timestamp_ns"] * 1e-9)
ref_c, ref_R, ref_t = np.array(ref_c), np.array(ref_R), np.array(ref_t)
# The bundle's keyframes were rotated upright (manifest bundle_rotation_deg) for the
# depth model, so its camera axes are the upright image's; undo that about the optical axis.
rot = float(manifest["stages"].get("bundle_rotation_deg", 0))
if rot:
    Rz = lambda d: Rotation.from_euler("z", d, degrees=True).as_matrix()
    ref_R = np.array([R @ Rz(rot) for R in ref_R])
rec = Recording.load(rec_dir)
order = np.argsort(ref_t); ref_c, ref_R, ref_t = ref_c[order], ref_R[order], ref_t[order]

# match by timestamp (VIO writes one pose per frame at the frame's timestamp)
j = np.searchsorted(t_vio, ref_t)
j = np.clip(j, 1, len(t_vio) - 1)
j = np.where(np.abs(t_vio[j - 1] - ref_t) < np.abs(t_vio[j] - ref_t), j - 1, j)
ok = np.abs(t_vio[j] - ref_t) < 0.005
print(f"\nreference keyframes {len(ref_t)}, with a VIO pose within 5 ms: {ok.sum()}")
A = c_vio[j[ok]]; B = ref_c[ok]; RA = R_CtoG[j[ok]]; RB = ref_R[ok]

def umeyama(src, dst, with_scale):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    d = np.eye(3); d[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ d @ Vt
    s = (np.trace(np.diag(sig) @ d) / (S ** 2).sum() * len(src)) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t

for label, with_scale in (("Sim(3)", True), ("SE(3), VIO scale kept", False)):
    s, R, t = umeyama(A, B, with_scale)
    aligned = (s * (R @ A.T)).T + t
    err = np.linalg.norm(aligned - B, axis=1)
    if with_scale:
        vio_sim3 = aligned
    else:
        vio_se3 = aligned
    print(f"{label}: scale VIO->ref {s:.4f}  ATE rmse {np.sqrt((err**2).mean())*100:.1f} cm  median {np.median(err)*100:.1f} cm  max {err.max()*100:.1f} cm")
    if with_scale:
        print(f"  (ref scale is floor-based with an assumed 1.3 m camera height; VIO says the walk is {1/s:.3f}x that)")
    # rotation residual after the same global rotation
    rel = [Rotation.from_matrix(RB[i].T @ R @ RA[i]).magnitude() for i in range(len(A))]
    rel = np.degrees(rel)
    print(f"  rotation residual vs COLMAP: median {np.median(rel):.2f}°  p95 {np.percentile(rel,95):.2f}°")

# against the phone's own orientation sensor, as align.py does for COLMAP
track = rec.orientation_track("gamerotv")
R_wd = track.rotation_at((ref_t[ok] * 1e9).astype(np.int64))
R_wc_sensor = R_wd @ device_to_camera(90)
# one global rotation between VIO's G and the sensor world (Wahba)
M = sum(R_wc_sensor[i] @ RA[i].T for i in range(len(A)))
U, _, Vt = np.linalg.svd(M); d = np.eye(3); d[2, 2] = np.sign(np.linalg.det(U @ Vt)); Rg = U @ d @ Vt
res = np.degrees([Rotation.from_matrix(R_wc_sensor[i].T @ Rg @ RA[i]).magnitude() for i in range(len(A))])
print(f"\nrotation residual VIO vs phone gamerotv: median {np.median(res):.2f}°  p95 {np.percentile(res,95):.2f}°")
print(f"  (reference run COLMAP vs gamerotv: median {manifest['stages']['align']['rotation_residual_median_deg']:.2f}°  p95 {manifest['stages']['align']['rotation_residual_p95_deg']:.2f}°)")
# drift: distance between first and last camera centres, VIO vs reference
print(f"\npath length ref {np.linalg.norm(np.diff(B,axis=0),axis=1).sum():.2f} m, VIO {np.linalg.norm(np.diff(A,axis=0),axis=1).sum():.2f} m")
np.savez(traj_path.with_suffix(".aligned.npz"), vio=vio_sim3, vio_se3=vio_se3, ref=B, t=ref_t[ok])
