"""videoscan recording -> OpenVINS offline input (grayscale PNGs + imu.csv + timestamps)."""
import sys, json, csv
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, "" + str(Path(__file__).resolve().parents[1]) + "")
from videorecon.recording import Recording, video_frame_pts_ns, match_video_to_capture_results
from videorecon.imu import device_to_camera

rec_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
out.mkdir(parents=True, exist_ok=True); (out / "cam0").mkdir(exist_ok=True)
rec = Recording.load(rec_dir)
pts = video_frame_pts_ns(rec.video_path)
timing = match_video_to_capture_results(pts, rec.frames[:, 1])
print(f"frames {len(pts)}, matched {timing.matched_fraction:.4f}")

# IMU: gyro interpolated onto accelerometer timestamps
ta, a = rec.imu["acc"]; tg, g = rec.imu["gyr"]
a = a[:, :3]; g = g[:, :3]
lo, hi = max(ta[0], tg[0]), min(ta[-1], tg[-1])
keep = (ta >= lo) & (ta <= hi)
ta, a = ta[keep], a[keep]
gi = np.stack([np.interp(ta, tg, g[:, k]) for k in range(3)], axis=1)
with open(out / "imu.csv", "w") as f:
    for t, w, acc in zip(ta, gi, a):
        f.write(f"{int(t)},{w[0]:.7f},{w[1]:.7f},{w[2]:.7f},{acc[0]:.6f},{acc[1]:.6f},{acc[2]:.6f}\n")
print(f"imu {len(ta)} samples, {(len(ta)-1)/((ta[-1]-ta[0])*1e-9):.0f} Hz")

cap = cv2.VideoCapture(str(rec.video_path))
names = []
i = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    t = int(timing.timestamps_ns[i])
    cv2.imwrite(str(out / "cam0" / f"{t}.png"), gray)
    names.append(t); i += 1
h, w = gray.shape
print(f"wrote {i} frames at {w}x{h}")
with open(out / "cam0.csv", "w") as f:
    for t in names: f.write(f"{t}\n")

# calibration
R_CtoI = device_to_camera(rec.sensor_orientation_deg)
meta = {"R_CtoI": R_CtoI.tolist(), "width": w, "height": h, "scale": scale,
        "intrinsics_prior": rec.intrinsics().scaled(w, h).__dict__,
        "sensor_orientation_deg": rec.sensor_orientation_deg}
(out / "calib_meta.json").write_text(json.dumps(meta, indent=1, default=str))
print(json.dumps(meta, indent=1, default=str))
