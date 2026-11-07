"""Stream an existing Video Scan recording through the live native estimator.

No extracted image files, server, pose labels, or reconstruction inputs are used.
The result is diagnostic evidence, not a trajectory-accuracy score.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from videorecon.recording import Recording, video_frame_pts_ns, match_video_to_capture_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    rec = Recording.load(args.recording)
    timing = match_video_to_capture_results(video_frame_pts_ns(rec.video_path), rec.frames[:, 1])
    if timing.matched_fraction < .99:
        raise ValueError("Camera timestamp matching below 99%; cannot trust this replay")
    ta, accel = rec.imu["acc"]
    tg, gyro = rec.imu["gyr"]
    print(f"Video/capture matching {timing.matched_fraction:.2%}; "
          f"max accel/gyro interval {np.diff(ta).max()/1e6:.2f}/{np.diff(tg).max()/1e6:.2f} ms", flush=True)
    if max(np.diff(ta).max(), np.diff(tg).max()) > 50_000_000:
        raise ValueError("Recorded IMU has a >50 ms gap; live app would restart, never interpolate it")
    keep = (ta >= tg[0]) & (ta <= tg[-1])
    ta, accel = ta[keep], accel[keep, :3]
    interpolated = np.stack([np.interp(ta, tg, gyro[:, k]) for k in range(3)], axis=1)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(ROOT / ".deps/native-host/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(rec.video_path))
    # Keep native stdout away from the binary input protocol; detailed failures
    # are retained next to the per-frame diagnostics.
    with args.output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen([str(ROOT / ".deps/host-build/live_recording_replay"),
            str(ROOT / "android/app/src/main/assets/vio/estimator_config.yaml"), str(args.output)],
            stdin=subprocess.PIPE, stdout=log, stderr=log, env=env)
        try:
            imu_index = 0
            for frame, ns in enumerate(timing.timestamps_ns):
                if args.max_frames and frame >= args.max_frames:
                    break
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"Video ended before timestamp {frame}")
                # Supply interpolation endpoints ahead of the camera, as the live
                # queue does. This is offline replay, not a latency benchmark.
                while imu_index < len(ta) and ta[imu_index] <= ns + 100_000_000:
                    values = [ta[imu_index] * 1e-9, *interpolated[imu_index], *accel[imu_index]]
                    process.stdin.write(("I " + " ".join(f"{x:.12f}" for x in values) + "\n").encode())
                    imu_index += 1
                gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (960, 540), interpolation=cv2.INTER_AREA)
                process.stdin.write(f"C {ns*1e-9:.12f} 960 540\n".encode())
                process.stdin.write(gray.tobytes())
                if frame % 300 == 0:
                    print(f"replayed {frame} frames", flush=True)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"Native replay failed; see {args.output.with_suffix('.log')}")
        finally:
            cap.release()
            if process.poll() is None:
                process.terminate()
                process.wait()
    data = np.genfromtxt(args.output, delimiter=",", names=True)
    initialized = data["initialized"] == 1
    print(f"Frames {len(data)}; initialized {initialized.sum()}; "
          f"accepted visual updates {data['accepted'].sum():.0f}; "
          f"processing median/p95 {np.percentile(data['camera_ms'], [50,95])} ms")
    if initialized.any():
        print(f"First initialized frame {data['t'][initialized][0]-data['t'][0]:.3f} s; "
              f"max speed {data['speed'][initialized].max():.3f} m/s; "
              f"max position sigma {data['position_sigma'][initialized].max():.3f} m")


if __name__ == "__main__":
    main()
