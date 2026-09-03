# Video + sensors → room geometry (no ARCore)

The phone records ordinary **video**, its **raw motion sensors** (accelerometer,
gyroscope, magnetometer and Android's fused rotation vectors) and the camera's
own **intrinsic calibration**. Nothing from ARCore is used anywhere.

Offline, each source does the one thing it is good at:

| Quantity            | Comes from                                        |
| ------------------- | ------------------------------------------------- |
| Orientation (tilt, pan, roll) per frame | rotation-vector sensor, slerped to the frame's timestamp |
| Gravity / which way is up | the same orientation                          |
| Heading (north)     | magnetometer, via the difference between the magnetic and game rotation vectors |
| Position per frame  | structure-from-motion on the video (COLMAP)       |
| Metric scale        | the floor: gravity from the sensors says which way is down, the lowest dense slab of sparse points is the floor, and the phone was held ~1.5 m above it (`--camera-height`). The accelerometer estimate is computed too but rejected when its spread is large, which on a real walk it always was |
| Camera intrinsics   | `LENS_INTRINSIC_CALIBRATION`, centre-cropped and scaled to the video; SfM refines the focal length |
| Dense depth         | Depth Anything 3, pose-conditioned, anchored to COLMAP's sparse points |
| Surface             | TSDF fusion (Open3D) → mesh + point cloud in the existing viewers |

The sensors cannot give position on their own: double-integrating a phone
accelerometer drifts by metres within seconds. That is why the video carries
position and the IMU is used only where it is reliable — orientation and
gravity. Measured on a real 67 s walk, the sensors' orientation agreed with
COLMAP's to a median 2.7° over 430 keyframes; the accelerometer scale was off
50x (walking steps alias against 4 Hz keyframes), while the floor-based scale
agreed with DA3's metric checkpoint to within 3 %.

## Layout

```
video-recon/
  android/            Camera2 recorder app (com.recon.videoscan), standalone Gradle project
  videorecon/         Python package
    recording.py        reads meta.json / frames.csv / imu.csv; frame ↔ IMU timing; intrinsics
    imu.py              sensor conventions, slerp, gravity removal, integration
    frames.py           blur scoring, keyframe selection by rotation/time, frame extraction
    sfm.py              COLMAP (pycolmap) wrapper → poses + sparse points
    align.py            SfM ↔ sensor rotation alignment, IMU scale, metric frame
    depth.py            writes a SessionBundle and runs the repo's DA3 adapter
    fuse.py             TSDF fusion, mesh/cloud export, viewer copies
    pipeline.py         the stages in order, cached, with manifest.json
  scripts/reconstruct.py   CLI
  scripts/pull.sh          adb pull the newest recording
  tests/                   unit tests (synthetic data; no phone needed)
  docs/                    design and recording format
```

## Record

```bash
cd video-recon/android
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
./gradlew :app:assembleDebug
~/Android/Sdk/platform-tools/adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Open *Video Scan*, press **Start recording**, walk the room slowly with the phone
upright, overlap everything you look at, come back to where you started, press
**Stop**. The HUD shows the camera clock (must be `REALTIME`), whether the camera
publishes intrinsics, and the live sensor rates.

Recording rules that matter for the geometry:

* **Translate as well as turn.** SfM needs baseline. Turning on the spot gives a
  panorama, not a room. A slow lap of the room, camera pointed inwards, is ideal.
* **Loop back** to where you started so the trajectory can close.
* Stabilisation is off and focus is fixed by the app; do not zoom.

## Watch it happen

```bash
python video-recon/scripts/monitor.py out/video/<stamp> --log run.log --port 8778
```

A read-only dashboard at http://localhost:8778 (reachable on the LAN) that
polls the output directory and COLMAP's database: stage, keypoints per frame
with overlays, the keyframe × keyframe match matrix filling in live, the most
recently verified pair with its inlier lines, the chain of neighbour matches,
mapping progress, the sparse model and camera path in 3D (grows from the
mapper's snapshots), the dense cloud republished every 10 fused frames, the
alignment and scale numbers, and the final viewers. Its **Walk** page
(`/walk`) plays the recording upright next to a WebGL view of the fused cloud
whose camera follows the reconstructed path in sync with the video clock —
first-person, follow, or orbit with the path drawn.

## Reconstruct

```bash
video-recon/scripts/pull.sh                 # -> data/recordings/rec_<stamp>
source ~/ai/env.sh kepler
python video-recon/scripts/reconstruct.py data/recordings/rec_<stamp> out/video/<stamp>
cd out/video/<stamp> && python -m http.server 8777 --bind 0.0.0.0    # index.html = cloud, mesh.html = mesh
```

Stages and their outputs inside `out/video/<stamp>/`:

1. `keyframes.json`, `images/` — frames chosen by orientation change (default every
   8°) or every 0.6 s, sharpest of the neighbours, long edge 1280 px.
2. `colmap/`, `sfm.json` — poses and sparse points from COLMAP.
3. `bundle/` — a `SessionBundle` in the repo's `recon_world_v1` frame (Y up, metric),
   with COLMAP's points as `sparse_points`. Anything in the repo that reads a
   bundle works on it.
4. `depth.npz` — DA3 depth per keyframe (`--variant small|base|large|metric-large`).
5. `mesh.ply`, `mesh.bin/json`, `cloud.bin/json`, `index.html`, `mesh.html`.
6. `manifest.json` — every number: sensor rates, timing match, keyframe count,
   SfM registration and reprojection error, sensor-vs-SfM rotation residual,
   both scale estimates (IMU and floor) and which was used, fusion sizes, warnings.

Stages are cached; `--force` redoes them all. To redo only from alignment on
(for a different `--camera-height`, say), delete `depth.npz` and rerun without
`--force`: keyframes and COLMAP are reused, the bundle, depth and fusion are
rebuilt. `--no-depth` stops after the bundle, which is enough to inspect poses.

Keyframe density matters more than anything else for COLMAP on a room: the
defaults (every 3° of rotation or 0.25 s, 1600 px) registered 430 of 434
keyframes on the reference walk where a sparser 8° / 0.6 s / 1280 px selection
registered 41 of 160 and broke into seven models.

If SfM cannot register the images (a static or purely rotating capture) the
pipeline falls back to **rotation only**: every keyframe at the origin, oriented
by the sensors, no scale. That is what the sensors alone can honestly provide,
and the manifest says so.

## Test

```bash
~/ai/envs/kepler-gpu/bin/python -m pytest video-recon/tests -q
cd video-recon/android && ./gradlew :app:testDebugUnitDebug
```

Python dependencies beyond the repo's: `pycolmap`, `open3d` (both installed
into the Kepler environment). `ffprobe` must be on the PATH.

See `docs/design.md` for the conventions (device, camera and world frames, the
scale estimator) and `docs/recording-format.md` for the on-disk format.
