# Design notes

## Why not the IMU for position

A phone accelerometer has a bias of a few cm/s² and its orientation estimate a
degree or so of error; both integrate twice. After one second the position
error is centimetres, after ten it is metres. No filtering fixes that without
another position observation, which is what the camera provides. Every
visual-inertial system, ARCore included, works this way; this pipeline just
does the visual part offline and transparently with COLMAP so its failures can
be inspected rather than guessed at.

## Frames

Three right-handed frames, converted in exactly one place each (`videorecon/imu.py`):

* **Device** (Android): X right, Y up along the screen (portrait), Z out of
  the screen. The back camera looks along −Z.
* **Camera** (OpenCV): x right in the image, y down, z forward. The image is
  the sensor-native one, so for `SENSOR_ORIENTATION = 90°` the camera's x axis
  is the device's −Y and its y axis is the device's −X (`device_to_camera`).
* **World**: the sensors' East-North-Up, converted to the repo's `recon_world_v1`
  (X east, Y up, Z south) so that `harness/recon` reads the result unchanged.
  Poses in the bundle are world←camera with the camera looking down −Z (OpenGL),
  the same as every other bundle in the repo.

## Timing

Camera frames and sensor samples share `elapsedRealtimeNanos` when the camera
reports its timestamp source as REALTIME. Container PTS from `ffprobe` are
matched to the logged `SENSOR_TIMESTAMP`s by nearest neighbour after fitting
the offset, and the orientation is slerped to each frame's timestamp. The
manifest reports the fraction matched within a quarter frame.

## Keyframes

Due when the sensors say the camera has turned 8° since the last keyframe, or
0.6 s has passed (translation is invisible to the IMU, so time is the fallback).
Among the ±3 frames around a due point the one with the highest variance of
Laplacian wins. Frames are resized to a 1280 px long edge for both SfM and DA3.

## Alignment (Wahba's problem)

For every registered keyframe there are two rotations of the camera: COLMAP's
and the sensors'. One global rotation `R` minimising Σ‖R_imu − R·R_sfm‖² is
found by SVD. The per-frame residual after alignment is the direct measure of
how much the two agree; the manifest warns above 15° at p95. With the game
rotation vector the yaw of the world is arbitrary; the magnetic rotation vector
supplies the median yaw offset so that −Z in the output points north when the
magnetometer is trusted (its spread across the capture is reported).

## Scale

Two estimators run; the manifest records both and which was used.

**Floor** (used by default). Gravity from the sensors fixes "down". Below the
cameras the sparse points pile into horizontal slabs — table tops, counters,
seats, and lowest of all the floor. Tables are often denser than the floor, so
the floor is the *lowest* slab holding at least 40 % of the densest slab's
points; only outliers lie under it. With the phone held at `--camera-height`
(1.5 m by default) the scale follows. On the reference walk this gave
0.477 m/unit; DA3's metric checkpoint, run pose-free on eight keyframes as an
independent check, gave 0.47–0.49.

**Accelerometer** (computed, rejected when its spread exceeds 35 %).
For three consecutive keyframes i, j, k closer together than 0.8 s,
p(t) = p_i + v_i (t − t_i) + ∬a, so

    (p_k − p_i)/Δt_ik − (p_j − p_i)/Δt_ij = D_ik/Δt_ik − D_ij/Δt_ij

where D are double integrals of gravity-free world acceleration from t_i. The
unknown velocity cancels. The left side is in SfM units, the right in metres;
one robust least squares over all triplets (30 % weakest dropped, 3-MAD band)
gives metres per unit. On a synthetic smooth walk with 5 cm/s² noise and a
3 cm/s² bias the estimate is within 5 %. On the real walk it was wrong by 50x
with a 2900 % spread: a person's steps put ~2 Hz oscillations into the
accelerometer that keyframes at ~4 Hz cannot sample, so the second difference
of keyframe positions measures aliasing, not motion. Fixing that properly means
IMU preintegration with bias states between *every* video frame, which is a
visual-inertial estimator, not a post-hoc fit.

## Dense depth and fusion

The metric bundle goes through `recon.depth.da3.DA3Depth` unchanged:
pose-conditioned DA3 in windows, then a per-frame scale anchor against the
sparse points, which are now COLMAP's. Fusion is Open3D's scalable TSDF at 2 cm
with a 12 cm truncation band, the same flying-pixel removal and debris
clustering as `scripts/capture/mesh.py`, and the same `cloud.bin` / `mesh.bin`
layout so `viewer/cloud.html` and `viewer/mesh.html` open the result directly.
