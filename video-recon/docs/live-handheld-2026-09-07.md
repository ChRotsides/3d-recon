# Handheld policy: permissive gates, reachable zero-velocity updates, regular frame cadence

## Decision

The live scanner must keep producing geometry for an ordinary handheld walk. About
one metre of drift is acceptable for now; a paused or frozen cloud is not. Accuracy
work is deferred. Every change below is additive or a relaxation: nothing new can
pause fusion, restart the estimator, or hold the map.

## Gates removed or relaxed (2026-09-07, first pass)

- `TrackingHealth`: fusion is allowed in TRACKING and STATIONARY; the restart latch is
  disabled (recovery timeout infinite); position sigma limit 0.75 m -> 3 m; the image
  track count, accepted/attempted ratio, recent-visual-update and stationary-settle
  gates are gone. Finite pose, monotonic time, speed < 3 m/s, camera gap < 0.75 s and
  the 0.25 m + 3 m/s * dt step check remain and pause fusion only frame by frame.
- A pose discontinuity no longer invalidates map alignment (`mapFrame` receives only
  the RESTARTING flag, which never fires), so relocalization is never required in a
  session; RESET still starts a new map.
- Depth: metric models are trusted (no feature-geometry veto); every valid pixel is
  fused with full weight (no cross-view confirmation); relative-model fits need 4
  anchors in 2 regions at 50% agreement; TSDF surfaces appear at weight 1.
- Startup: `init_dyn_min_deg` 3, the rotation-compensated parallax gate is gone.

## Drift analysis of the 21:32 session (logcat only, see below)

Healthy for 110 s (position inside a 5 m box, sigma 1 -> 8 cm). SLAM landmarks were
last present at 85 s. Four tracker collapses (every track lost) at 96, 113, 117 and
118 s. At 120.5-125 s the phone was held nearly still with ~200 long-lived tracks:
zero visual updates were attempted (no baseline), the one attempt failed
triangulation, and the stationary detector reported `evidence=0 reject=1` as it had
for the entire session. Five seconds of IMU-only propagation took sigma from 9 cm to
78 cm. When motion resumed every update was chi-square rejected, a bad accepted
update jumped the position 1.9 m at 128 s, and the velocity locked at ~2 m/s in a
fixed direction, later 10 m/s. Fusion gates were not involved: they engaged only
after the pose was already wrong.

Root causes: the stationary detector never fired handheld (its history check needs
frames no more than 0.12 s apart over 2.5 s, and 45% of frames were being dropped);
no SLAM landmarks in the second half; the VIO step (30-56 ms) exceeded the camera
interval, so the input queue dropped frames irregularly.

## Changes (second pass)

- `StationaryDetector`: window 1.0-2.7 s instead of exactly 2.5-2.7 s; camera gaps up
  to 0.25 s; track spans are a fraction of the window rather than 40 samples; flow
  limits 5/8 px long, 1.5/3 px per frame, trend 1.5/2.5 px/s (960 px width); IMU
  limits gyro RMS 0.08, gyro mean 0.012 rad/s, accel scatter 0.35 m/s^2, |a|-g 0.6.
  Velocity evidence needs trend < 1.0 px/s (~2.5 mm/s at 2 m with a 750 px focal),
  so a zero-velocity update (sigma 0.03 m/s) costs centimetres per hold. `SpatialMotion`
  accepts 12 tracks. Host tests updated: 42 pass, including a 12 Hz decimated feed.
- `LiveActivity`: adaptive decimation feeds VIO every k-th camera frame where
  k * camera interval > VIO time + 6 ms (step down with 12 ms slack), so gaps are
  regular and the queue no longer drops. Skipped frames are not preprocessed. The RGB
  crop uses bulk row reads. The HUD shows `vio N ms (1/k frames)`; the capture log
  line reports camera interval, decimation and skipped frames.
- `livescan.cpp` applies in memory (the on-device YAML is copied only on first
  install): `init_dyn_min_deg` <= 3, `dt_slam_delay` <= 1 s, `num_pts` <= 200,
  `max_msckf_in_update` <= 30. The bundled YAML matches.
- Diagnostics root quota 200 MiB -> 2 GiB. The 205 MiB of September 6 scans had
  silenced every recording since, which is why the drift session has no IMU CSV.

## Not changed

- Other estimator YAML variants under `sim/config` and `vio/config` keep 10 degrees
  so published simulator results stay reproducible.
- The provisional-weight branch in `tsdf.cpp` is dead code now and is left in place.
- The pose transformer remains a display fallback only.

## Verification

- 42 host native stationary/covariance tests pass (`g++ ... stationary_test.cpp`).
- 67 JVM unit tests pass.
- The native library and the physical handheld behaviour are not verified here:
  build `:app:assembleDebug`, install, walk a lit room, pause for ten seconds, and
  check the log for `stationary evidence=1`, `decimation=`, and steady `sigma`.
