# Live feature tracking repair

Feature detection was already enabled (`use_klt=true`). In the last complete
handheld run, all 1876 visual candidates failed before an EKF update: 631 lacked
surviving observations, 1232 failed triangulation, 11 failed refinement and two
failed the residual gate. Enabling fusion would not fix this.

## Changes under test

- The local OpenVINS source collected oldest-clone tracks when clone count was
  either above the configured maximum **or above five**. The latter clause used
  active tracks prematurely. The app-generated copy now waits until exceeding
  the configured clone window. Lost tracks are still processed normally.
- The window is increased from 11 to 31 clones (about one second at 30 Hz).
  An on-phone synthetic slow translation produced 0/637 accepted visual feature
  updates with 11 clones versus 4451/4871 with 31. The normal covariance-managed
  clone lifecycle is retained; triangulation thresholds are not relaxed. The
  larger covariance costs processing time, which must also be checked live.
  Existing saved YAML with the legacy value 11 is upgraded in memory to 31;
  saved calibration files and other explicitly custom window values are preserved.
- Newly detected tracks now retain their detection-frame observation, including
  features added to the previous image before tracking forward. Only newly
  allocated IDs receive the added observation; measurements already consumed by
  the filter are not reinserted. Tests check monotonically increasing timestamps.
- Forward optical-flow status, image bounds, finite values, patch appearance and
  round-trip consistency are checked before geometric fitting. Invalid pairs no
  longer participate in RANSAC. Geometry checks remain enabled; no visual update
  is forced through the estimator's residual gate.
- Stationary/planar image geometry is considered separately from the fundamental
  matrix used for general two-view geometry. This selects correspondences, not
  a camera pose or a zero-motion state.
- Startup-window diagnostics are copied from the joined initialization worker
  back to the calling frame. This fixes the earlier all-NaN startup columns
  without changing the startup guard's estimator decisions.

The existing timing, per-frame calibration, rectification, stationary correction,
tracking-health checks and fusion permits are retained. External dependency source
and archive files are not modified: eight checked translation units are compiled
into the app (`VioManager`, `VioManagerHelper`, three updaters, `StaticInitializer`,
`TrackKLT`, `FeatureInitializer`). No class layout changes are introduced.

The matching checks use a 30-intensity-unit mean patch-error cap and a 1 px
round-trip limit. RANSAC sees only surviving finite pairs. A homography is preferred
when at least ten matches and 80% of the candidates support it within 1 px;
otherwise the existing 2 px fundamental-matrix threshold is used. A dominant
plane can therefore exclude some valid foreground tracks; multi-depth synthetic
coverage checks guard against always forcing a planar model. These operating
limits still need evaluation under real lighting, motion blur and scene depth.

The occlusion fixture exposed a 51 px wrong match that passed round-trip checking,
patch filtering and the fundamental-matrix fit. Being close to an epipolar line
does not establish the correct position along it. The stronger dominant-plane
model is intended to reject that case without bypassing the later VIO checks.

## Evidence recorded by the next build

The frame CSV has 91 columns. New counters distinguish forward flow, acceptable
patch appearance, round-trip checks, geometric inliers, track history lengths,
clone count, seeded/removed observations, and triangulation failure categories.
Triangulation categories are mutually exclusive in this priority order:
non-finite values, conditioning, behind camera, too close, too far. Successful
triangulation is separate from successful refinement and actual EKF acceptance.
These detailed triangulation counters cover MSCKF candidates, not viewer-only
re-triangulations or SLAM delayed initialization. Depth/conditioning thresholds
are not loosened; non-finite geometry is explicitly rejected.

Two additional bounded local CSVs are sampled at most five times per second:

- `features.csv`: timestamp, ID and normalized image ray coordinates, at most
  64 points per sampled frame.
- `geometry.csv`: matching timestamp, camera/IMU offset, gyro bias and row-major
  rotation from IMU to camera.

These can compare measured feature-ray motion with recorded gyro motion. They
contain no images, patches, descriptors or audio and are not uploaded. The same
120 s / 12 MiB session cap, bounded queue and total-log quota apply. They do not
provide ground-truth camera motion, and translation/rolling shutter can also
contribute to a ray-motion discrepancy.

## Learned feature replenishment

The Android live tracker can replenish KLT tracks with the official pretrained
XFeat model from `verlab/accelerated_features` at commit
`e92685f57f8318b18725c5c8c0bd28c7fe188d9a`. The source and model are covered
by the bundled Apache-2.0 license. The checked weight file has SHA-256
`0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b`.

The bundled ONNX graph uses a fixed 320x192 grayscale input. A 960x540 camera
image is resized to 320x180 and centered with six pixels of vertical padding.
Model coordinates are mapped back using pixel centers, so scale and padding are
both accounted for. The graph exports the full official descriptor, 65-channel
keypoint-logit and reliability outputs. Ranking uses the official non-maximum-
suppressed keypoint probability multiplied by bilinearly sampled reliability;
the graph also exports the top ranked indices and scores so Android does not
copy the three large tensors through JNI. The graph preserves the official descriptor output, while the Android runtime
requests only ranked scores and indices and persistent feature identity still
comes from the existing KLT tracks.

Each result is keyed by its exact camera timestamp and source dimensions. The
OpenVINS top-up for an old image can consume only the result made from that same
image. Candidates then pass the existing mask, edge, 12-pixel spacing and 6x4
grid limits before receiving normal OpenVINS IDs and observation timestamps.
The live target remains 250 spatially distributed tracks. KLT follows accepted
points at camera rate.

Inference is scheduled only when the latest diagnostics report at most 230
tracks. The initial interval is 200 ms (a 5 Hz cap), and the interval adapts to
measured detector wall time to keep sustained learned work near a 35% duty
budget during prolonged low-track scenes. ONNX Runtime uses two intra-op
threads, one inter-op thread and sequential execution. If model setup or
inference fails, XFeat is disabled and the existing native FAST replenishment
resumes. A scheduled result that is absent or mismatched is never substituted
with a result from another frame.

On the SM-S918B, the standalone synthetic-frame benchmark measured median/p95
times of 11.5/11.7 ms for resize and preprocessing, 95.2/96.8 ms for ONNX
inference, and 4.5/4.6 ms for ranked-coordinate mapping, with 928 valid
candidates from 1,000 ranked outputs. This benchmark does not include the live
camera, KLT, depth or fusion workload and is not evidence of improved physical
tracking or reduced drift. Existing calibration, timing, stationary, health,
fusion and recovery gates are unchanged.

## Verification

`feature_tracking_test.cpp` is an Android-native executable linked to the actual
app library. It exercises stationary/translated/rotated/occluded synthetic images,
track-history preservation, triangulation with and without baseline, and an
optional synthetic camera-plus-IMU estimator sequence. It is excluded from APK
packaging via the Gradle native target list. Build it explicitly with the generated
CMake/Ninja `feature_tracking_test` target and run through ADB with the matching
`liblivescan.so`, OpenCV and C++ runtime. Passing the optional estimator YAML path
enables the synthetic full-estimator test.

### Measured results, 2026-09-05

- Gradle debug APK build succeeded; all 47 JVM tests pass. The existing 25 native
  stationary/startup/velocity tests also pass.
- All 15 Android feature/geometry tests pass, including the full VIO sequence.
  Translation retains 99 accurate tracks (mean error 0.00046 px); the occlusion
  fixture retains 29 (worst error 0.00131 px). These very small errors are from
  generated images, not physical camera accuracy. A two-depth fixture chooses
  the fundamental matrix and preserves tracks in both depth layers.
- Actual TrackKLT retains 101 tracks for at least 15 observations in a 20-frame
  sequence. Zero-baseline triangulation is still rejected.
- The full VIO fixture uses 240 camera frames at 30 Hz, IMU at 200 Hz, a textured
  plane at 3 m, a smooth 0.4 m translation over four seconds, and a final hold.
  With identical inputs, 11 clones produce 0/637 accepted feature updates;
  31 clones produce 4451/4871. These are accumulated feature-update counts,
  including repeated SLAM observations, not unique features or camera frames.
  Startup diagnostic handoff is observed exactly once.
- Final 0.97 s hold: raw estimator displacement 0.0000212 m and final speed
  0.000861 m/s. No viewer or fusion is present in this native test, so these
  numbers cannot result from a paused display. They are not real-world drift
  measurements. Estimated final IMU displacement is about 0.272 m versus the
  synthetic 0.4 m translation: scale/bias/calibration/startup observability
  remains unresolved by this fixture.
- Camera processing median/p95 on the SM-S918B is 16.58/35.03 ms in this fixture,
  without simultaneous live camera/depth/fusion load. Live timing still needs
  checking, especially with the larger covariance and extra backward LK pass.

Synthetic tests are not a substitute for a marked live scan. The next physical
test should establish accepted visual corrections during slow translation, then
measure raw estimator motion during the stop separately from fusion status.

### Deployment

Installed over wireless ADB and cold-launched on SM-S918B. Final APK SHA-256:
`6fc33252378db22c52328a2443630085ec8a7a081407438d9b1a569ab7e6b363`.
PID 27893 startup confirms bidirectional KLT and **clone window=31**, with the
improved 960x720-to-960x540 calibrated crop and stationary policy retained.
The first unmarked live frames show roughly 170–200 feature tracks, but no
accepted visual updates yet; health remains LOST and fusion remains paused.
No controlled movement was requested during this smoke check. Real-world pose
stability is therefore **not** established. Recording:
`scan-1788638623713-1`; reset before the requested short motion test.

Generated native build cache was preserved at `/tmp/vio-cxx-feature.B8v289/cxx`
and tracked cache files restored; user source changes and recordings retained.
