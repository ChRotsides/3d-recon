# Startup bias and stationary-detection repair

This records the earlier bias build. The subsequent recovery implementation and
its newer test/deployment results are documented in [live-recovery.md](live-recovery.md).

The analysis of `scan-1788638765960-4` found a startup gyro bias differing from
later quiet readings by 0.001916 rad/s. Assuming that later window was physically
quiet, this predicts gravity-compensated acceleration growing at 0.018743 m/s³;
the recorded fitted growth is 0.018880 m/s³. Estimated attitude changed 1.063°
while averaged raw acceleration direction changed only 0.048°. This supports
bias-induced tilt/gravity leakage, but is not a controlled ground-truth result.
Read-only analysis and ray/gyro checks are in `/tmp/vio-feature-live.7l0tJu/`.

## Changes

- Require a three-second startup window, both half-window gyro means agreeing
  within 0.0005 rad/s, and six non-overlapping half-second block means agreeing
  with the overall mean within 0.00075 rad/s. Existing gyro magnitude/RMS,
  acceleration, timestamp-gap and image-disparity checks remain. The saved old
  phone configuration is overridden for this startup duration in memory.
- Initialize gyro bias from the complete accepted window, not just its older
  half. Initialize its covariance from block dispersion with a conservative
  0.001 rad/s standard-deviation floor, not a hard-zero uncertainty. Orientation,
  acceleration-bias and velocity initialization remain otherwise unchanged.
- Stationary image evidence now uses signed per-feature slopes over 2.5 seconds
  and a robust affine spatial flow model. Translation, rotation and expansion
  cannot cancel as they could in a simple mean-flow vector. Residual-dependent
  prediction margins, conditioning checks and a scatter cap cause uncertain fits
  to abstain. At least 24 effective tracks and four grid cells with three tracks
  each are required. This is a detector heuristic, not a calibrated statistical
  confidence certificate. Geometry gates in the VIO tracker are unchanged.
- Independent feature jitter no longer gets rectified into positive motion
  before spatial aggregation. Existing motion thresholds, fixed-anchor excursion
  limits, IMU gates, and the stricter direct-velocity gate remain. Legacy
  gyro/gravity and explicit velocity corrections still use the full-state EKF.
  No running state variable is hard-zeroed or visually smoothed.
- A successful stationary EKF correction marks the first completed state update
  in VioManager bookkeeping. Previously, `initialized()` could remain false until
  the first visual propagation/update even with a valid current stationary state.
- Health no longer permanently latches RESET_REQUIRED merely because initial
  stationary readiness precedes a slow first visual acquisition. Before any
  TRACKING state, no fusion permit/map could exist. All uncertainty, speed,
  discontinuity and visual-acceptance checks still reject unsafe poses; fusion
  remains paused. Once visual tracking has occurred, long-loss reset and stale
  depth protections remain unchanged and are regression-tested.
- Startup guidance explicitly requests support and a textured scene, and says to
  wait for initialization rather than an arbitrary five-second countdown.

## Verification

- 37 host native stationary/covariance tests pass. Noisy independent track walks
  (0.12 px per-axis increments) support weak rest in 143/143 evaluated frames,
  but direct zero velocity in 0/143: noisy evidence is not enough for that stronger
  assertion. Adding coherent 0.5 px/s translation yields 0/143 rest or velocity
  acceptances. Existing tilt, jitter, acceleration, missing data, sparse/clustered
  features, slow translation, and full-covariance tests remain passing.
- Recorded-IMU startup-guard replay first accepts near 5.942 s instead of the
  previous app's 4.153 s startup. Its estimated bias is
  `[-0.00006810, 0.00056566, 0.00026787]` rad/s, about 15 times closer to the later
  quiet-window mean than the original bias. This replay does not include image
  startup gates and is not a full VIO replay; sparse saved rays cannot reconstruct
  every full-rate track history for a faithful stationary-detector replay.
- All 48 JVM tests and the debug APK build pass, including empty-map acquisition
  and unchanged post-tracking reset/fusion protections.
- All 16 Android-native feature/geometry tests pass against the actual app native
  library. Full synthetic sequence: 300 frames / 10 seconds, four initial seconds
  of rest, smooth 0.4 m translation over four seconds, then hold. Startup readiness
  before motion is explicitly asserted. Accepted feature-update counts are
  6401/7061, including repeated SLAM observations (not unique points/frames).
  Final 0.97 s raw hold displacement is 0.001079 m and final speed 0.002979 m/s.
  These are estimator outputs without a viewer or fusion—not physical accuracy.
  Final displacement remains about 0.348 m instead of 0.4 m; scale/calibration
  observability remains unresolved. Camera processing median/p95: 9.47/17.59 ms
  without simultaneous real camera/depth/fusion; not a comparable performance
  benchmark against the earlier shorter fixture.

## Remaining validation

The larger/longer stationary witness may still abstain with short-lived tracks,
low texture or genuine hand motion. Three-second startup deliberately rejects
settling and can take longer than three seconds. A marked still → tilt → hold →
translate → stop physical test is still needed, reporting raw estimator pose,
speed, accepted updates and fusion state separately. Do not infer stable pose
from a held viewer or paused fusion.

## Deployment

Installed successfully over wireless ADB and cold-launched on SM-S918B, PID 10768.
Startup log confirms `3s / six stable gyro blocks` and `clone window=31`.
APK SHA-256: `fef1ae5a8d4fc93f68d473f35b41dced2366ecce51574d9b85ec37f9d5fd108d`.
At handoff the phone is keyguard-locked/asleep and no live camera frames have
arrived, so physical startup/pose stability cannot yet be checked. Unlock and
RESET before the supported-still test. No user recordings/calibration were removed.
Generated native cache is preserved at `/tmp/vio-cxx-bias.ocb559/cxx`; tracked
cache files are restored.
