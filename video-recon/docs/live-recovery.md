# Automatic tracking recovery (improved pipeline, not rollback)

## Observed failure

The previous bias build (`scan-1788639781580-4`) initialized after approximately
80.1 seconds. It subsequently obtained real visual support: SLAM reached 33/33
accepted observations and fusion was allowed. Tracking then deteriorated and the
health policy latched RESET_REQUIRED after only one second. At that transition
position sigma was about 0.069 m and speed about 0.126 m/s. Some earlier rejected
frames actually had seven accepted observations but a poor overall acceptance
ratio, because many other candidates were immature.

This was not merely a false warning: after sustained loss of visual support the
raw estimator eventually diverged severely. Removing the warning or permitting
fusion throughout the loss would corrupt the cloud. The earlier quiet interval
had roughly 2.9 mm raw net displacement over 28.67 seconds with fusion paused;
that is neither controlled ground truth nor evidence of stable moving tracking.

## Implementation

- Pause fusion immediately when existing quality checks fail, but allow three
  seconds to regain support. Six or more accepted constraints are no longer
  rejected merely because many other candidates failed. Existing uncertainty,
  speed, step, timestamp and asynchronous fusion-permit checks remain.
- A persistent failure triggers automatic recovery, not a permanent manual-reset
  latch. Under the session write lock, save the complete extracted TSDF surface
  (or raw comparison cloud), then destroy/recreate the estimator and volume.
  Generation and health epochs prevent old depth jobs entering the new segment.
- Save binary little-endian XYZ/RGB PLY files under the app's external
  `files/segments/` folder. Files explicitly declare independent local frames:
  **there is no relocalization or automatic alignment between segments**.
  Full TSDF surface export is not limited by the viewer point cap and does not
  archive internal TSDF weights. Temporary files are synced and renamed before
  the old map is destroyed. Save failures retain that map and report an error.
  The 512 MiB archive quota never automatically deletes completed segments.
- Camera callbacks carry a separate lifecycle generation and serialize with
  closure. A late `onConfigured` callback after pause previously crashed with
  `CameraDevice was already closed`; the first integration run exposed this.
  Old callbacks now close/abstain instead of starting a closed camera session.
- In the same process, retain a recent gyro-bias prior only after a second of
  confirmed low-speed stationary evidence and a successful explicit velocity
  correction. Reinitialization can use that bias, with a variance floor and age
  inflation, rather than learning bias from the new settling motion. It expires
  after 120 seconds, checked again at actual initialization. Both IMU-window
  quietness checks and the existing image-disparity gate still apply. No running
  estimator state is hard-zeroed. The conservative cold-start path is unchanged.

## Verification

- 55 JVM tests pass, including brief recovery, long-loss restart, stale fusion
  rejection, accepted-observation counting, file atomicity, quota/failure
  handling and warm-prior lifetime/evidence.
- 41 host native stationary/covariance tests pass. Additional cases verify that
  warm initialization retains the supplied bias/variance rather than relearning
  settling motion, and rejects expiration, invalid values and clear rotation.
- 17 Android-native feature/geometry tests pass against this build's library.
  Complete TSDF export round-trips all 2067 fixture points and preserves the map
  after an intentionally failed save. The 300-frame synthetic VIO sequence still
  accepts 6401/7061 visual updates; final 0.97-second hold displacement is
  0.001079 m and final speed 0.002979 m/s. End displacement remains about 0.348 m
  for the synthetic 0.4 m translation; the scale error is not resolved here.
- `RecoveryInstrumentation` is a separate test-APK runner exercising actual
  Activity/JNI recovery with camera/IMU producers paused and synthetic inputs.
  It checks full archive creation, empty new volume, generation transition,
  rejection of stale fusion permission, and return to INITIALIZING.
  It passed on SM-S918B: 2067 points saved to
  `segment-3038450067747048981.ply`; generation advanced to 3, the new volume
  was empty and the old fusion permit was rejected. Logs confirm
  `automatic recovery: generation=3 warm_bias=true`. The synthetic restart
  request at 23:42:31.605 reached native-ready at 23:42:31.670 (65 ms), which
  excludes subsequent estimator initialization and is not handheld recovery time.

These are software regression checks, not a new measured physical motion test.
Cold-start time, warm-restart time during actual handheld operation, moving pose
accuracy and frequency of segment breaks remain to be measured. This change
addresses recovery usability; it does not claim to cure all underlying VIO drift.

Build APK SHA-256:
`e0ed04be1669a6b4aaba8fc5907abc58cfd6a6bf4bfc2af0e42a899ce839b1d6`.
Installed successfully over wireless ADB; the ordinary live activity was launched
again after the test. Existing recordings/calibration and completed clouds were
not deleted. App installation restarted the prior volatile live session.

The subsequent uncontrolled live check (PID 23755) repeatedly initialized in
roughly three seconds but obtained zero accepted visual updates, then restarted
with rising uncertainty. Fusion remained paused. A screenshot at approximately
23:43 shows an almost-black camera preview (`/tmp/vio-recovery-screen.png`).
This confirms the automatic path also triggers live, but **does not validate
usable tracking**. A covered/dark scene is plausible, not confirmed; a visible,
well-lit textured scene is required before assessing normal handheld behavior.

## Reproduce the integration check

Build `:app:assembleDebug :app:assembleDebugAndroidTest`, install both APKs, then:

```sh
adb shell am instrument -w com.recon.videoscan.test/com.recon.videoscan.live.RecoveryInstrumentation
```

This restarts the app for a synthetic scene and leaves a small independent test
PLY in `segments/`; do not run during a valuable unsaved live scan. It is not
included in the ordinary app APK and does not delete recordings or saved clouds.
