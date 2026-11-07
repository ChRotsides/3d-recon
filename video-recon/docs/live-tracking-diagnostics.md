# Improved live diagnostic build

This build retains synchronized sensor feeding, per-frame calibration, rectified
depth, and the native-width capture/crop path. It is not the rollback APK.

Sections below retain the experiment history. The latest policy is described in
"Handheld/startup revision" at the end; it supersedes the first strict detector.

## Health and fusion

The estimator is allowed to continue running when unhealthy so its failure can be
measured. Raw estimates always go to the diagnostic CSV. The viewer holds its last
reliable pose with an explicit status; that held display is **not evidence that
the estimator stopped drifting**.

Current-frame active 2D tracks, attempted MSCKF/SLAM feature measurements, and
measurements actually passed through `StateHelper::EKFUpdate` are separate counts.
Attempted minus accepted is an aggregate of all non-acceptances (including
triangulation/gating); it is not a count of chi-square rejections alone. Delayed
landmark initialization and exported visualization landmarks are not counted as
accepted updates. Stationary updates are reported independently.

The conservative guard requires 20 image tracks, at least 6 accepted feature
measurements within 0.5 seconds, and three supported frames before enabling
fusion. It rejects high rejection rates, non-finite/mismatched state timestamps,
large pose discontinuities, speed above 3 m/s, or position marginal standard
deviation above 0.75 m. Accepted stationary constraints can support a stationary
pose without triangulated landmarks, but speed must settle below 0.15 m/s;
fusion remains paused during stationary frames. These thresholds are diagnostic
heuristics, not guarantees of pose accuracy or statistically calibrated confidence.

Loss invalidates all outstanding depth-job permits. Both raw point accumulation
and TSDF commits check the permit under the same lock used for health changes.
A camera/IMU stall is checked by the UI watchdog and at commit. A loss lasting a
second after reliable tracking latches RESET_REQUIRED; automatic recovery cannot
join a possibly displaced coordinate system to the existing map. Relative-depth
models additionally require current-frame anchors, not a reused historical scale.

## Local recording

Each reset starts `files/diagnostics/scan-<wall-clock-ms>-<generation>/`:

- `imu.csv`: raw Android accelerometer and gyroscope samples with boot-clock timestamps.
- `frames.csv`: camera timestamp, health/reason, **raw** pose, track/update counts,
  stationary flag, speed, position uncertainty, offset and calibration; also velocity
  and gravity-compensated world acceleration before/after stationary corrections,
  and the trace of the velocity marginal covariance.
- `summary.txt`: recording termination reason, dropped records and data bytes.
- `events.csv`: boot-clock phase markers from the on-screen test button.

Recording is asynchronous with a 4096-row bounded queue, stops after 120 seconds or
12 MiB of rows, and does not start a new recording if existing logs already exceed
200 MiB. No captures are deleted automatically. There are no images, audio or
uploads. NaN post-stationary fields mean no completed stationary correction.

The four instrumented OpenVINS translation units are generated at CMake configure
time. Hook locations and occurrence counts are checked; changed upstream sources
fail configuration. Original external sources and the static archive are not
modified, and the estimator's class layout is unchanged. Instrumented definitions
are linked ahead of the archive. Camera feeding waits for initialization under
the native session lock, but OpenVINS still runs initialization in a joined worker
thread. Startup thread-local diagnostics therefore need an explicit handoff (see
the latest marked result below); ordinary update diagnostics run on the VIO thread.

## Stationary-velocity hypothesis: validation baseline

The stock active branch uses angular-rate and gravity-compensated acceleration
residuals, with Jacobian blocks for orientation and gyro/accelerometer bias. It
does not directly observe velocity. Cross-covariance can change velocity indirectly,
but a zero residual does not remove an existing false velocity. The propagator
already subtracts gravity and includes the previous velocity in position advance.

A numerical linear-Kalman example with initial velocity 0.4 m/s, zero inertial
residual, and independent velocity covariance retains 0.4 m/s under the existing
measurement structure. A direct zero-velocity measurement with variance 0.0004
(m/s)^2 and prior velocity variance 0.04 (m/s)^2 instead gives 0.0039604 m/s, with
positive posterior covariance. This establishes a possible mechanism, not its
frequency or sole causality on the phone.

The baseline diagnostic build kept the stationary estimator math unchanged.
Its measured results supported adding an explicit velocity observation; see below.
No untested branch, hard-zero state assignment, or display smoothing is used.

## Baseline phone evidence (2026-09-05)

The complete baseline recording contains 3549 frame rows over approximately 119 s,
with zero reported dropped records. It remains on the phone under
`files/diagnostics/scan-1788635067188-1` and was pulled to
`/tmp/vio-baseline-1788635067188/` on the development machine.

The user's start acknowledgement was aligned approximately with boot time
106365.37 s. In the following 15 s (phone held upright **in the hand**, not fixed
to a ground-truth reference), raw estimated position moved a net 1.241 m. Median
estimated speed was 0.408 m/s. Across 304 accepted stationary updates in this
window, mean speed immediately before/after was 0.38883 / 0.38814 m/s; mean
gravity-compensated acceleration norm was 0.22466 / 0.22307 m/s². Only two visual
feature measurements were accepted in this interval. This is direct evidence
that the existing stationary correction did not substantially remove accumulated
velocity in that run, consistent with the inspected measurement structure.

The full recording had 27 INITIALIZING and 3522 LOST frames: fusion was paused
while the **raw estimator** drifted. The held viewer pose is not a stability result.
Timing of the user's start has several seconds of uncertainty. The subsequent
motion was approximately 15 s of forward tilt, some holding, then approximately
10 s of rotation and another hold. No intentional translation was performed.
Those intervals cannot honestly be reported as a labelled translate/stop test.
Orientation, IMU bias, camera/IMU timing, weak visual updates, and the cause of the
initial false velocity remain unresolved contributors.

## Explicit velocity correction build

The app-generated stationary updater now requires sustained image **and** IMU
evidence rather than the original disparity override/estimated-speed gate:

- At least 0.75 s of image history; at least 40 matching features spanning four
  of nine image cells; strict long/short displacement limits. A fixed reference
  while stationary prevents an indefinitely sliding comparison from overlooking
  accumulated slow movement.
- At least 0.70 s of IMU coverage within a 0.75 s window, sufficient samples,
  bounded gaps, gyro RMS below 0.012 rad/s, acceleration standard deviation below
  0.08 m/s², and mean acceleration magnitude within 0.35 m/s² of gravity.
- Estimated velocity is not used as evidence that the device is physically still.

After the existing orientation/bias correction, a separate measurement `z=0`,
`H=I` on world velocity uses the full-state `StateHelper::EKFUpdate`, including
cross-covariances. Measurement standard deviation is 0.03 m/s, at most 5 Hz.
Innovation gating uses the velocity marginal covariance and a 16.27 threshold
(three degrees of freedom). Invalid or inconsistent innovations are rejected,
not forcibly repaired. Both original experimental flags remain false.

Additional frame columns report detector evidence, image motion, gyro RMS,
acceleration variability, feature coverage, velocity after the legacy correction,
and explicit-update result (1 applied, 0 not applied, -1 rejected). Original
before/after acceleration, velocity and covariance fields encompass both updates.
NaNs mean that stage was not reached. Legacy OpenVINS chi-square log text does
not describe the new physical-stationarity gate; use these CSV fields.

Fourteen standalone native detector/measurement tests pass, including translation,
rotation, poor coverage, sensor gaps, translation-to-stop, accumulated slow creep,
innovation rejection, covariance positivity and cross-covariance correction.
In the synthetic independent-covariance example, the actual 0.03 m/s setting
reduces 0.4 m/s to 0.00880196 m/s. All 46 JVM tests and the debug APK build pass.
These are software checks, **not measured accuracy of the new build on the phone**.
The APK was installed successfully over wireless ADB and cold-launched on the
Samsung SM-S918B (process 9404). Startup logs confirm the 960x720 capture with
central 960x540 crop and ongoing stationary-update policy. This launch alone is
not a successful tracking test: early frames still reported rejected visual
measurements and growing uncertainty while fusion remained paused.

The detector is deliberately conservative and may decline handheld holds. Slow
translation against distant scenery remains weakly observable; synthetic tests
do not guarantee its preservation in every real scene. Noise/threshold choices
need physical validation. A strongly overconfident divergent state may reject the
new measurement and require reset rather than recover automatically.

## Required controlled comparison

After installation, press RESET to start a fresh two-minute recording. Press the
phase button at the beginning of each phase: STILL (15 s), TILT (5 s), HOLD
(15 s), TRANSLATE (move about 30 cm sideways over 5 s without rotating), STOP
(15 s), DONE. Support the phone during STILL/HOLD/STOP where practical; keep a
well-lit textured scene in view. Allow for touch-induced motion when analysing
the first two seconds after each marker. Report raw pose displacement, velocity
before/legacy/after, detector and innovation rejections, accepted visual updates,
and health/fusion status separately. The first marked new-build test is reported
below; it failed and does not validate deployment.

## First marked new-build result: FAILED (2026-09-05)

Recording `scan-1788635773301-2`, pulled to
`/tmp/vio-marked-1788635773301/`, contains 2460 frame rows over 81.94 s and all
six phase markers. Summary: `status=saved`, `dropped_records=0`. Analysis excludes
the first two seconds after each phase marker to reduce tap-related motion.
The user subsequently confirmed the phone was held in the hand during the rests;
these labels are not external ground truth.

| Marked phase | Full duration (s) | Analysed raw net displacement (m) | Median estimated speed (m/s) |
| --- | ---: | ---: | ---: |
| STILL | 18.10 | 301.93 | 15.42 |
| TILT | 8.02 | 500.45 | 83.01 |
| HOLD | 21.56 | 4342.06 | 218.97 |
| TRANSLATE | 10.25 | 3465.61 | 422.43 |
| STOP | 22.08 | 13396.48 | 672.01 |

These absurd distances are **estimator divergence**, not physical motion or an
accuracy benchmark. All 2376 initialized frames were LOST (84 earlier frames
INITIALIZING), with zero stationary corrections and zero explicit velocity
measurements. Across 2578 attempted visual feature measurements, zero were
accepted. Fusion remained paused throughout. The test cannot validate preservation
of the intended 30 cm translation or the effect of the velocity measurement,
because tracking had already failed and that measurement never ran.

The detector rejected even the labelled rests. During STILL, median long-window
image displacement was 0.979 px (limit 0.30), short displacement 0.262 px (limit
0.12), gyro RMS 0.01781 rad/s (limit 0.012), and acceleration standard deviation
0.10505 m/s² (limit 0.08), despite approximately 140 witness features across all
nine cells. HOLD and STOP showed similar failures. The new strict gate also
suppressed the legacy bias/gravity corrections: it was not just declining the
additional velocity measurement. This is a regression in the proposed detector's
applicability to this test, and must not be presented as a successful fix.

An additional, earlier failure mechanism is supported by startup telemetry:

- Initial gyro bias was `(0.0292, 0.0049, 0.0147)` rad/s. During analysed STILL,
  raw Android gyro means were approximately `(-0.000072, 0.000639, -0.001048)`
  rad/s. Their difference is consistent with startup motion being absorbed as
  bias; it is not a laboratory measurement of the true sensor bias.
- Estimated orientation changed 30.89 degrees during analysed STILL. Integrating
  the recorded Android gyro rates over the corresponding interval gives only
  1.11 degrees of net rotation (not ground-truth orientation). HOLD similarly
  gives 36.98 estimated versus 1.67 gyro-integrated degrees.
- Gravity-compensated acceleration norm rose from about 0.139 m/s² at the first
  valid pose to 5.662 m/s² approximately 20 s after reset, and 18.731 m/s² after
  80 s. Before/after correction comparisons are unavailable because no correction
  was accepted.
- Inspected `ov_init/static/StaticInitializer.cpp` sets gyro bias to the mean rate
  in the older half of the initialization window, without a gyro-motion check.
  Its stillness checks use acceleration variability; the app's bundled settings
  are a 1.5 s window, 0.4 m/s² acceleration threshold and 40 px disparity limit.
  This allows some startup rotation to look acceptable. The actual logged
  initialization was during the beginning of this marked run.

The working causal hypothesis is contaminated startup bias → orientation error
→ erroneous gravity compensation → velocity/position runaway, left uncorrected
by absent accepted visual updates and the over-strict stationary gate. The missing
direct velocity observation remains supported by the earlier baseline, but it
alone is not a sufficient fix for this run. Why visual candidates failed has not
yet been separated into triangulation, residual gating and other causes.

Next validation should address initialization-window motion rejection, collect
supported-rest noise evidence for the stationary detector, and expose the reasons
for visual-update rejection. Do not simply broaden all thresholds using this
single sequence: doing so could suppress legitimate slow translation. No new
estimator changes or deployment were made while analysing this recording.

## Handheld/startup revision

The user authorized this revision after confirming the test was handheld.
The improved timing/calibration pipeline, raw-pose recording, covariance-based
velocity update and fusion safeguards remain in place. No state hard-zeroing,
pose smoothing or APK rollback was introduced.

### Startup protection

A fifth app-generated translation unit, `StaticInitializer.cpp`, checks the exact
older/newer IMU half-windows before they are used to set initial orientation and
bias. Both halves must have sufficient continuous finite samples, mean gyro norm
below 0.005 rad/s, gyro RMS below 0.04 rad/s, acceleration standard deviation below
0.20 m/s² and mean acceleration magnitude within 0.4 m/s² of gravity. The two gyro
means must agree within 0.005 rad/s. A transient tap, persistent rotation, sample
gap or invalid input postpones initialization; it does not overwrite the bias.
The app supplies Android calibrated gyro measurements, so this conservative guard
may refuse a device whose residual bias exceeds its limits.

The startup image-disparity limit is reduced from 40 to 2 pixels in bundled YAML
and explicitly at runtime, including for older on-device configurations. These
checks supplement rather than replace the upstream initializer checks. The UI
asks the user to hold steady while initialization settles.

Read-only replay using the production `StartupMotionCheck` helper and interpolated
gyro at recorded accelerometer timestamps rejects the contaminated startup
window. At 0.1 s sampling of candidate windows, the first eligible IMU window ends
at boot time 107031.019307 (5.2647 s after the first frame). Its older/newer gyro
mean norms are 0.00320 / 0.00442 rad/s, compared with approximately 0.032 rad/s
in the contaminated startup window. Excluding two seconds after each marker:

| Phase | Eligible IMU windows / examined |
| --- | ---: |
| STILL | 124 / 158 |
| TILT | 0 / 59 |
| HOLD | 149 / 192 |
| TRANSLATE | 0 / 81 |
| STOP | 127 / 198 |

This is an **IMU startup-guard replay**, not a full estimator replay and not proof
that a camera-supported startup or stable trajectory would occur at that time.
Reproduce with the host executable built from `app/src/test/cpp/startup_replay.cpp`
and `/tmp/vio-marked-1788635773301/imu.csv` (reference time 107025.75456).

### Handheld detector

The detector now estimates a per-feature signed linear motion trend over 1.5 s.
Zero-mean tremor can have noticeable instantaneous displacement without sustained
motion; translation produces a persistent trend. The detector retains feature
coverage, bounded short/long displacement, a fixed stationary reference, and
continuous camera/IMU history requirements. It uses both mean gyro and RMS, not
just the instantaneous/noise magnitude. Estimated velocity is not motion evidence.

Bias/gravity support requires median/p75 image trend below 0.35/0.60 px/s, gyro
mean below 0.006 rad/s, RMS below 0.04 rad/s and acceleration standard deviation
below 0.18 m/s². Image excursions remain bounded (median 3 px from the fixed
reference, 0.8 px between frames, with additional p75 limits). A separate stronger
velocity gate requires median/p75 trend below 0.20/0.35 px/s, gyro mean below
0.004 rad/s and smaller fixed-reference excursions. Declining that stronger gate
does not automatically disable an otherwise supported bias/gravity correction.
The 0.03 m/s measurement noise, 5 Hz limit and covariance innovation gate are unchanged.

These are provisional operating limits, not a guarantee that all slow motion can
be distinguished from rest. Sub-threshold motion, especially against distant
scenery, can still be missed. The legacy stationary branch also skips normal
pose propagation, so its weaker gate is not equivalent to permitting arbitrary
slow translation. The synthetic tests cover 0.5 px/s sustained translation under
tremor and 2 px/s quiet translation, not every physical speed/depth combination.

The original recording lacks per-feature signed history, so it cannot fully
replay the new image-trend decision. The next physical test must verify actual
translation and stop behavior, not merely successful stationary corrections.

### Diagnostics and verification

`frames.csv` now has 71 columns, adding trend median/p75, mean gyro, the stronger
velocity-evidence flag, startup-window checks and separate MSCKF rejection counts
(short tracks, triangulation, refinement after successful triangulation, chi-square).
SLAM counts remain aggregate. Stationary rejection bits are: 1 history, 2 IMU
coverage, 4 feature coverage, 8 image excursion, 16 sustained trend, 32 inertial
motion. NaN startup fields mean earlier initializer checks have not reached the
new guard. Legacy disparity logs are no longer presented as update acceptance.

All 46 JVM tests, 25 native tests and the debug APK build pass. Native tests cover
startup bias contamination, taps, persistent rotation, finite
input/history, observed-scale zero-mean tremor, slow translation under tremor,
the two correction gates, camera gaps, fixed-reference creep, and the previous
covariance/innovation checks. The new physical trajectory remains unvalidated
until a marked phone run is analysed.

The revision was installed over wireless ADB and cold-launched successfully on
the SM-S918B (process 14357, initial diagnostic folder
`scan-1788636560745-1`). Logs confirm the new startup policy and the retained
960x720 → central 960x540 capture path. Early smoke-check frames remained
INITIALIZING, so launch success is not being reported as estimator stability.

## Marked handheld/startup revision result: STILL FAILED

The latest complete test is `scan-1788636664195-3`, pulled to
`/tmp/vio-handheld-1788636664195/`. All six markers are present. It has 2793 frame
rows over 93.03 s, `status=saved`, and zero reported dropped records. The preceding
`scan-1788636601884-2` contains only STILL/TILT/HOLD and was not substituted for
the completed sequence. The analysis again excludes two seconds after each marker.

| Phase | Full marked duration (s) | Raw net position change (m) | Median estimated speed (m/s) | Stationary / direct velocity updates |
| --- | ---: | ---: | ---: | ---: |
| STILL | 17.95 | 17.02 | 0.749 | 1 / 0 |
| TILT | 10.82 | 25.57 | 2.918 | 0 / 0 |
| HOLD | 22.88 | 51.02 | 0.679 | 19 / 1 |
| TRANSLATE | 9.82 | 10.57 | 1.366 | 0 / 0 |
| STOP | 19.16 | 41.94 | 2.450 | 0 / 0 |

These changes include EKF corrections and are not physical distances travelled.
All 2479 initialized frames were LOST; 314 earlier frames were INITIALIZING.
Fusion remained paused. Neither accurate translation nor stable handheld pose is
validated. No zero-velocity correction was applied during the analysed translation
interval, but that alone does not demonstrate accurate preservation of movement.

### Direct observation of the proposed velocity mechanism

At boot time 107965.021202105, in HOLD, speed was 2.34427 m/s before the legacy
correction and 2.46987 m/s after it. The explicit velocity measurement reduced it
to 0.001315 m/s. The velocity covariance trace changed from 135.8674 before both
corrections to 0.00269883 afterward; gravity-compensated acceleration norm changed
from 0.12322 to 0.11248 m/s². This is actual phone evidence that the direct
measurement removes velocity which survives the legacy measurement, not only a
synthetic Kalman example.

However, the same frame's raw position changed 41.20 m relative to the preceding
frame. Full-state cross-covariance updates can change position when correcting
velocity; the already highly uncertain trajectory was not recovered reliably.
This frame contains both corrections, so the entire position jump cannot be
attributed to the direct correction alone from the current recorded fields.
Subsequent propagation drifted again. Do not claim a successful fix on the basis
of the low post-update speed.

### Startup and visual findings

The latest startup log reports gyro bias approximately `(0, 0.0003, -0.0014)`
rad/s, substantially smaller than the earlier contaminated approximately 0.03
rad/s component. The first valid pose appears 10.46 s after the recording begins.
During analysed STILL, net estimated orientation change is 1.80 degrees and mean
gravity-compensated acceleration norm is 0.1764 m/s². These support improvement in
the earlier gross startup failure, but the runs are not matched ground-truth
experiments and position remains unacceptable.

All 1876 attempted MSCKF feature measurements are accounted for:

- 631 insufficient surviving observations;
- 1232 triangulation failures;
- 11 refinement failures after successful triangulation;
- 2 chi-square rejections;
- 0 accepted updates; no SLAM updates were attempted.

Even during analysed TRANSLATE, 241 of 352 candidates failed triangulation and
110 lacked observations. The main failure is upstream of the final residual
gate. Inspected triangulation code combines conditioning, depth-range and NaN
checks into one rejection; current counters cannot distinguish these subcauses.
Poor baseline, track lifetime, drifted clone poses, camera/IMU orientation or timing
inconsistency remain candidates, not proven individual root causes.

The handheld image-trend detector still rarely accepted rests: median trend was
1.38 px/s in STILL, 1.12 in HOLD and 3.21 in STOP, versus the 0.35 px/s weaker
threshold. Sustained image trend may come from small real hand movement as well as
tracking error. Simply increasing thresholds risks freezing real slow movement.

### Diagnostic limitation discovered

All `startup_accepted` frame fields are NaN. Source inspection confirms that even
with `use_multi_threading_subs=false`, `try_to_initialize` spawns a worker and
joins it. The startup guard executes there, but its thread-local diagnostic
fields are not transferred to the caller. The guard itself still runs; successful
startup bias comes from OpenVINS logs, not these missing CSV fields. Correct the
diagnostic handoff before relying on startup rejection counts. This is a logging
defect, not evidence that the guard was bypassed.

The next implementation should focus on camera/IMU feature-ray consistency,
observation lifetime and the individual triangulation failure reasons, plus the
startup diagnostic handoff. The existing logs lack full tracked-ray history, so
they cannot establish the camera/IMU geometry fault, if any. No estimator settings
were changed and no new APK was installed while analysing this completed test.
