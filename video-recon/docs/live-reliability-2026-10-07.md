# Live Scan reliability build — 2026-10-07

Scope: the Android **Live Scan** activity, with reconstruction on the phone.
Version: `0.7-live-npu` / version code `7`, arm64, Android 12 or later.
This document supersedes the permissive tracking/fusion recommendations in the
September handheld notes. Existing calibration and saved scans are preserved.

## Install and test

The APK is `android/app/build/outputs/apk/debug/app-debug.apk`, relative to
`video-recon`. It includes the depth and feature models; no reconstruction server
or model download is required.

From this repository's root:

```sh
adb install -r video-recon/android/app/build/outputs/apk/debug/app-debug.apk
```

Open **Video Scan → Live Scan** and keep **fusion: on**. The installed S23 test
configuration uses **UniDepth accelerated**; fresh installations include the
CPU Depth Anything default. The accelerated model files are prepared separately
from the user’s existing UniDepth weights using [these tools](../tools/qnn/README.md).
Use the calibrated rear
camera; the existing profile targets the S23 Ultra. A different S23 variant still
needs its camera/IMU calibration verified.

1. Start in good light facing textured furniture, shelving, or a doorway. Hold
   still for about three seconds, then move sideways gently while keeping the
   same objects visible. Wait for `TRACKING` or `STATIONARY`.
2. Scan one room, moving sideways as well as turning. Relative depth needs at
   least eight consistent tracked landmarks across four image regions. A pause
   asking for anchors means no new depth is being fused.
3. Stop for ten seconds. The camera pose should settle and the existing cloud
   should remain fixed. Walk back to the starting view and inspect whether walls
   overlap without a second displaced surface.
4. Briefly cover the camera. Fusion should pause. After a longer tracking loss,
   revisit a saved view and move sideways to allow relocalization. The old cloud
   is held until the new coordinate frame is confirmed.
5. Press **Save scan**. Verify a PLY exists in the `segments` directory below.
   Reset and leaving the activity also save confirmed geometry. Returning to
   the activity starts a separate segment; it does not reload and resume the
   previous map. Automatic tracking recovery retains the current map in memory.

Collect the result and automatic diagnostics with:

```sh
adb pull /sdcard/Android/data/com.recon.videoscan/files/segments ./live-segments
adb pull /sdcard/Android/data/com.recon.videoscan/files/diagnostics ./live-diagnostics
adb pull /sdcard/Android/data/com.recon.videoscan/files/depth-evidence ./live-depth-evidence
```

For a longer acceptance test, scan connected rooms for 10–15 minutes, revisit the
start, and compare a measured doorway/wall length with the exported cloud. Record
pose drift, duplicated walls, recovery time, sustained VIO/depth processing time,
memory use, and thermal slowdown. The on-screen counters and captures expose
these failures rather than hiding them behind unconstrained geometry.

## Repairs

- Camera/IMU tracking no longer waits for XFeat. Descriptors run through a bounded
  mailbox with matching image, pose, landmark, and calibration snapshots. Missing
  XFeat output cannot stop native feature replenishment after occlusion.
- Missing IMU intervals and queue overflow trigger recovery. The app stops
  integrating unreliable poses, implausible jumps, stale depth, and estimates
  without recent accepted visual/stationary constraints. Stationary updates are
  enabled throughout the scan, including upgrades with older saved settings.
  Startup position corrections before the first trusted pose are allowed to
  converge; subsequent discontinuities still force map-frame recovery.
- The unvalidated learned pose fallback is disabled. Tracking loss holds the last
  reliable pose and cloud while the existing saved-view relocalizer recovers.
- Dynamic initialization uses scaled polynomial roots, rejects nonfinite and
  non-positive-definite gravity solves, and eliminates feature variables with
  QR instead of an explicit measurement-space projector. Translation parallax
  and covariance checks remain required before committing the candidate state.
- Active, reprojection-checked triangulated tracks now provide current depth
  anchors. Relative depth requires distributed inliers; metric models are checked
  against available tracked geometry. Cross-view confidence and confirmed-surface
  thresholds are restored.
- Surface extraction starts at the 3 cm voxel size instead of discarding detail
  at 10 cm. Preview and export have separate point budgets. TSDF allocation is
  bounded; reaching capacity holds the map and asks for Save/New segment.
- Each GL view gets its own buffer position and a consistent cloud/count snapshot.
  A recreated GL context reuploads its data. Model sessions close on their worker
  threads and failed model switching retains the working session.
- A default depth model is bundled and uses a 322×182 grid. ONNX output dimensions
  are read independently of input tensor rank. The model is relative inverse
  depth, not a direct metric distance sensor. Model provenance and licenses are
  included in the APK.

## Verification

Completed on the development machine:

| Check | Result |
| --- | --- |
| Android JVM tests | 88 passed, zero failures/errors |
| Native CTest suites | 5/5 passed |
| Full stop/move/hold VIO with actual tracker | 6,617/6,987 accepted visual updates; 0.63 mm net displacement during the final 0.97 s hold |
| Full rendered moving-start VIO | Initialized at 3.2 s; first six accepted visual updates at 3.77 s |
| Initializer scenarios | Exact motion and three noisy seeds accepted; pure rotation and missing IMU rejected |
| Noisy initializer body velocity error | 0.015–0.033 m/s for the tested seeds |
| Surface export | Complete synthetic surface round-trip; failed save retained map |
| Packaged ONNX weights on desktop | Finite 322×182 output; approximately 142–177 ms per inference with four CPU threads |
| Android build | App APK and instrumentation APK built; debug APK signature checked |
| Android lint | Zero errors; existing/product UI, compatibility, and style warnings remain |

Three existing SM-S918B recordings were also replayed through the instrumented
native estimator, with real camera/IMU timestamps and no pose labels:

| Recording | Replay result |
| --- | --- |
| `rec_20260903-070045` (67 s) | Initialized at 2.63 s; 45,644 accepted visual updates; maximum estimated speed 0.73 m/s and position sigma 0.11 m |
| `rec_20260903-232445` (77 s) | Initialized at 2.70 s; 46,552 accepted visual updates; early corrections reached 0.62 m and uncertainty 1.74 m, which require withholding fusion/recovery |
| `rec_20260902-225022` (12 s) | Did not initialize; too few persistent features for dynamic initialization, and motion failed the static-start guard |

The short clip remains a failed acquisition case: this input requires a clearer
view or a new startup attempt. The second recording is not an accuracy pass.
These raw estimator replays do not execute the Activity's automatic restarts,
health guard, descriptor recovery, or depth fusion, and they have no ground-truth
trajectory. They identified the initial-position-correction guard regression,
which now has a JVM regression test. Timestamp matching was 99.45–99.96%; no IMU
gap exceeded the live queue's 50 ms rejection threshold.

The initial October 7 timing and accuracy figures above are **not S23 device
performance claims**. No ADB device was connected for that initial validation.
The October 8 device validation is recorded below. There is no global loop-closure optimization or dense
map reintegration to correct accumulated whole-building drift. Saved-view PnP
relocalization and memory limits do not replace that functionality. This is a
testable reliability build; whole-house production readiness is not established.

Claude independently reviewed the input pipeline and initialization mathematics.
Recommendations were checked against executable tests. The suggested shorter
clone window was rejected: 15 clones eliminated accepted visual updates in the
slow-motion fixture, whereas the retained 31-clone window passed. A dangling Eigen
expression in the old initializer fixture was also fixed before trusting its
results.

## Repeat verification

```sh
cd video-recon/android
JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 ./gradlew \
  testDebugUnitTest lintDebug assembleDebug assembleDebugAndroidTest \
  --no-daemon --max-workers=2
cd ../..
video-recon/tools/test_live_native.sh
```

To replay an existing recorder capture through the same native estimator without
writing thousands of extracted images:

```sh
cmake --build video-recon/.deps/host-build --target live_recording_replay
.venv/bin/python video-recon/tools/replay_live_recording.py \
  data/recordings/rec_20260903-070045 /tmp/live-replay.csv
```

The CSV records pose, uncertainty, visual-update counts, active landmarks, and
processing time. A neighboring `.log` file retains native failures. Python needs
NumPy and OpenCV, and `ffprobe` must be installed. Set `LIVE_REPLAY_LOG_LEVEL=DEBUG`
for detailed startup rejection reasons. Recordings with missing IMU intervals
are rejected instead of replaying fabricated interpolation through the gap.

The native script uses `.deps/native-host`. This machine's tested dependencies
are Eigen 5.0.1, Boost headers 1.85, host OpenCV 4.12, and host Ceres 2.2. Android
uses OpenCV 4.10 and Ceres 2.1 built against Eigen 5.0.1. The ignored
`.deps/native-headers` prefix supplies Android's matching Eigen/Boost headers.
OpenVINS source and Android dependencies retain their configured external paths;
they are not downloaded automatically. Override `nativeHeaders`, `OV_ROOT`,
`OV_FULL_ROOT`, `OPENCV_SDK`, and `Ceres_DIR` with Gradle `-P` arguments as needed.

To recreate the host dependency prefix with conda-forge:

```sh
conda create -y -p video-recon/.deps/native-host -c conda-forge \
  'eigen=5.0.1' 'libboost-headers=1.85.0' \
  'libopencv=4.12.0=headless_py313ha5796b8_12' \
  'ceres-solver=2.2.0=cpugplh3250042_211'
conda create -y -p video-recon/.deps/native-headers -c conda-forge \
  'eigen=5.0.1' 'libboost-headers=1.85.0'
```

After installing both APKs, the bundled-model device smoke test is:

```sh
adb install -r video-recon/android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w -e depth_smoke 1 \
  com.recon.videoscan.test/com.recon.videoscan.live.RecoveryInstrumentation
```

It runs the actual Android depth runtime, validates finite output and dimensions,
and checks that a corrupt model switch preserves the active model. The same
runner supports `-e overlay_smoke 1`, `-e empty_map_recovery 1`, and
`-e landmark_reference 1`. Run those after opening Live Scan once to install its
configuration. Running the instrumentation with no extra arguments exercises
the Activity/JNI map-retention and stale-fusion recovery path.


## October 8: reconstruction stopped after one patch

The SM-S918B phone log contained 155 depth attempts: 135 failed the relative-depth
calibration, 14 integrated only provisional depth, and six had cross-view support.
A bounded evidence recording retained 41 frames; its queue dropped 114, so the
replay below is a subset, not a replay of the whole scan. Pose and depth are
separate pipelines: apparently good tracking does not imply usable dense depth.

Changes in `0.4-live-depth`:

- Choose only calibration hypotheses with sufficient support and distance range.
  The previous lowest-cost hypothesis could veto another usable distributed fit.
- Calibrate within 12 cm / 10%, with at least eight anchors, four image regions,
  65% consensus, and median relative anchor error at most 8%. Weighted inverse-depth
  refinement must retain the support checks. No previous scale is reused.
- Exclude predictions beyond the supported disparity range (10% margin), also
  bounded by the inlier metric depth range (10% margin). Excluded pixels stay empty.
- Keep up to four comparison views for eight seconds, cleared on generation,
  map revision, or model changes. Continue requiring at least 4 cm translation
  and agreement within 6 cm / 2.5% before depth is confirmed.
- Only confirmed foreground can block an older matching view. Rejected pixels
  never become references for later confirmation. Provisional TSDF weights still
  cannot create visible surfaces by themselves.
- Show distinct pause reasons for insufficient feature coverage, unobservable
  distance range, and inconsistent depth. Clear stale support counters per attempt.

Claude Opus 5.5 reviewed the captured predictions, revised logic, and performance
independently. The bundled default remains Depth Anything V2. The S23 test phone
uses an optional quantized UniDepth model, calibrated to same-frame VIO landmarks
with a single robust scale. The selected model persists across restarts.

Verification: 88 JVM tests pass, Android lint has zero errors (49 warnings), and both debug APKs build. Tests include numerical anchors from the
phone capture, unsupported-depth exclusion, expired/reset comparison history,
confirmed occlusion, and provisional/invalid reference handling. The numeric
fixture contains no images, camera poses, or room coordinates.

The test APK also supports replay through the actual Android calibration and JNI
TSDF, using a captured evidence directory relative to the app's external files:

```sh
adb shell am instrument -w \
  -e depth_replay depth-evidence/map-1791434991165-1 \
  com.recon.videoscan.test/com.recon.videoscan.live.RecoveryInstrumentation
```

It uses isolated native maps and writes no scan segments. The baseline integrates
saved depth/confidence; the updated path recalibrates saved model predictions.
Before each integration, the existing visible map is compared against that
frame's landmarks. This measures later-frame VIO consistency, not ground-truth
accuracy or the quality of a newly recorded room scan.


## October 8: no points despite good tracking

The subsequent physical test exposed a second blocker: UniDepth took 1.8–2.5 s
under live tracking load, but the final fusion guard discarded results older than
1.5 s as “tracking changed.” In version 0.5, completed depth may use its measured
inference time plus 0.75 s, bounded by an absolute 3 s maximum. It still requires
the original healthy tracking epoch, map revision, session generation, captured
pose, and continuously fresh camera updates. Queued work retains the 1.5 s bound.
Tests cover loss, reset, stale cameras, delayed inference, and the absolute cap.

Evidence writes now buffer before compression; the old stream compressed individual
four-byte values. The binary format, quotas, and atomic publication are unchanged.
The on-phone v0.5 scan reached 36 integrated depth frames and about 79,000 preview
points, visibly rendered. The user confirmed a growing cloud but reported it was
too slow. The final saved segment contained 49,269 vertices. This is evidence of
working fusion, not a room accuracy or drift acceptance test.

## October 8: sustained speed

The phone reported thermal status 3 during the long scan. Profiling found that
the nominal 322×182 UniDepth input was internally resized to 602×336 before the
encoder and decoder. A 490×280 internal grid was tested while retaining the 322×182 input/output
grid and per-frame camera intrinsics, then rejected by the map comparison below.
Version 0.6 retains the working original grid. ORT depth
and XFeat workers sleep between jobs instead of spinning. The two GL views render
on pose, geometry, or touch changes. Loading a saved model no longer first loads
and probes the bundled default model. Frames with fewer than eight landmarks
are rejected before inference, avoiding work that could never pass calibration.

Runtime comparisons used the same real recorded image on the SM-S918B, an awake
foreground benchmark activity, two warmups, and eight measured inferences. These
are isolated depth timings, not end-to-end live scan frame rates:

| Model/runtime | Measured inference |
| --- | --- |
| Original quantized UniDepth, CPU 4, no spinning | 885–950 ms |
| Internal 490×280, CPU 4, no spinning | 555–570 ms |
| Internal 490×280, XNNPACK 4 | 678–731 ms |
| Original quantized UniDepth, XNNPACK 4 | 1,202–1,210 ms |
| Full UniDepth, NNAPI with FP16 enabled | 6,106–7,110 ms |

Smaller 392×224 and 322×182 internal grids were rejected for this build because
fewer captured views passed calibration. Later-frame surface consistency on the
old 41-frame subset has not met every geometry acceptance threshold: the original
quantized UniDepth replay had 8 growth frames and 8,231 surface points, but median
relative anchor error was 5.76%, above the 5% test limit (p90 11.67%). Anchor checks
are not independent ground truth and only score overlapping surfaces. Do not
report this as a passed geometry regression or production accuracy validation.

The UniDepth weights were already supplied separately on the test phone. They are
optional and are **not bundled in the APK**; they retain upstream's
[CC BY-NC 4.0 license](https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE).
They are unsuitable for an unrestricted commercial release under that license.
The APK's bundled Depth Anything model remains available as a fallback.

Runtime benchmark (separate instrumentation APK; requires the frozen capture):

```sh
adb shell am instrument -w \
  -e depth_runtime_benchmark unidepth_vits_182x322_quantized.onnx \
  -e runtime_configuration cpu4 \
  com.recon.videoscan.test/com.recon.videoscan.live.RecoveryInstrumentation
```

Other configurations are `cpu1`, `cpu2`, `cpu4spin`, `xnn2`, `xnn4`, `nnapi`, and
`nnapi16`. Add `-e runtime_profile 1` to record an ORT operator trace. Benchmark
outputs go to the app's `runtime-benchmark` directory and never modify scan maps.


The final paired JNI comparison rejected the 490×280 candidate, including its
corrected per-axis camera scaling. Across the same 21 later-frame map/landmark
samples, the original model had median/p90 relative error 3.99%/9.08%, while the
candidate had 14.43%/20.03%. Original-only coverage was 65 samples; candidate-only
coverage was 114 samples. This small, imperfect map-consistency check does not
establish absolute accuracy, but it is sufficient reason not to deploy the
smaller model as the speed fix. The host neural-depth anchor check alone had
missed this regression after fusion. The new replay audit CSV preserves matching
frame/anchor identities for future comparisons.

`tools/quantize_unidepth.py --live-grid SOURCE OUTPUT` reproduces the rejected
experiment for investigation, with a source/output hash manifest; the flag is
not a recommended scanning configuration. With the flag omitted, it reproduces
the original full-grid quantized optional model.


## October 8: original model on the NPU (version 0.7)

The full-resolution model now has a QNN backend on the S23 Ultra. A small CPU graph
computes camera conditioning from the current frame's intrinsics; the main graph
runs on the Hexagon NPU. Shapes are fixed while K values remain dynamic. No camera
calibration is baked in. The exact GELU rewrite and graph split were verified
against the original fp32 model on all 41 captures: maximum absolute difference
1.91e-6 m and maximum relative difference 6.64e-7 on CPU.

The first hardware test required the entire main graph to run on QNN, with CPU
fallback disabled inside that graph. Profiling confirmed five QNN executions and
no CPU main-graph nodes. Across five real images, camera + main inference took
242–274 ms. Compared with the original fp32 output, per-image median relative
error was 0.11–0.62%; per-image p99 was 0.29–1.34%. Some edge pixels differed more,
so full reconstruction checks remain necessary.

First compilation took 203.7 s. The app caches the compiled context atomically in
its private **no-backup files** directory. Model, runtime-library and device hashes
control cache reuse. The compilation session closes before the cached session
opens, limiting peak memory. An intact cached context that fails to load causes
prompt CPU fallback rather than a repeat compilation. Corrupt/incomplete contexts
are quarantined and rebuilt; quarantine and interrupted-build cleanup are confined
to the app-owned context directory.

`DepthRunner` preserves a working model during failed switches and falls back to
the original quantized UniDepth CPU model if accelerated inference fails. The
Activity clears confirmation history when that model changes and updates the
saved model choice. First-time setup runs on the depth worker, with a preparation
message; camera/IMU processing continues. New depth can be scheduled after 5 cm of
movement, above the 4 cm minimum used for independent-view confirmation.

Normal builds use ONNX Runtime QNN 1.22.0 with the CPU provider retained. Native
libraries are extracted for the DSP loader. `-PcpuOnly=true` produces a diagnostic
build with the previous CPU runtime. The prepared UniDepth weights remain optional
and are not bundled. QNN runtime licensing is described in the preparation notes.

Device checks after both APKs are installed:

```sh
adb shell am instrument -w \
  -e depth_replay depth-evidence/map-1791434991165-1 \
  -e depth_replay_model qnn_main_static_gelu.onnx \
  com.recon.videoscan.test/com.recon.videoscan.live.RecoveryInstrumentation
adb shell am instrument -w -e qnn_recovery_smoke 1 \
  com.recon.videoscan.test/com.recon.videoscan.live.RecoveryInstrumentation
```

The second check requires a compiled cache. It opens the cached graph twice,
asserts no recompilation, deliberately invalidates an NPU session, and verifies
that the actual CPU fallback returns finite depth. Neither test changes saved scans.


Integrated NPU validation on the phone:

- All 41 recorded images ran through `DepthRunner`, calibration, confirmation and
  the actual JNI TSDF. 26 frames calibrated, 15 had confirmed pixels, the cloud
  grew on 9 subsequent frames, and the final subset map had 7,211 points.
- Across all overlapping samples: median relative anchor error 4.09%, p90 12.55%.
  **The geometry test still failed its 12% p90 threshold.** Across the same 40
  samples shared with the previous quantized CPU map: median/p90 improved from
  4.94%/10.59% to 4.12%/8.99%. These remain VIO consistency checks, not ground truth.
- The full 41-frame FP16 comparison found per-image median differences up to 3.22%
  and per-image p99 up to 4.30% against fp32; the initial five images understated
  the worst median difference. The proposed uniform 1% median equivalence check
  did not pass. After each model's VIO scale fit (Python port, 26 shared frames),
  maximum per-image median/p99 differences were 2.49%/5.41%.
- End-to-end depth inference during replay took 337–684 ms as the phone's power
  state changed. The 242–274 ms isolated hardware measurement is not a promise
  of that rate under every live/thermal condition.
- Persistent context compilation took 292.6 s on this run. Subsequent complete
  session opens took 1,390 and 1,187 ms, with zero recompilation. Deliberately
  closing the NPU session caused successful CPU fallback with finite predictions.
- The bundled model smoke test passed on ORT 1.22, including failed-switch retention.

Live observation then exposed a separate recovery corner case: a single
provisional depth frame allocated TSDF blocks but had neither visible surface nor
saved reference. Using block allocation as “map exists” made recovery wait for an
impossible reference match. Recovery now checks for an actual extracted surface;
if there is no surface and no reference, it clears only those provisional voxels
before starting the new coordinate frame. Confirmed surfaces remain preserved.
The JNI recovery regression covers an empty map, this provisional-only case,
and retention of an existing confirmed cloud.

Final-build checks: 88 JVM tests, all 5 native CTest suites, the tracking and
initializer fixtures, Android lint (0 errors, 49 warnings), and the on-device
empty/provisional/confirmed-map recovery regression passed. One cached-start
run immediately after reinstall took 11.83 s and failed its 10 s timing target,
without recompiling. The subsequent Live Scan launch loaded that same context
in 581 ms (roughly 2.2 s including hashing and warmup). Startup latency is therefore
not yet validated under every phone power/memory condition.

The final repeated on-device cache/fallback check passed: cached starts took
1,712 and 1,567 ms without recompiling, and an invalidated NPU session fell back
to the actual CPU model with finite depth. Live Scan was reopened afterward.

Final APK SHA-256: `b340e852a55ae08706610bbdec344d3b290142debdefc909721875daa1beeca2`.
