# Per-frame calibration for live depth

The JNI image update returns pose, landmarks and an 11-double calibration snapshot
under the same VIO lock: `fx,fy,cx,cy,d0,d1,d2,d3,width,height,model`. Model 0 is
OpenVINS radtan (`k1,k2,p1,p2`); model 1 is equidistant (`k1,k2,k3,k4`). Unsupported
models, non-finite calibration and camera/configuration dimension mismatches are
rejected. The Kotlin depth job owns an immutable copy so subsequent online
calibration updates cannot change the camera associated with an older pose.

The phone was observed delivering 960x720 for the previous unsupported 960x540
request. An initial revision selected 1280x720 and used nearest-neighbour
decimation to 960x540. Pose instability was reported after that revision.
Capture now prefers an advertised native-width stream: 960x720 on the tested
phone. Both tracking and RGB use its central 960x540 crop (top offset 90),
preserving the calibrated field of view without resampling tracking pixels.
For devices without a native-width mode, an advertised same-aspect mode is
area-filtered using precomputed footprints. Unexpected geometries are rejected.
The phone's reported factory K (fx 2716.61, fy 2717.06, cx 2044.42, cy 1523.48
in 4080x3060 pixels) is consistent with the configured central crop:
focal lengths approximately 639.2/639.3 and cy approximately 268.5.

Depth preprocessing maps a
virtual pinhole image through the captured distortion model into the small RGB
buffer, with bilinear sampling. Axes and field-of-view convention are preserved;
there is no upright rotation or additional crop after capture preprocessing. The camera-to-RGB
downsample samples nearest pixel centres. All subsequent image/calibration
resizes use `f' = scale*f`, `c' = scale*(c+0.5)-0.5`, separately in x and y.

Intrinsics-aware depth models receive this virtual camera at model-input size.
Metric3D's canonical-depth conversion uses its input focal length. The model's
output grid carries a resized version of the same camera and matching rectified
RGB. Landmark anchoring, TSDF projection, and raw-cloud projection/colour all use
that grid, removing the fixed factory intrinsics from the depth path.

Out-of-image rectification pixels have invalid alpha. Invalidity propagates
through output resizing and sets predictions to NaN before landmark fitting.
The final millimetre depth array explicitly zeros invalid predictions, including
for inverse-depth models whose fitted offset could otherwise create surfaces in
unsupported borders. This mask excludes unsupported pixels; it is not a model
confidence or cross-view consistency filter.

`CameraCalibrationTest` checks pinhole resize/ray correspondence, independent
OpenCV radtan/equidistant projection fixtures, colour alignment, invalid borders,
and immutable calibration snapshots. Build/test with Java 17 from
`video-recon/android`: `./gradlew :app:testDebugUnitTest :app:assembleDebug`.

The app logs `calibrated depth frame=...` every 20 fused/processed depth frames,
including output intrinsics, source intrinsics, model, anchor count and runtime.
Phone scans are still required to assess wall alignment and room dimensions.
This change cannot correct existing fused geometry after later pose/calibration
refinement; reintegration and loop closure remain separate work.

## Pose-regression safeguards

The live JNI initializer enforces `zupt_only_at_beginning=false`, including when
an older configuration remains in the app's external files. A startup-only policy
was tried and then reverted after severe stationary drift was observed on the
phone (hundreds of metres of estimated displacement). In this OpenVINS version,
low image disparity can override the IMU chi-square test and estimated velocity
limit. Estimated velocity is not actual motion, so accepting a stationary update
above that limit is not by itself evidence of a false stationary detection.
Stationary constraints must remain available when the user stops during a scan.
The JNI bridge logs timestamp, world position, estimated speed and landmark count
once per second to support actual stationary/movement testing.
The camera path logs preprocessing time, VIO time and dropped-frame counts.
No pose smoothing or jump clipping is applied: verification must assess the
underlying tracker, not just hide corrections in the display.

## Deployment history after stationary-drift report

Restoring ongoing stationary updates did not resolve the regression in a fresh
phone run: position continued drifting by metres, with long intervals reporting
zero exported 3D landmarks. The shared-calibration/crop revision is therefore
**not validated for deployment**. The phone was restored and relaunched with the exact
pre-calibration APK the user previously reported worked better (the sensor-timing
revision). The newer source is retained for diagnosis, not marked as a successful
fix. Before another rollout, record a controlled stationary and stop/move/stop
test with synchronized images/IMU and compare to that APK. Nonzero gyro readings
were observed during the attempted test, so motion versus bias still needs to be
distinguished; the current measurements are not a controlled accuracy benchmark.

Subsequently, the user also observed severe drift in the earlier APK and requested
the improved version again, explicitly not a rollback. The improved timing,
calibration and crop path has therefore been restored in the diagnostic build,
with raw estimator recording and tracking-health/fusion safeguards. Baseline
measurements support the missing direct velocity constraint as a contributor,
not a complete root-cause diagnosis. The subsequent explicit velocity measurement
and physical-test status are documented in `live-tracking-diagnostics.md`.
