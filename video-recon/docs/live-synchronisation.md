# Live camera and IMU synchronisation

The live Android pipeline pairs motion readings on accelerometer timestamps.
`VioInputQueue` waits for gyro readings on both sides of each timestamp and
linearly interpolates all three gyro axes. It preserves the accelerometer values,
rejects duplicate/out-of-order/non-finite input, and does not extrapolate.

Sensor callbacks run on their own thread. A coalesced VIO worker delivers the
paired IMU samples before processing eligible camera frames. The camera queue
holds at most three frames, dropping the oldest when processing falls behind;
each raw sensor queue holds at most 2,048 samples. This bounds memory while a
sensor is missing or processing is delayed. It is not a recovery mechanism for
long sensor outages.

OpenVINS uses `t_imu = t_camera + offset`. The worker reads the estimator's
current offset before each frame and waits until delivered IMU data reaches
that time plus a 2 ms interpolation margin. It keeps the Camera2 image timestamp
unchanged, in the same convention used by the estimator's time calibration; it
does not independently apply an exposure midpoint correction. Cameras without
the REALTIME timestamp source are refused for live scanning because their clock
cannot be assumed to match the Android motion sensors.

The native bridge independently checks IMU coverage and increasing camera time.
It only returns a pose and landmarks when the estimator's state timestamp matches
the requested camera timestamp within 1 microsecond. Initialisation or an update
that leaves an older state cannot attach that state to a newer depth image.

Reset and pause invalidate queued frames with a session generation. Native
reset/destruction waits for in-flight VIO and depth work; VIO and depth can still
run concurrently during normal capture. Resuming starts a fresh scan rather than
propagating through an unobserved sensor gap. Old-session depth jobs are discarded.

The live status shows the estimated camera–IMU offset and dropped camera-frame
count. This change does not recalibrate the sensors, change the depth model, or
add loop closure.

## Validation

From `video-recon/android`, using Java 17:

```sh
./gradlew :app:testDebugUnitTest :app:assembleDebug
```

`VioInputQueueTest` exercises delayed and batched sensor callbacks, interpolation,
invalid input, positive/negative/changing offsets, actual IMU delivery before
camera release, bounded queues and reset barriers. The APK build compiles the
native timestamp guard against the local Android OpenVINS library.

These checks establish software behaviour, not localisation or cloud accuracy.
Compare repeat scans of the same room loop before and after installing the APK:
use the same model, lighting and route, return to the same view, and compare
return-to-start displacement, wall thickness/doubling and measured room dimensions.
Also exercise reset during inference and pause/resume on the device, and check
that tracking initialises and the depth-frame counter advances.
