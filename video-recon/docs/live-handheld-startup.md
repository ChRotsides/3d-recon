# Handheld initialization

The Android archive previously removed OpenVINS' dynamic initializer, leaving
only a guarded still-start path. Shortening a timer cannot make that path work
while the phone moves. This change restores the existing visual/inertial dynamic
initializer and its covariance recovery; it does not replace the estimator or
roll back the timing, calibration, feature tracking, stationary measurements,
tracking-health gates or cloud-preserving recovery.

## Implementation

- Three-second sensor window; dynamic optimization limited to 50 iterations,
  0.5 seconds and two threads. This optimizer limit does **not** bound all
  preintegration, linear initialization, covariance recovery or camera startup.
- Request 120 startup features. Keep calibration fixed during initialization.
  The existing 10-degree accumulated rotation prerequisite remains: the intended
  action is a gentle sideways movement and turn, not a supported stationary hold.
- Quiet observations may still use the validated static initializer. If that
  path cannot accept ordinary hand motion, try the joint dynamic solver instead.
- Dynamic attempts use a temporary IMU state. Only finite state, finite positive
  semidefinite full covariance, valid timestamp and expected state ordering may
  be committed. Missing/non-monotonic IMU data are rejected. Never hard-zero the
  running velocity or grant fusion permission because a timer expired.
- A converged pure-rotation fixture exposed that an initialized IMU is not
  necessarily a useful scan start. Moving initialization also requires at least
  24 tracks spanning 0.8 seconds and median rotation-compensated displacement of
  1.5 pixels (normalized to image width 960). This is an observability heuristic,
  not a calibrated guarantee of scale accuracy and not a stationary correction.
- The first-reliable-pose wall time is logged separately from solver time.
  The UI removes “Support phone”, shows handheld guidance for at most five
  seconds, then explicitly reports that tracking is not ready if necessary.
  Automatic retries before first acquisition do not restart that visible clock.
  This is truthful deadline reporting, **not a guarantee of tracking in five
  seconds for every scene/motion**. Darkness, occlusion, lack of translational
  information and poor calibration can prevent initialization.

## Build dependencies

`HandheldInitialization.cmake` imports matching full sources from
`OV_FULL_ROOT=/home/charalambos/ai/vio/open_vins` (the existing local source tree),
and checks that initializer options exactly match the Android header. The
`shared_ptr<void>` helper in the original Android class retains its ABI; an
app-generated translation unit instantiates/casts the typed helper. Neither the
original Android sources nor `libopenvins.a` are modified.

Android Ceres 2.1.0 is built from commit
`f68321e7de8929fbcdb95dd42877531e64f72f66` under
`/home/charalambos/ai/phone/ceres-android-2.1.0`, installed in `install-arm64`.
It uses the same NDK 28.2.13676358, API 31, `c++_shared`, and Eigen 5.0.1 headers
as the existing app. One local CMake adaptation removes the Eigen major-version
constraint (`find_package(Eigen3 3.3 REQUIRED)` becomes
`find_package(Eigen3 REQUIRED)`); installed consumers still require the exact
Eigen version used to build Ceres. This combination requires regression tests,
not an assumption of upstream certification. Android Ceres/miniglog headers
explicitly precede the host environment's desktop Ceres/glog headers.

CMake options for the Ceres build:

```text
CMAKE_TOOLCHAIN_FILE=<NDK>/build/cmake/android.toolchain.cmake
ANDROID_ABI=arm64-v8a
ANDROID_PLATFORM=android-31
ANDROID_STL=c++_shared
CMAKE_BUILD_TYPE=Release
CMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG -g0
CMAKE_POSITION_INDEPENDENT_CODE=ON
Eigen3_DIR=/home/charalambos/ai/vio/env/share/eigen3/cmake
MINIGLOG=ON
BUILD_TESTING=OFF
BUILD_EXAMPLES=OFF
BUILD_BENCHMARKS=OFF
SUITESPARSE=OFF
CXSPARSE=OFF
LAPACK=OFF
BUILD_SHARED_LIBS=OFF
SCHUR_SPECIALIZATIONS=OFF
```

The new dependency's license is included in APK assets. Build and install it
before building the app; `Ceres_DIR` can point to another compatible Android
installation. Documentation: [OpenVINS dynamic initialization](https://docs.openvins.com/namespaceov__init.html)
and [Ceres Android build](https://ceres-solver.org/installation.html#android).

## Tests

`handheld_init_test` links the actual app native library. It exercises a
three-axis moving synthetic trajectory with 3.2 seconds of observations,
compares recovered body-frame velocity/gravity/bias to the fixture, and checks
that pure rotation and missing IMU data do not commit a state. This is not a
measured physical handheld startup and does not exercise image tracking itself.
The separate full-VIO rendered-image regression and Android Activity recovery
integration tests cover those respective paths.

On-phone checks so far: 3.2 s sensor history plus 0.274 s computation for the
moving initializer fixture. Body-frame velocity error was 0.0000024 m/s and
gravity-vector error 0.000023 m/s² in this idealized fixture. Pure rotation and
missing-IMU cases rejected without committing state. The unchanged 17-test
feature/geometry regression passes with the new startup settings: 5871/6347
accepted visual updates, 0.000304 m raw displacement over the final 0.97 s hold,
final speed 0.001055 m/s, and about 0.384 m displacement for its 0.4 m synthetic
translation. These are synthetic estimates, not measured handheld accuracy.
All 57 JVM tests and 41 host stationary/covariance tests pass.

The added full rendered moving-start test also passes on Android. It moves from
the first frame (no initial rest), renders a textured plane with the correct
camera homography, feeds matching IMU readings through VioManager and the actual
tracker, and checks continued visual updates. Initialization became available
at 3.2 s of simulated time; the first frame with six accepted visual updates was
at 4.1 s. The slowest camera-processing call took 0.160 s. There were 349 accepted
updates by 5.97 s. This is not the Activity's first-reliable-pose wall time or a
real camera/handheld measurement; it does validate that moving startup reaches
visual updates within the five-second target in that controlled fixture.

## APK

SHA-256: `f389ba66f772fcd70dd48dc963144eb7b613a124282841a61ee397c7efc2fe7b`.
The previous recovery build is not a handheld-start build; only this new native
library contains the restored dynamic initializer. The startup log must say
`handheld dynamic enabled`.

Installed successfully over wireless ADB on SM-S918B, 2026-09-06 00:05:43 local.
The actual Activity recovery regression passed again, archiving all 2067 fixture
points to `segment-5441138280219974257.ply`, rejecting the old fusion permit and
starting an empty independent segment. The normal scanner was relaunched as
PID 10036 and logs confirm `handheld dynamic enabled`. No fresh controlled
physical handheld motion measurement has been performed. Generated native build
cache is retained at `/tmp/vio-cxx-handheld.cR0uGz/cxx`; tracked cache files are
restored. Recordings, saved calibration and completed clouds were not deleted.
