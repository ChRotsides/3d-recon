# Android pose logger

The throwaway ARCore capture app from the plan's task 3. It writes a plan §5
`SessionBundle` the Phase-0 harness can reconstruct directly.

## Build

Needs a **JDK**, not a JRE. On this machine Java 21 is JRE-only, so point Gradle
at the Java 17 JDK:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export GRADLE_USER_HOME=~/.gradle          # needs ~2 GB free
echo "sdk.dir=$HOME/Android/Sdk" > local.properties

./gradlew :app:assembleDebug       # -> app/build/outputs/apk/debug/app-debug.apk
./gradlew :app:testDebugUnitTest   # 6 JVM tests, no device needed
./gradlew installDebug             # to a connected ARCore device
```

## What the unit tests cover

`FrameConversionTest` runs on a plain JVM — no device, no emulator — which is
what lets the coordinate-frame guard from plan §7 run on every commit rather
than only when someone plugs a phone in. It pins:

* column-major `Pose.toMatrix` is read without an accidental transpose,
* a mirrored pose (det = -1) is **refused**, not uploaded,
* a pose carrying scale is refused,
* the camera looks down -Z,
* the serialised JSON uses snake_case wire names, since the server's pydantic
  `SessionBundle` forbids unknown fields and a camelCase slip would fail the
  upload rather than be silently dropped.

## Using a capture

The app writes to `Android/data/com.recon.poselogger/files/captures/bundle_<uuid>/`.
Pull it and check it before relying on it:

```bash
adb pull /sdcard/Android/data/com.recon.poselogger/files/captures/bundle_<uuid> .
recon validate bundle_<uuid>
recon reconstruct bundle_<uuid> --out runs/room --depth da3:base
```

`recon validate` is the pre-flight: it catches a pure-rotation capture (no
parallax, will not reconstruct), a missing VIO scale prior, intrinsics that were
not rescaled after downscaling, and a tape measure entered in the wrong units —
all before someone spends a day capturing fifteen rooms with a broken setup.

## Not done

No guidance overlay, no next-best-station ring. That is Phase 3 and it belongs
in the shared Rust core (`core/`), not in an Activity — see ADR-0001.
