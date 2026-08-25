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
./gradlew :app:testDebugUnitTest   # JVM tests, no device needed
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
Pull it, then build the panorama and the cloud from it:

```bash
adb pull /sdcard/Android/data/com.recon.poselogger/files/captures/bundle_<uuid> \
    data/captures/
python scripts/capture/equirect.py  data/captures/bundle_<uuid> out/sessions/equirect.jpg 4096
python scripts/capture/da3_cloud.py data/captures/bundle_<uuid> out/cloud.bin small 900000
```

`docs/capture-to-cloud.md` records what a good capture looks like in numbers —
27 keyframes, a full circle of yaw, 1.23 m of baseline — and why a capture that
pans inside a 1.3 m box produces a floor plan of the furniture. **For a
panorama, turn on the spot**; for geometry, translate. The two goals pull in
opposite directions and one capture cannot be optimal for both.

## Capture guidance

Phase 3 is on screen. All of it is decided by the shared Rust core and merely
drawn here, per ADR-0001 — the Activity chooses how to render, never what is
true:

* **Coverage paint** (`CoverageRenderer`) — the core's coverage voxels over the
  real surfaces, amber where a voxel has been seen but has not cleared `q_min`,
  green where it has. It answers "which wall still needs work" in place, rather
  than as a number to interpret.
* **Station ring** (`StationRenderer`) — the §6.1 next-best-station on the floor,
  with a pulse so it is findable, plus an on-screen arrow (`StationArrow`) when
  it is behind the user.
* **HUD** (`GuidanceHud`) — floor/wall/ceiling bars against the core's own
  targets, the keyframe budget, and the core's nag wording verbatim.

The bars read *quality of what you have seen*, not *share of the room*: the
core measures against observed voxels because the room's extent is unknown
mid-capture. The HUD says so out loud, because "90%" invites the other reading.

The camera preview intentionally has no room-outline drawing interaction. Raw
AR-space lines moved whenever tracking refined its map and disappeared during
brief tracking pauses, turning capture into an editing task. Room boundaries
are inferred from the reconstructed geometry; manual corrections belong in a
stable post-capture top-down editor. The existing outline wire schema remains
compatible with that future editor.

### Where the geometry comes from

`DepthSampler` back-projects **ARCore's depth map** into the world-space
oriented samples the coverage grid consumes. This is not §3.6's on-device
DA3-Small pass, which still needs a GPU to train and a device to run. It drives
guidance, and accepted keyframes retain the aligned sensor map as 16-bit
millimetre PNG evidence. The offline pipeline may still choose inferred depth,
but validation can now prove whether device depth was actually present.

Before production capture is enabled, the app must acquire several real depth
frames over time. A device without sustained depth support can only enter the
separately labelled RGB-only mode through a confirmation dialog; its bundle is
written with `capture_mode: rgb_only`, and the HUD says coverage is unmeasurable
rather than drawing zeros that look like a successful target sequence.

## Not done

Occlusion for the coverage paint. There is no depth buffer for the real world
in this renderer, so coverage on a far wall shows through a near one. Left as
is deliberately — seeing that the wall behind you is still amber is useful, and
a half-right occlusion test reads as missing coverage, which is worse.
