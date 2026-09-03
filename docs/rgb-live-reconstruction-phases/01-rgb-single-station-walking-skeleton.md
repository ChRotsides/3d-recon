# Phase 1 — RGB-only single-station walking skeleton

## Vertical outcome

The phone starts a scan without ARCore, captures RGB photographs at one optical center, uploads them to the reconstruction service, finishes the station, and displays the resulting 3D point cloud on the phone.

This phase is deliberately batch-after-Finish. It establishes the complete usable path before making any stage incremental.

## Non-negotiable RGB-only boundary

The production capture activity uses Camera2 or CameraX. The shipping manifest must not require AR hardware, and the shipping runtime dependency graph must not contain `com.google.ar:core`.

The reconstruction API accepts only session/station/frame IDs, JPEG bytes, and an ordering timestamp. Its schema uses `additionalProperties: false` and rejects pose, depth, intrinsics, sparse points, planes, anchors, tracking state, heading, IMU data, and any AR-derived geometry.

## End-to-end slice

```text
Camera2/CameraX RGB shutter
  -> local JPEG with stable frame ID
  -> idempotent upload to one station
  -> Finish
  -> established RootSIFT overlap graph and robust rotation solve
  -> narrow-owner spherical panorama and exact 2:1 wrap
  -> six exact 512x512 virtual cameras
  -> DA3 Small at 504 processing resolution
  -> exact-camera back-projection, up to 800,000 points
  -> versioned cloud asset
  -> phone point-cloud viewer
```

The existing image-stitch, panorama-wrap, cubemap, and DA3 primitives are reused rather than reimplemented. Phase 1 creates a new RGB schema; it does not adapt the old AR session bundle into a live protocol.

The current Android “RGB-only” fallback is not this phase's starting point: it still obtains images through an ARCore `Session`. The new production path starts at Camera2/CameraX.

## Deliverables

- A new Camera2/CameraX RGB scan activity with a manual shutter and Finish action.
- Removal of ARCore requirements and runtime dependencies from the shipping variant.
- A strict session/station/frame API with idempotent upload and finalize operations.
- Persistent service storage for JPEGs, job state, diagnostics, and immutable result versions.
- A batch reconstruction worker wrapping the proven RGB-only pipeline.
- An Android decoder and orbitable viewer for `cloud.bin`/`cloud.json`.
- A replay adapter for the 42 RGB images in `capture-065a`; the adapter itself may read only IDs and JPEG paths.
- A fast deterministic validation path using a depth-model stub and a separately marked real-DA3 acceptance path.
- A JSON acceptance report under `out/live-rgb-validation/phase-01/`.
- Tests under `harness/tests/live_rgb/phase_01/` and Android unit/instrumentation tests.

## Automated validation

Run the deterministic suite:

```bash
PYTHONPATH=harness uv run pytest -q harness/tests/live_rgb/phase_01 -m "not real_da3"
cd clients/android
./gradlew testDebugUnitTest assembleDebug lintDebug connectedDebugAndroidTest
cd ../..
PYTHONPATH=harness uv run python scripts/live_rgb/validate_phase.py 1 \
  --bundle data/captures/bundle_065a6e7a-e8e3-449f-aa0d-61fc540d6e41 \
  --depth-backend stub \
  --output out/live-rgb-validation/phase-01/stub
```

Run the separately reported real-model gate:

```bash
PYTHONPATH=harness uv run pytest -q harness/tests/live_rgb/phase_01 -m real_da3
PYTHONPATH=harness uv run python scripts/live_rgb/validate_phase.py 1 \
  --bundle data/captures/bundle_065a6e7a-e8e3-449f-aa0d-61fc540d6e41 \
  --depth-backend da3-small \
  --output out/live-rgb-validation/phase-01/real-da3
```

The validator also replays a poisoned copy of the reference bundle whose non-image metadata contains deliberately invalid poses, depth, intrinsics, headings, and sparse points.

## Automated acceptance criteria

- [ ] Python, Android unit, instrumentation, lint, and debug assembly tasks pass.
- [ ] The shipping APK manifest contains camera permission but no AR hardware feature or ARCore metadata.
- [ ] The shipping runtime classpath and APK contain no ARCore SDK classes or native library.
- [ ] Static validation finds no ARCore imports in the new RGB scan package.
- [ ] One shutter press creates one local JPEG and stable frame ID before network I/O.
- [ ] Duplicate upload and duplicate Finish requests produce one logical frame and one finalization job.
- [ ] The service rejects undeclared request fields, including all forbidden geometric fields.
- [ ] Poisoning every non-RGB reference-bundle field leaves selected images, graph metrics, panorama, virtual cameras, and cloud numerically unchanged.
- [ ] The `capture-065a` graph contains the established 30-image largest component and 65 verified edges, with median cycle error at most `1.30°`.
- [ ] The wrapped panorama is exactly 2:1, horizontally continuous, and has no transparent pixels.
- [ ] Diagnostics describe six named 512×512 faces, 90° horizontal/vertical FOV, one shared center, and exact virtual matrices.
- [ ] DA3 receives `extrinsics=None` and `intrinsics=None`; exported geometry uses the exact virtual cameras instead of predicted cameras.
- [ ] Confidence filtering removes no finite positive depth direction.
- [ ] The final cloud has no non-finite coordinates, contains at most 800,000 points, and its declared bounds enclose every point.
- [ ] Android verifies asset checksum/schema and renders the accepted final asset in instrumentation state.
- [ ] The JSON acceptance report records every threshold, model revision, dataset checksum, and result.

## Deferred

Incremental matching, live coverage, draft 3D during capture, confidence-driven corrective photography, robust recovery, and translated stations are later vertical phases.
