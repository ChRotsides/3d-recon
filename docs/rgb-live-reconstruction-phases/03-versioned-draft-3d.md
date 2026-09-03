# Phase 3 — Versioned draft 3D

## Vertical outcome

While the Phase 2 scan is still active, a sufficiently connected station periodically produces an RGB-only draft point cloud. The phone downloads and displays each complete version in an orbitable 3D viewer without interrupting capture.

The draft is allowed to move when the image graph improves. It is never incrementally fused with an older draft.

## End-to-end slice

```text
new stable panorama version
  -> exact 2:1 wrap
  -> six 256x256, 90° pinhole views at one optical centre
  -> DA3 Small draft inference
  -> replace predicted cameras with exact virtual cameras
  -> back-project and quantize draft cloud
  -> checksum-addressed version event
  -> phone decodes and atomically swaps the 3D scene
```

The inference worker consumes only the RGB panorama produced by Phase 1. No mobile pose, depth, camera intrinsics, or tracking output is introduced.

## Deliverables

- A queued draft-reconstruction worker with coalescing: stale pending panorama versions are skipped.
- A low-resolution six-face cubemap renderer using one shared virtual optical center.
- DA3 Small inference with explicit processed-resolution intrinsic rescaling.
- Point records containing XYZ, RGB, DA3 confidence, panorama support, and source version.
- A bounded, versioned binary transport with metadata and checksums.
- An Android point-cloud decoder and orbitable OpenGL viewer.
- Atomic scene replacement while camera capture and upload continue.
- Geometry round-trip fixtures and tests under `harness/tests/live_rgb/phase_03/`.

## Automated validation

Run:

```bash
PYTHONPATH=harness uv run pytest -q harness/tests/live_rgb/phase_03 -m "not real_da3"
PYTHONPATH=harness uv run python scripts/live_rgb/validate_phase.py 3 \
  --bundle data/captures/bundle_065a6e7a-e8e3-449f-aa0d-61fc540d6e41 \
  --depth-backend stub \
  --output out/live-rgb-validation/phase-03
cd clients/android && ./gradlew testDebugUnitTest connectedDebugAndroidTest
```

The replay pauses at configured frame counts so the validator can assert that drafts are produced before the final photograph and superseded safely afterward.

The separately marked real-model gate runs on the frozen GPU profile:

```bash
PYTHONPATH=harness uv run pytest -q harness/tests/live_rgb/phase_03 -m real_da3
PYTHONPATH=harness uv run python scripts/live_rgb/validate_phase.py 3 \
  --bundle data/captures/bundle_065a6e7a-e8e3-449f-aa0d-61fc540d6e41 \
  --depth-backend da3-small \
  --output out/live-rgb-validation/phase-03-real-da3
```

## Automated acceptance criteria

- [ ] The six virtual faces cover every sampled sphere direction exactly once, except shared mathematical boundaries.
- [ ] Pinhole projection followed by back-projection has maximum angular error below `1e-5` radians on the geometry fixture.
- [ ] All six cameras have identical optical centers and exact 90° horizontal and vertical fields of view.
- [ ] DA3-predicted camera matrices are absent from exported geometry; exported matrices equal the virtual camera specification.
- [ ] Intrinsics are rescaled by the exact rendered-to-processed resolution ratio.
- [ ] Every finite positive draft depth pixel remains eligible; confidence is exported as evidence rather than used to erase directions.
- [ ] `capture-065a` produces at least one valid draft before its final input image and a newer draft after loop closure.
- [ ] A draft contains at most 250,000 points and its compressed transport is at most 12 MiB.
- [ ] On the frozen GPU profile, p95 stable-panorama-to-published-draft latency is at most `15 s`, with hardware/model revisions recorded.
- [ ] A newer panorama coalesces queued stale work; an inference already running may finish but cannot become current if superseded.
- [ ] Each point-cloud asset passes checksum, finite-coordinate, point-count, and declared-bounds validation on server and Android.
- [ ] Android swaps an entire version in one render-thread transaction; mixed-version points never appear in instrumentation state.
- [ ] A synthetic low-support wedge changes only the corresponding points' evidence state; unrelated points retain their prior channels.
- [ ] Capture, JPEG persistence, and upload tests continue to pass while draft downloads and rendering are active.
- [ ] The phase validator records inference time, transport size, decode time, point count, and peak worker memory without using those measurements as hidden correctness gates.

## Deferred

This phase displays evidence but does not yet drive corrective photography from 3D, declare the room complete, align translated stations, or run the full 512→504 final export.
