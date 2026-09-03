# Phase 4 — Corrective rescan, resilience, and finalization

## Vertical outcome

The live 3D view identifies a weak region, maps it to a spherical target, guides the user with RGB localization, accepts a corrective photograph, and publishes a measurably improved draft. The complete flow survives phone/network/worker interruption and finishes with the established full-quality 512→504, 800,000-point reconstruction.

## End-to-end slice

```text
tap or automatically select weak 3D region
  -> point/face/pixel/direction target mapping
  -> RGB-localized camera guidance
  -> unconditional shutter and durable outbox
  -> incremental graph/confidence update
  -> coalesced replacement draft
  -> explicit target result
  -> completion gates
  -> global graph optimization and Phase 1 full-quality export
```

DA3 confidence influences color and priority but never deletes a finite positive direction. Targets that cannot improve from the current optical center become reason-coded `station_limited`; they do not loop forever.

## Deliverables

- 3D-region-to-spherical-direction targeting with stable target IDs.
- Target priority over the separate coverage, graph, image, and depth confidence channels.
- A durable checksum-addressed Android outbox with retry, resume, and duplicate-safe acknowledgement.
- Result-event reconnection, monotonic version application, and invalid-asset fallback.
- Worker restart recovery and job coalescing under capture bursts.
- Completion gates for coverage, connectivity, loop closure, ceiling/floor evidence, independent image support, and unresolved recoverable targets.
- Full final graph optimization and the proven 512×512 face / 504 processing / 800,000-point export.
- A withheld-image correction harness plus network, process-kill, duplicate, and out-of-order fixtures.
- JSON acceptance report under `out/live-rgb-validation/phase-04/`.

## Automated validation

```bash
PYTHONPATH=harness uv run pytest -q harness/tests/live_rgb/phase_04 -m "not real_da3"
PYTHONPATH=harness uv run python scripts/live_rgb/validate_phase.py 4 \
  --bundle data/captures/bundle_065a6e7a-e8e3-449f-aa0d-61fc540d6e41 \
  --withhold-manifest harness/tests/fixtures/live_rgb/rescan-cases.json \
  --fault-manifest harness/tests/fixtures/live_rgb/recovery-cases.json \
  --depth-backend stub \
  --output out/live-rgb-validation/phase-04/stub
cd clients/android && ./gradlew testDebugUnitTest connectedDebugAndroidTest
```

Run the separately marked full-quality model gate:

```bash
PYTHONPATH=harness uv run pytest -q harness/tests/live_rgb/phase_04 -m real_da3
PYTHONPATH=harness uv run python scripts/live_rgb/validate_phase.py 4 \
  --bundle data/captures/bundle_065a6e7a-e8e3-449f-aa0d-61fc540d6e41 \
  --depth-backend da3-small \
  --output out/live-rgb-validation/phase-04/real-da3
```

## Automated acceptance criteria

- [ ] Phases 1–3 remain green through the same production path.
- [ ] Point→cubemap face/pixel→world-direction target round-trip error is no more than one draft pixel.
- [ ] Each target contains a stable ID, source version, spherical bounds, priority, channel values, and reason code.
- [ ] Withheld-image fixtures select a target intersecting the known withheld sector.
- [ ] Supplying the designated corrective image improves the intended failing channel without crediting unseen sectors.
- [ ] Improvements elsewhere cannot falsely resolve the target.
- [ ] A blank or ambiguous preview suppresses directional guidance while the shutter remains enabled.
- [ ] Repeated non-improving corrective images transition a target to `station_limited` with recorded evidence.
- [ ] Killing and restarting phone app, service, or worker loses no accepted JPEG and creates no logical duplicate.
- [ ] Delayed events and checksum-invalid assets cannot replace the last valid newer view.
- [ ] A burst of photographs results in exactly one current draft after work coalesces; stale drafts may finish but never publish as current.
- [ ] Completion remains false when any required gate is programmatically removed and becomes true for the complete frozen reference replay.
- [ ] Finalization globally optimizes all accepted images and satisfies the Phase 1 graph, cubemap, DA3-camera, confidence-retention, and cloud gates.
- [ ] Interrupting finalization and resuming yields the same content hashes as an uninterrupted run.
- [ ] Full-quality output uses 512×512 faces, DA3 processing at 504, all six exact virtual cameras, and at most 800,000 finite points.
- [ ] ARCore absence and RGB-only payload guards remain green.

## Deferred

Translation is still forbidden inside a station. Explicit station separation, alignment, and merged-room provenance arrive in Phase 5.
