# Phase 2 — Incremental RGB graph and live guidance

## Vertical outcome

The Phase 1 flow remains usable from capture through final 3D, but the service now processes each photograph as it arrives. The phone receives a live spherical map showing missing directions in red, weak directions in amber, sufficient directions in green, and a reasoned next-photo target.

## End-to-end slice

```text
new RGB photograph
  -> cached spatially balanced RootSIFT extraction
  -> match only new neighbour and loop-closure candidates
  -> strict geometric and photometric edge verification
  -> versioned focal/rotation graph update
  -> spherical coverage plus reason masks
  -> RGB-localized current preview and next target
  -> phone guidance
  -> Finish still runs Phase 1 final 3D
```

Global reoptimization may move rotations and coverage. Therefore every graph, panorama, localization, target, and UI asset carries a version. Coverage is not required to be monotonic.

## Confidence channels

The service keeps separate angular channels for:

- photographed coverage and independent-view count;
- edge strength, feature spread, reprojection error, and cycle consistency;
- sharpness, exposure, and parallax risk;
- seam ownership and source-image support;
- current-preview localization confidence.

The UI state is derived from these channels and carries a reason mask. It is not an unexplained aggregate score.

## Deliverables

- Cached feature sets keyed by JPEG checksum and cached verified pair results keyed by image/configuration checksums.
- Incremental neighbour and loop-closure candidate selection.
- Versioned focal and robust rotation-graph optimization.
- Versioned spherical coverage/confidence/reason assets.
- RGB-only current-preview localization against the station graph.
- Stable target IDs and deterministic target prioritization.
- A red/amber/green spherical renderer on Android with an explicit unlocalized state.
- Process-restart persistence and idempotent result events.
- Prefix-by-prefix replay comparison against a batch graph.
- JSON acceptance output under `out/live-rgb-validation/phase-02/`.

## Automated validation

```bash
PYTHONPATH=harness uv run pytest -q harness/tests/live_rgb/phase_02
PYTHONPATH=harness uv run python scripts/live_rgb/validate_phase.py 2 \
  --bundle data/captures/bundle_065a6e7a-e8e3-449f-aa0d-61fc540d6e41 \
  --output out/live-rgb-validation/phase-02
cd clients/android && ./gradlew testDebugUnitTest connectedDebugAndroidTest
```

The validator compares every deterministic replay prefix with a fresh batch solve, spies on feature/match cache calls, localizes held-out preview images, and injects blur, exposure failure, false links, textureless images, and parallax-risk fixtures.

Latency is reported only against a frozen reference hardware profile. It is not silently enforced on arbitrary CI machines.

## Automated acceptance criteria

- [ ] Phase 1 end-to-end finalization and 3D-viewer gates continue to pass.
- [ ] Feature extraction runs once per unique JPEG checksum.
- [ ] Previously verified pairs are not rematched unless their matching configuration changes.
- [ ] Repeating an ingested frame creates no second frame, edge, graph version, or target event.
- [ ] Restarting the service mid-replay produces the same final graph as uninterrupted replay.
- [ ] The final incremental selected component and accepted edges equal the final batch graph.
- [ ] Incremental and batch rotations differ by at most `0.50°` median on the reference replay.
- [ ] The final reference graph still has 30 selected images, 65 verified edges, and cycle error at most `1.30°`.
- [ ] Every coverage cell contains a state enum, contributing channel values, and reason mask.
- [ ] A textureless or ambiguous preview returns `unlocalized`, emits no directional arrow, and does not mutate the graph.
- [ ] Blur, exposure, false-link, and parallax-risk fixtures produce their configured rejection reasons.
- [ ] Result versions increase strictly; delayed older graph, localization, or target assets cannot replace newer state.
- [ ] Restart recovery cannot expose a partially written graph or panorama asset.
- [ ] On the frozen benchmark profile, p95 photograph-to-guidance latency is at most `2.0 s`; the report names the hardware and configuration.
- [ ] Request and data-flow guards still prove that no ARCore or mobile geometric field enters reconstruction.

## Deferred

Live draft 3D, corrective targets selected from 3D, hardened offline recovery, full-quality live finalization, and multiple stations remain deferred.
