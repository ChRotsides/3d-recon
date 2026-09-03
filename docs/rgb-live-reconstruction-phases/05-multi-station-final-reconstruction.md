# Phase 5 — Multi-station alignment and final reconstruction

## Vertical outcome

The user completes one RGB-only panorama station, moves to another explicit station, captures a second panorama, and receives a merged room reconstruction. Stopping the scan runs the established full-quality six-face pipeline and exports a final cloud, scene, confidence data, and reproducibility diagnostics.

Photographs from translated stations are never inserted into one panorama graph.

## End-to-end slice

```text
Station A RGB graph -> panorama -> DA3 cube A
Station B RGB graph -> panorama -> DA3 cube B
  -> shared verified RGB correspondences
  -> back-project matched pixels through each exact cube
  -> robust Sim(3) station edge
  -> globally optimized station graph
  -> confidence-aware merged draft
  -> full 512x512 faces / DA3 504 processing / final 800k cloud
```

An unalignable station remains a separate component with an explicit diagnostic. The system must never force a plausible-looking merge.

## Deliverables

- Explicit station lifecycle in Android and the service.
- Per-station image graphs, panoramas, confidence maps, cubes, and immutable versions.
- Cross-station RGB matching and 3D correspondence lifting through each station's exact virtual cameras.
- Robust similarity-transform estimation with held-out correspondence scoring.
- Global station-graph optimization, component handling, and merge provenance.
- Guidance that distinguishes “photograph again here” from “start/move to another station.”
- Full-quality final reconstruction using 512×512 virtual faces, DA3 processing at 504, and an 800,000-point budget.
- Final `cloud.bin`, `cloud.json`, `scene.glb`, confidence assets, graph diagnostics, checksums, and run manifest.
- Synthetic Sim(3), adversarial false-link, restart/resume, and final-replay fixtures.
- Tests under `harness/tests/live_rgb/phase_05/` plus complete Android regression tests.

## Automated validation

Run:

```bash
PYTHONPATH=harness uv run pytest -q harness/tests/live_rgb/phase_05
PYTHONPATH=harness uv run python scripts/live_rgb/validate_phase.py 5 \
  --fixture-manifest harness/tests/fixtures/live_rgb/multi-station.json \
  --output out/live-rgb-validation/phase-05
cd clients/android
./gradlew testDebugUnitTest assembleDebug lintDebug connectedDebugAndroidTest
```

The multi-station manifest includes known synthetic similarity transforms, a disconnected station, a false high-count planar match, and an interrupted final export.

## Automated acceptance criteria

- [ ] Images are namespaced by station and no matching path can insert a translated image into another station's panorama graph.
- [ ] Synthetic station transforms recover scale within `2%`, rotation within `1.0°`, and translation within `2%` of the fixture scene diagonal.
- [ ] Station edges require spatially distributed RGB inliers and pass a held-out 3D correspondence check.
- [ ] The adversarial planar/false-link fixture is rejected despite having more raw matches than the valid edge.
- [ ] The robust transform fixture rejects at least `90%` of injected correspondence outliers while retaining a valid station edge.
- [ ] Global optimization lowers held-out station-graph error without worsening any protected valid edge beyond its configured tolerance.
- [ ] A disconnected station remains a separate component and is labelled unmerged in all outputs.
- [ ] Every merged point records source station, source result version, and contributing confidence channels.
- [ ] Final cubemap faces are 512×512, DA3 processed intrinsics are scaled to 504, and all six exact cameras share one center per station.
- [ ] The final exported cloud contains at most 800,000 finite points and its declared bounds exactly enclose them.
- [ ] `cloud.bin`, `cloud.json`, `scene.glb`, confidence assets, graph diagnostics, checksums, and run manifest all pass schema and checksum validation.
- [ ] Interrupting and resuming final export yields the same content hashes as an uninterrupted run.
- [ ] The complete single-station `capture-065a` compatibility run still satisfies Phase 1 graph gates and the established six-face geometry gates.
- [ ] Repository-wide Python, Rust, and Android automated suites pass after the production RGB-only flow is enabled.
- [ ] Production APK manifest and dependency validation still prove that ARCore is absent.

## Completion definition

Phase 5 is complete when every automated gate above passes and the validation directory contains enough immutable evidence to reproduce each result. Visual quality and real-room usability are intentionally left for the user's later manual acceptance pass.
