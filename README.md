# Room-to-House 3D Scanner

Phone capture → cloud reconstruction → photorealistic walkthrough + dimensioned
floor plan. No LiDAR required.

The approved architecture is [`room-scanner-implementation-plan.md`](room-scanner-implementation-plan.md).

- **[`docs/status-summary.md`](docs/status-summary.md)** — one page: what is done,
  what is blocked, and on what.
- [`docs/execution-status.md`](docs/execution-status.md) — deviations from the
  plan and the reasoning behind each.
- [`docs/licensing-audit.md`](docs/licensing-audit.md) — verified licences; three
  findings change the plan.

**Measured against the plan's §1.3 targets:** wall-length error 0.06–0.15%
(target 1.5%), floor area 0.05–0.16% (target 3%), whole-house drift 12.6 cm over
six rooms (target 15 cm). 188 tests green.

## Layout

```
harness/recon/       Phase-0 offline validation harness (Python)
  schemas.py           SessionBundle / HouseGraph / RoomLayout  (plan §5)
  geom.py              meshes, triangulation, ray casting
  synth.py             synthetic rooms with exact ground truth
  depth/               DA3, MapAnything, ground-truth adapters   (plan §3.1-3.2)
  masking/             SAM 3, ground-truth, null maskers         (plan §3.5)
  fusion/              TSDF volume                               (plan §6.4)
  metric/              scale solver                              (plan §6.2)
  layout/              planes, Atlanta regularization, floor plan(plan §6.4)
  splat/               PGSR / 2DGS trainers on gsplat            (plan §3.3-3.4)
  eval/                metrics + benchmark grid                  (plan §8)
  stitch/              room alignment, house pose graph          (plan §6.3)
core/                Shared Rust guidance core + UniFFI          (plan §3.8, §6.1)
server/              FastAPI reconstruction service              (plan §2, Phase 2)
viewer/              Babylon.js walkthrough                      (plan §3.8)
clients/ios          Swift + ARKit pose logger                   (plan task 3)
clients/android      Kotlin + ARCore pose logger                 (plan task 3)
docs/                Status, ADRs, licensing audit               (plan §10)
```

## Quick start

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -e '.[geometry,dev]'

# Render five synthetic rooms with exact ground truth
recon synth data/scenes --preset all --vio-scale-error 1.025

# Reconstruct one and compare against truth
recon reconstruct data/scenes/l_shaped --out runs/l_shaped

# Run the Phase-0 benchmark grid
recon bench data/scenes --out runs/bench
```

`recon bench` writes `report.md` (the plan §4 table), `report.json` (the
regression baseline) and evaluates the Phase-0 exit criterion.

## Rust guidance core

```bash
cd core && cargo test
```

## Status

Everything the plan calls for that can be built and verified **without a GPU, a
phone or a Mac is built and verified**. What remains is hardware, not unwritten
code:

| Blocked on | What |
|---|---|
| a GPU | DA3 / MapAnything / SAM 3 / PGSR actually running |
| a device | on-device DA3-Small, capture-time metrics |
| a Mac | the iOS Xcode project |

See [`docs/status-summary.md`](docs/status-summary.md).
