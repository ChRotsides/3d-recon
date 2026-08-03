# Room-to-House 3D Scanner

Phone capture → cloud reconstruction → photorealistic walkthrough + dimensioned
floor plan. No LiDAR required.

The approved architecture is [`room-scanner-implementation-plan.md`](room-scanner-implementation-plan.md).
Deviations, decisions and live build status live in
[`docs/execution-status.md`](docs/execution-status.md).

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
core/                Shared Rust guidance core                   (plan §3.8, §6.1)
clients/ios          Swift + ARKit pose logger                   (plan task 3)
clients/android      Kotlin + ARCore pose logger                 (plan task 3)
docs/                Execution status, ADRs, licensing audit     (plan §10)
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

The GPU stages (DA3, MapAnything, SAM 3, PGSR) are implemented behind adapters
but are **not validated** — the dev box has Kepler-era GPUs that PyTorch 2.x no
longer supports. See [`docs/execution-status.md`](docs/execution-status.md) for
what that means and what hardware unblocks it.
