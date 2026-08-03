#!/usr/bin/env bash
# One-command demo: render synthetic rooms with exact ground truth, reconstruct
# them, and print the accuracy against that truth.
#
#   ./scripts/demo.sh            # one room, ~1 minute
#   ./scripts/demo.sh all        # all five rooms + the benchmark table, ~10 min
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ ! -d .venv ]]; then
  echo "no .venv — run:  uv venv --python 3.12 .venv && . .venv/bin/activate && uv pip install -e '.[geometry,dev]'"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

MODE="${1:-one}"
OUT=runs/demo
mkdir -p "$OUT"

if [[ "$MODE" == "all" ]]; then
  echo "==> rendering five synthetic rooms (exact ground truth)"
  recon synth data/scenes --preset all --width 240 --height 180 \
    --stations 3 --max-keyframes 90 --vio-scale-error 1.025

  echo
  echo "==> reconstructing each and comparing against truth"
  for scene in data/scenes/*/; do
    name=$(basename "$scene")
    echo "--- $name"
    recon reconstruct "$scene" --out "$OUT/$name"
  done

  echo
  echo "==> benchmark grid (with and without masking)"
  recon bench data/scenes --out "$OUT/bench" --depth gt --mask gt,none
  echo "table -> $OUT/bench/report.md"
else
  echo "==> rendering one L-shaped room with a 2.5% VIO scale error"
  recon synth "$OUT/scene" --preset l_shaped --width 240 --height 180 \
    --stations 3 --max-keyframes 90 --vio-scale-error 1.025

  echo
  echo "==> checking the bundle is usable (what you would run on a real capture)"
  recon validate "$OUT/scene"

  echo
  echo "==> reconstructing"
  recon reconstruct "$OUT/scene" --out "$OUT/room"

  echo
  echo "artifacts:"
  ls -1 "$OUT/room"
  echo
  echo "open $OUT/room/floorplan.svg in a browser to see the dimensioned plan"
fi
