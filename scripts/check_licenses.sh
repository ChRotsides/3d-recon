#!/usr/bin/env bash
# Re-run the licensing audit (docs/licensing-audit.md).
#
# Model and repo licences change. This prints the current state so it can be
# diffed against the audit table before any launch decision.
set -uo pipefail

repos=(
  ByteDance-Seed/Depth-Anything-3
  facebookresearch/map-anything
  facebookresearch/sam3
  zju3dv/PGSR
  nerfstudio-project/gsplat
  nianticlabs/spz
  hbb1/2d-gaussian-splatting
  autonomousvision/gaussian-opacity-fields
  isl-org/Open3D
  borglab/gtsam
  colmap/colmap
  pytorch/executorch
)

models=(
  depth-anything/DA3-SMALL
  depth-anything/DA3-BASE
  depth-anything/DA3-LARGE-1.1
  depth-anything/DA3-GIANT-1.1
  depth-anything/DA3METRIC-LARGE
  facebook/map-anything
  facebook/map-anything-apache
)

echo "== code repositories =="
for r in "${repos[@]}"; do
  lic=$(curl -fsS --max-time 20 "https://api.github.com/repos/$r/license" 2>/dev/null \
    | python3 -c 'import sys,json;d=json.load(sys.stdin);print((d.get("license") or {}).get("spdx_id","?"))' 2>/dev/null)
  printf '  %-52s %s\n' "$r" "${lic:-LOOKUP FAILED}"
done

echo
echo "== model weights (HuggingFace card metadata) =="
for m in "${models[@]}"; do
  lic=$(curl -fsS --max-time 20 "https://huggingface.co/api/models/$m" 2>/dev/null \
    | python3 -c 'import sys,json;d=json.load(sys.stdin);print((d.get("cardData") or {}).get("license","?"))' 2>/dev/null)
  printf '  %-52s %s\n' "$m" "${lic:-LOOKUP FAILED}"
done

echo
echo "== DA3 model-zoo table (the upstream README, which has disagreed with the cards) =="
curl -fsS --max-time 20 https://raw.githubusercontent.com/ByteDance-Seed/Depth-Anything-3/main/README.md 2>/dev/null \
  | grep -E '^\| \[DA3' \
  | sed -E 's#\| \[([^]]*)\][^|]*#  \1#; s#\|#|#g' \
  | awk -F'|' '{printf "  %-34s %s\n", $1, $(NF-1)}'

echo
echo "Compare against docs/licensing-audit.md. Investigate any change."
