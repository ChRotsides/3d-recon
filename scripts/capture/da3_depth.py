"""Run DA3 once and cache the corrected depth maps.

Meshing wants Open3D and Open3D is not installed in the Kepler GPU environment;
DA3 wants the GPU and is not much use without it. Rather than fight that, the
model runs here and writes what it produced, and `mesh.py` reads the file in
whichever environment can fuse it.

That split is worth having on its own terms: fusion has parameters worth
sweeping — voxel size, truncation, how far to trust the depth — and none of them
should cost 40 seconds of GPU inference to try.

    python scripts/capture/da3_depth.py BUNDLE OUT.npz [variant] [space]

The maps written are **after** the sparse anchor and the reference-depth affine
fit, i.e. exactly the metres the cloud builder places.
"""

import sys
from pathlib import Path

import numpy as np
from recon.depth.da3 import DA3Config, DA3Depth
from recon.io.bundle import BundleDir

BUNDLE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
VARIANT = sys.argv[3] if len(sys.argv) > 3 else "large"
SPACE = sys.argv[4] if len(sys.argv) > 4 else "depth"

bundle = BundleDir.load(BUNDLE)
kfs = bundle.bundle.keyframes
print(f"{len(kfs)} keyframes · DA3-{VARIANT}", flush=True)

result = DA3Depth(DA3Config(variant=VARIANT, reference_anchor_space=SPACE)).estimate(bundle)
print(f"depth done · model {result.model_name} · {result.runtime_s:.1f}s", flush=True)

fitted = {
    d["keyframe"] for d in result.diagnostics.get("reference_anchor", []) if d.get("fitted")
}
print(f"reference anchor ({SPACE}): {len(fitted)}/{len(kfs)} frames", flush=True)

maps = {kf.id: result.depth[kf.id].astype(np.float32) for kf in kfs if kf.id in result.depth}
OUT.parent.mkdir(parents=True, exist_ok=True)
# `model` and `anchored` ride along so a cached file can always answer what
# produced it — a stale .npz that silently meshes as if it were the current
# settings is the obvious way for this split to go wrong.
np.savez_compressed(
    OUT,
    model=np.array(result.model_name),
    anchor_space=np.array(SPACE),
    anchored=np.array(sorted(fitted)),
    **maps,
)
print(f"wrote {OUT} · {len(maps)} maps · {OUT.stat().st_size / 1e6:.1f} MB", flush=True)
