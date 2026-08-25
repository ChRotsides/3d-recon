"""Segment every keyframe with SAM and cache one label map per frame.

Separate from any use of the segments, and cached, because segmenting is the
slow irreversible part (~10 s a frame on the TITANs) while what to *do* with the
segments took several attempts. Same reasoning as `--depth-cache`.

SAM 1 rather than SAM 3: automatic "segment everything" mode is what per-object
alignment needs, and it is SAM 1 that has it. SAM 3's promptable concept
segmentation answers a different question (mirrors, windows) and its checkpoint
is gated.

Weights resolve through HF_HOME from `source ~/ai/env.sh kepler`, and the model
is forced to fp32 with eager attention because the TITANs are compute capability
3.5 — no native FP16, no fused attention kernels.

    python scripts/capture/sam_masks.py BUNDLE OUT.npz [points_per_crop]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from recon.io.bundle import BundleDir
from recon.segments import composite_labels, segment_sizes

args = sys.argv[1:]
if len(args) < 2:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} BUNDLE OUT.npz [points_per_crop]")
BUNDLE = Path(args[0])
OUT = Path(args[1])
POINTS_PER_CROP = int(args[2]) if len(args) > 2 else 32
MODEL = "facebook/sam-vit-base"

# Deferred, so the rest of the harness stays importable without torch.
import torch  # noqa: E402
from recon.torch_compat import configure_cuda_for_legacy_device  # noqa: E402
from transformers import pipeline  # noqa: E402

device = 0 if torch.cuda.is_available() else -1
if device == 0:
    configure_cuda_for_legacy_device(torch, "cuda:0")
    capability = torch.cuda.get_device_capability(0)
    print(f"cuda:0 · {torch.cuda.get_device_name(0)} · capability {capability}", flush=True)
else:
    print("no CUDA, running on CPU", flush=True)

bundle = BundleDir.load(BUNDLE)
kfs = bundle.bundle.keyframes
print(
    f"{len(kfs)} keyframes · {MODEL} · {POINTS_PER_CROP}x{POINTS_PER_CROP} prompt grid", flush=True
)

generator = pipeline(
    "mask-generation",
    model=MODEL,
    device=device,
    dtype=torch.float32,
    model_kwargs={"attn_implementation": "eager"},
)

maps: dict[str, np.ndarray] = {}
meta: dict[str, dict] = {}
for i, kf in enumerate(kfs):
    rgb = bundle.rgb(kf)
    out = generator(
        Image.fromarray(rgb), points_per_batch=64, points_per_crop=POINTS_PER_CROP
    )
    masks = [np.asarray(m, dtype=bool) for m in out["masks"]]
    labels = composite_labels(masks)
    sizes = segment_sizes(labels)
    maps[kf.id] = labels
    meta[kf.id] = {
        "segments": int(len(sizes)),
        "raw_masks": len(masks),
        "covered": float((labels >= 0).mean()),
        "median_area_px": float(np.median(sizes)) if len(sizes) else 0.0,
    }
    print(
        f"  [{i + 1:2d}/{len(kfs)}] {kf.id} · {len(masks):3d} masks -> "
        f"{len(sizes):3d} segments · {meta[kf.id]['covered']:.0%} of the frame",
        flush=True,
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, model=MODEL, meta=json.dumps(meta), **maps)
covered = np.mean([m["covered"] for m in meta.values()])
segments = np.sum([m["segments"] for m in meta.values()])
print(
    f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) · "
    f"{segments} segments over {len(maps)} frames · {covered:.0%} mean coverage",
    flush=True,
)
