"""Masking protocol (PLAN.md §3.5).

Two masks come out of this stage and they are *not* the same mask:

* `fusion_mask` — pixels safe to fuse into TSDF. Mirrors, windows, glass doors,
  TVs and people are excluded. This is the phantom-room fix and it must be
  applied *before* fusion, never after (PLAN.md §7).
* `geometric_weight` — a soft weight for PGSR's geometric loss. Masked regions
  are down-weighted but stay in the photometric loss, so a mirror still renders
  as a mirror in the walkthrough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from recon.geom import Semantic
from recon.io.bundle import BundleDir

#: Text prompts from PLAN.md §3.5, verbatim.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "mirror",
    "window",
    "glass door",
    "television screen",
    "person",
)

#: Weight applied to masked pixels in the geometric loss. Not zero — a fully
#: zeroed region lets Gaussians drift there unconstrained.
MASKED_GEOMETRIC_WEIGHT = 0.05


@dataclass
class MaskResult:
    #: keyframe id -> (H, W) bool, True == safe to fuse.
    fusion_mask: dict[str, np.ndarray]
    #: keyframe id -> (H, W) float32 weight for the geometric loss.
    geometric_weight: dict[str, np.ndarray] = field(default_factory=dict)
    #: keyframe id -> {prompt: (H, W) bool}, kept for debugging and opening detection.
    per_prompt: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    model_name: str = "unknown"
    runtime_s: float = 0.0

    def masked_fraction(self) -> float:
        if not self.fusion_mask:
            return 0.0
        return float(np.mean([1.0 - m.mean() for m in self.fusion_mask.values()]))


@runtime_checkable
class Masker(Protocol):
    name: str

    def mask(self, bundle: BundleDir) -> MaskResult:
        ...


class NullMasker:
    """Masks nothing. The control arm — use it to measure what masking is worth."""

    name = "none"

    def mask(self, bundle: BundleDir) -> MaskResult:
        out = {}
        for kf in bundle.bundle.keyframes:
            out[kf.id] = np.ones((kf.intrinsics.height, kf.intrinsics.width), dtype=bool)
        return MaskResult(fusion_mask=out, model_name=self.name)


class GroundTruthMasker:
    """Perfect masks from the synthetic scene's semantic buffer.

    This is the ceiling on what SAM 3 can buy us. Running the benchmark with
    `none` / `gt` / `sam3` brackets the real model between a control and an
    oracle, which is the only way to tell a masking failure from a depth failure.
    """

    name = "gt"

    def __init__(self, dilate_px: int = 2) -> None:
        self.dilate_px = dilate_px

    def mask(self, bundle: BundleDir) -> MaskResult:
        fusion, weight = {}, {}
        for kf in bundle.bundle.keyframes:
            sem = bundle.semantic(kf)
            if sem is None:
                raise RuntimeError(
                    f"keyframe {kf.id} has no semantic buffer; GroundTruthMasker needs a "
                    "synthetic bundle"
                )
            bad = np.isin(sem, [int(s) for s in (*Semantic.__members__.values(),) if s in _BAD])
            bad = _dilate(bad, self.dilate_px)
            fusion[kf.id] = ~bad
            weight[kf.id] = np.where(bad, MASKED_GEOMETRIC_WEIGHT, 1.0).astype(np.float32)
        return MaskResult(fusion_mask=fusion, geometric_weight=weight, model_name=self.name)


_BAD = frozenset({Semantic.MIRROR, Semantic.WINDOW, Semantic.TV, Semantic.OUTSIDE})


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation. Masks should err generous — a few extra masked pixels
    cost coverage, a few missed ones cost a phantom room."""
    if radius <= 0:
        return mask
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out
