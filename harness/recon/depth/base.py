"""Depth estimator protocol.

Everything downstream consumes `DepthResult`, so DA3, MapAnything, a ground-truth
oracle and any future model are interchangeable. This is what lets the whole
pipeline be developed and regression-tested without a GPU (see
docs/execution-status.md) and then run unchanged on an L40S.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from recon.io.bundle import BundleDir
from recon.schemas import Keyframe


@dataclass
class DepthResult:
    """Per-keyframe metric depth, in the bundle's (drifted) reconstruction frame."""

    #: keyframe id -> (H, W) float32 metres, 0 == no return.
    depth: dict[str, np.ndarray]
    #: keyframe id -> (H, W) float32 confidence, if the model reports it.
    #: Higher must mean more reliable, but scores need not be calibrated or in
    #: [0, 1]; fusion robustly normalises each frame before using them.
    confidence: dict[str, np.ndarray] = field(default_factory=dict)
    #: Independent metric-scale estimate (MapAnything's scale token, PLAN.md §3.2).
    #: Multiplies the reconstruction to reach metric. None if the model has no opinion.
    scale_estimate: float | None = None
    scale_sigma: float | None = None
    model_name: str = "unknown"
    runtime_s: float = 0.0
    peak_vram_gb: float = 0.0
    #: Backend-specific, JSON-serialisable evidence such as sparse-anchor support.
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def stack(self, keyframes: list[Keyframe]) -> np.ndarray:
        return np.stack([self.depth[k.id] for k in keyframes])


@runtime_checkable
class DepthEstimator(Protocol):
    name: str

    def estimate(self, bundle: BundleDir) -> DepthResult:
        """Predict metric depth for every keyframe in the bundle.

        Implementations must honour PLAN.md §3.1: when the bundle carries poses
        and intrinsics, run the model *pose-conditioned* rather than re-solving
        the cameras.
        """
        ...


class GroundTruthDepth:
    """Reads the depth already stored in the bundle.

    For synthetic scenes this is the exact answer, which turns every stage after
    depth into something with a machine-precision oracle. For RGBD rigs (ZED 2i)
    it is the sensor depth. It is *not* a model — a config using it reports
    itself as `gt` in the benchmark table so results are never confused with a
    real prediction.
    """

    name = "gt"

    def __init__(self, noise_sigma_m: float = 0.0, seed: int = 0) -> None:
        self.noise_sigma_m = noise_sigma_m
        self._rng = np.random.default_rng(seed)

    def estimate(self, bundle: BundleDir) -> DepthResult:
        out: dict[str, np.ndarray] = {}
        for kf in bundle.bundle.keyframes:
            d = bundle.depth(kf)
            if d is None:
                raise RuntimeError(
                    f"keyframe {kf.id} has no stored depth; GroundTruthDepth needs an "
                    "RGBD or synthetic bundle"
                )
            if self.noise_sigma_m > 0:
                # Depth noise grows with range, as it does on any real sensor.
                sigma = self.noise_sigma_m * np.maximum(d, 0.5) ** 2
                d = np.where(d > 0, d + self._rng.normal(0.0, 1.0, d.shape) * sigma, 0.0)
            out[kf.id] = d.astype(np.float32)
        return DepthResult(depth=out, model_name=self.name)
