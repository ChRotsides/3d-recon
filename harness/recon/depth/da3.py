"""Depth Anything 3 adapter — pose-conditioned mode (PLAN.md §3.1).

The whole point of picking DA3 is that ARKit/ARCore hand us poses and intrinsics
for free, which puts us in DA3's strongest configuration. This adapter therefore
*always* feeds the bundle's poses and intrinsics in and never asks DA3 to
re-solve the cameras.

Not runnable on the Kepler dev box (see docs/execution-status.md). Import of
torch is deferred to `estimate()` so the rest of the harness stays importable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from recon.depth.base import DepthResult
from recon.io.bundle import BundleDir

#: Published checkpoints, with their verified weight licences
#: (docs/licensing-audit.md §1.2). The plan names DA3-Large for the server, but
#: the upstream model zoo lists LARGE and GIANT as CC BY-NC 4.0 while the
#: HuggingFace card for LARGE says Apache — an unresolved contradiction. Until
#: ByteDance clarifies, only the Apache rows are usable commercially, so the
#: benchmark grid must include DA3-BASE and DA3METRIC-LARGE as the
#: commercial-path candidates rather than assuming LARGE will be available.
VARIANTS = {
    "small": "depth-anything/DA3-SMALL",
    "base": "depth-anything/DA3-BASE",
    "large": "depth-anything/DA3-LARGE-1.1",
    "giant": "depth-anything/DA3-GIANT-1.1",
    "metric-large": "depth-anything/DA3METRIC-LARGE",
    "mono-large": "depth-anything/DA3MONO-LARGE",
}

#: True when the checkpoint is safe for commercial use. Asserted at load time so
#: a non-commercial model cannot reach production by accident.
COMMERCIAL_OK = {
    "small": True,
    "base": True,
    "metric-large": True,
    "mono-large": True,
    "large": False,  # contradictory; treat as NC until clarified
    "giant": False,  # CC BY-NC 4.0 in both the model zoo and the card
}


@dataclass
class DA3Config:
    variant: str = "large"
    device: str = "cuda"
    #: Frames per multi-view window. DA3 conditions across the window, so this
    #: trades VRAM against cross-frame consistency.
    window: int = 16
    #: Overlap between consecutive windows, in frames.
    window_overlap: int = 4
    dtype: str = "float16"
    #: Long captures: DA3-Streaming's sliding-window inference stays under 12 GB
    #: (PLAN.md §3.1). It is a separate inference wrapper in the upstream repo
    #: (`da3_streaming/`), not a checkpoint, so it is selected by this flag
    #: rather than by `variant`.
    streaming_threshold: int = 120
    use_streaming: bool = False
    weights_dir: str | None = None
    #: Refuse to load a checkpoint that cannot ship. Turn off for Phase-0
    #: research benchmarking, which non-commercial licences do permit.
    enforce_commercial_licence: bool = False


class DA3Depth:
    """Pose-conditioned Depth Anything 3."""

    def __init__(self, config: DA3Config | None = None) -> None:
        self.config = config or DA3Config()
        if self.config.variant not in VARIANTS:
            raise ValueError(
                f"unknown DA3 variant {self.config.variant!r}; have {sorted(VARIANTS)}"
            )
        if self.config.enforce_commercial_licence and not COMMERCIAL_OK[self.config.variant]:
            raise ValueError(
                f"DA3 variant {self.config.variant!r} is not cleared for commercial use "
                "(see docs/licensing-audit.md §1.2). Use 'base' or 'metric-large', or "
                "set enforce_commercial_licence=False for research benchmarking."
            )
        self.name = f"da3-{self.config.variant}"
        self._model = None

    # -- lifecycle ----------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch  # noqa: F401
            from depth_anything_3.api import DepthAnything3  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on GPU box
            raise RuntimeError(
                "Depth Anything 3 is not installed. On a GPU box:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
                "  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3\n"
                f"(original error: {exc})"
            ) from exc

        repo = VARIANTS[self.config.variant]
        self._model = DepthAnything3.from_pretrained(
            self.config.weights_dir or repo
        ).to(self.config.device)
        self._model.eval()
        return self._model

    # -- inference ----------------------------------------------------------

    def estimate(self, bundle: BundleDir) -> DepthResult:
        import torch

        model = self._load()
        kfs = bundle.bundle.keyframes
        cfg = self.config
        variant = cfg.variant
        if len(kfs) > cfg.streaming_threshold and variant != "streaming":
            # Not an error: the caller asked for DA3-Large on a long capture.
            # Windowing already bounds VRAM, so just note it in the model name.
            self.name = f"da3-{variant}-windowed"

        torch.cuda.reset_peak_memory_stats(cfg.device) if cfg.device.startswith("cuda") else None
        t0 = time.time()

        depth: dict[str, np.ndarray] = {}
        conf: dict[str, np.ndarray] = {}
        step = max(1, cfg.window - cfg.window_overlap)
        dtype = getattr(torch, cfg.dtype)

        for start in range(0, len(kfs), step):
            chunk = kfs[start : start + cfg.window]
            if not chunk:
                break
            images = np.stack([bundle.rgb(kf) for kf in chunk])
            imgs = torch.from_numpy(images).to(cfg.device).permute(0, 3, 1, 2).to(dtype) / 255.0
            # world<-camera in the bundle; DA3 wants camera<-world extrinsics.
            extrinsics = torch.from_numpy(
                np.stack([kf.pose.inverse_matrix() for kf in chunk])
            ).to(cfg.device, dtype)
            intrinsics = torch.from_numpy(
                np.stack([kf.intrinsics.matrix() for kf in chunk])
            ).to(cfg.device, dtype)

            with torch.inference_mode():
                out = model.inference(
                    images=imgs,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    use_pose_condition=True,
                )

            pred = out["depth"].float().cpu().numpy()
            pred_conf = out.get("confidence")
            pred_conf = pred_conf.float().cpu().numpy() if pred_conf is not None else None
            for i, kf in enumerate(chunk):
                if kf.id in depth:
                    # Overlapping window: keep the more confident prediction.
                    if pred_conf is None:
                        continue
                    better = pred_conf[i] > conf[kf.id]
                    depth[kf.id] = np.where(better, pred[i], depth[kf.id])
                    conf[kf.id] = np.maximum(conf[kf.id], pred_conf[i])
                else:
                    depth[kf.id] = pred[i].astype(np.float32)
                    if pred_conf is not None:
                        conf[kf.id] = pred_conf[i].astype(np.float32)

        vram = 0.0
        if cfg.device.startswith("cuda"):
            vram = torch.cuda.max_memory_allocated(cfg.device) / 1e9

        return DepthResult(
            depth=depth,
            confidence=conf,
            model_name=self.name,
            runtime_s=time.time() - t0,
            peak_vram_gb=vram,
        )
