"""MapAnything adapter (PLAN.md §3.2).

Kept in the stack for two things DA3 does not give us:

1. A **metric scale token** — an independent estimate of global scale, which
   enters the §6.2 solver as one more constraint alongside VIO and the tape
   measure. Its sigma is marked TBD in the plan; `estimate_scale_only()` exists
   so Phase 0 can measure it before anyone trusts it.
2. **Registration** — feeding two rooms' partial reconstructions in and getting
   a relative pose out, used as the ICP initialisation in §6.3 step 5.

Not runnable on the Kepler dev box; torch import is deferred.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from recon.depth.base import DepthResult
from recon.io.bundle import BundleDir

#: The default *must* be the Apache variant. `facebook/map-anything` is
#: CC BY-NC 4.0 — the plan's "Apache-licensed model" is only true of this one.
#: See docs/licensing-audit.md §1.3.
APACHE_CHECKPOINT = "facebook/map-anything-apache"
NONCOMMERCIAL_CHECKPOINT = "facebook/map-anything"


@dataclass
class MapAnythingConfig:
    checkpoint: str = APACHE_CHECKPOINT
    device: str = "cuda"
    window: int = 24
    dtype: str = "bfloat16"
    #: Feed poses/intrinsics as optional geometric inputs rather than solving them.
    condition_on_poses: bool = True
    weights_dir: str | None = None


class MapAnythingDepth:
    name = "mapanything"

    def __init__(self, config: MapAnythingConfig | None = None) -> None:
        self.config = config or MapAnythingConfig()
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch  # noqa: F401
            from mapanything.models import MapAnything  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on GPU box
            raise RuntimeError(
                "MapAnything is not installed. On a GPU box:\n"
                "  pip install git+https://github.com/facebookresearch/map-anything\n"
                f"(original error: {exc})"
            ) from exc
        self._model = MapAnything.from_pretrained(
            self.config.weights_dir or self.config.checkpoint
        ).to(self.config.device)
        self._model.eval()
        return self._model

    def _views(self, bundle: BundleDir, kfs) -> list[dict]:
        import torch

        cfg = self.config
        dtype = getattr(torch, cfg.dtype)
        views = []
        for kf in kfs:
            img = bundle.rgb(kf)
            v = {
                "img": torch.from_numpy(img).permute(2, 0, 1).to(cfg.device, dtype) / 255.0,
            }
            if cfg.condition_on_poses:
                v["intrinsics"] = torch.from_numpy(kf.intrinsics.matrix()).to(cfg.device, dtype)
                v["camera_poses"] = torch.from_numpy(kf.pose.matrix()).to(cfg.device, dtype)
            views.append(v)
        return views

    def estimate(self, bundle: BundleDir) -> DepthResult:
        import torch

        model = self._load()
        cfg = self.config
        kfs = bundle.bundle.keyframes
        if cfg.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(cfg.device)
        t0 = time.time()

        depth: dict[str, np.ndarray] = {}
        conf: dict[str, np.ndarray] = {}
        scales: list[float] = []

        for start in range(0, len(kfs), cfg.window):
            chunk = kfs[start : start + cfg.window]
            with torch.inference_mode():
                preds = model.infer(self._views(bundle, chunk), memory_efficient_inference=True)
            for kf, p in zip(chunk, preds, strict=True):
                depth[kf.id] = p["depth_along_ray"].squeeze().float().cpu().numpy().astype(
                    np.float32
                )
                if "conf" in p:
                    conf[kf.id] = p["conf"].squeeze().float().cpu().numpy().astype(np.float32)
                if "metric_scaling_factor" in p:
                    scales.append(float(p["metric_scaling_factor"].float().mean().cpu()))

        vram = (
            torch.cuda.max_memory_allocated(cfg.device) / 1e9
            if cfg.device.startswith("cuda")
            else 0.0
        )
        scale = float(np.median(scales)) if scales else None
        # Spread of the per-window scale tokens is the only honest sigma we have
        # until Phase 0 measures the real thing against the laser set.
        sigma = float(np.std(scales)) if len(scales) > 2 else None

        return DepthResult(
            depth=depth,
            confidence=conf,
            scale_estimate=scale,
            scale_sigma=sigma,
            model_name=self.name,
            runtime_s=time.time() - t0,
            peak_vram_gb=vram,
        )

    # -- §6.3 step 5: registration -----------------------------------------

    def register(self, bundle_a: BundleDir, bundle_b: BundleDir, max_views: int = 12):
        """Estimate the rigid transform taking room B into room A's frame.

        Returns a 4x4 world_a<-world_b matrix, for use as the ICP initialisation
        in PLAN.md §6.3. Sub-sampled to `max_views` per room to bound VRAM.
        """
        import torch

        model = self._load()

        def pick(b: BundleDir):
            kfs = b.bundle.keyframes
            idx = np.linspace(0, len(kfs) - 1, min(max_views, len(kfs))).astype(int)
            return [kfs[i] for i in idx]

        kfs_a, kfs_b = pick(bundle_a), pick(bundle_b)
        views = self._views(bundle_a, kfs_a) + self._views(bundle_b, kfs_b)
        # Registration means: do NOT tell it where room B's cameras are.
        for v in views[len(kfs_a) :]:
            v.pop("camera_poses", None)

        with torch.inference_mode():
            preds = model.infer(views, memory_efficient_inference=True)

        pred_b = np.stack(
            [p["camera_poses"].squeeze().float().cpu().numpy() for p in preds[len(kfs_a) :]]
        )
        # Least-squares similarity between MapAnything's estimate of B's cameras
        # (already in A's frame) and B's own reported poses.
        src = np.stack([kf.pose.matrix() for kf in kfs_b])
        return _umeyama(src[:, :3, 3], pred_b[:, :3, 3])


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid (scale-free) Umeyama fit: returns 4x4 taking src points to dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    u, _, vt = np.linalg.svd(cov)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    r = u @ d @ vt
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = mu_d - r @ mu_s
    return m
