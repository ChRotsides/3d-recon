"""Splat trainer protocol and the Phase-0 evaluation split.

Plan §1.3 measures novel-view PSNR on *held-out* keyframes. That split has to be
made here, once, and used by every backend — a backend evaluated on frames it
trained on will report 35 dB and mean nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from recon.depth.base import DepthResult
from recon.io.bundle import BundleDir
from recon.masking.base import MaskResult
from recon.schemas import Keyframe


def holdout_split(
    keyframes: list[Keyframe], every: int = 8
) -> tuple[list[Keyframe], list[Keyframe]]:
    """Every Nth keyframe is held out.

    Strided rather than random: keyframes arrive along a trajectory, so a random
    split leaves each held-out frame sandwiched between two near-identical
    training frames and the "novel" view is not novel at all. A stride puts real
    distance between train and test.
    """
    train, test = [], []
    for i, kf in enumerate(keyframes):
        (test if (i % every == every // 2) else train).append(kf)
    return train, test


@dataclass
class SplatResult:
    ply_path: str | None = None
    spz_path: str | None = None
    n_gaussians: int = 0
    psnr_db: float = float("nan")
    ssim: float = float("nan")
    lpips: float = float("nan")
    n_holdout_views: int = 0
    train_seconds: float = 0.0
    peak_vram_gb: float = 0.0
    model_name: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "n_gaussians": self.n_gaussians,
            "psnr_db": self.psnr_db,
            "ssim": self.ssim,
            "lpips": self.lpips,
            "n_holdout_views": self.n_holdout_views,
            "train_seconds": self.train_seconds,
            "peak_vram_gb": self.peak_vram_gb,
            "ply": self.ply_path,
            "spz": self.spz_path,
            **self.extra,
        }


@runtime_checkable
class SplatTrainer(Protocol):
    name: str

    def train(
        self,
        bundle: BundleDir,
        depth: DepthResult,
        masks: MaskResult,
        *,
        scale: float = 1.0,
    ) -> dict[str, Any]:
        ...


def seed_points_from_depth(
    bundle: BundleDir,
    depth: DepthResult,
    masks: MaskResult,
    *,
    scale: float = 1.0,
    max_points: int = 300_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Initial Gaussian positions and colours, from the masked depth maps.

    Initialising from DA3 depth instead of a random cloud or COLMAP's sparse
    points is the main reason our per-room training budget is minutes rather
    than hours: the Gaussians start on the surface.
    """
    from recon.geom import unproject

    rng = np.random.default_rng(seed)
    pts_all, col_all = [], []
    for kf in bundle.bundle.keyframes:
        d = depth.depth.get(kf.id)
        if d is None:
            continue
        d = d.astype(np.float64)
        m = masks.fusion_mask.get(kf.id)
        if m is not None:
            d = np.where(m, d, 0.0)
        if scale != 1.0:
            d = d * scale
        if not (d > 0).any():
            continue
        rgb = bundle.rgb(kf)
        pts = unproject(
            d,
            kf.intrinsics.matrix(),
            np.asarray(kf.pose.R),
            np.asarray(kf.pose.t) * scale,
        )
        cols = rgb.reshape(-1, 3)[d.reshape(-1) > 0] / 255.0
        pts_all.append(pts)
        col_all.append(cols)

    if not pts_all:
        return np.zeros((0, 3)), np.zeros((0, 3))
    pts = np.vstack(pts_all)
    cols = np.vstack(col_all)
    if len(pts) > max_points:
        idx = rng.choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols
