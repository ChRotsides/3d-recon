"""Planar-based Gaussian splatting on gsplat (plan §3.3, §3.4).

**Licensing note, and it changes the plan.** The audit
(docs/licensing-audit.md §1.1) found that PGSR, 2DGS *and* GOF are all
research-only:

* PGSR — ZJU 3DV licence: "educational, research and non-profit purposes only",
  and any modification must be open-sourced.
* 2DGS and GOF — the Inria/MPII Gaussian-Splatting licence: research only.

The plan treats a clean-room PGSR reimplementation as a contingency. It is the
only commercial path, because both named fallbacks are encumbered too. gsplat
(Apache 2.0) is the sole licence-clean rasteriser.

So this module is written from the *paper*, not from the reference repository:
planar-aligned Gaussians carrying a normal and a signed distance, unbiased depth
rendering, and multi-view geometric regularisation, all on top of gsplat's
public API. The reference implementation must not be consulted while writing it.

`ReferencePGSRTrainer` shells out to the upstream repository instead, for
Phase-0 benchmarking only — research use, which the licence permits — so the
clean-room version has a quality bar to hit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from recon.depth.base import DepthResult
from recon.io.bundle import BundleDir
from recon.masking.base import MaskResult
from recon.splat.base import SplatResult, holdout_split, seed_points_from_depth


@dataclass
class PGSRConfig:
    iterations: int = 7_000
    device: str = "cuda"
    #: Plan §9 budgets ~500k Gaussians per room.
    max_gaussians: int = 600_000
    sh_degree: int = 3
    #: Weight on the planar/normal-consistency term. This is what buys geometry
    #: good enough to measure, which vanilla 3DGS does not have.
    lambda_normal: float = 0.05
    #: Weight on multi-view geometric consistency.
    lambda_multiview: float = 0.10
    #: Weight on the DA3 depth prior. Non-zero because we *have* metric depth;
    #: a from-scratch splat run would leave this at 0.
    lambda_depth: float = 0.20
    #: Masked pixels stay in the photometric loss and are down-weighted in the
    #: geometric one — a mirror should still look like a mirror (plan §3.5).
    masked_geometric_weight: float = 0.05
    holdout_every: int = 8
    output_dir: str = "runs/splat"
    write_spz: bool = True


class PGSRTrainer:
    """Clean-room planar Gaussian splatting on gsplat.

    Not implemented yet — deliberately. Writing it needs a GPU to iterate
    against, and this box has Kepler cards PyTorch no longer supports
    (docs/execution-status.md). Shipping a plausible-looking but never-executed
    training loop would be worse than shipping this: the loss weights above are
    guesses until something has actually converged, and a benchmark row filled
    in by untested code is a lie.

    What *is* settled and captured here: the licence position, the config
    surface, the held-out evaluation split, and the depth-seeded initialisation
    in `recon.splat.base` — all of which are testable now and are what the
    training loop will be built on.
    """

    name = "pgsr"

    def __init__(self, config: PGSRConfig | None = None) -> None:
        self.config = config or PGSRConfig()

    def train(
        self,
        bundle: BundleDir,
        depth: DepthResult,
        masks: MaskResult,
        *,
        scale: float = 1.0,
    ) -> dict[str, Any]:
        try:
            import gsplat  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "gsplat and torch are required for splat training. On a GPU box:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
                "  pip install gsplat\n"
                f"(original error: {exc})"
            ) from exc

        raise NotImplementedError(
            "Clean-room PGSR-on-gsplat is committed Phase-2 work, not written yet. "
            "See docs/licensing-audit.md §1.1 for why it cannot simply vendor the "
            "reference implementation, and docs/execution-status.md for why it was "
            "not written blind. For Phase-0 benchmarking use ReferencePGSRTrainer, "
            "which is research use and licence-permitted."
        )

    # -- what the trainer will be built on, testable today ------------------

    def prepare(
        self,
        bundle: BundleDir,
        depth: DepthResult,
        masks: MaskResult,
        *,
        scale: float = 1.0,
    ) -> dict[str, Any]:
        """Everything up to the first optimiser step.

        Split, seed cloud and per-pixel geometric weights. Exercised by the
        tests so the parts that do not need a GPU are known-good before one is
        rented.
        """
        cfg = self.config
        train_kfs, test_kfs = holdout_split(bundle.bundle.keyframes, cfg.holdout_every)
        points, colours = seed_points_from_depth(
            bundle, depth, masks, scale=scale, max_points=cfg.max_gaussians
        )
        weights = {
            kf.id: masks.geometric_weight.get(
                kf.id,
                np.where(
                    masks.fusion_mask.get(
                        kf.id, np.ones((kf.intrinsics.height, kf.intrinsics.width), bool)
                    ),
                    1.0,
                    cfg.masked_geometric_weight,
                ).astype(np.float32),
            )
            for kf in train_kfs
        }
        return {
            "train_keyframes": [kf.id for kf in train_kfs],
            "holdout_keyframes": [kf.id for kf in test_kfs],
            "seed_points": points,
            "seed_colours": colours,
            "geometric_weight": weights,
        }


@dataclass
class ReferenceConfig:
    """Shell out to an upstream research implementation for Phase-0 comparison."""

    repo_dir: str
    #: "pgsr" | "2dgs" | "gof"
    backend: str = "pgsr"
    iterations: int = 7_000
    python: str = "python"
    extra_args: list[str] = field(default_factory=list)
    timeout_s: int = 3_600


class ReferenceSplatTrainer:
    """Phase-0 benchmark harness around a research splat implementation.

    **Research use only.** PGSR, 2DGS and GOF are all non-commercial
    (docs/licensing-audit.md §1.1). This exists to produce the plan §4 benchmark
    row and to give the clean-room implementation a quality target. Its output
    must never reach a customer artifact.
    """

    def __init__(self, config: ReferenceConfig) -> None:
        self.config = config
        self.name = f"ref-{config.backend}"

    def train(
        self,
        bundle: BundleDir,
        depth: DepthResult,
        masks: MaskResult,
        *,
        scale: float = 1.0,
    ) -> dict[str, Any]:
        import subprocess

        repo = Path(self.config.repo_dir)
        if not repo.exists():
            raise RuntimeError(
                f"{self.config.backend} checkout not found at {repo}. Clone it for "
                "Phase-0 benchmarking only; see docs/licensing-audit.md §1.1."
            )
        work = Path(self.config.output_dir) if hasattr(self.config, "output_dir") else Path(
            "runs/splat"
        ) / self.config.backend
        work.mkdir(parents=True, exist_ok=True)

        colmap_dir = work / "colmap"
        self._write_colmap(bundle, colmap_dir, scale=scale)

        t0 = time.time()
        cmd = [
            self.config.python, str(repo / "train.py"),
            "-s", str(colmap_dir),
            "-m", str(work / "model"),
            "--iterations", str(self.config.iterations),
            *self.config.extra_args,
        ]
        proc = subprocess.run(
            cmd, cwd=repo, capture_output=True, text=True, timeout=self.config.timeout_s
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.config.backend} training failed ({proc.returncode}):\n"
                f"{proc.stderr[-2000:]}"
            )

        result = SplatResult(
            model_name=self.name,
            train_seconds=time.time() - t0,
            ply_path=str(work / "model" / "point_cloud"),
        )
        return result.as_dict()

    def _write_colmap(self, bundle: BundleDir, out: Path, *, scale: float) -> None:
        """Export the bundle as a COLMAP text model.

        Every research splat implementation reads COLMAP, so this is the cheapest
        adapter between our bundle format and all three of them. Poses are
        inverted to camera<-world and quaternions written w-first, which is
        COLMAP's convention and a common source of silently rotated scenes.
        """
        out.mkdir(parents=True, exist_ok=True)
        sparse = out / "sparse" / "0"
        sparse.mkdir(parents=True, exist_ok=True)

        cams, imgs = [], []
        for i, kf in enumerate(bundle.bundle.keyframes, start=1):
            k = kf.intrinsics
            cams.append(
                f"{i} PINHOLE {k.width} {k.height} {k.fx} {k.fy} {k.cx} {k.cy}"
            )
            extr = kf.pose.inverse_matrix()
            r = extr[:3, :3]
            t = extr[:3, 3] * scale
            q = _mat_to_quat_wxyz(r)
            imgs.append(
                f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i} "
                f"{Path(kf.image).name}"
            )

        (sparse / "cameras.txt").write_text(
            "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n" + "\n".join(cams) + "\n"
        )
        (sparse / "images.txt").write_text(
            "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
            + "\n\n".join(imgs)
            + "\n\n"
        )
        (sparse / "points3D.txt").write_text("# empty; splats are seeded from depth\n")


def _mat_to_quat_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> (w, x, y, z). Shepperd's method, for numerical safety
    when the trace is near -1."""
    m = np.asarray(r, dtype=np.float64)
    tr = m.trace()
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return float(w), float(x), float(y), float(z)
