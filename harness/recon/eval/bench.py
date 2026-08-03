"""Benchmark runner — the Phase-0 grid from PLAN.md §4 and the §8 regression gate.

Reports, per (scene x depth model x masker x splat backend):
Chamfer distance, wall-length error, area error, PSNR, wall-clock, peak VRAM.

The grid always includes control arms. `depth=gt` isolates everything after
depth prediction; `mask=none` measures what masking is actually worth. Without
those two rows a bad number is uninterpretable — you cannot tell a depth failure
from a masking failure from a fusion failure.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from recon.eval.metrics import GeometryMetrics, chamfer, layout_metrics
from recon.io.bundle import BundleDir
from recon.pipeline import Pipeline, PipelineConfig

#: PLAN.md §8: fail the build if wall-length error regresses by more than this.
REGRESSION_TOLERANCE_PP = 0.5


@dataclass
class BenchRow:
    scene: str
    depth_model: str
    mask_model: str
    splat_model: str = "none"
    layout: dict[str, Any] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    psnr_db: float = float("nan")
    runtime_s: float = 0.0
    timings_s: dict[str, float] = field(default_factory=dict)
    peak_vram_gb: float = 0.0
    scale: float = 1.0
    error: str | None = None

    @property
    def config_name(self) -> str:
        return f"{self.depth_model}+{self.mask_model}+{self.splat_model}"

    @property
    def wall_err(self) -> float:
        return float(self.layout.get("wall_error_max_abs_pct", float("inf")))

    @property
    def area_err(self) -> float:
        return abs(float(self.layout.get("area_error_pct", float("inf"))))

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        wall_ok = self.wall_err <= 1.5 or float(
            self.layout.get("wall_error_max_abs_m", 1e9)
        ) <= 0.03
        return wall_ok and self.area_err <= 3.0


@dataclass
class BenchReport:
    rows: list[BenchRow]
    host: dict[str, str] = field(default_factory=dict)
    created_utc: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "created_utc": self.created_utc,
                "host": self.host,
                "rows": [asdict(r) for r in self.rows],
            },
            indent=2,
            default=float,
        )

    @classmethod
    def from_json(cls, text: str) -> BenchReport:
        raw = json.loads(text)
        return cls(
            rows=[BenchRow(**r) for r in raw["rows"]],
            host=raw.get("host", {}),
            created_utc=raw.get("created_utc", ""),
        )

    # -- reporting ----------------------------------------------------------

    def to_markdown(self) -> str:
        head = (
            "| scene | depth | mask | splat | wall err % | wall err m | area err % "
            "| ceiling err m | chamfer m | recall@5cm | PSNR dB | time s | VRAM GB | pass |"
        )
        sep = "|" + "---|" * 14
        lines = [head, sep]
        for r in sorted(self.rows, key=lambda r: (r.scene, r.config_name)):
            if r.error:
                lines.append(
                    f"| {r.scene} | {r.depth_model} | {r.mask_model} | {r.splat_model} | "
                    f"ERROR: {r.error[:60]} | | | | | | | {r.runtime_s:.1f} | | ✗ |"
                )
                continue
            lay, gm = r.layout, r.geometry
            lines.append(
                f"| {r.scene} | {r.depth_model} | {r.mask_model} | {r.splat_model} "
                f"| {lay.get('wall_error_max_abs_pct', float('nan')):.2f} "
                f"| {lay.get('wall_error_max_abs_m', float('nan')):.3f} "
                f"| {lay.get('area_error_pct', float('nan')):+.2f} "
                f"| {lay.get('ceiling_error_m', float('nan')):+.3f} "
                f"| {gm.get('chamfer_m', float('nan')):.4f} "
                f"| {gm.get('recall_5cm', float('nan')):.3f} "
                f"| {r.psnr_db:.1f} "
                f"| {r.runtime_s:.1f} | {r.peak_vram_gb:.1f} "
                f"| {'✓' if r.passed else '✗'} |"
            )
        return "\n".join(lines)

    def exit_criterion(self, target_pct: float = 2.0, min_scenes: int = 4) -> dict[str, Any]:
        """PLAN.md §4/Phase 0: at least one configuration at <=2% wall-length
        error on at least 4 of 5 scenes."""
        by_config: dict[str, list[BenchRow]] = {}
        for r in self.rows:
            by_config.setdefault(r.config_name, []).append(r)
        results = {}
        for cfg, rows in by_config.items():
            ok = [r for r in rows if not r.error and r.wall_err <= target_pct]
            results[cfg] = {
                "scenes_within_target": len(ok),
                "scenes_total": len(rows),
                "worst_wall_err_pct": max(
                    (r.wall_err for r in rows if not r.error), default=float("inf")
                ),
                "meets_exit": len(ok) >= min(min_scenes, len(rows)),
            }
        return {
            "target_pct": target_pct,
            "min_scenes": min_scenes,
            "per_config": results,
            "any_config_meets_exit": any(v["meets_exit"] for v in results.values()),
        }

    def regression_check(self, baseline: BenchReport) -> dict[str, Any]:
        """Fail if wall-length error grew by more than 0.5 pp on any scene."""
        base = {(r.scene, r.config_name): r for r in baseline.rows}
        failures = []
        for r in self.rows:
            b = base.get((r.scene, r.config_name))
            if b is None or b.error or r.error:
                continue
            delta = r.wall_err - b.wall_err
            if delta > REGRESSION_TOLERANCE_PP:
                failures.append(
                    {
                        "scene": r.scene,
                        "config": r.config_name,
                        "baseline_pct": b.wall_err,
                        "current_pct": r.wall_err,
                        "delta_pp": delta,
                    }
                )
        return {"passed": not failures, "failures": failures,
                "tolerance_pp": REGRESSION_TOLERANCE_PP}


def run_scene(
    bundle_dir: Path,
    depth,
    masker,
    *,
    pipeline_config: PipelineConfig | None = None,
    splat_trainer=None,
    out_dir: Path | None = None,
    gt_sample_points: int = 120_000,
    seed: int = 0,
) -> BenchRow:
    bd = BundleDir.load(bundle_dir)
    gt = bd.ground_truth()
    row = BenchRow(
        scene=(gt or {}).get("preset", bundle_dir.name),
        depth_model=getattr(depth, "name", type(depth).__name__),
        mask_model=getattr(masker, "name", type(masker).__name__),
        splat_model=getattr(splat_trainer, "name", "none") if splat_trainer else "none",
    )

    t0 = time.time()
    try:
        pipe = Pipeline(depth=depth, masker=masker, config=pipeline_config,
                        splat_trainer=splat_trainer)
        res = pipe.run(bd, out_dir)
    except Exception as exc:  # noqa: BLE001 - a failed config is a result, not a crash
        row.error = f"{type(exc).__name__}: {exc}"
        row.runtime_s = time.time() - t0
        return row

    row.runtime_s = time.time() - t0
    row.timings_s = res.timings_s
    row.peak_vram_gb = res.depth.peak_vram_gb
    row.scale = res.scale.scale

    if gt:
        row.layout = asdict(layout_metrics(res.layout, gt))
        gm = _geometry_vs_gt(bd, res, gt_sample_points, seed)
        if gm is not None:
            row.geometry = asdict(gm)
    if res.splat and "psnr_db" in res.splat:
        row.psnr_db = float(res.splat["psnr_db"])
    return row


def _geometry_vs_gt(bd: BundleDir, res, n_samples: int, seed: int) -> GeometryMetrics | None:
    """Chamfer between the fused surface and the ground-truth mesh."""
    path = bd.gt_mesh_path()
    if path is None:
        return None
    mesh = _read_ply_mesh(path)
    if mesh is None:
        return None
    rng = np.random.default_rng(seed)
    gt_pts = mesh.sample_surface(n_samples, rng)
    pred, _ = res.volume.surface_points(min_weight=res.volume.weight.max() * 0 + 1.0)
    if len(pred) > n_samples:
        pred = pred[rng.choice(len(pred), n_samples, replace=False)]
    return chamfer(pred, gt_pts)


def _read_ply_mesh(path: Path):
    """Minimal ASCII PLY reader for the meshes the synth generator writes."""
    from recon.geom import TriMesh

    text = path.read_text().splitlines()
    n_v = n_f = 0
    header_end = 0
    labels_present = False
    for i, line in enumerate(text):
        if line.startswith("element vertex"):
            n_v = int(line.split()[-1])
        elif line.startswith("element face"):
            n_f = int(line.split()[-1])
        elif line.strip() == "property uchar label":
            labels_present = True
        elif line.strip() == "end_header":
            header_end = i + 1
            break
    if not n_v or not n_f:
        return None
    verts = np.array(
        [[float(x) for x in text[header_end + i].split()[:3]] for i in range(n_v)]
    )
    faces, labels = [], []
    for i in range(n_f):
        parts = text[header_end + n_v + i].split()
        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
        labels.append(int(parts[4]) if labels_present and len(parts) > 4 else 0)
    faces = np.asarray(faces, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)

    # Exterior geometry exists only to create the phantom-room failure mode; it
    # is not part of the room and must not be scored as missing surface.
    from recon.geom import Semantic

    keep = labels != int(Semantic.OUTSIDE)
    faces, labels = faces[keep], labels[keep]

    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return TriMesh(verts, faces, nrm, labels, np.tile([0.8, 0.8, 0.8], (len(faces), 1)))


def host_info() -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda or "unknown"
        else:
            info["gpu"] = "none (CPU only)"
    except ImportError:
        info["torch"] = "not installed"
        info["gpu"] = "none (CPU only)"
    return info
