"""The reconstruction pipeline — steps 1-7 of PLAN.md §2.

    1. Masking      SAM 3 (mirror/window/TV/person)
    2. Geometry     Depth Anything 3, pose-conditioned
    3. Pose refine  pose graph + scale constraints
    4. Render       PGSR via gsplat            [optional, GPU]
    5. Metric layer TSDF fusion -> Atlanta planes
    6. Layout       wall/floor/ceiling -> 2D floor plan
    7. Package      SPZ + GLB + SVG + layout JSON

Each step is a separate method with its own artifact, so the Phase-2 Celery
worker can retry any one of them independently (PLAN.md §4/Phase 2) and the
Phase-0 harness can run subsets.

Step order note: masking runs *before* fusion, never after — that ordering is
the phantom-room fix and it is asserted in the tests.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from recon.depth.base import DepthEstimator, DepthResult, GroundTruthDepth
from recon.fusion.tsdf import TSDFConfig, TSDFVolume, fuse
from recon.io.bundle import BundleDir
from recon.layout.floorplan import FloorplanConfig, build_layout, to_svg
from recon.layout.planes import PlaneConfig, PlaneSet, extract_planes
from recon.masking.base import Masker, MaskResult, NullMasker
from recon.metric.scale import ScaleSolution, build_constraints, solve_scale
from recon.schemas import RoomLayout


@dataclass
class PipelineConfig:
    tsdf: TSDFConfig = field(default_factory=TSDFConfig)
    planes: PlaneConfig = field(default_factory=PlaneConfig)
    floorplan: FloorplanConfig = field(default_factory=FloorplanConfig)
    #: Apply the §6.2 scale correction before fusing, so all downstream metres
    #: are corrected metres.
    apply_scale: bool = True
    #: Train a splat model. Off by default: it is the only GPU-bound step.
    train_splats: bool = False
    write_artifacts: bool = True


@dataclass
class PipelineResult:
    room_id: str
    layout: RoomLayout
    scale: ScaleSolution
    planes: PlaneSet
    volume: TSDFVolume
    depth: DepthResult
    masks: MaskResult
    timings_s: dict[str, float] = field(default_factory=dict)
    splat: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "depth_model": self.depth.model_name,
            "mask_model": self.masks.model_name,
            "scale": self.scale.scale,
            "scale_sigma": self.scale.sigma,
            "masked_fraction": self.masks.masked_fraction(),
            "area_m2": self.layout.area_m2,
            "perimeter_m": self.layout.perimeter_m,
            "ceiling_height_m": self.layout.ceiling_height_m,
            "n_walls": len(self.layout.wall_lengths_m),
            "wall_lengths_m": self.layout.wall_lengths_m,
            "n_openings": len(self.layout.openings),
            "timings_s": self.timings_s,
            "total_s": sum(self.timings_s.values()),
        }


class Pipeline:
    def __init__(
        self,
        depth: DepthEstimator | None = None,
        masker: Masker | None = None,
        config: PipelineConfig | None = None,
        splat_trainer=None,
    ) -> None:
        self.depth = depth or GroundTruthDepth()
        self.masker = masker or NullMasker()
        self.config = config or PipelineConfig()
        self.splat_trainer = splat_trainer

    def run(self, bundle: BundleDir, out_dir: Path | None = None) -> PipelineResult:
        cfg = self.config
        timings: dict[str, float] = {}

        # 1. Masking — before anything consumes depth.
        t = time.time()
        masks = self.masker.mask(bundle)
        timings["mask"] = time.time() - t

        # 2. Geometry.
        t = time.time()
        depth = self.depth.estimate(bundle)
        timings["depth"] = time.time() - t

        # 3. Scale. (Full pose-graph refinement is Phase 4/5; the scale variable
        #    is solved here and handed to the house graph as a per-room prior.)
        t = time.time()
        constraints = build_constraints(
            bundle.bundle.reference_measurements,
            mapanything_scale=depth.scale_estimate,
            mapanything_sigma=depth.scale_sigma,
        )
        scale = solve_scale(constraints)
        timings["scale"] = time.time() - t
        s = scale.scale if cfg.apply_scale else 1.0

        # 5. Metric layer.
        t = time.time()
        volume = fuse(bundle, depth.depth, masks.fusion_mask, cfg.tsdf, scale=s)
        points, normals = volume.surface_points(min_weight=cfg.tsdf.min_weight)
        timings["fuse"] = time.time() - t

        t = time.time()
        planes = extract_planes(points, normals, bundle.bundle.up, cfg.planes)
        timings["planes"] = time.time() - t

        # 6. Layout.
        t = time.time()
        # Camera positions pick this room out of everything visible through its
        # doorways (see FloorplanConfig.door_sever_radius_m). They are in the
        # corrected metric frame, so they must carry the same scale as fusion.
        cams = np.array([kf.pose.position for kf in bundle.bundle.keyframes]) * s
        layout = build_layout(
            planes, bundle.bundle.room_id, bundle.bundle.up, cfg.floorplan,
            scale_applied=s, camera_positions=cams,
        )
        timings["layout"] = time.time() - t

        # 4. Render + surface (GPU).
        splat = None
        if cfg.train_splats and self.splat_trainer is not None:
            t = time.time()
            splat = self.splat_trainer.train(bundle, depth, masks, scale=s)
            timings["splat"] = time.time() - t

        result = PipelineResult(
            room_id=bundle.bundle.room_id,
            layout=layout,
            scale=scale,
            planes=planes,
            volume=volume,
            depth=depth,
            masks=masks,
            timings_s=timings,
            splat=splat,
        )

        # 7. Package.
        if out_dir is not None and cfg.write_artifacts:
            self.write_artifacts(result, Path(out_dir))
        return result

    @staticmethod
    def write_artifacts(result: PipelineResult, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "layout.json").write_text(result.layout.model_dump_json(indent=2))
        (out_dir / "floorplan.svg").write_text(to_svg(result.layout))
        (out_dir / "summary.json").write_text(json.dumps(result.summary(), indent=2))
        (out_dir / "scale_report.txt").write_text(result.scale.report())

        pts, nrm = result.volume.surface_points(min_weight=result.volume.weight.max() * 0.0 + 1.5)
        _write_point_ply(out_dir / "fused_points.ply", pts, nrm)

        mesh = result.volume.extract_mesh()
        if mesh is not None:
            _write_mesh_ply(out_dir / "fused_mesh.ply", mesh)


def _write_point_ply(path: Path, pts: np.ndarray, nrm: np.ndarray) -> None:
    lines = [
        "ply\nformat ascii 1.0\n",
        f"element vertex {len(pts)}\n",
        "property float x\nproperty float y\nproperty float z\n",
        "property float nx\nproperty float ny\nproperty float nz\n",
        "end_header\n",
    ]
    lines += [
        f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n"
        for p, n in zip(pts, nrm, strict=True)
    ]
    path.write_text("".join(lines))


def _write_mesh_ply(path: Path, mesh) -> None:
    v, f = mesh.vertices, mesh.faces
    lines = [
        "ply\nformat ascii 1.0\n",
        f"element vertex {len(v)}\n",
        "property float x\nproperty float y\nproperty float z\n",
        f"element face {len(f)}\n",
        "property list uchar int vertex_indices\n",
        "end_header\n",
    ]
    lines += [f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n" for p in v]
    lines += [f"3 {t[0]} {t[1]} {t[2]}\n" for t in f]
    path.write_text("".join(lines))
