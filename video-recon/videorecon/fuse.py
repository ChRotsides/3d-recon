# ruff: noqa: E501
"""Depth maps -> one surface. TSDF fusion with Open3D, then the viewer files.

Outputs, in the layout `viewer/cloud.html` and `viewer/mesh.html` already read:

    mesh.ply             full-resolution fused mesh
    mesh.bin, mesh.json  decimated mesh for the browser
    cloud.bin, cloud.json  coloured point cloud sampled from the TSDF surface
    index.html, mesh.html  the viewers themselves, copied next to the data
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWER_DIR = REPO_ROOT / "viewer"


def _ensure_harness_importable() -> None:
    try:
        import recon  # noqa: F401
    except ImportError:
        harness = REPO_ROOT / "harness"
        if harness.is_dir():
            sys.path.insert(0, str(harness))


@dataclass(frozen=True)
class FusionReport:
    frames_used: int
    voxel_m: float
    sdf_trunc_m: float
    depth_trunc_m: float
    mesh_vertices: int
    mesh_triangles: int
    cloud_points: int
    extent_m: tuple[float, float, float]


def fuse(
    bundle_dir: Path,
    depth_npz: Path,
    out_dir: Path,
    *,
    voxel_m: float = 0.02,
    sdf_trunc_m: float | None = None,
    depth_trunc_m: float = 6.0,
    target_triangles: int = 600_000,
    max_points: int = 900_000,
    min_component_fraction: float = 0.002,
    #: Publish a partial cloud.bin after this many integrated frames so a viewer
    #: can watch the room appear; 0 disables it.
    progress_every: int = 10,
    #: Keep a depth pixel only when this many neighbouring keyframes see a
    #: surface within `consistency_tol` of it. 0 disables the test.
    min_support: int = 1,
    consistency_tol: float = 0.10,
    #: Neighbours checked for support, as keyframe offsets in capture order.
    support_offsets: tuple[int, ...] = (-12, -6, -3, 3, 6, 12),
    #: Cap each frame's depth at this multiple of the farthest sparse point it
    #: observes; None disables the cap.
    sparse_depth_cap: float | None = 1.3,
    #: Drop each frame's least-confident fraction of pixels when the depth
    #: cache carries confidence maps.
    confidence_drop_fraction: float = 0.2,
    log=print,
) -> FusionReport:
    _ensure_harness_importable()
    import open3d as o3d
    from recon.cloud import flying_pixel_mask
    from recon.frames import world_to_camera_cv
    from recon.io.bundle import BundleDir

    if sdf_trunc_m is None:
        # Set from the disagreement the depth actually has, not the voxel size
        # (see scripts/capture/mesh.py for the measurement behind 16 cm).
        sdf_trunc_m = max(4.0 * voxel_m, 0.12)

    bundle = BundleDir.load(bundle_dir)
    cache = np.load(depth_npz)
    model = str(cache["model"]) if "model" in cache else "unknown"
    kfs = [kf for kf in bundle.bundle.keyframes if kf.id in cache]
    depths = {kf.id: np.array(cache[kf.id], dtype=np.float32) for kf in kfs}
    cams = {kf.id: _camera(kf, depths[kf.id].shape) for kf in kfs}
    caps = _sparse_depth_caps(bundle, cams, sparse_depth_cap)
    for kf in kfs:
        d = depths[kf.id]
        if f"conf_{kf.id}" in cache and confidence_drop_fraction > 0:
            conf = np.asarray(cache[f"conf_{kf.id}"], dtype=np.float32)
            if conf.shape == d.shape:
                d[conf < np.quantile(conf[d > 0], confidence_drop_fraction)] = 0.0
        cap = caps.get(kf.id)
        if cap is not None:
            d[d > cap] = 0.0
    dropped_support = 0
    if min_support > 0:
        for i, kf in enumerate(kfs):
            d = depths[kf.id]
            neighbours = [kfs[i + o] for o in support_offsets if 0 <= i + o < len(kfs)]
            if not neighbours:
                continue
            support = _support_count(
                d, cams[kf.id], [(depths[n.id], cams[n.id]) for n in neighbours], consistency_tol
            )
            unsupported = (d > 0) & (support < min_support)
            dropped_support += int(unsupported.sum())
            d[unsupported] = 0.0
        log(
            f"fuse: consistency test dropped {dropped_support:,} pixels "
            f"({100 * dropped_support / max(1, sum(int((v > 0).sum()) for v in depths.values()) + dropped_support):.1f}%)"
        )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_m,
        sdf_trunc=sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_viewers(out_dir)
    used = 0
    flying_total = 0
    for kf in kfs:
        depth = depths[kf.id]
        flying = flying_pixel_mask(depth)
        depth[flying] = 0.0
        flying_total += int(flying.sum())
        depth = np.ascontiguousarray(depth)
        rgb = bundle.rgb(kf)
        ih, iw = depth.shape
        if rgb.shape[:2] != (ih, iw):
            ys = (np.arange(ih) * rgb.shape[0] / ih).astype(np.int32)
            xs = (np.arange(iw) * rgb.shape[1] / iw).astype(np.int32)
            rgb = rgb[ys][:, xs]
        intr = kf.intrinsics
        sx, sy = iw / intr.width, ih / intr.height
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            iw, ih, intr.fx * sx, intr.fy * sy, intr.cx * sx, intr.cy * sy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb, dtype=np.uint8)),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=depth_trunc_m,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, world_to_camera_cv(kf))
        used += 1
        if progress_every and used % progress_every == 0:
            partial = volume.extract_point_cloud()
            _publish_cloud(
                out_dir / "cloud", partial, voxel_m, max_points, model, used, partial=True
            )
    if used == 0:
        raise RuntimeError("no keyframe had a depth map to fuse")
    log(
        f"fuse: integrated {used} frames at {voxel_m * 100:.1f} cm, "
        f"trunc {sdf_trunc_m * 100:.0f} cm, "
        f"{flying_total:,} flying pixels removed"
    )

    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts):
        keep = counts >= max(1, int(min_component_fraction * counts.max()))
        mesh.remove_triangles_by_mask(~keep[labels])
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    log(f"fuse: mesh {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    out_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(
        str(out_dir / "mesh.ply"), mesh, write_vertex_normals=True, write_vertex_colors=True
    )

    web = mesh
    if target_triangles and len(mesh.triangles) > target_triangles:
        web = mesh.simplify_quadric_decimation(target_triangles)
        web.remove_unreferenced_vertices()
        web.compute_vertex_normals()
    _write_mesh_bin(out_dir / "mesh", web, mesh, voxel_m, depth_trunc_m, model, used)

    cloud = volume.extract_point_cloud()
    pts = _publish_cloud(out_dir / "cloud", cloud, voxel_m, max_points, model, used, partial=False)

    v = np.asarray(mesh.vertices)
    extent = (v.max(axis=0) - v.min(axis=0)) if len(v) else np.zeros(3)
    log(f"fuse: cloud {len(pts):,} points; extent {np.round(extent, 2).tolist()} m")
    return FusionReport(
        used,
        voxel_m,
        sdf_trunc_m,
        depth_trunc_m,
        len(mesh.vertices),
        len(mesh.triangles),
        len(pts),
        tuple(float(e) for e in extent),
    )


class _Cam:
    __slots__ = ("k", "w2c", "c2w")

    def __init__(self, k: np.ndarray, w2c: np.ndarray) -> None:
        self.k = k
        self.w2c = w2c
        self.c2w = np.linalg.inv(w2c)


def _camera(kf, shape: tuple[int, int]) -> _Cam:
    from recon.frames import world_to_camera_cv

    h, w = shape
    intr = kf.intrinsics
    sx, sy = w / intr.width, h / intr.height
    k = np.array([[intr.fx * sx, 0, intr.cx * sx], [0, intr.fy * sy, intr.cy * sy], [0, 0, 1.0]])
    return _Cam(k, world_to_camera_cv(kf))


def _sparse_depth_caps(bundle, cams: dict[str, _Cam], factor: float | None) -> dict[str, float]:
    """Farthest sparse point each frame observes, times `factor`.

    Depth beyond anything the reconstruction itself triangulated in that view
    is the model guessing through a window or off a screen.
    """
    if factor is None or not bundle.bundle.sparse_points:
        return {}
    per: dict[str, list[float]] = {}
    for p in bundle.bundle.sparse_points:
        xyz = np.asarray(p.xyz)
        for fid in p.observations:
            cam = cams.get(fid)
            if cam is None:
                continue
            z = cam.w2c[2, :3] @ xyz + cam.w2c[2, 3]
            if z > 0:
                per.setdefault(fid, []).append(float(z))
    return {fid: factor * float(np.percentile(zs, 98)) for fid, zs in per.items() if len(zs) >= 20}


def _support_count(
    depth: np.ndarray, cam: _Cam, others: list[tuple[np.ndarray, _Cam]], tol: float
) -> np.ndarray:
    """How many other views see a surface within `tol` of each pixel's point."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth
    ok = z > 0
    x = (u - cam.k[0, 2]) / cam.k[0, 0] * z
    y = (v - cam.k[1, 2]) / cam.k[1, 1] * z
    pw = np.stack([x, y, z], axis=-1) @ cam.c2w[:3, :3].T + cam.c2w[:3, 3]
    support = np.zeros((h, w), dtype=np.int16)
    for od, oc in others:
        pc = pw @ oc.w2c[:3, :3].T + oc.w2c[:3, 3]
        zo = pc[..., 2]
        vis = ok & (zo > 0.05)
        uo = np.where(vis, oc.k[0, 0] * pc[..., 0] / np.where(vis, zo, 1) + oc.k[0, 2], -1)
        vo = np.where(vis, oc.k[1, 1] * pc[..., 1] / np.where(vis, zo, 1) + oc.k[1, 2], -1)
        oh, ow = od.shape
        inside = vis & (uo >= 0) & (uo < ow - 1) & (vo >= 0) & (vo < oh - 1)
        ui = np.clip(uo.astype(np.int32), 0, ow - 1)
        vi = np.clip(vo.astype(np.int32), 0, oh - 1)
        seen = od[vi, ui]
        agree = inside & (seen > 0) & (np.abs(zo - seen) <= tol * np.maximum(seen, 1e-3))
        support += agree.astype(np.int16)
    return support


def _copy_viewers(out_dir: Path) -> None:
    for src, dst in (("cloud.html", "index.html"), ("mesh.html", "mesh.html")):
        if (VIEWER_DIR / src).exists() and not (out_dir / dst).exists():
            shutil.copyfile(VIEWER_DIR / src, out_dir / dst)


def _publish_cloud(
    base: Path, cloud, voxel_m: float, max_points: int, model: str, used: int, *, partial: bool
):
    pts = np.asarray(cloud.points, dtype=np.float32)
    cols = (
        np.clip(np.asarray(cloud.colors) * 255.0, 0, 255).astype(np.uint8)
        if len(cloud.colors)
        else np.full((len(pts), 3), 200, np.uint8)
    )
    cloud_voxel = voxel_m
    while len(pts) > max_points:
        cloud_voxel *= 1.25
        down = cloud.voxel_down_sample(cloud_voxel)
        pts = np.asarray(down.points, dtype=np.float32)
        cols = np.clip(np.asarray(down.colors) * 255.0, 0, 255).astype(np.uint8)
    _write_cloud_bin(base, pts, cols, cloud_voxel, model, used, partial=partial)
    return pts


def _write_mesh_bin(
    base: Path, web, full, voxel_m: float, depth_trunc_m: float, model: str, used: int
) -> None:
    v = np.asarray(web.vertices, dtype=np.float32)
    n = np.asarray(web.vertex_normals, dtype=np.float32)
    c = np.clip(np.asarray(web.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
    if len(c) != len(v):
        c = np.full((len(v), 3), 200, dtype=np.uint8)
    tri = np.asarray(web.triangles, dtype=np.uint32)
    # positions f32[3V] · normals f32[3V] · indices u32[3T] · colours u8[3V]
    with open(base.with_suffix(".bin"), "wb") as f:
        f.write(v.tobytes())
        f.write(n.tobytes())
        f.write(tri.tobytes())
        f.write(c.tobytes())
    lo, hi = (v.min(axis=0), v.max(axis=0)) if len(v) else (np.zeros(3), np.zeros(3))
    meta = {
        "vertices": int(len(v)),
        "triangles": int(len(tri)),
        "full_vertices": len(full.vertices),
        "full_triangles": len(full.triangles),
        "min": lo.tolist(),
        "max": hi.tolist(),
        "centre": ((lo + hi) / 2).tolist(),
        "extent": (hi - lo).tolist(),
        "voxel_m": voxel_m,
        "depth_trunc_m": depth_trunc_m,
        "model": model,
        "keyframes": used,
    }
    base.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def _write_cloud_bin(
    base: Path,
    pos: np.ndarray,
    col: np.ndarray,
    voxel_m: float,
    model: str,
    used: int,
    *,
    partial: bool = False,
) -> None:
    # Write next to the target and rename, so a viewer never reads a half file.
    tmp = base.with_suffix(".bin.tmp")
    with open(tmp, "wb") as f:
        f.write(pos.astype(np.float32).tobytes())
        f.write(col.astype(np.uint8).tobytes())
    tmp.replace(base.with_suffix(".bin"))
    if len(pos):
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        view_lo, view_hi = np.percentile(pos, [1.0, 99.0], axis=0)
    else:
        lo = hi = view_lo = view_hi = np.zeros(3)
    meta = {
        "count": int(len(pos)),
        "min": lo.tolist(),
        "max": hi.tolist(),
        "centre": ((lo + hi) / 2).tolist(),
        "extent": (hi - lo).tolist(),
        "view_min": view_lo.tolist(),
        "view_max": view_hi.tolist(),
        "view_centre": ((view_lo + view_hi) / 2).tolist(),
        "view_extent": (view_hi - view_lo).tolist(),
        "model": model,
        "keyframes": used,
        "voxel_m": float(voxel_m),
        "units": "m",
        "partial": partial,
    }
    tmp = base.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(base.with_suffix(".json"))
