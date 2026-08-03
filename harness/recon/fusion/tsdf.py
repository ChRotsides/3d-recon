"""TSDF fusion (PLAN.md §6.4 step 1) — the robustness net under PGSR.

A dense truncated signed distance volume over the room's bounding box, 2 cm
voxels by default. Pure NumPy so it runs anywhere; a 5x3x5 m room at 2 cm is
~9 M voxels, which is a few hundred MB of float32 and a couple of seconds.

Masking is applied *here*, at integration time, not to the output. Masked
pixels never contribute a weight, so a mirror leaves a hole rather than a
phantom wall (PLAN.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recon.geom import TriMesh
from recon.io.bundle import BundleDir
from recon.schemas import Keyframe


@dataclass
class TSDFConfig:
    voxel_size_m: float = 0.02
    #: Truncation distance; 4 voxels is the usual compromise between thin-surface
    #: resolution and noise rejection.
    truncation_voxels: float = 4.0
    #: Ignore returns outside this range — beyond it a phone depth map is noise.
    min_depth_m: float = 0.25
    max_depth_m: float = 8.0
    #: Padding around the observed point cloud when sizing the volume.
    padding_m: float = 0.25
    #: Drop voxels seen by fewer than this many rays before meshing. Weights are
    #: normalised so a return at `reference_depth_m` counts as exactly 1.0, which
    #: makes this threshold mean "seen well by roughly this many frames"
    #: independent of room size.
    min_weight: float = 1.0
    #: Weight returns by (d_ref/d)^2, matching the falling reliability of stereo
    #: and monocular depth with range. Un-normalised 1/d^2 silently starves far
    #: surfaces — in a 4 m room that deletes the floor.
    inverse_square_weighting: bool = True
    reference_depth_m: float = 2.0  # d_opt from PLAN.md §6.1
    #: Never let a very close return dominate.
    max_weight_ratio: float = 4.0
    max_voxels: int = 40_000_000
    #: When the volume would exceed `max_voxels`, coarsen instead of raising.
    #: Unmasked glass inflates the bounding box to the whole neighbourhood, and
    #: a crash there is not a measurement — the control arm has to produce
    #: numbers so the value of masking can be quantified rather than asserted.
    auto_coarsen: bool = True
    max_voxel_size_m: float = 0.25

    @property
    def truncation_m(self) -> float:
        return self.truncation_voxels * self.voxel_size_m


@dataclass
class TSDFVolume:
    origin: np.ndarray  # (3,) world position of voxel (0,0,0) centre
    dims: tuple[int, int, int]
    voxel_size: float
    tsdf: np.ndarray  # (nx, ny, nz) float32 in [-1, 1]
    weight: np.ndarray  # (nx, ny, nz) float32

    def grid_points(self) -> np.ndarray:
        nx, ny, nz = self.dims
        ax = self.origin[0] + np.arange(nx) * self.voxel_size
        ay = self.origin[1] + np.arange(ny) * self.voxel_size
        az = self.origin[2] + np.arange(nz) * self.voxel_size
        gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
        return np.stack([gx, gy, gz], axis=-1)

    def surface_points(self, min_weight: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
        """Zero-crossing voxels as an oriented point cloud (points, normals).

        Cheaper and more robust than meshing when all we need downstream is
        RANSAC plane fitting.
        """
        w = self.weight
        valid = w >= min_weight
        # Gradient of the TSDF is the surface normal; also gives sub-voxel offset.
        gx, gy, gz = np.gradient(self.tsdf)
        grad = np.stack([gx, gy, gz], axis=-1)
        norm = np.linalg.norm(grad, axis=-1)
        near = valid & (np.abs(self.tsdf) < 0.5) & (norm > 1e-6)
        idx = np.argwhere(near)
        if len(idx) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        g = grad[near] / norm[near][:, None]
        d = self.tsdf[near][:, None]
        pts = self.origin + idx * self.voxel_size
        # Newton step to the zero crossing using the *measured* gradient rather
        # than the nominal 1/truncation slope. Assuming the nominal slope biases
        # every surface outward by a few millimetres, which shows up directly as
        # a room that is systematically too large.
        step_voxels = np.clip(d / norm[near][:, None], -2.0, 2.0)
        pts = pts - g * step_voxels * self.voxel_size
        # TSDF is positive on the camera side of the surface, so +grad points
        # into the room. That is the inward normal the layout stage expects.
        return pts, g

    #: Set by `fuse` so surface_points can convert TSDF units back to metres.
    truncation_scale: float = 4.0
    #: True when `fuse` had to enlarge the voxel to fit the observed extent —
    #: a strong signal that unmasked glass inflated the scene.
    coarsened: bool = False

    def extract_mesh(self, min_weight: float = 1.5) -> TriMesh | None:
        """Marching cubes via scikit-image, if available."""
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            return None
        vol = np.where(self.weight >= min_weight, self.tsdf, 1.0).astype(np.float32)
        if not ((vol.min() < 0) and (vol.max() > 0)):
            return None
        verts, faces, normals, _ = marching_cubes(vol, level=0.0)
        verts = self.origin + verts * self.voxel_size
        faces = faces.astype(np.int32)
        return TriMesh(
            vertices=verts,
            faces=faces,
            face_normals=normals[faces[:, 0]],
            face_labels=np.zeros(len(faces), np.int32),
            face_colors=np.tile([0.8, 0.8, 0.8], (len(faces), 1)),
        )


def fuse(
    bundle: BundleDir,
    depths: dict[str, np.ndarray],
    masks: dict[str, np.ndarray] | None = None,
    config: TSDFConfig | None = None,
    *,
    scale: float = 1.0,
) -> TSDFVolume:
    """Integrate masked depth maps into a TSDF volume.

    `scale` multiplies both poses and depths, so the caller can fuse directly in
    corrected metric space once the §6.2 solver has run.
    """
    cfg = config or TSDFConfig()
    kfs = [kf for kf in bundle.bundle.keyframes if kf.id in depths]
    if not kfs:
        raise ValueError("no keyframes with depth to fuse")

    lo, hi = _bounds(bundle, kfs, depths, masks, cfg, scale)
    origin = lo - cfg.padding_m
    extent = (hi + cfg.padding_m) - origin

    voxel = cfg.voxel_size_m
    dims = tuple(int(np.ceil(e / voxel)) + 1 for e in extent)
    n_vox = int(np.prod(dims))
    if n_vox > cfg.max_voxels:
        if not cfg.auto_coarsen:
            raise MemoryError(
                f"TSDF volume would be {n_vox/1e6:.1f} M voxels at {voxel*100:.0f} cm; "
                f"raise voxel_size_m or max_voxels"
            )
        # Solve for the voxel size iteratively: the closed form underestimates
        # because of the +1 per dimension, and landing 0.2% over the cap is
        # still a hard failure.
        while n_vox > cfg.max_voxels and voxel < cfg.max_voxel_size_m:
            target = voxel * (n_vox / cfg.max_voxels) ** (1.0 / 3.0)
            voxel = min(cfg.max_voxel_size_m, max(target * 1.02, voxel * 1.05))
            dims = tuple(int(np.ceil(e / voxel)) + 1 for e in extent)
            n_vox = int(np.prod(dims))
        if n_vox > cfg.max_voxels:
            raise MemoryError(
                f"TSDF volume is {n_vox/1e6:.1f} M voxels even at the coarsest allowed "
                f"{voxel*100:.0f} cm — the observed extent is "
                f"{np.round(extent, 1).tolist()} m, which is not a room."
            )

    tsdf = np.ones(dims, dtype=np.float32)
    weight = np.zeros(dims, dtype=np.float32)

    trunc = cfg.truncation_voxels * voxel
    for kf in kfs:
        d = depths[kf.id]
        if scale != 1.0:
            d = d * scale
        m = masks.get(kf.id) if masks else None
        _integrate(tsdf, weight, origin, dims, kf, d, m, cfg, trunc, scale, voxel)

    vol = TSDFVolume(origin=origin, dims=dims, voxel_size=voxel, tsdf=tsdf, weight=weight)
    vol.truncation_scale = cfg.truncation_voxels
    vol.coarsened = voxel > cfg.voxel_size_m
    return vol


#: Voxels processed per chunk. Bounds peak memory independent of room size.
_CHUNK = 2_000_000


def _integrate(
    tsdf: np.ndarray,
    weight: np.ndarray,
    origin: np.ndarray,
    dims: tuple[int, int, int],
    kf: Keyframe,
    depth: np.ndarray,
    mask: np.ndarray | None,
    cfg: TSDFConfig,
    trunc: float,
    scale: float,
    voxel: float,
) -> None:
    intr = kf.intrinsics
    extr = kf.pose.inverse_matrix()
    if scale != 1.0:
        extr = extr.copy()
        extr[:3, 3] *= scale
    rot = extr[:3, :3].astype(np.float32)
    tr = extr[:3, 3].astype(np.float32)

    nx, ny, nz = dims
    vs = np.float32(voxel)
    org = origin.astype(np.float32)
    flat_t = tsdf.reshape(-1)
    flat_w = weight.reshape(-1)
    n_vox = nx * ny * nz

    # Precompute the y/z sub-grid once; x is walked in slabs so a chunk is a
    # contiguous index range and scatter stays a simple flat assignment.
    yz = np.stack(
        np.meshgrid(np.arange(ny, dtype=np.float32), np.arange(nz, dtype=np.float32),
                    indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    per_slab = ny * nz
    slab_step = max(1, _CHUNK // max(per_slab, 1))

    for x0 in range(0, nx, slab_step):
        x1 = min(nx, x0 + slab_step)
        n_slabs = x1 - x0
        xs = np.repeat(np.arange(x0, x1, dtype=np.float32), per_slab)
        yzs = np.tile(yz, (n_slabs, 1))
        world = np.empty((n_slabs * per_slab, 3), dtype=np.float32)
        world[:, 0] = org[0] + xs * vs
        world[:, 1] = org[1] + yzs[:, 0] * vs
        world[:, 2] = org[2] + yzs[:, 1] * vs
        base = x0 * per_slab

        cam = world @ rot.T + tr
        z = -cam[:, 2]  # camera looks down -Z, so depth is -z_cam
        near = z > cfg.min_depth_m
        if not near.any():
            continue

        loc = np.flatnonzero(near)
        zz = z[loc]
        u = intr.fx * (cam[loc, 0] / zz) + intr.cx
        v = -intr.fy * (cam[loc, 1] / zz) + intr.cy
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        inside = (ui >= 0) & (ui < intr.width) & (vi >= 0) & (vi < intr.height)
        if not inside.any():
            continue
        loc, zz, ui, vi = loc[inside], zz[inside], ui[inside], vi[inside]

        dm = depth[vi, ui]
        ok = (dm > cfg.min_depth_m) & (dm < cfg.max_depth_m)
        if mask is not None:
            ok &= mask[vi, ui]
        if not ok.any():
            continue
        loc, zz, dm = loc[ok], zz[ok], dm[ok]

        sdf = dm - zz
        keep = sdf >= -trunc
        if not keep.any():
            continue
        loc, zz, sdf = loc[keep], zz[keep], sdf[keep]

        idx = base + loc
        assert idx.max() < n_vox
        val = np.clip(sdf / trunc, -1.0, 1.0).astype(np.float32)
        if cfg.inverse_square_weighting:
            w = np.minimum(
                (cfg.reference_depth_m / np.maximum(zz, 0.3)) ** 2, cfg.max_weight_ratio
            ).astype(np.float32)
        else:
            w = np.ones(len(idx), dtype=np.float32)

        w_old = flat_w[idx]
        w_new = w_old + w
        flat_t[idx] = (flat_t[idx] * w_old + val * w) / w_new
        flat_w[idx] = w_new


def _bounds(
    bundle: BundleDir,
    kfs: list[Keyframe],
    depths: dict[str, np.ndarray],
    masks: dict[str, np.ndarray] | None,
    cfg: TSDFConfig,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box of the *masked* observed points.

    Deriving it from masked points is what keeps a phantom room from inflating
    the volume to the whole street.
    """
    from recon.geom import unproject

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for kf in kfs:
        d = depths[kf.id].copy()
        m = masks.get(kf.id) if masks else None
        if m is not None:
            d = np.where(m, d, 0.0)
        d = np.where((d > cfg.min_depth_m) & (d < cfg.max_depth_m), d, 0.0)
        if not (d > 0).any():
            continue
        pts = unproject(
            d * scale,
            kf.intrinsics.matrix(),
            np.asarray(kf.pose.R),
            np.asarray(kf.pose.t) * scale,
        )
        if len(pts) == 0:
            continue
        # Trim the extremes so one stray return can't blow up the volume.
        lo = np.minimum(lo, np.percentile(pts, 0.1, axis=0))
        hi = np.maximum(hi, np.percentile(pts, 99.9, axis=0))
    if not np.isfinite(lo).all():
        raise ValueError("no valid depth returns after masking")
    return lo, hi
