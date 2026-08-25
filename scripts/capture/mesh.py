"""Fuse the posed depth maps into a surface, clean it, and write a mesh.

**TSDF fusion rather than meshing the point cloud.** The cloud is 8.2M points
with no notion of which side of a surface is air, so anything meshing it has to
guess that back from normals it also has to guess. The depth maps still have
what the cloud threw away: each pixel is a *ray*, and everything along it in
front of the hit is known-empty. Fusing in that space averages 27 disagreeing
views into one surface instead of stacking 27 noisy shells, and marching cubes
comes out watertight where the room was seen and open where it was not.

    python scripts/capture/mesh.py BUNDLE DEPTH.npz OUT.ply [voxel_m] [depth_trunc_m]

Needs Open3D, which lives in the CPU environment; `da3_depth.py` produces the
input from the GPU one.
"""

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from recon.cloud import flying_pixel_mask
from recon.frames import world_to_camera_cv
from recon.io.bundle import BundleDir

BUNDLE = Path(sys.argv[1])
DEPTH = Path(sys.argv[2])
OUT = Path(sys.argv[3])
VOXEL_M = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
#: Beyond this, depth is not evidence. 99% of the measured returns on this
#: capture are inside 5.92 m, and what lies past it is one doorway seen through
#: at an angle no other frame corroborates — fusing it builds a wing onto the
#: room out of a single view.
DEPTH_TRUNC_M = float(sys.argv[5]) if len(sys.argv) > 5 else 7.0
#: Triangles in the copy the browser downloads. 0 keeps the full mesh.
TARGET_TRIANGLES = int(sys.argv[6]) if len(sys.argv) > 6 else 600_000
#: How far either side of a surface a depth reading votes.
#:
#: The textbook default is 4x the voxel, and at 2 cm voxels that is 8 cm — which
#: is *half* the disagreement the depth actually has. Measured after the affine
#: anchor, frames still differ from the sensor by a median 0.156 m, so two views
#: of the same wall land outside each other's band entirely, and TSDF does what
#: it is asked: it builds two surfaces. That is the layered-slab look, and no
#: amount of cleaning fixes it downstream. Truncation has to be set from the
#: error the data actually has, not from the resolution you wish it had.
SDF_TRUNC_M = float(sys.argv[7]) if len(sys.argv) > 7 else max(4.0 * VOXEL_M, 0.16)
#: Connected components smaller than this fraction of the largest are debris:
#: flying-pixel shells, a chair leg seen once, the far side of a doorway.
MIN_COMPONENT_FRACTION = 0.002

bundle = BundleDir.load(BUNDLE)
kfs = bundle.bundle.keyframes
cache = np.load(DEPTH)
print(f"{len(kfs)} keyframes · depth from {DEPTH.name} · model {cache['model']}", flush=True)

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL_M,
    sdf_trunc=SDF_TRUNC_M,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
)

used = 0
flying_total = 0
for kf in kfs:
    if kf.id not in cache:
        continue
    depth = np.array(cache[kf.id], dtype=np.float32)
    # The same removal the cloud builder does, and for a sharper reason here:
    # a flying pixel is a lone reading out in free space, and TSDF renders it
    # as a cone stretched along the view ray back towards the camera. The cache
    # holds what the model produced; this is a fusion-time decision.
    flying = flying_pixel_mask(depth)
    depth[flying] = 0.0
    flying_total += int(flying.sum())
    depth = np.ascontiguousarray(depth)
    rgb = bundle.rgb(kf)
    ih, iw = depth.shape
    if rgb.shape[:2] != (ih, iw):
        # Same rule as the cloud builder: sample the image onto the depth grid
        # rather than resizing depth across its discontinuities.
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
        depth_scale=1.0,  # already metres
        depth_trunc=DEPTH_TRUNC_M,
        convert_rgb_to_intensity=False,
    )
    # Open3D wants world-to-camera in OpenCV axes, which is what the bundle's
    # OpenGL-convention pose is *not*. See recon.frames.
    volume.integrate(rgbd, intrinsic, world_to_camera_cv(kf))
    used += 1

print(f"integrated {used} frames · voxel {VOXEL_M * 100:.1f} cm · sdf_trunc "
      f"{SDF_TRUNC_M * 100:.0f} cm · depth trunc {DEPTH_TRUNC_M:.1f} m · "
      f"{flying_total:,} flying pixels removed", flush=True)

mesh = volume.extract_triangle_mesh()
print(f"{len(mesh.vertices):,} vertices · {len(mesh.triangles):,} triangles raw", flush=True)

mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_non_manifold_edges()

# Drop the debris. Marching cubes over a hand-held capture always leaves
# islands — a shell around a flying pixel, a sliver of something seen once —
# and they are visually loud out of all proportion to their size.
labels, counts, _ = mesh.cluster_connected_triangles()
labels = np.asarray(labels)
counts = np.asarray(counts)
if len(counts):
    keep = counts >= max(1, int(MIN_COMPONENT_FRACTION * counts.max()))
    mesh.remove_triangles_by_mask(~keep[labels])
    mesh.remove_unreferenced_vertices()
    print(f"kept {int(keep.sum())} of {len(counts)} connected components", flush=True)

mesh.compute_vertex_normals()
print(f"{len(mesh.vertices):,} vertices · {len(mesh.triangles):,} triangles cleaned", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
o3d.io.write_triangle_mesh(str(OUT), mesh, write_vertex_normals=True, write_vertex_colors=True)
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, full resolution)", flush=True)

# The full mesh is the archive copy. The browser gets a decimated one, because
# a 97 MB download is not a viewer — and quadric decimation on a fused surface
# is nearly free visually: it collapses the dense flat regions first, which is
# most of a room.
web = mesh
if TARGET_TRIANGLES and len(mesh.triangles) > TARGET_TRIANGLES:
    web = mesh.simplify_quadric_decimation(TARGET_TRIANGLES)
    web.remove_unreferenced_vertices()
    web.compute_vertex_normals()
    print(f"{len(web.vertices):,} vertices · {len(web.triangles):,} triangles for the web",
          flush=True)

v = np.asarray(web.vertices, dtype=np.float32)
n = np.asarray(web.vertex_normals, dtype=np.float32)
c = np.clip(np.asarray(web.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
if len(c) != len(v):  # a mesh with no colour still has to render
    c = np.full((len(v), 3), 200, dtype=np.uint8)
tri = np.asarray(web.triangles, dtype=np.uint32)

# Same idea as cloud.bin: one flat buffer straight into GPU buffers, no parsing.
# positions f32[3V] · normals f32[3V] · indices u32[3T] · colours u8[3V]
#
# Colours go last precisely because they are the only 1-byte field: with them in
# the middle, `3V` bytes of colour leaves the index array at an offset that is
# not a multiple of 4, and a Uint32Array cannot be created on a misaligned
# offset — the reader would have to copy the whole index block to look at it.
with open(OUT.with_suffix(".bin"), "wb") as f:
    f.write(v.tobytes())
    f.write(n.tobytes())
    f.write(tri.tobytes())
    f.write(c.tobytes())

lo, hi = v.min(axis=0), v.max(axis=0)
meta = {
    "vertices": int(len(v)),
    "triangles": int(len(tri)),
    "full_vertices": len(mesh.vertices),
    "full_triangles": len(mesh.triangles),
    "min": lo.tolist(),
    "max": hi.tolist(),
    "centre": ((lo + hi) / 2).tolist(),
    "extent": (hi - lo).tolist(),
    "voxel_m": VOXEL_M,
    "depth_trunc_m": DEPTH_TRUNC_M,
    "model": str(cache["model"]),
    "keyframes": used,
}
OUT.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT.with_suffix('.bin')} "
      f"({OUT.with_suffix('.bin').stat().st_size / 1e6:.1f} MB, for the viewer)", flush=True)
print("extent m:", np.round(hi - lo, 2).tolist(), flush=True)
