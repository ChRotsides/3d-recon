#!/usr/bin/env python3
"""Render matching orthographic views and report basic mesh statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict[str, float | int]:
    triangles = np.asarray(mesh.triangles)
    clusters, counts, _ = mesh.cluster_connected_triangles()
    return {
        "vertices": len(mesh.vertices),
        "triangles": len(triangles),
        "components": len(counts),
        "surface_area_m2": float(mesh.get_surface_area()),
        "largest_component_fraction": (
            float(np.max(counts) / len(triangles)) if len(counts) and len(triangles) else 0.0
        ),
    }


def sampled_vertices(mesh: o3d.geometry.TriangleMesh, count: int, seed: int):
    vertices = np.asarray(mesh.vertices)
    colours = np.asarray(mesh.vertex_colors)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(vertices), min(count, len(vertices)), replace=False)
    if colours.shape != vertices.shape:
        colours = np.full(vertices.shape, 0.7)
    return vertices[selected], np.clip(colours[selected], 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("refined", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--points", type=int, default=80_000)
    args = parser.parse_args()

    paths = [("Captured poses", args.baseline), ("Refined poses", args.refined)]
    meshes = [(label, o3d.io.read_triangle_mesh(str(path))) for label, path in paths]
    for label, mesh in meshes:
        print(label, mesh_stats(mesh))

    projections = [
        (0, 2, "top: X / Z"),
        (0, 1, "front: X / Y"),
        (2, 1, "side: Z / Y"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor="#111318")
    for row, (label, mesh) in enumerate(meshes):
        vertices, colours = sampled_vertices(mesh, args.points, row)
        for column, (horizontal, vertical, title) in enumerate(projections):
            axis = axes[row, column]
            axis.set_facecolor("#111318")
            axis.scatter(
                vertices[:, horizontal],
                vertices[:, vertical],
                s=0.12,
                c=colours,
                linewidths=0,
                rasterized=True,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.tick_params(colors="#aeb5c0", labelsize=7)
            axis.set_title(f"{label} — {title}", color="white", fontsize=11)
            for spine in axis.spines.values():
                spine.set_color("#3a3f49")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170, facecolor=figure.get_facecolor())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
