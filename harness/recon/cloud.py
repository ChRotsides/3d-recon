"""Turn per-frame depth into a point cloud that is a surface rather than a spray.

Two operations, both aimed at the same complaint — that the cloud looks full of
holes.

**The holes were mostly self-inflicted.** The builder used to keep a uniform
random sample of each frame, `max_points / frames` pixels out of 307,200, and
random sampling of a surface leaves gaps at every scale: on
bundle_e4a5da81, 890,991 random points filled **19%** of the 1 cm cells the full
8.2M points fill. Voxelising instead of sampling keeps one point per occupied
cell, which is the same budget spent on covering the surface once rather than on
covering a fraction of it repeatedly. At 3 cm it is a *smaller* file than the
random sample it replaces, and it has no holes the data did not have.

**Flying pixels are the opposite failure** — surface that is not there. At a
depth discontinuity a pixel straddles both sides and lands somewhere between,
strung along the view ray as a comet tail behind every object edge. They are
easy to find precisely because they are the pixels whose depth disagrees sharply
with their neighbours'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A pixel whose depth differs from a 4-neighbour by more than this *fraction*
#: of its own depth is on a discontinuity. Relative rather than absolute because
#: depth noise and quantisation both grow with range: 5 cm at 1 m is a real edge,
#: 5 cm at 8 m is the sensor's last digit.
FLYING_PIXEL_REL = 0.05


@dataclass(frozen=True)
class VoxelCleanupStats:
    """Counts that make point removal visible to the caller and the manifest."""

    input_points: int
    finite_points: int
    frame_voxel_samples: int
    occupied_voxels: int
    kept_voxels: int
    dropped_isolated: int
    dropped_spread: int
    low_support_kept_by_neighbours: int
    min_frame_support: int
    min_neighbours: int
    max_spread_m: float | None


def flying_pixel_mask(depth: np.ndarray, *, rel_threshold: float = FLYING_PIXEL_REL) -> np.ndarray:
    """True where a pixel sits on a depth discontinuity and should be dropped.

    Compares each pixel with its four neighbours and flags it when the largest
    relative difference exceeds `rel_threshold`. Both sides of an edge are
    flagged, which is deliberate: the interpolated pixel could be on either, and
    a one-pixel-thick border of a real surface is a much cheaper loss than a
    tail of invented geometry stretching metres into the room.
    """
    if depth.ndim != 2:
        raise ValueError(f"expected a 2-D depth map, got shape {depth.shape}")

    d = depth.astype(np.float32)
    valid = d > 0
    worst = np.zeros_like(d)

    for axis in (0, 1):
        diff = np.abs(np.diff(d, axis=axis))
        both = np.take(valid, np.arange(d.shape[axis] - 1), axis=axis) & np.take(
            valid, np.arange(1, d.shape[axis]), axis=axis
        )
        diff = np.where(both, diff, 0.0)
        # A difference belongs to the pixels on both sides of it.
        pad_lo = [(0, 0), (0, 0)]
        pad_hi = [(0, 0), (0, 0)]
        pad_lo[axis] = (0, 1)
        pad_hi[axis] = (1, 0)
        worst = np.maximum(worst, np.pad(diff, pad_lo))
        worst = np.maximum(worst, np.pad(diff, pad_hi))

    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(valid, worst / np.maximum(d, 1e-6), 0.0)
    return (relative > rel_threshold) & valid


def voxel_downsample(
    points: np.ndarray, colours: np.ndarray, voxel_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """One point per occupied cell, at the mean position and colour of its members.

    Averaging rather than picking a representative: several frames hitting the
    same patch of wall disagree by their own noise, and the mean of those is a
    better estimate of the wall than any one of them. It is also what makes this
    a *denoising* step and not only a thinning one.
    """
    if voxel_m <= 0:
        raise ValueError(f"voxel size must be positive, got {voxel_m}")
    if len(points) != len(colours):
        raise ValueError(f"{len(points)} points but {len(colours)} colours")
    if len(points) == 0:
        return points.astype(np.float32), colours.astype(np.uint8)

    p = points.astype(np.float64)
    grid = np.floor((p - p.min(axis=0)) / voxel_m).astype(np.int64)
    # Exact linear index rather than a hash: a collision here would merge two
    # unrelated patches of the room into one averaged point, which is invisible
    # in the output and impossible to explain later.
    dims = grid.max(axis=0) + 1
    key = (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]

    _, inverse = np.unique(key, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    out_p = np.stack(
        [np.bincount(inverse, weights=p[:, i]) / counts for i in range(3)], axis=1
    )
    out_c = np.stack(
        [
            np.bincount(inverse, weights=colours[:, i].astype(np.float64)) / counts
            for i in range(3)
        ],
        axis=1,
    )
    return out_p.astype(np.float32), np.clip(np.rint(out_c), 0, 255).astype(np.uint8)


def _weighted_group_mean(
    values: np.ndarray, groups: np.ndarray, weights: np.ndarray, n_groups: int
) -> np.ndarray:
    totals = np.bincount(groups, weights=weights, minlength=n_groups)
    return np.stack(
        [
            np.bincount(groups, weights=weights * values[:, axis], minlength=n_groups)
            / totals
            for axis in range(values.shape[1])
        ],
        axis=1,
    )


def _occupied_neighbour_counts(
    voxel_keys: np.ndarray,
    voxel_grid: np.ndarray,
    dims: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Count occupied cells in each queried cell's 26-neighbourhood."""
    counts = np.zeros(len(query), dtype=np.uint8)
    if len(query) == 0:
        return counts

    base = voxel_grid[query]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                neighbour = base + np.array([dx, dy, dz], dtype=np.int64)
                valid = np.all((neighbour >= 0) & (neighbour < dims), axis=1)
                valid_at = np.flatnonzero(valid)
                if len(valid_at) == 0:
                    continue
                cell = neighbour[valid_at]
                key = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
                found_at = np.searchsorted(voxel_keys, key)
                in_range = found_at < len(voxel_keys)
                matched = np.zeros(len(key), dtype=bool)
                matched[in_range] = voxel_keys[found_at[in_range]] == key[in_range]
                counts[valid_at[matched]] += 1
    return counts


def frame_aware_voxel_filter(
    points: np.ndarray,
    colours: np.ndarray,
    frame_ids: np.ndarray,
    voxel_m: float,
    *,
    min_frame_support: int = 2,
    min_neighbours: int = 3,
    huber_delta_m: float | None = None,
    max_spread_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, VoxelCleanupStats]:
    """Robustly fuse frames and remove only unsupported, isolated voxels.

    Points are first averaged within each (frame, voxel) pair. Those per-frame
    samples are then fused with a Huber mean, so a frame gets one vote however
    many pixels it contributed. A voxel is retained when either enough distinct
    frames support it or it has enough occupied neighbours to be part of a
    coherent single-view surface. Thin isolated tails therefore disappear while
    a wall seen from only one angle survives.

    ``max_spread_m`` applies to the RMS distance between per-frame samples and
    their robust centre. Leave it as ``None`` to report dispersion without
    filtering it; the cloud builder uses a conservative fraction of a voxel.
    """
    points = np.asarray(points)
    colours = np.asarray(colours)
    frame_ids = np.asarray(frame_ids)
    if voxel_m <= 0:
        raise ValueError(f"voxel size must be positive, got {voxel_m}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected points shaped (N, 3), got {points.shape}")
    if colours.ndim != 2 or colours.shape[1] != 3:
        raise ValueError(f"expected colours shaped (N, 3), got {colours.shape}")
    if len(points) != len(colours) or len(points) != len(frame_ids):
        raise ValueError(
            f"{len(points)} points, {len(colours)} colours, {len(frame_ids)} frame ids"
        )
    if frame_ids.ndim != 1 or not np.issubdtype(frame_ids.dtype, np.integer):
        raise ValueError("frame_ids must be a one-dimensional integer array")
    if min_frame_support < 1:
        raise ValueError("min_frame_support must be at least one")
    if min_neighbours < 0 or min_neighbours > 26:
        raise ValueError("min_neighbours must be between zero and 26")
    if huber_delta_m is not None and huber_delta_m <= 0:
        raise ValueError("huber_delta_m must be positive")
    if max_spread_m is not None and max_spread_m <= 0:
        raise ValueError("max_spread_m must be positive")

    input_points = len(points)
    finite = np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(colours), axis=1)
    p = points[finite].astype(np.float64)
    c = colours[finite]
    frames = frame_ids[finite]
    if len(p) == 0:
        stats = VoxelCleanupStats(
            input_points=input_points,
            finite_points=0,
            frame_voxel_samples=0,
            occupied_voxels=0,
            kept_voxels=0,
            dropped_isolated=0,
            dropped_spread=0,
            low_support_kept_by_neighbours=0,
            min_frame_support=min_frame_support,
            min_neighbours=min_neighbours,
            max_spread_m=max_spread_m,
        )
        return p.astype(np.float32), c.astype(np.uint8), stats

    origin = p.min(axis=0)
    grid = np.floor((p - origin) / voxel_m).astype(np.int64)
    dims = grid.max(axis=0) + 1
    point_keys = (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]
    voxel_keys, first_point, point_voxel = np.unique(
        point_keys, return_index=True, return_inverse=True
    )
    n_voxels = len(voxel_keys)
    voxel_grid = grid[first_point]
    del first_point, grid

    # Make frame ids dense so the composite key is compact and collision-free.
    _, dense_frame = np.unique(frames, return_inverse=True)
    del frames
    n_frames = int(dense_frame.max()) + 1
    frame_voxel_key = point_voxel * n_frames + dense_frame
    frame_voxel_keys, point_frame_voxel = np.unique(
        frame_voxel_key, return_inverse=True
    )
    del dense_frame, frame_voxel_key, point_voxel, point_keys
    n_frame_voxels = len(frame_voxel_keys)
    pair_counts = np.bincount(point_frame_voxel, minlength=n_frame_voxels).astype(np.float64)
    pair_points = np.stack(
        [
            np.bincount(point_frame_voxel, weights=p[:, axis], minlength=n_frame_voxels)
            / pair_counts
            for axis in range(3)
        ],
        axis=1,
    )
    pair_colours = np.stack(
        [
            np.bincount(point_frame_voxel, weights=c[:, axis], minlength=n_frame_voxels)
            / pair_counts
            for axis in range(3)
        ],
        axis=1,
    )
    del c, p, pair_counts, point_frame_voxel
    pair_voxel = frame_voxel_keys // n_frames
    support = np.bincount(pair_voxel, minlength=n_voxels)

    # Two Huber reweighting rounds are enough here: all samples are already in
    # the same voxel, and this only has to stop one frame pulling the centre to
    # its edge rather than solve an unconstrained registration problem.
    weights = np.ones(n_frame_voxels, dtype=np.float64)
    centre = _weighted_group_mean(pair_points, pair_voxel, weights, n_voxels)
    delta = huber_delta_m if huber_delta_m is not None else 0.5 * voxel_m
    for _ in range(2):
        residual = np.linalg.norm(pair_points - centre[pair_voxel], axis=1)
        weights = np.minimum(1.0, delta / np.maximum(residual, 1e-12))
        centre = _weighted_group_mean(pair_points, pair_voxel, weights, n_voxels)
    out_colours = _weighted_group_mean(pair_colours, pair_voxel, weights, n_voxels)

    residual = np.linalg.norm(pair_points - centre[pair_voxel], axis=1)
    spread = np.sqrt(
        np.bincount(pair_voxel, weights=residual**2, minlength=n_voxels) / support
    )
    supported = support >= min_frame_support
    unsupported_at = np.flatnonzero(~supported)
    neighbour_count = _occupied_neighbour_counts(
        voxel_keys, voxel_grid, dims, unsupported_at
    )
    coherent = neighbour_count >= min_neighbours
    support_keep = supported.copy()
    support_keep[unsupported_at] = coherent
    spread_keep = np.ones(n_voxels, dtype=bool)
    if max_spread_m is not None:
        spread_keep = spread <= max_spread_m
    keep = support_keep & spread_keep

    dropped_isolated = int((~support_keep).sum())
    dropped_spread = int((support_keep & ~spread_keep).sum())
    stats = VoxelCleanupStats(
        input_points=input_points,
        finite_points=int(finite.sum()),
        frame_voxel_samples=n_frame_voxels,
        occupied_voxels=n_voxels,
        kept_voxels=int(keep.sum()),
        dropped_isolated=dropped_isolated,
        dropped_spread=dropped_spread,
        low_support_kept_by_neighbours=int(coherent.sum()),
        min_frame_support=min_frame_support,
        min_neighbours=min_neighbours,
        max_spread_m=max_spread_m,
    )
    return (
        centre[keep].astype(np.float32),
        np.clip(np.rint(out_colours[keep]), 0, 255).astype(np.uint8),
        stats,
    )


def choose_voxel_size(
    points: np.ndarray, max_points: int, *, start_m: float = 0.01, limit_m: float = 0.5
) -> float:
    """Smallest voxel from `start_m` upwards whose cell count fits `max_points`.

    The caller asks for a point budget, not a resolution, because the budget is
    what a browser has to download. Doubling rather than solving: occupancy is
    not a smooth function of cell size, and four cheap counting passes beat a
    model of a surface nobody has measured.
    """
    if len(points) == 0 or max_points <= 0:
        return start_m

    p = points.astype(np.float64)
    origin = p.min(axis=0)
    size = start_m
    while size <= limit_m:
        grid = np.floor((p - origin) / size).astype(np.int64)
        dims = grid.max(axis=0) + 1
        key = (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]
        if len(np.unique(key)) <= max_points:
            return size
        size *= 2.0
    return limit_m
