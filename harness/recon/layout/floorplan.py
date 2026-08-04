"""Floor-plan extraction (PLAN.md §6.4 steps 5-6).

Naively cycling the fitted wall lines and intersecting them pairwise breaks on
any room that is not convex, and rooms very often are not (L-shapes, alcoves).
So the footprint is recovered in three stages:

1. **Occupancy** — rasterise observed floor points into a 2D grid, close and
   fill, keep the largest connected component. This gets the *topology* right
   for any shape.
2. **Contour + simplify** — trace the component boundary and reduce it with
   Douglas-Peucker. This gets the *corner count* right.
3. **Snap** — replace each simplified edge with the fitted wall line it matches
   and re-derive corners by intersecting consecutive lines. This gets the
   *metric* right, because the final numbers come from planes fitted to
   thousands of points rather than from a 5 cm raster.

Wall lengths and area therefore inherit plane-fit accuracy, not grid accuracy,
which is what makes the PLAN.md §1.3 targets (<=1.5% wall length, <=3% area)
reachable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recon import geom
from recon.layout.planes import PlaneSet, _basis
from recon.schemas import Opening, RoomLayout


@dataclass
class FloorplanConfig:
    grid_m: float = 0.05
    #: Morphological closing radius, in cells, to bridge gaps under furniture.
    close_cells: int = 3
    #: A plane is only used as a room barrier if its inliers reach within this
    #: of the ceiling, and carry at least this share of the best-supported
    #: wall's inliers. See `_wall_barrier`.
    barrier_ceiling_gap_m: float = 0.35
    barrier_min_support: float = 0.12
    #: Radius of the opening used to sever doorway necks, in metres.
    #:
    #: A capture sees straight through an open doorway into the next room, so
    #: the observed floor is not one room but a chain of them joined by
    #: door-width necks. Taking the largest connected component then returns
    #: *the whole house* as one footprint. A morphological opening with a radius
    #: above half a door width cuts every neck narrower than a door while
    #: leaving room-sized areas intact; the component containing the camera
    #: track is then this room. Corner rounding from the opening does not reach
    #: the output, because edges are snapped back to fitted wall planes.
    door_sever_radius_m: float = 0.45
    #: Douglas-Peucker tolerance.
    simplify_m: float = 0.08
    #: An edge snaps to a wall line within this angle and distance.
    snap_angle_deg: float = 18.0
    snap_dist_m: float = 0.30
    #: Drop polygon edges shorter than this after snapping (raster artefacts).
    min_edge_m: float = 0.25
    #: Opening detection.
    opening_bin_m: float = 0.05
    opening_min_width_m: float = 0.40
    opening_min_height_m: float = 0.40
    #: A hole whose bottom is below this is door-shaped, above it window-shaped.
    door_sill_max_m: float = 0.30
    #: Minimum plausible heights per class. Without these, the gap left where a
    #: grazing view missed the wall-floor junction reads as a door: it starts at
    #: the floor, it is a metre wide, and it is surrounded by wall. The only
    #: thing separating it from a real doorway is that it is 50 cm tall.
    #: Interior doors cluster near 2.03 m (PLAN.md §6.2), so 1.5 m is a generous
    #: floor that still rejects every data gap seen so far.
    door_min_height_m: float = 1.50
    window_min_height_m: float = 0.40
    #: A door must reach close to the floor; a gap starting 40 cm up that is
    #: door-height is more likely two stacked data gaps than a doorway.
    door_max_sill_m: float = 0.30


def build_layout(
    planes: PlaneSet,
    room_id: str,
    up: np.ndarray,
    config: FloorplanConfig | None = None,
    *,
    scale_applied: float = 1.0,
    camera_positions: np.ndarray | None = None,
) -> RoomLayout:
    cfg = config or FloorplanConfig()
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    e0, e1 = _basis(up)

    pts2d = _floor_footprint_points(planes, up, e0, e1)
    cams2d = None
    if camera_positions is not None and len(camera_positions):
        c = np.asarray(camera_positions, dtype=np.float64)
        cams2d = np.column_stack([c @ e0, c @ e1])
    poly = _extract_polygon(pts2d, planes, up, e0, e1, cfg, cams2d)

    floor_h = planes.floor_height if planes.floor else 0.0
    ceil_h = planes.ceiling_height if planes.ceiling else 0.0
    openings = detect_openings(planes, poly, up, e0, e1, floor_h, ceil_h, cfg)

    lengths = geom.edge_lengths(poly) if len(poly) >= 3 else np.zeros(0)
    return RoomLayout(
        room_id=room_id,
        footprint=poly.tolist(),
        floor_height_m=float(floor_h),
        ceiling_height_m=float(ceil_h),
        area_m2=float(abs(geom.polygon_area(poly))) if len(poly) >= 3 else 0.0,
        perimeter_m=float(geom.polygon_perimeter(poly)) if len(poly) >= 3 else 0.0,
        wall_lengths_m=lengths.tolist(),
        planes=planes.all_planes(),
        openings=openings,
        scale_applied=scale_applied,
    )


def _floor_footprint_points(
    planes: PlaneSet, up: np.ndarray, e0: np.ndarray, e1: np.ndarray
) -> np.ndarray:
    """2D points defining the occupied floor area.

    Wall inliers are included alongside floor points: furniture hides large
    parts of the floor, but the walls that bound it are almost always visible,
    and their footprint traces the same boundary.
    """
    parts = []
    if len(planes.floor_points):
        parts.append(planes.floor_points)
    for p in planes.wall_inliers:
        parts.append(p)
    if not parts:
        return np.zeros((0, 2))
    pts = np.vstack(parts)
    return np.column_stack([pts @ e0, pts @ e1])


def _extract_polygon(
    pts2d: np.ndarray,
    planes: PlaneSet,
    up: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    cfg: FloorplanConfig,
    cams2d: np.ndarray | None = None,
) -> np.ndarray:
    if len(pts2d) < 10:
        return np.zeros((0, 2))

    lo = pts2d.min(axis=0) - 4 * cfg.grid_m
    hi = pts2d.max(axis=0) + 4 * cfg.grid_m
    dims = np.maximum(np.ceil((hi - lo) / cfg.grid_m).astype(int) + 1, 3)

    ij = np.floor((pts2d - lo) / cfg.grid_m).astype(int)
    ij = ij[(ij[:, 0] >= 0) & (ij[:, 0] < dims[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < dims[1])]
    occ = np.zeros(dims, dtype=bool)
    occ[ij[:, 0], ij[:, 1]] = True

    occ = _close_and_fill(occ, cfg.close_cells)
    occ = _select_room(occ, lo, dims, cfg, planes, e0, e1, up, cams2d)
    contour = _largest_contour(occ)
    if contour is None or len(contour) < 4:
        return np.zeros((0, 2))
    poly = lo + (contour + 0.5) * cfg.grid_m

    poly = _douglas_peucker(poly, cfg.simplify_m)
    if len(poly) < 3:
        return np.zeros((0, 2))
    poly = _snap_to_walls(poly, planes, e0, e1, cfg)
    return geom.ensure_ccw(poly)


def _close_and_fill(occ: np.ndarray, radius: int) -> np.ndarray:
    """Bridge gaps under furniture and fill interior holes."""
    from scipy import ndimage

    if radius > 0:
        st = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        occ = ndimage.binary_closing(occ, structure=st, border_value=0)
    return ndimage.binary_fill_holes(occ)


def _disk(radius_cells: int) -> np.ndarray:
    r = max(int(radius_cells), 1)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _wall_barrier(
    planes: PlaneSet,
    lo: np.ndarray,
    dims: np.ndarray,
    cfg: FloorplanConfig,
    e0: np.ndarray,
    e1: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Rasterise each fitted wall as a solid line segment.

    A wall is drawn across its **full fitted extent**, including the gap where
    its doorway is. A doorway is part of the room's boundary — the room ends
    there — so closing it is correct, and it is what stops a flood fill from
    escaping into the room next door.

    Only planes that **reach the ceiling** are used. RANSAC finds vertical
    planes, and a wardrobe is a vertical plane: on the furnished scene the
    wardrobe was drawn as a barrier straight across the room, the flood fill
    stopped at it, and the reported wall length was 41% out. Furniture stops
    short of the ceiling and real walls do not, which separates them cleanly —
    measured spans were 1.96 m for the wardrobe against 2.44 m for every wall,
    in a 2.46 m room.
    """
    barrier = np.zeros(tuple(dims), dtype=bool)
    step = cfg.grid_m * 0.4
    ceiling = planes.ceiling_height
    floor_h = planes.floor_height
    best_support = max((w.inlier_count for w in planes.walls), default=0)
    for wall, pts in zip(planes.walls, planes.wall_inliers, strict=False):
        if len(pts) < 20:
            continue
        if ceiling > 0:
            top = float((pts @ up).max()) - floor_h
            if top < ceiling - cfg.barrier_ceiling_gap_m:
                continue  # furniture, not a wall
        if best_support and wall.inlier_count < cfg.barrier_min_support * best_support:
            # A minor plane that happens to reach the ceiling. On the furnished
            # scene RANSAC produced one tilted 5 degrees off the real back wall
            # with 5% of its support; drawn as a barrier it sliced the room
            # diagonally and cost 41% of the wall length. A room *boundary* is
            # one of the major planes, by construction.
            continue
        n3 = wall.normal
        n2 = np.array([n3 @ e0, n3 @ e1])
        norm = float(np.linalg.norm(n2))
        if norm < 1e-6:
            continue
        n2 = n2 / norm
        along = np.array([-n2[1], n2[0]])
        p2 = np.column_stack([pts @ e0, pts @ e1])
        u = p2 @ along
        c = float(np.median(p2 @ n2))
        u0, u1 = float(u.min()), float(u.max())
        if u1 - u0 < cfg.min_edge_m:
            continue
        ts = np.arange(u0, u1 + step, step)
        line = c * n2[None, :] + ts[:, None] * along[None, :]
        ij = np.floor((line - lo) / cfg.grid_m).astype(int)
        ok = (
            (ij[:, 0] >= 0) & (ij[:, 0] < dims[0])
            & (ij[:, 1] >= 0) & (ij[:, 1] < dims[1])
        )
        barrier[ij[ok, 0], ij[ok, 1]] = True
    return barrier


def _select_room(
    occ: np.ndarray,
    lo: np.ndarray,
    dims: np.ndarray,
    cfg: FloorplanConfig,
    planes: PlaneSet,
    e0: np.ndarray,
    e1: np.ndarray,
    up: np.ndarray,
    cams2d: np.ndarray | None,
) -> np.ndarray:
    """Isolate *this* room from everything visible through its doorways.

    A capture sees straight through an open doorway into the next room, and the
    further the sweep gets the more of the neighbour it picks up — a dense
    capture leaks *more* than a sparse one, so this cannot be left to coverage.
    Nor is the leak a narrow neck that a morphological opening could sever: the
    side walls of adjoining rooms are usually collinear, so the two regions meet
    along a wide front.

    What does separate them is the walls themselves. Every fitted wall is drawn
    as a barrier across its full extent (`_wall_barrier`), the barrier is
    subtracted from the occupied floor, and the result is flood-filled from
    where the camera actually stood. The flood cannot cross a wall, so it stops
    at the doorway, and the room comes back bounded by its own walls.

    Falls back to the untouched occupancy whenever the flood looks wrong — no
    cameras, no walls, or a result so small it is clearly a mis-fill — because a
    slightly-too-large footprint is a better failure than an empty one.
    """
    from scipy import ndimage

    if cams2d is None or not len(cams2d) or not planes.walls:
        return occ

    barrier = _wall_barrier(planes, lo, dims, cfg, e0, e1, up)
    if not barrier.any():
        return occ

    free = occ & ~barrier
    lab, n = ndimage.label(free)
    if n == 0:
        return occ

    ij = np.floor((cams2d - lo) / cfg.grid_m).astype(int)
    inside = (
        (ij[:, 0] >= 0) & (ij[:, 0] < dims[0])
        & (ij[:, 1] >= 0) & (ij[:, 1] < dims[1])
    )
    if not inside.any():
        return occ
    hits = lab[ij[inside, 0], ij[inside, 1]]
    hits = hits[hits > 0]
    if not len(hits):
        return occ
    room = lab == int(np.bincount(hits).argmax())

    # Grow back onto the wall cells the barrier removed, so the traced contour
    # sits on the wall surface rather than one cell inside it.
    grown = ndimage.binary_dilation(
        room, structure=ndimage.generate_binary_structure(2, 2), iterations=2
    ) & occ

    if grown.sum() < 0.05 * occ.sum():
        return occ  # implausibly small; trust the raw occupancy instead
    return ndimage.binary_fill_holes(grown)


def _largest_contour(occ: np.ndarray) -> np.ndarray | None:
    from skimage import measure

    padded = np.pad(occ.astype(float), 1)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None
    c = max(contours, key=len)
    return c - 1.0


def _douglas_peucker(poly: np.ndarray, tol: float) -> np.ndarray:
    """Closed-polygon Douglas-Peucker. Seeded from the two most distant points
    so the split is stable regardless of where the contour tracer started."""
    if len(poly) < 4:
        return poly
    # Drop the duplicated closing vertex if present.
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    start = int(np.argmax(np.linalg.norm(poly - poly.mean(axis=0), axis=1)))
    poly = np.roll(poly, -start, axis=0)
    far = int(np.argmax(np.linalg.norm(poly - poly[0], axis=1)))
    keep = np.zeros(len(poly), dtype=bool)
    keep[0] = keep[far] = True
    _dp(poly, 0, far, tol, keep)
    _dp(poly, far, len(poly) - 1, tol, keep)
    keep[-1] = True
    out = poly[keep]
    if len(out) > 3 and np.linalg.norm(out[0] - out[-1]) < tol:
        out = out[:-1]
    return out


def _dp(poly: np.ndarray, i: int, j: int, tol: float, keep: np.ndarray) -> None:
    if j <= i + 1:
        return
    a, b = poly[i], poly[j]
    ab = b - a
    n = np.linalg.norm(ab)
    seg = poly[i + 1 : j]
    if n < 1e-12:
        d = np.linalg.norm(seg - a, axis=1)
    else:
        d = np.abs(geom.cross2(np.broadcast_to(ab, seg.shape), seg - a)) / n
    k = int(np.argmax(d))
    if d[k] <= tol:
        return
    m = i + 1 + k
    keep[m] = True
    _dp(poly, i, m, tol, keep)
    _dp(poly, m, j, tol, keep)


def _snap_to_walls(
    poly: np.ndarray, planes: PlaneSet, e0: np.ndarray, e1: np.ndarray, cfg: FloorplanConfig
) -> np.ndarray:
    """Replace raster edges with fitted wall lines and re-derive the corners."""
    lines = []
    for w in planes.walls:
        n3 = w.normal
        n2 = np.array([n3 @ e0, n3 @ e1])
        norm = np.linalg.norm(n2)
        if norm < 1e-6:  # a wall that is not actually vertical; skip it
            continue
        n2 = n2 / norm
        # Plane is n.x + d = 0 in 3D; in the 2D basis that is n2.p = c.
        c = -w.offset / norm
        lines.append((n2, c, w.inlier_count))
    if not lines:
        return poly

    n_pts = len(poly)
    assigned: list[tuple[np.ndarray, float] | None] = []
    cos_tol = np.cos(np.radians(cfg.snap_angle_deg))

    for i in range(n_pts):
        a, b = poly[i], poly[(i + 1) % n_pts]
        d = b - a
        length = np.linalg.norm(d)
        if length < 1e-9:
            assigned.append(None)
            continue
        edge_n = np.array([-d[1], d[0]]) / length
        mid = 0.5 * (a + b)
        best = None
        best_key = None
        for n2, c, count in lines:
            if abs(float(edge_n @ n2)) < cos_tol:
                continue
            dist = abs(float(n2 @ mid - c))
            if dist > cfg.snap_dist_m:
                continue
            key = (dist, -count)
            if best_key is None or key < best_key:
                best_key, best = key, (n2, c)
        assigned.append(best)

    # An unmatched edge keeps its own line, so the polygon never loses a side.
    resolved: list[tuple[np.ndarray, float]] = []
    for i in range(n_pts):
        if assigned[i] is not None:
            resolved.append(assigned[i])  # type: ignore[arg-type]
        else:
            a, b = poly[i], poly[(i + 1) % n_pts]
            d = b - a
            length = max(np.linalg.norm(d), 1e-9)
            n2 = np.array([-d[1], d[0]]) / length
            resolved.append((n2, float(n2 @ a)))

    # Collapse consecutive edges that landed on the same line.
    dedup: list[tuple[np.ndarray, float]] = []
    for n2, c in resolved:
        if dedup:
            pn, pc = dedup[-1]
            if abs(float(pn @ n2)) > np.cos(np.radians(2.0)) and abs(
                (c if float(pn @ n2) > 0 else -c) - pc
            ) < 0.05:
                continue
        dedup.append((n2, c))
    if len(dedup) > 2:
        fn, fc = dedup[0]
        ln, lc = dedup[-1]
        if abs(float(fn @ ln)) > np.cos(np.radians(2.0)) and abs(
            (lc if float(fn @ ln) > 0 else -lc) - fc
        ) < 0.05:
            dedup.pop()
    if len(dedup) < 3:
        return poly

    corners = []
    for i in range(len(dedup)):
        n1, c1 = dedup[i - 1]
        n2, c2 = dedup[i]
        det = geom.cross2(n1, n2)
        if abs(det) < 1e-6:  # parallel: fall back to the raster corner
            corners.append(poly[i % n_pts])
            continue
        x = (c1 * n2[1] - c2 * n1[1]) / det
        y = (n1[0] * c2 - n2[0] * c1) / det
        corners.append(np.array([x, y]))
    out = np.asarray(corners)

    # Drop degenerate sides created by near-parallel neighbours.
    keep = geom.edge_lengths(out) >= cfg.min_edge_m
    if keep.sum() >= 3:
        out = out[keep]
    return out


# ---------------------------------------------------------------------------
# Openings (PLAN.md §6.4 step 5)
# ---------------------------------------------------------------------------


def detect_openings(
    planes: PlaneSet,
    poly: np.ndarray,
    up: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    floor_h: float,
    ceil_h: float,
    cfg: FloorplanConfig,
) -> list[Opening]:
    """Find doors and windows as *holes* in the observed wall surface.

    A depth-based detector, not a semantic one: a door is a rectangle of wall
    with no returns, reaching the floor. SAM 3's `"window"` / `"glass door"`
    masks refine this (PLAN.md §6.4), but geometry alone already separates the
    two classes by sill height, and geometry does not need a GPU.

    Openings are reported against **polygon edge** indices with `u` measured
    from that edge's start vertex, so a consumer can place them without knowing
    anything about which RANSAC plane they came from.
    """
    out: list[Opening] = []
    if ceil_h <= 0 or len(poly) < 3:
        return out

    for wi in range(len(poly)):
        a2, b2 = poly[wi], poly[(wi + 1) % len(poly)]
        edge = b2 - a2
        edge_len = float(np.linalg.norm(edge))
        if edge_len < cfg.min_edge_m:
            continue
        along = (edge[0] * e0 + edge[1] * e1) / edge_len
        origin3 = a2[0] * e0 + a2[1] * e1

        pts = _inliers_for_edge(planes, a2, b2, e0, e1, cfg)
        if pts is None or len(pts) < 200:
            continue

        u = (pts - origin3) @ along
        h = pts @ up - floor_h
        # Parameterise over the polygon edge itself, not the inlier extent, so
        # u is directly comparable with the edge the SVG draws.
        u0, u1 = 0.0, edge_len
        keep = (u >= -0.05) & (u <= edge_len + 0.05)
        if keep.sum() < 200:
            continue
        u, h = u[keep], h[keep]
        nu = max(int(np.ceil((u1 - u0) / cfg.opening_bin_m)), 1)
        nh = max(int(np.ceil(ceil_h / cfg.opening_bin_m)), 1)
        occ = np.zeros((nu, nh), dtype=bool)
        iu = np.clip(((u - u0) / cfg.opening_bin_m).astype(int), 0, nu - 1)
        ih = np.clip((h / cfg.opening_bin_m).astype(int), 0, nh - 1)
        occ[iu, ih] = True

        # Only trust columns that were observed at all — an unobserved strip at
        # the end of a wall is missing data, not a doorway.
        seen_col = occ.any(axis=1)
        holes = _rect_holes(occ, seen_col, cfg)
        for c0, c1, r0, r1 in holes:
            width = (c1 - c0 + 1) * cfg.opening_bin_m
            height = (r1 - r0 + 1) * cfg.opening_bin_m
            if width < cfg.opening_min_width_m or height < cfg.opening_min_height_m:
                continue
            sill = r0 * cfg.opening_bin_m
            kind = _classify_opening(sill, height, cfg)
            if kind is None:
                continue
            out.append(
                Opening(
                    kind=kind,
                    wall_index=wi,
                    u_start_m=float(u0 + c0 * cfg.opening_bin_m),
                    u_end_m=float(u0 + (c1 + 1) * cfg.opening_bin_m),
                    height_min_m=float(sill),
                    height_max_m=float((r1 + 1) * cfg.opening_bin_m),
                )
            )
    return out


def _classify_opening(sill: float, height: float, cfg: FloorplanConfig) -> str | None:
    """Door, window, or neither.

    Returning `None` is the important case: a hole in the observed wall is only
    an opening if its dimensions are physically plausible for one. Everything
    else is missing data, and calling it a doorway is worse than missing a real
    one — the house graph joins rooms *at doorways* (PLAN.md §6.3), so a
    phantom door is a phantom join.
    """
    if sill <= cfg.door_max_sill_m:
        return "door" if height >= cfg.door_min_height_m else None
    return "window" if height >= cfg.window_min_height_m else None


def _inliers_for_edge(
    planes: PlaneSet,
    a2: np.ndarray,
    b2: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    cfg: FloorplanConfig,
) -> np.ndarray | None:
    """The wall plane's 3D inliers backing a given footprint edge."""
    d = b2 - a2
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return None
    edge_n = np.array([-d[1], d[0]]) / n
    mid = 0.5 * (a2 + b2)
    cos_tol = np.cos(np.radians(cfg.snap_angle_deg))

    best, best_key = None, None
    for wall, pts in zip(planes.walls, planes.wall_inliers, strict=False):
        n3 = wall.normal
        n2 = np.array([n3 @ e0, n3 @ e1])
        norm = np.linalg.norm(n2)
        if norm < 1e-6:
            continue
        n2 = n2 / norm
        if abs(float(edge_n @ n2)) < cos_tol:
            continue
        dist = abs(float(n2 @ mid + wall.offset / norm))
        if dist > cfg.snap_dist_m:
            continue
        key = (dist, -wall.inlier_count)
        if best_key is None or key < best_key:
            best_key, best = key, pts
    return best


def _rect_holes(
    occ: np.ndarray, seen_col: np.ndarray, cfg: FloorplanConfig
) -> list[tuple[int, int, int, int]]:
    """Maximal empty rectangles in an observed wall's occupancy grid."""
    from scipy import ndimage

    nu, nh = occ.shape
    empty = ~occ
    empty[~seen_col, :] = False
    # An opening is bounded on both sides by wall, so erode away the fringe
    # produced by the wall simply ending.
    lab, n = ndimage.label(empty)
    out = []
    for i in range(1, n + 1):
        cells = np.argwhere(lab == i)
        if len(cells) < 9:
            continue
        c0, r0 = cells.min(axis=0)
        c1, r1 = cells.max(axis=0)
        # Reject components that touch both ends of the wall: that is an
        # unobserved region, not an opening.
        if c0 == 0 and c1 == nu - 1:
            continue
        fill = len(cells) / max((c1 - c0 + 1) * (r1 - r0 + 1), 1)
        if fill < 0.55:  # not rectangle-shaped; probably occlusion shadow
            continue
        if r1 >= nh - 1 and r0 == 0:  # full-height gap == wall end, not a door
            continue
        out.append((int(c0), int(c1), int(r0), int(r1)))
    return out


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------


def to_svg(layout: RoomLayout, *, px_per_m: float = 120.0, margin_px: float = 70.0) -> str:
    """Dimensioned floor plan. Every wall carries its length in metres."""
    poly = np.asarray(layout.footprint, dtype=np.float64)
    if len(poly) < 3:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="60">' \
               '<text x="10" y="30">no footprint</text></svg>'

    lo, hi = poly.min(axis=0), poly.max(axis=0)
    size = (hi - lo) * px_per_m + 2 * margin_px

    def xy(p: np.ndarray) -> tuple[float, float]:
        # Flip the second axis so the plan reads north-up rather than mirrored.
        return (
            (p[0] - lo[0]) * px_per_m + margin_px,
            (hi[1] - p[1]) * px_per_m + margin_px,
        )

    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(p) for p in poly))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size[0]:.0f}" '
        f'height="{size[1]:.0f}" viewBox="0 0 {size[0]:.0f} {size[1]:.0f}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<polygon points="{pts}" fill="#eef2f6" stroke="#1f2933" stroke-width="3"/>',
    ]

    centroid = poly.mean(axis=0)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        length = float(np.linalg.norm(b - a))
        mid = 0.5 * (a + b)
        # Push the label outward, away from the room centre.
        outward = mid - centroid
        nrm = np.linalg.norm(outward)
        outward = outward / nrm if nrm > 1e-9 else np.array([0.0, 1.0])
        lx, ly = xy(mid + outward * 0.22)
        angle = np.degrees(np.arctan2(-(b[1] - a[1]), b[0] - a[0]))
        if angle > 90 or angle < -90:
            angle += 180
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-family="Inter,Helvetica,sans-serif" '
            f'font-size="13" fill="#1f2933" text-anchor="middle" '
            f'transform="rotate({angle:.1f} {lx:.1f} {ly:.1f})">{length:.2f} m</text>'
        )

    for op in layout.openings:
        if op.wall_index >= len(poly):
            continue
        a, b = poly[op.wall_index], poly[(op.wall_index + 1) % len(poly)]
        d = b - a
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d = d / n
        p0, p1 = a + d * op.u_start_m, a + d * op.u_end_m
        x0, y0 = xy(p0)
        x1, y1 = xy(p1)
        colour = "#c2410c" if op.kind == "door" else "#0369a1"
        parts.append(
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
            f'stroke="{colour}" stroke-width="6" stroke-linecap="round"/>'
        )

    cx, cy = xy(centroid)
    parts.append(
        f'<text x="{cx:.1f}" y="{cy:.1f}" font-family="Inter,Helvetica,sans-serif" '
        f'font-size="17" font-weight="600" fill="#1f2933" text-anchor="middle">'
        f'{layout.area_m2:.2f} m²</text>'
    )
    parts.append(
        f'<text x="{cx:.1f}" y="{cy + 20:.1f}" font-family="Inter,Helvetica,sans-serif" '
        f'font-size="12" fill="#52606d" text-anchor="middle">'
        f'ceiling {layout.ceiling_height_m:.2f} m</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)
