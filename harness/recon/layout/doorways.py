"""Doorway extraction (PLAN.md §5, §6.3).

A `Doorway` is the join between two rooms in the house graph, so it carries
exactly what the 3-DoF alignment in §6.3 needs: the wall plane it sits in, the
centroid of its threshold (the floor-level midpoint), and its width and height.

Doorways come from the opening detector in `floorplan.py` — a door is a hole in
a wall that reaches the floor. That makes them free: no extra model, no extra
capture step. It also makes them the right thing to ask the user to measure
(§6.2), since they are planar, unambiguous and already located.
"""

from __future__ import annotations

import numpy as np

from recon.layout.planes import _basis
from recon.schemas import Doorway, RoomLayout


def extract_doorways(layout: RoomLayout, up: np.ndarray, prefix: str = "dw") -> list[Doorway]:
    """Turn a layout's detected doors into house-graph joinable doorways.

    The plane is in Hessian normal form with the normal pointing *into* the
    room, matching the wall-plane convention. The centroid sits at floor level
    in the middle of the opening, which is the point §6.3 step 3 matches across
    rooms.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    e0, e1 = _basis(up)
    poly = np.asarray(layout.footprint, dtype=np.float64)
    out: list[Doorway] = []

    for i, op in enumerate(layout.openings):
        if op.kind != "door":
            continue
        if op.wall_index >= len(poly):
            continue
        a2 = poly[op.wall_index]
        b2 = poly[(op.wall_index + 1) % len(poly)]
        d2 = b2 - a2
        n = float(np.linalg.norm(d2))
        if n < 1e-9:
            continue
        d2 = d2 / n

        # Along-wall direction and inward normal, lifted to 3D.
        along = d2[0] * e0 + d2[1] * e1
        normal = np.cross(up, along)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        # Orient into the room: the footprint is CCW, so the interior lies to
        # the left of each edge.
        centre2 = poly.mean(axis=0)
        mid2 = 0.5 * (a2 + b2)
        inward3 = (centre2 - mid2)[0] * e0 + (centre2 - mid2)[1] * e1
        if float(normal @ inward3) < 0:
            normal = -normal

        u_mid = 0.5 * (op.u_start_m + op.u_end_m)
        origin3 = a2[0] * e0 + a2[1] * e1
        centroid = origin3 + along * u_mid + up * layout.floor_height_m

        out.append(
            Doorway(
                id=f"{prefix}_{i:02d}",
                plane=[*normal.tolist(), float(-(normal @ centroid))],
                centroid=centroid.tolist(),
                width_m=float(op.u_end_m - op.u_start_m),
                height_m=float(op.height_max_m - op.height_min_m),
            )
        )
    return out


def doorway_frame(dw: Doorway, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centroid, inward normal, along-wall direction) for a doorway.

    The along direction is `up x normal`, so the three form a right-handed
    frame and the sign convention is fixed rather than incidental.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    n = np.asarray(dw.plane[:3], dtype=np.float64)
    n = n - up * float(n @ up)  # a doorway normal is horizontal by construction
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    along = np.cross(up, n)
    along = along / max(float(np.linalg.norm(along)), 1e-12)
    return np.asarray(dw.centroid, dtype=np.float64), n, along
