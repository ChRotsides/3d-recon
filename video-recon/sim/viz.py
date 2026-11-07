"""Top-down pictures of the discovered map: occupied black, free white, unknown grey,
frontiers green, the current path blue, the estimated pose a red arrow and the ground-truth
pose (mapped into the estimate frame) a faint orange one."""

from __future__ import annotations

import cv2
import numpy as np

from .navgrid import FREE, OCCUPIED


def costmap_png(state, cost, origin_xy, res, path=None, est_pose=None, gt_pose=None,
                frontier_mask=None, scale=4, target=None):
    ny, nx = state.shape
    img = np.full((ny, nx, 3), 150, np.uint8)
    img[state == FREE] = 245
    img[state == OCCUPIED] = 25
    img[(state == FREE) & ~np.isfinite(cost)] = 190  # free but inside the inflated footprint
    if frontier_mask is not None:
        img[frontier_mask] = (60, 200, 60)
    img = cv2.resize(img, (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST)

    def to_px(xy):
        return (int((xy[0] - origin_xy[0]) / res * scale), int((xy[1] - origin_xy[1]) / res * scale))

    if path is not None and len(path) > 1:
        pts = np.array([to_px(p) for p in path], np.int32)
        cv2.polylines(img, [pts], False, (235, 120, 40), max(1, scale // 2))
    if target is not None:
        cv2.circle(img, to_px(target), 3 * scale, (235, 120, 40), 1)
    for pose, colour, w in ((gt_pose, (120, 180, 245), 2), (est_pose, (40, 40, 230), 3)):
        if pose is None:
            continue
        p0 = to_px(pose[:2])
        p1 = to_px((pose[0] + 0.45 * np.cos(pose[2]), pose[1] + 0.45 * np.sin(pose[2])))
        cv2.arrowedLine(img, p0, p1, colour, w, tipLength=0.35)
        cv2.circle(img, p0, max(2, scale // 2), colour, -1)
    return np.flipud(img)  # +y up


def depth_png(depth_m, vmax=6.0):
    d = np.clip(np.nan_to_num(depth_m) / vmax, 0, 1)
    img = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth_m <= 0.01] = 0
    return img
