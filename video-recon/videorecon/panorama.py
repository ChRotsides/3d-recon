"""Panoramas by pose projection, one per station along the walk.

A hand-held walk has no single viewpoint, so the trajectory is cut into
stations (runs of keyframes within `station_radius_m` of a centre) and each
station's keyframes are projected onto a sphere by their *orientation* alone,
blended with a centre-weighted feather. Near a station centre the parallax is
small and the panorama is clean; where the stitch is bad, the poses or the
keyframe density are what is wrong, which is exactly what this is for.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np

_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}


def stations(
    positions: np.ndarray, *, radius_m: float = 0.6, min_frames: int = 12
) -> list[list[int]]:
    """Greedy runs of consecutive keyframes staying within `radius_m` of a centre."""
    runs: list[list[int]] = []
    start = 0
    n = len(positions)
    while start < n:
        centre = positions[start].copy()
        end = start
        while end + 1 < n:
            candidate = positions[start : end + 2].mean(axis=0)
            if np.linalg.norm(positions[start : end + 2] - candidate, axis=1).max() > radius_m:
                break
            end += 1
            centre = candidate
        runs.append(list(range(start, end + 1)))
        start = end + 1
    return [r for r in runs if len(r) >= min_frames]


def render_equirect(
    images: list[np.ndarray],
    rotations_gl: list[np.ndarray],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    width: int = 2048,
    feather: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend images onto an equirectangular sphere (Y up, -Z at longitude 0)."""
    height = width // 2
    lon = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    lon_g, lat_g = np.meshgrid(lon, lat)
    dirs = np.stack(
        [np.cos(lat_g) * np.sin(lon_g), np.sin(lat_g), -np.cos(lat_g) * np.cos(lon_g)], axis=-1
    ).astype(np.float32)
    # "Nearest image centre wins": averaging frames taken from slightly
    # different positions ghosts every edge, while a winner-takes-all mosaic
    # shows a pose or parallax error as a visible seam offset — which is the
    # diagnostic value of the picture. A short feather softens the seams.
    best = np.zeros((height, width, 3), dtype=np.float32)
    wsum = np.zeros((height, width), dtype=np.float32)
    for img, r in zip(images, rotations_gl, strict=True):
        h, w = img.shape[:2]
        d = dirs @ np.asarray(r, dtype=np.float32)  # R^T d  (rows: d · columns of R)
        z = -d[..., 2]
        vis = z > 1e-3
        u = np.where(vis, fx * d[..., 0] / np.where(vis, z, 1) + cx, -1).astype(np.float32)
        v = np.where(vis, cy - fy * d[..., 1] / np.where(vis, z, 1), -1).astype(np.float32)
        inside = vis & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        if not inside.any():
            continue
        sample = cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # centre-weighted feather so seams fade instead of cut
        du = np.minimum(u / max(w - 1, 1), 1 - u / max(w - 1, 1))
        dv = np.minimum(v / max(h - 1, 1), 1 - v / max(h - 1, 1))
        edge = np.clip(np.minimum(du, dv) / feather, 0, 1)
        wgt = np.where(inside, edge * edge, 0).astype(np.float32)
        take = wgt > wsum
        # feather: where the new frame barely beats the old one, blend the two
        close = inside & ~take & (wgt > 0.7 * wsum)
        best[take] = sample[take].astype(np.float32)
        best[close] = 0.5 * best[close] + 0.5 * sample[close].astype(np.float32)
        wsum = np.maximum(wsum, wgt)
    out = np.where(wsum[..., None] > 0, best, 0).astype(np.uint8)
    return out, wsum


def render_all(root: Path, *, width: int = 2048, radius_m: float = 0.4, log=print) -> dict:
    bundle = json.loads((root / "bundle" / "bundle.json").read_text())
    kfs = bundle["keyframes"]
    positions = np.asarray([k["pose"]["t"] for k in kfs])
    out_dir = root / "panoramas"
    out_dir.mkdir(exist_ok=True)
    intr = kfs[0]["intrinsics"]
    result = {"width": width, "stations": []}
    for si, run in enumerate(stations(positions, radius_m=radius_m)):
        name = f"station_{si:02d}.jpg"
        path = out_dir / name
        if not path.exists():
            images, rots = [], []
            for i in run:
                img = cv2.imread(str(root / "bundle" / kfs[i]["image"]))
                if img is None:
                    continue
                images.append(img)
                rots.append(np.asarray(kfs[i]["pose"]["R"]))
            pano, cover = render_equirect(
                images, rots, intr["fx"], intr["fy"], intr["cx"], intr["cy"], width=width
            )
            cv2.imwrite(str(path), pano, [cv2.IMWRITE_JPEG_QUALITY, 88])
            coverage = float((cover > 0).mean())
        else:
            coverage = None
        centre = positions[run].mean(axis=0)
        result["stations"].append(
            {
                "index": si,
                "file": f"panoramas/{name}",
                "frames": [int(kfs[run[0]]["id"][2:]), int(kfs[run[-1]]["id"][2:])],
                "keyframes": len(run),
                "centre": [round(float(c), 2) for c in centre],
                "coverage": coverage,
            }
        )
        log(f"panorama: station {si} from {len(run)} keyframes -> {name}")
    (out_dir / "panoramas.json").write_text(json.dumps(result, indent=1))
    return result


def status(root: Path) -> dict:
    """Kick off rendering once; report what exists."""
    index = root / "panoramas" / "panoramas.json"
    if index.exists():
        return {"ready": True, **json.loads(index.read_text())}
    if not (root / "bundle" / "bundle.json").exists():
        return {"ready": False, "error": "no bundle yet"}
    key = str(root)
    with _LOCK:
        job = _JOBS.get(key)
        if job is None:
            job = {"state": "running", "error": None}
            _JOBS[key] = job

            def work() -> None:
                try:
                    render_all(root, log=lambda *_: None)
                    job["state"] = "done"
                except Exception as exc:  # noqa: BLE001
                    job["state"] = "failed"
                    job["error"] = repr(exc)

            threading.Thread(target=work, daemon=True).start()
    return {"ready": False, "state": job["state"], "error": job["error"]}
