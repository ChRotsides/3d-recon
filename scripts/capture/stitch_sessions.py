"""Stitch each capture, trying a ladder of confidence thresholds.

A looser threshold is not simply more permissive — it admits marginal pairs to
the bundle adjustment, and the camera-parameter solve then diverges. One
capture here stitched at the default 1.0 and failed at 0.6 for exactly that
reason. So each is tried strict-first and the first success wins, rather than
picking one threshold and calling the rest unstitchable.

Writes a manifest the viewer reads, so adding a capture needs no HTML edit.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

LADDER = [1.0, 0.8, 0.6, 0.4]
STATUS = {
    cv2.Stitcher_OK: "OK",
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "too little overlap",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "homography failed",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "camera solve diverged",
}


def best_stitch(imgs):
    """First threshold that succeeds, strictest first."""
    for conf in LADDER:
        st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        st.setPanoConfidenceThresh(conf)
        t0 = time.time()
        status, pano = st.stitch(imgs)
        dt = time.time() - t0
        if status == cv2.Stitcher_OK:
            return pano, conf, dt, "OK"
        last = STATUS.get(status, str(status))
    return None, None, dt, last


def pose_stats(bundle: Path):
    """Baseline and step, so the mosaic can be read against how it was shot."""
    try:
        b = json.loads((bundle / "bundle.json").read_text())
    except Exception:
        return ""
    kf = b.get("keyframes", [])
    if len(kf) < 2:
        return ""
    t = np.array([k["pose"]["t"] for k in kf])
    d = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=-1)
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    return (f"max baseline {d.max():.2f} m · median step {np.median(steps):.3f} m"
            f" · {b.get('capture_mode', '?')}")


entries = []
for bundle in sorted(p for p in ROOT.iterdir() if (p / "images").is_dir()):
    paths = sorted((bundle / "images").glob("*.jpg"))
    imgs = [cv2.imread(str(p)) for p in paths]
    imgs = [i for i in imgs if i is not None]
    short = bundle.name[:20]
    if len(imgs) < 2:
        print(f"{short:22} {len(imgs):3d}  skipped", flush=True)
        continue
    pano, conf, dt, why = best_stitch(imgs)
    if pano is None:
        print(f"{short:22} {len(imgs):3d}  FAILED at every threshold — {why}  {dt:.1f}s",
              flush=True)
        continue
    name = f"{short}.jpg"
    cv2.imwrite(str(OUT / name), pano)
    print(f"{short:22} {len(imgs):3d}  OK conf={conf}  {pano.shape[1]}x{pano.shape[0]}  {dt:.1f}s",
          flush=True)
    entries.append({
        "file": f"sessions/{name}",
        "label": short,
        "meta": f"{len(imgs)} keyframes · stitched at conf {conf} · {pose_stats(bundle)}",
    })

(OUT / "manifest.json").write_text(json.dumps(entries, indent=2))
print(f"\n{len(entries)} mosaics -> {OUT/'manifest.json'}", flush=True)
