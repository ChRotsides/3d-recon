"""Build a 360 equirectangular panorama from poses, not from feature matching.

Feature-based stitching assumes every frame shares one centre of projection and
throws away whatever it cannot reconcile — on this capture it kept a handful of
27 frames and produced a 1.4:1 crop instead of a sphere. But the bundle already
records where each frame pointed: ARCore's rotation is good, and the room's
whole geometry problem (translation) barely affects *direction*.

So each pixel of the output sphere is traced back into every frame that can see
it and blended. Nothing is discarded for failing to match, which is the same
principle as the shutter: use what was captured rather than refuse it.

Convention is PLAN.md §5 — right-handed, Y-up, world<-camera poses, camera
looking down -Z with +X right and +Y up.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BUNDLE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
W = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
H = W // 2

bundle = json.loads((BUNDLE / "bundle.json").read_text())
kfs = bundle["keyframes"]
print(f"{len(kfs)} keyframes -> {W}x{H} equirectangular", flush=True)

# Ray direction for every output pixel.
#
# Yaw is measured so that the panorama's centre column looks along -Z, matching
# the camera convention; pitch runs +90 (up) at the top row to -90 at the bottom.
yaw = (np.arange(W, dtype=np.float32) + 0.5) / W * 2.0 * np.pi - np.pi
pitch = np.pi / 2.0 - (np.arange(H, dtype=np.float32) + 0.5) / H * np.pi
yaw_g, pitch_g = np.meshgrid(yaw, pitch)
cos_p = np.cos(pitch_g)
rays = np.stack([cos_p * np.sin(yaw_g), np.sin(pitch_g), -cos_p * np.cos(yaw_g)], axis=-1)
rays = rays.reshape(-1, 3)

acc = np.zeros((H * W, 3), dtype=np.float64)
wsum = np.zeros((H * W, 1), dtype=np.float64)
used = 0

for kf in kfs:
    img = cv2.imread(str(BUNDLE / kf["image"]))
    if img is None:
        continue
    ih, iw = img.shape[:2]
    intr = kf["intrinsics"]
    sx, sy = iw / intr["width"], ih / intr["height"]
    fx, fy = intr["fx"] * sx, intr["fy"] * sy
    cx, cy = intr["cx"] * sx, intr["cy"] * sy

    R = np.asarray(kf["pose"]["R"], dtype=np.float64)      # world<-camera
    cam = rays @ R                                          # camera<-world is R^T

    # The camera looks down -Z, so only rays with negative Z are in front of it.
    front = cam[:, 2] < -1e-6
    z = np.where(front, cam[:, 2], -1.0)
    u = cx + fx * (cam[:, 0] / -z)
    v = cy - fy * (cam[:, 1] / -z)

    inside = front & (u >= 0) & (u < iw - 1) & (v >= 0) & (v < ih - 1)
    if not inside.any():
        continue
    used += 1

    idx = np.nonzero(inside)[0]
    uu, vv = u[idx].astype(np.float32), v[idx].astype(np.float32)

    # Bilinear sample. The corner indices are clamped rather than trusted to
    # the mask above: that test runs in float64 and these coordinates are
    # float32, and a value a hair under the bound can round up past it.
    x0 = np.clip(np.floor(uu), 0, iw - 2).astype(np.int32)
    y0 = np.clip(np.floor(vv), 0, ih - 2).astype(np.int32)
    fxx = np.clip(uu - x0, 0.0, 1.0)[:, None]
    fyy = np.clip(vv - y0, 0.0, 1.0)[:, None]
    p00 = img[y0, x0].astype(np.float64)
    p10 = img[y0, x0 + 1].astype(np.float64)
    p01 = img[y0 + 1, x0].astype(np.float64)
    p11 = img[y0 + 1, x0 + 1].astype(np.float64)
    sample = (p00 * (1 - fxx) * (1 - fyy) + p10 * fxx * (1 - fyy)
              + p01 * (1 - fxx) * fyy + p11 * fxx * fyy)

    # Feather towards the frame edges so seams fade instead of stepping, and
    # weight by sharpness so a soft frame loses to a crisp one where they meet.
    fw = np.minimum(uu, iw - 1 - uu) / (iw * 0.5)
    fh = np.minimum(vv, ih - 1 - vv) / (ih * 0.5)
    feather = np.clip(np.minimum(fw, fh), 1e-3, 1.0) ** 2
    sharp = max(float(kf.get("blur_score", 1.0)), 1.0) ** 0.5
    w = (feather * sharp)[:, None]

    acc[idx] += sample * w
    wsum[idx] += w

covered = (wsum[:, 0] > 0).sum()
print(f"{used} frames contributed · {covered * 100.0 / (H * W):.1f}% of the sphere covered",
      flush=True)

pano = np.zeros((H * W, 3), dtype=np.uint8)
hit = wsum[:, 0] > 0
pano[hit] = np.clip(acc[hit] / wsum[hit], 0, 255).astype(np.uint8)
pano = pano.reshape(H, W, 3)

# Crop the empty top and bottom: a phone sweep never reaches the poles, and
# leaving them black makes the panorama look like it failed rather than like
# what it is — a band around the horizon.
rows = np.nonzero(hit.reshape(H, W).any(axis=1))[0]
if len(rows):
    pano = pano[rows[0]:rows[-1] + 1]

cv2.imwrite(str(OUT), pano)
print(f"wrote {OUT}  {pano.shape[1]}x{pano.shape[0]}", flush=True)
