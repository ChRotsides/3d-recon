# Image-only panorama to cubemap point cloud

This is the exact RGB-only process that produced the best point cloud so far.
It was finalized on 2026-08-22 using capture
`bundle_065a6e7a-e8e3-449f-aa0d-61fc540d6e41`.

The governing rule is:

> Use image features to build one wrapped 360° panorama, render that panorama
> through six known non-overlapping pinhole cameras, predict one depth map per
> camera, then use those same camera matrices to put every depth pixel back in
> its correct place in one shared 3D coordinate system.

No ARCore pose, translation, rotation, intrinsics, depth, sparse points, or
heading is used. The bundle contains some of that data, but this pipeline does
not read it. The feature stitcher reads only the keyframe IDs and RGB image
paths. The depth stage reads only the wrapped RGB panorama.

```text
42 portrait RGB photos
        |
        v
image features + verified overlaps
        |
        v
largest connected camera graph: 30 photos / 65 overlap edges
        |
        v
spherical feature stitch: 2667 x 1066 with alpha
        |
        v
close longitude + bring ceiling down + bring floor up
        |
        v
wrapped 2:1 equirectangular panorama: 2667 x 1334
        |
        v
six 512 x 512, 90° x 90° pinhole cameras
front | right | back | left | ceiling | floor
        |
        v
Depth Anything 3 Small: six 504 x 504 depth maps
        |
        v
back-project with the exact same six camera matrices
        |
        v
one shared, complete-sphere, 800,000-point cloud
```

## 1. Take the RGB photographs

The successful capture contained 42 JPEG frames. Each frame was 480 × 640,
portrait orientation. There was no fixed angular step and no fixed overlap
percentage enforced during capture. Instead, the later image-feature graph
decided which photographs really overlapped.

For another capture, preserve the conditions this method depends on:

1. Keep the phone near one optical center and rotate around it. Translation
   creates parallax, which no single panorama can represent exactly.
2. Give every neighboring photograph generous shared visual detail. Plain
   walls alone are weak; include corners, door frames, furniture, pictures, and
   other repeatable features in the overlap.
3. Complete the full horizontal circle and photograph the starting direction
   again so the left and right ends can be verified as neighbors.
4. Tilt upward and downward as well as around the walls. The wrapper can carry
   the last visible ceiling and floor pixels to the poles, but it cannot invent
   real detail that was never photographed.
5. Prefer sharp, similarly exposed photographs. The software compensates
   exposure and chooses the sharper centered source in an overlap, but it
   cannot undo motion blur.

## 2. Extract spatially balanced image features

Run:

```bash
.venv/bin/python scripts/capture/feature_stitch.py \
  data/captures/bundle_065a6e7a-e8e3-449f-aa0d-61fc540d6e41 \
  out/capture-065a/feature-stitch-v4.png \
  --diagnostics out/capture-065a/feature-stitch-v4.json \
  --confidence 0.6
```

For each RGB photograph, the stitcher did the following:

1. Converted BGR to grayscale.
2. Applied CLAHE with clip limit 2.0 and an 8 × 8 tile grid.
3. Detected SIFT with at most 10,000 features, contrast threshold 0.003,
   edge threshold 15, and sigma 1.2.
4. Split the portrait image into a 12-row × 8-column grid. It kept at most 80
   strongest features per cell and at most 6,000 total. This stopped one highly
   textured object from consuming the feature budget and gave walls, corners,
   and furniture independent votes.
5. Converted SIFT descriptors to RootSIFT by L1-normalizing each descriptor and
   taking its element-wise square root.

This stage uses no recorded camera focal length. The focal length is recovered
later from the image correspondences.

## 3. Verify every overlapping image pair

Every pair among the 42 images was considered. A pair became an edge in the
camera graph only after passing all of these checks:

1. FLANN KD-tree matching used 5 trees and 64 checks.
2. Two-neighbor Lowe ratio matching used threshold 0.65 in both directions.
3. A match survived only when the forward and reverse matches agreed.
4. At least 16 mutual matches were required before geometric fitting.
5. A homography was fitted with USAC MAGSAC: 3 px threshold, at most 10,000
   iterations, and 0.999 confidence.
6. At least 14 MAGSAC inliers were required.
7. Inliers had to occupy at least four image-grid cells in both images and at
   least 0.002 of each image's area.
8. Median symmetric reprojection error had to be at most 3 px.
9. Pair confidence was `inliers / (8 + 0.3 * matches)` and had to be at least
   0.6. The inlier ratio also had to be at least 0.30.
10. Warped pixel overlap had to pass three photometric checks: gradient
    correlation at least 0.20, intensity correlation at least 0.35, and overlap
    fraction at least 0.08.

The pair weight combined confidence, inlier count, gradient and intensity
agreement, overlap area, feature spread, and reprojection error. Weak accidental
links were therefore unable to drag a separate region into the panorama.

## 4. Recover the image-only camera rotations

The stitcher self-calibrated one focal length from the 50 strongest rotation
homographies. It searched from 0.28 to 1.6 times the longest image side and
chose the focal length whose normalized homographies were closest to rotations.

For this capture:

- Estimated focal length: 424.489083 px
- Implied FOV for a 480 × 640 image: approximately 58.97° horizontal and
  74.02° vertical
- Recorded phone/ARCore intrinsics used: none

Pair rotations above 95° were rejected. The code then:

1. Found connected components in the verified overlap graph.
2. Kept only the largest component.
3. Built a maximum-weight spanning tree, rooted at the image with the greatest
   sum of overlap-edge weights.
4. Rejected inconsistent graph loops using a limit of
   `max(10°, 3 * initial median cycle error)`.
5. Refined all remaining rotations together with robust `soft_l1` least
   squares, a 3° scale, sparse LSMR solving, and at most 80 evaluations.

The successful graph contained:

- 42 input photographs
- 30 selected photographs in the largest component
- 65 accepted overlap edges
- Component sizes: 30, 5, 2, 1, 1, 1, 1, 1
- Final median cycle error: 1.198278°

The selected frames were `kf_0000`–`kf_0007`, `kf_0009`–`kf_0013`,
`kf_0015`–`kf_0019`, `kf_0021`–`kf_0026`, and `kf_0028`–`kf_0033`.

## 5. Warp the selected photos onto a sphere

The refined rotations received horizontal wave correction. Each selected
photo was then projected with OpenCV's spherical rotation warper using the
image-only focal estimate.

Blending was deliberately narrow:

1. Gain-block exposure compensation was applied to the warped photographs.
2. At each panorama pixel, one source photograph became the owner. Preference
   went to a pixel far from that source image's boundary and to a globally
   sharper source, measured by Laplacian variance.
3. Ownership was feathered with only a 2 px Gaussian band.

This single-view ownership was important. Broad multiband blending had produced
translucent duplicates when nearby objects moved because of capture parallax.
The output was cropped to its non-transparent alpha region and saved as a BGRA
PNG:

```text
out/capture-065a/feature-stitch-v4.png
2667 x 1066
```

At this point the walls have been placed around a spherical panorama, but the
image is not yet a closed 2:1 sphere and it still has transparent ceiling/floor
borders.

## 6. Wrap the panorama: close the walls, lower the ceiling, raise the floor

Run:

```bash
.venv/bin/python scripts/capture/wrap_panorama.py \
  out/capture-065a/feature-stitch-v4.png \
  out/capture-065a/feature-stitch-v5-wrapped.png
```

The wrapper did exactly this:

1. For each panorama longitude/column, it found the rows containing real
   stitched pixels.
2. It filled transparent rows above and below by one-dimensional interpolation
   along that same column. An entirely empty column borrowed the nearest
   non-empty longitude first.
3. It made the canvas exactly 2:1 by increasing its height to `width / 2`:
   2667 × 1334. Padding was split between top and bottom.
4. The top and bottom stitched edges were horizontally blurred with sigma 24.
   Their median colors became the common north- and south-pole colors.
5. Smoothstep interpolation brought the upper edge down toward the north pole
   and the lower edge up toward the south pole. The outer 20% pole bands were
   faded into their shared pole colors to prevent longitude streaks where all
   longitudes meet.
6. The left and right longitude boundaries were closed across a 48 px band.
   Matching pixels from both edges were averaged and blended with smoothstep.

This is the operation described informally as "the walls come in, the ceiling
comes down, and the floor folds up." It closes the spherical image and makes it
safe to sample at every direction. It does not create unseen ceiling or floor
geometry; those pole areas are extrapolations of the nearest photographed
pixels.

## 7. Render a complete non-overlapping virtual camera cube

Do not cut six rectangles directly out of the equirectangular image. An
equirectangular rectangle is not a pinhole-camera photograph. Instead, render
six true rectilinear views by casting a ray for every output pixel.

The successful virtual camera had:

| Property | Value |
|---|---:|
| Camera model | rectilinear pinhole |
| Faces | front, right, back, left, ceiling, floor |
| Face size | 512 × 512 |
| Horizontal FOV | 90° |
| Vertical FOV | 90° |
| Focal length | 256 px |
| Principal point | (255.5, 255.5) px |
| Optical centers | one shared origin |
| Area overlap | 0°; only boundary rays are shared |
| Spherical coverage | complete |

For pixel `(u, v)`, the camera ray is:

```text
r_camera = [(u - cx) / f, (v - cy) / f, 1]
f = width / (2 * tan(horizontal_FOV / 2)) = 256
cx = cy = (512 - 1) / 2 = 255.5
```

The ray is rotated by that face's camera-to-world rotation. With world `+X`
right, `+Y` down, and `+Z` front, the six optical axes are:

| Face | World direction |
|---|---|
| front | `+Z` |
| right | `+X` |
| back | `-Z` |
| left | `-X` |
| ceiling | `-Y` |
| floor | `+Y` |

The world ray is sampled from the equirectangular panorama with:

```text
longitude = atan2(ray.x, ray.z)
latitude  = atan2(ray.y, sqrt(ray.x^2 + ray.z^2))
pano_u    = (longitude / (2*pi) + 0.5) * panorama_width
pano_v    = (latitude  / pi     + 0.5) * panorama_height
```

Bilinear sampling is used and longitude wraps at the horizontal boundary.
Each world-to-camera extrinsic is the transpose/inverse of the corresponding
camera-to-world rotation, with zero translation. Therefore every face has the
same optical center.

Six square 90° faces are essential: four side views alone cover only a belt and
omit ceiling/floor directions. These six faces tile the entire sphere. They do
not overlap in area, but their shared edge and corner rays meet exactly.

## 8. Predict depth from the six RGB faces

Run:

```bash
.venv/bin/python -u scripts/capture/panorama_depth_scene.py \
  out/capture-065a/feature-stitch-v5-wrapped.png \
  out/capture-065a/panorama-depth-anything-cubemap \
  --variant small \
  --face-width 512 \
  --face-height 512 \
  --process-res 504 \
  --max-points 800000
```

The model was the locally cached `depth-anything/DA3-SMALL`. It ran in CPU eval
and inference mode. The six RGB views were supplied in this order:

```text
front, right, back, left, ceiling, floor
```

DA3 received no ARCore data and no camera matrices as model inputs:

```text
extrinsics=None
intrinsics=None
align_to_input_ext_scale=False
ref_view_strategy="first"
process_res=504
```

DA3 produced six 504 × 504 relative-depth maps. Their medians in the successful
run were 1.129644, 1.060184, 1.032995, 1.147546, 0.842634, and 0.850284 in face
order.

After inference, we did not let predicted camera poses decide where the cube
faces belonged. We replaced the prediction's cameras with the exact virtual
cubemap cameras. Because DA3 resized 512 to 504, the intrinsics were rescaled
by `504 / 512 = 0.984375`:

```text
f_processed  = 252.0 px
cx_processed = cy_processed = 251.5078125 px
```

This rescaling is the camera-POV correction. The depth pixel grid and the
camera rays must use the same resolution and FOV or the reconstructed walls
will open, overlap, or land at the wrong angles.

Confidence filtering was intentionally disabled so a low-confidence wall or
pole could not disappear:

```text
conf_thresh=0.0
conf_thresh_percentile=0.0
ensure_thresh_percentile=100.0
```

The resulting threshold is the minimum predicted confidence, so every finite,
positive depth pixel is eligible.

## 9. Put every point back in its correct place

For each valid pixel `(u, v)` on face `i`, Depth Anything's exporter performed
this inverse pinhole projection:

```text
p_pixel = [u, v, 1]^T
r_camera = inverse(K_i) * p_pixel
p_camera = depth_i(u, v) * r_camera
p_world  = inverse(E_world_to_camera_i) * [p_camera, 1]^T
```

`K_i` is the rescaled 504 × 504 virtual-camera intrinsic matrix.
`E_world_to_camera_i` is the known face rotation with zero translation.
Consequently:

- front depths travel along front-camera rays;
- right, back, and left depths rotate into their correct walls;
- ceiling depths rotate upward;
- floor depths rotate downward;
- all six sets meet around one shared virtual camera center.

The RGB color for each point came from the same processed face pixel. Points
from all six faces were concatenated. If more than 800,000 survived, the
exporter randomly sampled 800,000 without replacement. The sample is therefore
not byte-for-byte deterministic across reruns, although the reconstruction
method and geometry are the same.

Finally, the exporter oriented the scene for glTF: it aligned to the first
(front) camera, flipped CV Y and Z to glTF Y-up/Z-backward, and translated the
median point to the origin. This final rigid transform only changes how the
scene is displayed; it does not change where the six faces sit relative to one
another.

## 10. Exact result and artifacts

The successful depth/export pass took 14.224 seconds and produced:

| Result | Value |
|---|---:|
| Point count | 800,000 |
| Coordinate units | relative, not meters |
| Exact extent | 2.706529 × 2.444535 × 3.036901 relative units |
| Robust 1–99% view extent | 2.611275 × 2.255570 × 2.797834 |
| ARCore used | no |
| Panorama directions omitted | none |
| Confidence-filtered directions | none |

Artifacts:

- `out/capture-065a/feature-stitch-v4.png` — feature-only spherical stitch
- `out/capture-065a/feature-stitch-v4.json` — selected frames and overlap graph
- `out/capture-065a/feature-stitch-v5-wrapped.png` — closed 2:1 panorama
- `out/capture-065a/panorama-depth-anything-cubemap/view-*.jpg` — six faces
- `out/capture-065a/panorama-depth-anything-cubemap/virtual-views.jpg` — face sheet
- `out/capture-065a/panorama-depth-anything-cubemap/prediction-panorama-only.npz`
- `out/capture-065a/panorama-depth-anything-cubemap/scene.glb`
- `out/capture-065a/panorama-depth-anything-cubemap/cloud.bin`
- `out/capture-065a/panorama-depth-anything-cubemap/cloud.json`
- `out/capture-065a/panorama-depth-anything-cubemap/diagnostics.json`

## 11. What must remain true when this is repeated

1. Use only RGB images for this path. Do not silently mix ARCore poses or depth
   into it.
2. Reject weak feature links; one false graph edge can deform a whole region.
3. Use narrow single-owner seams instead of broad blending when the capture has
   parallax.
4. Make the wrapped panorama exactly 2:1 and horizontally continuous.
5. Render rectilinear pinhole faces; never treat equirectangular crops as camera
   photographs.
6. Use all six cube faces, including ceiling and floor.
7. Use square faces and exactly 90° horizontal and vertical FOV for a
   non-overlapping cube.
8. Keep all six virtual cameras at one optical center.
9. Rescale the virtual intrinsics whenever the depth model changes resolution.
10. Back-project with the exact same intrinsics and extrinsics used to create
    the virtual views.
11. Do not confidence-filter away whole weak surfaces when evaluating spatial
    coverage.
12. Treat the result as relative-scale geometry. This pipeline has no metric
    reference.

The remaining limitations are upstream of the cubemap: panorama stitching
flattens any capture parallax, and the pole wrapper extrapolates where the
original photos did not see. The six-face stage now preserves every direction
present in that wrapped panorama, but it cannot recover perspectives or detail
that were already absent before the panorama was made.
