# What is wrong with the first video reconstruction, measured

Reference run: recording `rec_20260903-070045` (67 s walk), output
`out/video/rec_20260903-070045-dense/`. Symptoms the user saw in the viewer:
many points out of place, the floor in disconnected pieces, walls standing in
the middle of the room. Everything below is measured by
`video-recon/scripts/diagnose.py` on that output; the two figures it draws are
next to this file.

The short version: **the camera poses are good; the depth maps are not, and
fusion trusts them blindly.** Position and orientation from COLMAP agree with
the phone's rotation sensors to 2.7° median over 430 frames. The Depth Anything
maps that are hung on those poses disagree with each other by 6–20 % across the
room and with the reconstruction's own sparse points in shape, and the TSDF
fuses every disagreeing copy of every wall.

## 1. Per-frame depth is inconsistent — this is the doubled walls

*Measurement.* Project each keyframe's depth into a keyframe 20 later and
compare with that frame's own depth at the same pixel.

| keyframes apart | median disagreement | within 5 % | worse than 25 % |
| --- | --- | --- | --- |
| 1 | 1.4 % | 85 % | 1.6 % |
| 5 | 4.6 % | 52 % | 6.8 % |
| 20 | 6.4 % | 45 % | 17 % |

Neighbouring frames (same DA3 window) agree; frames a few seconds apart do
not. At 3 m, 6.4 % is 19 cm — larger than the 12 cm TSDF truncation band, so
the same wall seen twice becomes two walls. 17 % of pixels are off by more
than a quarter of their depth: those are the "points out of place".

*Why.* On the GTX TITAN the adapter runs DA3 in **windows of 4 frames**
(FP32 attention memory), so multi-view consistency only exists inside groups
of four; each window is then scaled to the sparse points by **one scalar per
frame** and nothing ties windows together. Peak VRAM was 0.7 GB of 6 GB, so
the clamp is far more conservative than the card requires (tested below).

## 2. The floor sits at a different height in different frames — the split floor

*Measurement.* For each keyframe, back-project its own depth map and find the
floor plane it implies (figure `findings-2026-09-03-per-frame.png`, top).

| | |
| --- | --- |
| floor height, p10–p90 spread across frames | **0.91 m** |
| standard deviation (with blow-ups) | 1.49 m |
| camera height above the floor the *depth* implies | 1.05 m |
| camera height above the floor the *sparse points* imply | 1.50 m (assumed) |

The floor moves in plateaus of 30–80 frames. Looking at single frames shows
the mechanism: in keyframe 10 the depth map's lowest band sits exactly on the
sparse floor (−1.44 m below the camera); in keyframes 100, 200 and 350 the
depth maps contain **no points below −0.8 to −1.0 m at all** although the
sparse floor is at −1.5 m. At the sparse points themselves the depth agrees
with COLMAP (ratio 0.99 on floor points, 1.00 on walls), so where COLMAP
found floor features the scalar anchor pulls the depth right, and where the
floor is bare tiles with no features the model's floor stays half a metre too
near. The error is spatially varying *within* a frame; one scale per frame
cannot represent it, and fusion stacks the resulting floors.

Three frames put the floor 3–22 m below the camera: depth blow-ups through
the window and on the TV screen, which fusion then paints as geometry outside
the room (the mass below −2 m and above +2.5 m in the side elevation).

## 3. The depth model compresses distance — near/far shape error

*Measurement.* At the sparse points each frame observes, the ratio
predicted/true depth for the nearest third versus the farthest third of
points.

| | |
| --- | --- |
| residual scale after anchoring, median | 0.999 (the anchor does its job) |
| within-frame dispersion of the ratio | 6.6 % |
| near→far slope of log(ratio) vs log(depth), median | **−0.36** |

A slope of −0.36 means predicted depth grows like true depth^0.64: a wall at
4 m is placed as if at ~3.3 m while a chair at 1 m is right. One scalar per
frame cannot fix that; a wall's position then depends on how far the camera
was from it when it was seen — again, doubled walls.

*Suspected causes* (tested below): the depth model was given the frames
**sideways** (sensor-native landscape with world-up to the left, which breaks
every indoor prior a monocular network has); a **5.6 % focal length mismatch**
(§4); the smallest checkpoint.

## 4. A plain bug: wrong focal length in the bundle

COLMAP refined the focal length from the calibration prior of 1065 px to
1008 px and solved every pose with 1008. The bundle handed to the depth model
and to fusion carries **1065** (`pipeline.py` wrote the keyframe intrinsics,
not `SfmResult.intrinsics`). Every depth map is therefore back-projected 5.6 %
too narrow, and the model was conditioned on inconsistent cameras.
Fixed in the pipeline; effect measured below.

## 5. Orientation drift along the walk, no loop closure

Sensor-vs-SfM orientation residual by thirds of the walk: 2.2°, 3.0°, 3.4°;
a 5–6° bump at 25–35 s and up to 14° in the last five seconds
(figure, bottom). The recording did not end on its starting view, so COLMAP
had nothing to close the loop with, and the match matrix shows no
first↔last block. Drift of a few degrees over 7 m of path is a few decimetres
of wall offset at the far end of the walk.

## 6. Fusion trusts everything

`fuse.py` integrates every depth pixel of every frame with unit weight up to
6 m. There is no confidence weighting (the adapter returns DA3's confidence;
the pipeline discards it), no cross-view consistency test before integration,
no per-frame depth cap from the sparse points, and the exported point cloud
is the raw TSDF surface with no debris removal. The archived pipeline
(`archive/harness/recon/fusion/tsdf.py`) had all four; none survived into the
new path.

## 7. Absolute scale is an assumption

Scale comes from the floor under the camera with an assumed 1.5 m phone
height. DA3's metric checkpoint, run pose-free on eight keyframes, said
0.47–0.61 m per unit (median 0.61) against the floor's 0.477. The shape is
unaffected, but every metre in the output is ±25 % until a known height or
dimension is supplied.

## What the experiments changed

(filled in from `exp_depth.py`; each variant re-predicts depth for all 430
keyframes and re-runs the same diagnostics)

| variant | floor spread p10–p90 | near/far slope | within-frame dispersion | 20-frame disagreement, median | 20-frame, worse than 25 % | DA3 time | VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (sideways, prior focal, window 4) | 0.91 m | −0.36 | 6.6 % | 6.4 % | 17.1 % | 109 s | 0.7 GB |
| + refined focal (fixK) | 0.93 m | −0.38 | 6.5 % | 6.5 % | 17.5 % | 174 s | 0.7 GB |
| + upright frames | 0.84 m | −0.18 | 4.2 % | 3.8 % | 8.6 % | 169 s | 0.8 GB |
| + upright, window 8 | 0.84 m | −0.17 | 4.0 % | 4.1 % | 6.7 % | 179 s | 2.2 GB |
| + upright, window 12 | 0.82 m | −0.16 | 3.8 % | 3.8 % | 6.3 % | 201 s | 4.6 GB |
| + upright, DA3-base, window 4 | 0.85 m | −0.13 | 3.5 % | 2.6 % | 5.2 % | 298 s | 1.7 GB |

Reading it:

* **Upright input is the single biggest win**: shape error halved, gross
  disagreement halved. It costs nothing. Adopted as the pipeline default.
* **Wider windows** help the gross-outlier rate (17 % → 6 %) but not the
  floor; the Kepler clamp to 4 was unnecessary (12 frames fit in 4.6 GB).
  Adopted: window 8 by default.
* **DA3-base** is the best of the lot at every metric except the floor:
  20-frame disagreement 2.6 %, gross outliers 5.2 %, for 5 minutes on this
  card. Adopted as the default checkpoint (Apache licence).
* **The refined focal** was a real bug with no measurable effect on these
  metrics — it scales all depth maps alike. Fixed anyway.
* **Nothing above touches the floor spread.** That needs an anchor that can
  vary across the image (fix 5) or a depth source that is not monocular.

Fusion changes made after these measurements, in `fuse.py`: a pixel is kept
only if at least one keyframe 3, 6 or 12 frames away sees a surface within
10 % of it; each frame's depth is capped at 1.3× the farthest sparse point it
observes (the window and the TV screen go away); the least-confident 20 % of
each frame's pixels are dropped when the cache carries DA3's confidence.
Effect on the reference walk, full pipeline before → after (refined focal,
upright frames, DA3-base, 8-frame windows, consistency fusion):

| | before | after |
| --- | --- | --- |
| near/far slope | −0.36 | −0.13 |
| within-frame dispersion at sparse points | 6.6 % | 3.2 % |
| 20-keyframe disagreement, median | 6.4 % | 2.7 % |
| 20-keyframe, worse than 25 % | 17.1 % | 5.5 % |
| floor height spread p10–p90 | 0.91 m | 0.83 m |
| cloud extent (m) | 7.4 × 6.9 × 8.2 | 7.0 × 2.8 × 6.1 |
| cloud points | 787 k | 332 k |
| pixels removed by the consistency test | — | 5.8 % |
| pipeline time after SfM | 4.5 min | 10 min |

The vertical extent going from 6.9 m to 2.8 m is the geometry outside the
room disappearing: the side elevation is now bounded by the floor at −1.5 m
and the ceiling at +1.2 m (`findings-2026-09-03-slices-after.png`). Walls in
the top-down slice are thinner and fewer, but not single lines yet, and a
second floor band remains at −1.0 m: that is the spatially varying floor
error of §2, which none of these changes addresses.

Not landed: letting COLMAP fit lens distortion. Re-mapping the existing
matches with the OPENCV model (radial + tangential) did not converge
("matrix not positive definite" in every bundle adjustment after the first
few hundred registrations); the pipeline now asks for the RADIAL model
(k1, k2 only) on fresh runs and undistorts keyframes to a pinhole before depth,
but that path has not yet been exercised on this recording.

## Ranked fixes

1. **Feed the depth model upright frames** (rotate images, intrinsics and
   poses by the sensor orientation before depth).
2. **Use the refined COLMAP intrinsics** in the bundle (done).
3. **Bigger DA3 windows** on this card (the clamp of 4 is not what the VRAM
   says), and the `base` checkpoint if it fits; on the H100 use `large` with
   16-frame windows.
4. **Fusion that doubts**: weight by DA3 confidence, keep a pixel only when a
   neighbouring keyframe agrees within 10 %, cap each frame's depth at 1.3× the
   farthest sparse point it sees, and reject frames whose implied floor is
   absurd. Then clean the cloud by connected components as the mesh already is.
5. **Per-frame affine anchor** (scale *and* offset against sparse points, as
   the harness already does against ARCore depth) instead of one scalar, to
   take the near/far slope out — if upright input does not remove it.
6. **Close the loop** when recording (end on the starting view) and add the
   sensors' orientation as a soft prior in a final pose-graph refinement so
   drift cannot exceed what the gyroscope allows.
7. **Pin the scale** with the real phone height or one measured dimension.


---

# Round 2 — what is still wrong after the fixes (measured on the corrected run)

## R1. The floor error is a per-frame plateau, not a gradient

Per pixel whose ray hits the known floor plane, predicted depth ÷ distance to
the plane, by distance:

| true distance to floor | median ratio | p25 | p75 |
| --- | --- | --- | --- |
| 0.5–1.5 m | 1.00 | 0.81 | 1.01 |
| 1.5–2.5 m | 0.68 | 0.66 | 0.97 |
| 2.5–3.5 m | 0.68 | 0.65 | 0.96 |
| 3.5–5 m | 0.68 | 0.64 | 0.85 |
| 5–8 m | 0.68 | 0.64 | 0.85 |

The ratio is not distance-dependent, so it is not depth compression. It is
**bimodal**: in roughly 40 % of frames the floor is right (ratio ≈ 0.97) and
in the rest it sits at 0.68 of its true distance — the model puts the floor
as if the camera were **1.0 m** above it instead of 1.5 m, uniformly across
that frame, while the walls in the same frame are anchored correctly. A
scalar or even an affine-per-frame anchor cannot express "walls right, floor
wrong". Only a correction that knows where the floor is can.

*Settled by the user:* 177 cm tall, phone at chest height, so the camera was
**1.30 m** above the floor. That rules out the depth model's 1.0 m and shows
the 1.5 m default was 15 % high; 1.30 m is now the default and the reference
run was rescaled (0.414 m per unit; room footprint 5.8 × 3.7 m, walk 17.7 m).
The sparse slab is therefore the real floor and the 0.68 plateau is the
model's error, as assumed above. (A CLI bug found on the way: `--camera-height`
and `--scale-source` were parsed but never passed to the pipeline; fixed, with
a test that every flag reaches `Options`.)

## R2. The remaining doubled walls come from pose drift, not from depth

Fit a plane to the floor band of the **sparse** points (SfM only, no depth
model): tilted **1.36°** from the sensors' gravity, 1.8 cm of height per metre
along the walk's main axis. The fused cloud's floor tilts by the same
1.29°. Depth cannot cause a tilt in COLMAP's own points; this is the
reconstruction slowly rotating along the walk, i.e. drift with no loop
closure, which the sensor-residual plot showed rising to 5–14° at the end.
Over the 7 m walk 1.3° is 16 cm of height and, at 3 m range, 7 cm of lateral
wall offset from tilt alone — plus whatever the end-of-walk 5–14° adds.

The sensors already know every frame's pitch and roll to about a degree, and
COLMAP never uses that. The fix is a post-SfM refinement with the gravity
direction as a per-frame prior (a pose graph with the sequential relative
poses from COLMAP plus a soft absolute-orientation term per frame), which
takes the rotational drift out; position drift then needs a loop closure in
the recording.

## R3. Motion blur is a first-order factor

5-keyframe depth disagreement, sharpest quarter of keyframes vs blurriest
quarter: **1.2 % vs 5.1 %**. The five worst keyframes (disagreement 7.5–8.9 %)
have blur scores 14–21 against a median of 341 — they are smeared and should
never have been chosen. The keyframe selector takes the sharpest of ±3
frames but has no absolute floor. Fix: reject candidates below a fraction of
the running median blur and widen the window during fast turns; and, at
recording time, slow down.

## R4. The mesh is one piece; coverage is the gap

630 k triangles in 4 components, 98.5 % in the largest (94 m² of surface).
Fragmentation is solved. Of horizontal surface, 28 % is at floor level and
40 % at ceiling level; the ceiling and the blank walls are where the depth
model has no anchors and the sparse cloud has no points, so their placement
is the model's guess. A ~7 × 6 m room with a 2.7 m ceiling has ~150 m² of
surface; 94 m² fused means about a third is missing or collapsed.

## What "production ready" is still missing, beyond the geometry

* **Runtime**: 65 of the 80 minutes are CPU SIFT matching of 94 k pairs.
  GPU SIFT or a vocabulary tree brings that to minutes; on the H100 the whole
  run is under ten.
* **Scale** is an assumption (§7) — needs the phone height or a measured
  object, or a metric depth model trusted enough to set it.
* **No loop closure** and no use of the sensors after alignment (R2).
* **No ground truth**. Every number above is self-consistency. One room with
  three tape-measured distances would turn "better" into "how far off".
* **Failure handling**: the pipeline stops on a failed SfM with a rotation-
  only bundle and no depth; there is no partial-model merging, no retry with
  different keyframe density, no report to the phone.
* **The app** has no pause, no storage check, no recording review, no
  guidance about speed or overlap while recording; it never tells the user
  that a stretch was too blurry to use.
* **The GPU**: the DA3 clamp, FP32 and small windows are Kepler-specific;
  large + 16-frame windows is the configuration the model was built for.

## Next work, in order

1. Gravity-prior pose refinement after COLMAP (R2). Removes the 1.3° tilt
   and most of the end-of-walk drift; all inputs already exist.
2. Floor-aware depth correction (R1): detect floor pixels by ray/plane
   geometry, snap the 0.68 plateau frames to the plane, re-fuse. Cheap; test
   against the ceiling-height answer first.
3. Blur gating in keyframe selection (R3). One line of policy.
4. Scale is now pinned by the camera height (±4 %); one measured distance in the room would confirm it.
5. Move depth to the H100 (large, 16-frame windows) and add COLMAP MVS
   there for textured surfaces; fuse both.
6. Record the next walk slower and ending on its starting view.
