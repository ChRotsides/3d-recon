# Handoff — capture alignment and latest reconstruction, 2026-08-19

Read this section first. The older 2026-08-10 investigation is preserved below
because its measurements still explain several failed approaches, but the active
capture, app and next task have changed.

## 0. Current objective and user assessment

The product goal remains a room point cloud and mesh, followed by SAM-based
furniture separation and automatic healing of walls, floor and ceiling. The
immediate problem is narrower: **frame alignment is not happening correctly**.
This is the next session's top priority. Do not move on to SAM, surface healing,
or mesh tuning until alignment is fixed and visually validated.

The newest reconstruction is **better according to the user**. Their visual
assessment is important and supersedes the initial component-count reading:

> Frames taken at position 1 align well with each other. Frames taken at
> position 2 align well with each other. Position 1 and position 2 do not align
> with one another.

Start by testing a two-rigid-group registration model. Do **not** disturb any
working within-position alignment with another unconstrained per-frame pose
optimization; first identify which frames actually form coherent groups.

The separate position renders later showed that the problem may not be limited
to one clean position-1/position-2 transform: the provisional second-position
subset still contains several components. Therefore the first next-session task
is to establish the real frame groups and measure which frame-to-frame links are
wrong. The frame-12 boundary is only a hypothesis, not a solution.

The user stopped for the night and was not sure exactly where the position
switch occurs. Confirm it if possible; otherwise infer it carefully from the
contact sheet and pose discontinuities described below.

## 1. Latest completed capture and outputs

Newest complete phone bundle:

```text
data/captures/bundle_fd1b39da-ba01-4dee-8616-da5ae329da4b
```

- Captured 2026-08-19 around 23:40 device time.
- 38 keyframes, 38 measured ARCore depth maps.
- All keyframes say `tracking_state: normal`.
- No skipped shots and no floor outline.
- Newer directories on the phone had no `bundle.json`; they are interrupted
  sessions and were deliberately ignored.
- A numbered image overview is at
  `out/capture-fd1b/contact-sheet.jpg`.

Build output:

```text
out/capture-fd1b/cloud.bin             736,844 points, 2 cm voxels
out/capture-fd1b/cloud.json            extent 5.91 x 5.63 x 7.28 m
out/capture-fd1b/mesh.ply              full mesh, 1,270,819 triangles
out/capture-fd1b/mesh.bin              web mesh, 599,999 triangles
out/capture-fd1b/mesh.json             extent 5.90 x 5.62 x 7.26 m
out/capture-fd1b/depth-arcore.npz      all 38 measured depth maps
out/capture-fd1b/pose-refine.json      rejected refinement diagnostics
out/capture-fd1b/mesh-comparison.png   previous best on top, latest on bottom
```

Separate diagnostic builds were also made using the leading frame-12 split
hypothesis (this is still an inference, not confirmed capture metadata):

```text
data/captures/bundle_fd1b39da-pos1    frames 0-11, 12 RGB-D keyframes
data/captures/bundle_fd1b39da-pos2    frames 12-37, 26 RGB-D keyframes
out/capture-fd1b/pos1/                309,709 cloud points; 532,469 mesh triangles
out/capture-fd1b/pos2/                431,596 cloud points; 766,786 full mesh triangles
out/capture-fd1b/positions-comparison.png
```

The position-1 mesh has 96.5% of its triangles in its largest component. Under
this split, position 2 still has 9 components and only 47.0% in the largest.
That is evidence the frame-12 split may not match the user's actual two groups;
use the separate viewers to identify the boundary rather than treating it as
settled. `scripts/capture/subset_bundle.py` was added to make alternate splits
cheap and reproducible without editing the source bundle.

An existing Python server has repository root as its working directory and is
listening on port 8080. On the local network:

```text
http://192.168.0.226:8080/out/capture-fd1b/cloud.html
http://192.168.0.226:8080/out/capture-fd1b/mesh.html
http://192.168.0.226:8080/out/capture-fd1b/pos1/cloud.html
http://192.168.0.226:8080/out/capture-fd1b/pos1/mesh.html
http://192.168.0.226:8080/out/capture-fd1b/pos2/cloud.html
http://192.168.0.226:8080/out/capture-fd1b/pos2/mesh.html
```

The previous best remains under `out/capture-065a/refined/` and came from
`bundle_065a6e7a-e8e3-449f-aa0d-61fc540d6e41-refined`.

## 2. What pose refinement found

The existing SIFT + ARCore-depth pose graph was run on all 38 frames. It found
59 metric edges spanning a 30/38-frame component, but correctly rejected the
solution:

```text
held-out median: 4.2 cm -> 8.1 cm
component: 30/38 frames
edges: 59
REJECTED: no refined bundle written
```

This failure now makes sense: the optimizer assigned a separate correction to
each frame when the observed error is mostly one transform between two already
coherent groups. The raw bundle was therefore used for the cloud and mesh.

Potential position boundary evidence:

- `kf_0011 -> kf_0012`: camera translation jumps **0.452 m**.
- `kf_0032 -> kf_0033`: another **0.560 m** jump.
- Frames 0-11 occupy a tight early trajectory; frame 12 begins the first large
  offset. This made frame 12 the leading hypothesis, but it is **not confirmed**.
- The current scripted center pass is 36 frames, so do not blindly assume
  frames 36-37 are the only second-position images. The user's actual movement
  and manual shutter use matter more than script phase, and shot phase is not
  persisted per keyframe.

Useful cross-frame clues from `pose-refine.json` include strong local edges and
several cross-boundary candidates. Recompute the actual `PoseEdge` matrices if
needed; the diagnostics store statistics, not `relative_ij` itself.

## 3. Recommended next experiment

**Priority: fix frame alignment.** The latest capture has better image overlap,
but the reconstruction still places some frames or frame groups incorrectly.
Treat the current cloud and mesh as diagnostics, not a successful aligned result.

1. Establish the position split. Ask how many shots were taken at position 1,
   or inspect `contact-sheet.jpg`, keyframe timestamps and pose jumps. Do not
   choose a split solely because it makes a metric improve.
2. Freeze every pose in group 1.
3. Preserve all relative poses inside group 2 and solve exactly one world-space
   SE(3) correction `C`, applying `T2' = C @ T2` to every group-2 pose.
4. Estimate `C` from cross-position SIFT correspondences lifted through each
   frame's measured ARCore depth. For an edge with
   `E = camera_i <- camera_j`, a group correction candidate is
   `C = T_i @ E @ inv(T_j)` for `i` in group 1 and `j` in group 2.
5. Robustly combine several cross-group candidates. Reject planar/degenerate
   edges, reserve held-out correspondences, and compare against a group-cloud
   global registration or constrained point-to-plane ICP only as a secondary
   check. Plain nearest-neighbour ICP is blind to sliding along walls; see the
   older findings below.
6. Write a sibling bundle rather than editing the raw capture. Rebuild cloud and
   mesh, then compare both visually and with held-out cross-group texture error.

Success means the two position reconstructions coincide while their internal
geometry remains bit-for-bit rigid. A lower ICP residual alone is not evidence.

## 4. Commands used for this build

```bash
# Pull newest completed bundle
adb -s 192.168.0.40:43369 pull \
  /sdcard/Android/data/com.recon.poselogger/files/captures/bundle_fd1b39da-ba01-4dee-8616-da5ae329da4b \
  data/captures/

# Attempt pose refinement; rejected, so output bundle was not created
PYTHONPATH=harness uv run python -u scripts/capture/refine_poses.py \
  data/captures/bundle_fd1b39da-ba01-4dee-8616-da5ae329da4b \
  --output data/captures/bundle_fd1b39da-ba01-4dee-8616-da5ae329da4b-refined \
  --diagnostics out/capture-fd1b/pose-refine.json

# Build from measured ARCore depth and original poses
PYTHONPATH=harness uv run python scripts/capture/arcore_depth_cache.py \
  data/captures/bundle_fd1b39da-ba01-4dee-8616-da5ae329da4b \
  out/capture-fd1b/depth-arcore.npz

PYTHONPATH=harness uv run python scripts/capture/da3_cloud.py \
  data/captures/bundle_fd1b39da-ba01-4dee-8616-da5ae329da4b \
  out/capture-fd1b/cloud.bin small 900000 \
  --depth-cache out/capture-fd1b/depth-arcore.npz --segments none

PYTHONPATH=harness uv run python scripts/capture/mesh.py \
  data/captures/bundle_fd1b39da-ba01-4dee-8616-da5ae329da4b \
  out/capture-fd1b/depth-arcore.npz out/capture-fd1b/mesh.ply
```

`--segments none` was intentional: the default `out/segments.npz` belongs to an
older capture and must not be silently applied to these frame IDs.

## 5. Android app state

The current debug APK is installed on the Samsung S23 Ultra. Wireless ADB was
`192.168.0.40:43369`; this port can change. Package/activity:
`com.recon.poselogger/.MainActivity`.

Capture UX now has:

- depth-warped previous-frame ghost and automatic capture once alignment is
  stable;
- candidate JPEG encoding off the render thread (observed around 59-60 fps);
- one prioritized HUD instruction at a time;
- no coverage cloud, diagnostic bars, generic nags or persistent station arrow;
- target square only for aim/hold, station ring only for move, depth ghost only
  for alignment, so these cues no longer compete.

The shot route was also changed after the user reported insufficient overlap:

- center pass: 12 yaw headings, 30-degree spacing, 36 shots;
- station pass: 10 offset headings, 20 shots;
- vertical serpentine order avoids the old ceiling-to-floor diagonal jump;
- regression test caps consecutive targets at 37 degrees (old worst case was
  about 78 degrees).

Relevant modified/new files include:

```text
core/src/script.rs
core/src/lib.rs
clients/android/app/src/main/java/com/recon/poselogger/AlignmentGuide.kt
clients/android/app/src/main/java/com/recon/poselogger/AlignmentGhostRenderer.kt
clients/android/app/src/main/java/com/recon/poselogger/CaptureRecorder.kt
clients/android/app/src/main/java/com/recon/poselogger/GuidanceHud.kt
clients/android/app/src/main/java/com/recon/poselogger/MainActivity.kt
clients/android/app/src/test/java/com/recon/poselogger/AlignmentGuideTest.kt
clients/android/app/src/main/jniLibs/*/librecon_core.so
```

Tests last run successfully: 90 Rust core tests and the Android unit-test/debug
build. The worktree already contains substantial user/pre-existing changes;
**do not reset or discard unrelated files**.

---

# Historical handoff — point-cloud quality, 2026-08-10

One session's work on "clean up the point cloud". Three approaches were built and
**two were backed out**. The lasting value is in the measurements, not the code:
they explain why the obvious fixes do not work, and they point at the one that
should. Read §1 and §2 before writing anything.

Subject capture throughout: `data/captures/bundle_e4a5da81-4e98-45a5-86e6-73a68f7ef5f2`
— one room, Samsung S23 Ultra, 27 keyframes, ARCore depth on every frame.

---

## 1. The finding that matters

**The frames disagree with each other by ~21 cm, and every geometric method only
ever measured the ~7.5 cm of it that points perpendicular to surfaces.**

| measured on the same 27 frames | |
|---|---|
| nearest-neighbour distance between overlapping frames | **7.5–9.6 cm** |
| distance between SIFT-matched *texture* points | **21.3 cm** |

Nearest-neighbour distance is blind to sliding: a wall shifted 20 cm sideways
along its own plane still has a neighbour a millimetre away for every point. A
matched texture point is not blind to it, because it names *which* piece of wall.

Consequences, all of them confirmed the hard way below:

- Any method that minimises nearest-neighbour distance is optimising a metric
  that ignores most of the error. That is why plane fitting, per-frame ICP and
  per-segment ICP all stalled at 3–5 cm while the cloud still looked doubled.
- **Do not quote a residual as evidence when the method minimises that residual.**
  This was done twice this session and was wrong both times (§3).
- The measurement was run on **ARCore's own metric depth**, not DA3 output, so it
  is not a monocular-model error. It is placement.

## 2. What to do next

**Refine the poses against cross-frame feature correspondences.** This is the only
constraint identified that can see the in-plane error, and it is a well-posed
problem (27 poses × 6 DoF against 1,806 correspondences), unlike the per-segment
free-for-all that failed.

Two ways to get the correspondences, and they are not exclusive:

1. **Offline, works today.** SIFT + ratio test + `findFundamentalMat` RANSAC over
   keyframe pairs already yields **30 matched pairs and 1,806 verified 3-D
   correspondences** on this capture. OpenCV is already a dependency. Caveat: only
   ~204 keypoints per frame — this room is largely textureless — so expect
   coverage to be patchy and weight accordingly.
2. **In the client, better.** `sparse_points` in the bundle has 234 entries and
   **every one has exactly one observation**, so there is no correspondence in the
   capture at all. The schema already supports multi-frame `observations`. ARCore's
   point cloud exposes persistent point IDs; recording them would give real
   correspondences for free and remove the need for SIFT.

Sanity check before building: whatever refinement is fitted, score it with the
**texture-match disagreement**, not with nearest-neighbour distance.

## 3. What was tried and removed — do not repeat

### 3a. Frame-aware voxel cleanup was inert (kept, but it does nothing)

`recon/cloud.py::frame_aware_voxel_filter` removed 1.3% of cells. Its own manifest
says why:

```
frame_voxel_samples 1,566,460 / occupied_voxels 1,525,996 = 1.027 frames per cell
low_support_kept_by_neighbours = 96% of all cells
```

96% of voxels hold exactly one frame, so frame voting has nothing to count and the
cross-frame spread test fired on **one cell in 1.5M** — a cell with one sample has
zero spread by construction. Disagreement larger than a voxel does not appear as
spread *inside* a cell; it appears as two separate cells.

### 3b. Plane detection and snapping — built, made it visibly worse, removed

Found consensus planes, fitted a rigid correction per frame, snapped planar points.
`recon/planes.py` and its tests were deleted. Three lessons:

- **A plane needs spatial extent.** Membership was decided by normal direction and
  offset alone, so a plane claimed every cell in the room sharing them. 87 of 128
  planes had footprint fill under 20%; one was 3,877 cells scattered over
  13.2 × 8.5 m at 3.1% fill; eight horizontal planes between them flattened every
  table top, chair seat and shelf onto a few global heights. A local-occupancy gate
  raised median fill 12% → 40% and was still not enough.
- **"Plane thickness 17.0 → 2.6 cm" was circular** — snapping points onto planes
  reduces plane thickness by construction.
- Detection details that did work, if anyone rebuilds this: normals must be
  estimated **per frame** (on the merged cloud, a 10 cm smear inside an 8 cm
  neighbourhood is not planar to PCA and nothing is detected); and a wall is a
  **peak** in the offset histogram, not a run of occupied bins — in a furnished room
  every direction is occupied continuously by floor, table, shelf and ceiling.

### 3c. TSDF fusion — built, worked on its own terms, rejected on sight

`recon/fusion.py` + `scripts/capture/tsdf_cloud.py`, deleted. 35.2M voxels at 3 cm,
truncation 15 cm, 80 s on CPU, **569,329 surface points at 2.22 frames per voxel**
against the point cloud's 1.027 — it genuinely fused. Single-surface output is
structural there, not tuned. User assessment: "fucked it up completely".

Worth keeping from it: with truncation *below* the disagreement you do not get two
sheets — the later frame's free space carves the earlier surface away entirely and
the answer depends on integration order.

## 4. Current pipeline and state

```
capture -> DA3 depth -> SAM segments -> segment alignment -> voxel fuse -> viewer
```

63 tests pass, ruff clean.

### In the tree (new/modified this session)

| path | state |
|---|---|
| `harness/recon/segments.py` | **new** — SAM mask compositing + per-segment alignment |
| `harness/tests/test_segments.py` | **new** — 10 tests |
| `scripts/capture/sam_masks.py` | **new** — segments a bundle, caches label maps |
| `scripts/capture/da3_cloud.py` | `--segments SEG.npz\|none`; uses `out/segments.npz` automatically |
| `harness/recon/schemas.py` | `Keyframe.heading_deg` (optional) |
| `clients/android/.../SessionBundle.kt` | `KeyframeWire.heading_deg` |
| `clients/android/.../CaptureRecorder.kt` | stamps latest compass reading onto each keyframe |
| `viewer/cloud.html` | slice / ceiling / far-trim / density levers |
| `pyproject.toml` | `scipy>=1.11` (KD-tree, already in both envs) |

### Segment alignment — honest status

```
27 keyframes segmented, 697 segments, 81% coverage, ~5 min on the TITANs
164 aligned · 72 without a partner · 320 REJECTED as too large
cells containing more than one frame: 4.0% -> 10.2%
```

**320 of 556 segments are still rejected**, and the user's verdict on the result is
"about the same". It is kept in the pipeline at the user's request, not because it
is demonstrated to work.

The reason for the rejections is §1: rigid ICP on a planar patch is degenerate —
it can slide within its own plane and spin about its normal at near-zero residual.
Mitigated by anchoring each fit to its starting position, scaled per segment by
degeneracy (full hold for a flat patch, none for a couch, which constrains all six
DoF itself). That took aligned segments 90 → 164. It is a workaround; the real fix
is the texture constraint in §2.

**The 2.4 cm "per-segment ceiling" measured earlier is optimistic** for the same
reason — it was nearest-neighbour residual, which a sliding patch minimises.

### Magnetometer

Recorded from this build onward; **needs a client rebuild and a new capture**,
nothing retrofits. Per keyframe, not per session: indoors the field is bent tens of
degrees by steel and wiring, so no single reading is a true bearing, but two
readings in one room share that distortion and their difference survives it. Uses
the existing `TYPE_ROTATION_VECTOR` (fused, gravity-aligned) that `Compass.kt`
already reads, not a raw magnetometer, which swings too much to read hand-held.

Temper expectations: for a 2-minute single-room sweep ARCore's visual-inertial yaw
drift is small, so this should matter more for long sessions and multi-room
stitching than for this capture.

Also audited, and **not** additional information: `gravity` is `[0,-1,0]`, already
applied by ARCore; pan/tilt are already fused inside the poses.

## 5. Environment and commands

```bash
source ~/ai/env.sh kepler      # GPU env (custom sm_35 torch) + HF_HOME
source ~/ai/env.sh cpu         # everything that does not need the GPU
```

- **Never `pip install` into `envs/kepler-gpu`.** No pip is present on purpose;
  use `uv pip install --python ~/ai/envs/kepler-gpu/bin/python --no-deps`. A stock
  torch wheel loses the Kepler cards and rebuilding is a multi-hour compile.
- GPUs are compute capability **3.5**: force fp32 and `attn_implementation="eager"`.
  `recon/torch_compat.py` has `cuda_requires_fp32` / `configure_cuda_for_legacy_device`.
- SAM runs fine on these cards: `facebook/sam-vit-base`, 94M params, ~10 s/frame
  for a 32×32 automatic prompt grid. `facebook/sam3` is **gated** and needs the
  licence accepted plus an `HF_TOKEN`; SAM 1 is the generation with automatic
  "segment everything" mode, which is what per-object work needs anyway.

```bash
# segment a bundle (cached; GPU)
python scripts/capture/sam_masks.py data/captures/bundle_<uuid> out/segments.npz 32

# build a cloud (picks up out/segments.npz automatically)
python scripts/capture/da3_cloud.py data/captures/bundle_<uuid> \
    out/seg/cloud.bin large 1500000 depth 0.02 --depth-cache out/depth-large.npz

# baseline, no segments
... --segments none

# serve the viewers
cd out && python -m http.server 8781 --bind 127.0.0.1
```

Caches that save a lot of time: `out/depth-large.npz` (DA3-large), `out/depth-arcore.npz`
(ARCore measured depth — **the reference for any alignment claim**), `out/segments.npz`.

## 6. Viewer levers

`viewer/cloud.html`, all client-side and all *hiding* points rather than moving
them, so nothing on screen can misrepresent the file:

- **slice** — axis (X / Y-plan / Z) + position + thickness. A thin plan cut turns
  the room into a floor plan and makes a doubled wall two measurable lines.
  **Arrow keys sweep it** — the fastest way to scan for doubling.
- **ceiling** — hide above a height, to look down into the room.
- **far trim** — hide beyond a radius; the far-1% trim, made interactive.
- **density** — stable random fraction, for seeing through the cloud.
- `t` texture · `c` clear cuts · `r` reset.

## 7. Method notes

Measurements this session were scratch scripts, now gone. The two worth being able
to redo:

- **Pairwise geometric disagreement.** For frame pairs sharing ≥150 cells of a
  32 cm grid, restrict A's points to shared cells, KD-tree nearest neighbour into
  B, take the median. Restricting to shared cells is essential — unrestricted ICP
  over partial overlap drags frames 20–40 cm to match geometry that is not shared,
  which produced one badly wrong "ceiling" number before it was caught.
- **Texture disagreement (§1).** SIFT (`nfeatures=4000`), ratio test at 0.75,
  `findFundamentalMat` RANSAC at 3 px, back-project both matched pixels through
  their own depth and pose, take the distance between the resulting 3-D points.

Ask before trusting a number: *does the method being scored minimise this quantity?*
If yes, it is not evidence.
