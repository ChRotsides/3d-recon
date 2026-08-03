# Execution status

Companion to [`room-scanner-implementation-plan.md`](../room-scanner-implementation-plan.md),
which stays the canonical architecture. This file records what has been built,
what deviates from the plan and why, and what is blocked.

**Last updated:** 2026-08-04

---

## 1. Hardware constraint, discovered at kickoff

The development box has **2× GTX TITAN (Kepler, sm_35, 6 GB)**. PyTorch 2.x
dropped Kepler support, so DA3, MapAnything, SAM 3 and gsplat CUDA kernels
**cannot run here at all**. This is not a slow-GPU problem; the kernels will not
load.

What that changed:

- The GPU stages are implemented behind adapter interfaces (`DepthEstimator`,
  `Masker`, `SplatTrainer`) and run unchanged on a rented L40S. They are
  **written but unvalidated**, and are marked as such everywhere.
- To keep Phase 0 moving, the harness ships a **synthetic room generator**
  (`recon synth`) that emits a real `SessionBundle` with exact ground truth:
  poses, intrinsics, ray-traced metric depth, semantic masks, and the room's
  true wall lengths, area and ceiling height. Depth is exact to machine
  precision (asserted in `test_raycast_depth_is_exact`).
- Everything downstream of depth — masking, fusion, plane extraction, Atlanta
  regularization, floor plan, scale solving, metrics — is therefore **fully
  validated today**, against an oracle stronger than the ZED 2i rig the plan
  specifies.

**To unblock:** one L40S or A100 40 GB box. Per plan §9, a Phase-0 benchmark
sweep is a few dollars of GPU time.

### What the synthetic oracle cannot tell us

It is an oracle for *geometry processing*, not for *models*. It says nothing
about whether DA3 predicts good depth on a real white wall, whether SAM 3 finds
a real mirror, or whether PGSR converges. Those are exactly the Phase-0
questions the GPU box has to answer. The synthetic set derisks everything else
so that GPU time is spent only on what needs it.

---

## 2. Deviations from the plan

Each of these is a place where the plan as written would not work. None change
the architecture.

### 2.1 `w_dist` is asymmetric, not symmetric (plan §6.1)

The plan specifies `w_dist = clamp(1 - |d - d_opt| / d_range, 0, 1)` with
`d_opt ≈ 2.0 m`.

Taken literally this makes ceilings permanently uncoverable. A 2.45 m ceiling
viewed from a 1.5 m eye height is **0.95 m away** and scores 0.53, so it can
never reach the coverage bar — `CEILING_MISSING` would nag forever and
room-complete would never fire.

Depth is not less reliable at 1 m than at 2 m. Only very close range genuinely
hurts, by killing parallax and shrinking the per-frame footprint. So the weight
is flat from 0.8 m to `d_opt` and falls away only beyond it.
(`CoverageGrid::distance_weight`, `core/src/coverage.rs`.)

### 2.2 `q_min` is 1.8, not 2.0 (plan §6.1)

The plan starts `q_min` at 2.0, "roughly two decent views", and says to tune in
Phase 3. It cannot stay at 2.0: `w_angle`, `w_dist` and `w_blur` are each
strictly less than 1, so **no** pair of views can ever sum to 2.0 and every
voxel reads as uncovered forever. 1.8 gives two good views real headroom.

### 2.3 Room-complete measures gain against total, not remaining, volume (plan §6.1)

The plan's stop condition is "no candidate station offers gain above 2% of total
volume". Implementing that against *remaining* volume never terminates: as the
room fills in, the bar shrinks with it, so the last few unreachable corners keep
the capture open indefinitely. The denominator is the whole observed grid.
(`next_best_station`, `core/src/station.rs`.)

### 2.4 The footprint is not traced by intersecting wall lines (plan §6.4 step 6)

The plan says "intersect wall planes pairwise for corners, trace the room
polygon". Done naively that breaks on any non-convex room, and L-shapes and
alcoves are common. The implementation instead rasterises floor and wall points
into an occupancy grid, traces and simplifies the contour to get the *topology*
and corner count right, then **snaps each edge to its fitted wall plane** and
derives corners by intersecting consecutive snapped lines.

The metric result still comes from planes fitted to thousands of points, not
from the 5 cm raster — which is why wall-length error lands at 0.05–0.15%
rather than at grid resolution. (`recon/layout/floorplan.py`.)

### 2.5 The unmasked control arm degrades instead of crashing

Without masking, a room with a window produces a bounding box tens of metres
across and the TSDF volume exceeds 180 M voxels. Raising `MemoryError` is
arguably correct product behaviour, but it makes the benchmark's control arm
unmeasurable. `TSDFConfig.auto_coarsen` enlarges the voxel until the volume
fits, so the control arm produces *numbers* — which is how we can say masking is
worth 91 percentage points of wall error rather than merely asserting it.

---

## 3. Corrections to the plan's technology facts

Found while building; all verified. See [`licensing-audit.md`](licensing-audit.md).

| Plan says | Actually |
|---|---|
| "Use **DA3-Large** on the server" | DA3-LARGE is **CC BY-NC 4.0** in the upstream model zoo (the HF card says Apache — an unresolved contradiction). Not commercially usable until ByteDance clarifies. **DA3-BASE** and **DA3METRIC-LARGE** are unambiguously Apache 2.0 and must be in the Phase-0 grid. |
| "DA3-Small on device" | ✅ DA3-SMALL exists, 0.08 B, Apache 2.0. The on-device plan is fine. |
| "`DA3-Streaming`" | Not a checkpoint — an inference wrapper (`da3_streaming/`) in the same repo. Selected by a flag, not a variant name. |
| MapAnything is "Apache-licensed model" | Only `facebook/map-anything-apache` is. The default `facebook/map-anything` is **CC BY-NC 4.0**. Adapter now defaults to the Apache variant. |
| PGSR "may need reimplementation"; benchmark against 2DGS and GOF | **All three are non-commercial.** Clean-room PGSR-on-gsplat is the only commercial path, not a contingency. Budget it as committed Phase-2 work. |
| SAM 3 "verify commercial terms" | ✅ Verified: royalty-free, no MAU ceiling, commercial use permitted. Three obligations apply — see the audit §3. |

---

## 4. Decisions taken

- **ADR-0001 — native shells + shared Rust core**, per the plan's own
  recommendation. This is a product, not a three-week demo.
  ([`decisions/0001-client-framework.md`](decisions/0001-client-framework.md))
- **Bundle frame is normative and asserted.** Right-handed, Y-up, world←camera,
  camera down −Z. `Pose` rejects non-orthonormal and mirrored rotations at
  parse time; `BundleDir.load` refuses a non-canonical frame; the Android client
  refuses a pose with negative determinant. Four independent guards on plan §7's
  flipped-scene failure mode, because it is silent when it happens.

---

## 5. Component status

| Component | Status | Validated |
|---|---|---|
| `SessionBundle` / `HouseGraph` / `RoomLayout` schemas | done | 9 tests |
| Synthetic generator + ground-truth oracle | done | depth exact to 1e-9 |
| Ray caster, triangulation, polygon ops | done | 7 tests |
| Scale solver (§6.2) | done | 10 tests |
| TSDF fusion (§6.4.1) | done | end-to-end |
| Plane extraction + Atlanta (§6.4.2–4) | done | angled-wall test |
| Floor plan + openings (§6.4.5–6) | done | 0.05–0.15% wall error |
| Eval metrics + benchmark grid (§8) | done | 5 scenes × 2 maskers |
| Bundle validator | done | 5 tests |
| Rust guidance core (§6.1) | done | 20 tests |
| Room stitching solver (§6.3) | done | 29 tests |
| Room isolation in a connected house | done | ±0.1% area |
| Integrated stitch -> drift target | done | 12.6 cm on 6 rooms, see §7b |
| GLB / SPZ / PDF export | done | 18 tests |
| Android pose logger | **builds** | APK + 6 JVM tests |
| DA3 / MapAnything / SAM 3 adapters | written | ❌ needs GPU |
| PGSR clean-room trainer | **not written** | see §6 |
| iOS pose logger | source only | ❌ needs a Mac |
| Licensing audit | done | re-runnable |
| Cloud service (Phase 2) | not started | |


---

## 6. Why the PGSR trainer is a `NotImplementedError`

It would have been easy to write a plausible-looking training loop. It would
also have been untestable here, and its loss weights are guesses until something
has actually converged. A benchmark row filled in by never-executed code is
worse than a gap in the table.

What *is* settled and tested: the licence position, the config surface, the
held-out evaluation split (strided, not random — see
`test_holdout_split_is_strided_not_random`), the depth-seeded initialisation,
and the COLMAP export that lets the reference implementations run for
comparison. The training loop is the remaining piece and it needs a GPU to
iterate against.

---

## 7. Phase-0 results so far

`recon bench data/scenes` over five synthetic rooms, ground-truth depth,
ground-truth masks. Targets are plan §1.3: wall ≤1.5% or ≤3 cm, area ≤3%.

| scene | wall err | area err | ceiling err | chamfer | recall@5cm |
|---|---|---|---|---|---|
| empty_rect | 0.07% | +0.14% | +12 mm | 13.7 mm | 0.985 |
| l_shaped | 0.15% | +0.05% | +14 mm | 16.3 mm | 0.972 |
| angled | 0.09% | +0.15% | +13 mm | 13.8 mm | 0.982 |
| furnished | 0.06% | +0.12% | +11 mm | 42.2 mm | 0.796 |
| mirror_heavy | 0.06% | +0.10% | +11 mm | 50.2 mm | 0.716 |

**The value of masking**, same scenes with `mask=none`:

| scene | wall err masked | wall err unmasked |
|---|---|---|
| mirror_heavy | 0.06% | **91.0%** (area +51.7%) |
| l_shaped | 0.15% | **86.8%** |
| furnished | 0.06% | **31.4%** |
| angled | 0.09% | volume too large to fuse |
| empty_rect (no glass) | 0.07% | 0.07% ← control behaves |

This says the geometry pipeline is sound and that mask-before-fusion is
load-bearing exactly as plan §7 claims. It says **nothing yet** about whether
DA3 depth or SAM 3 masks are good enough on real rooms — that is what the GPU
box is for.

---

## 7a. Room isolation: what a room sees through its own doorways

A capture sees straight through an open doorway into the next room, and a
**denser capture leaks more, not less** — better coverage means seeing further
in. Before this was handled, reconstructing a room in a connected house gave
+42% area and 30 walls in a 4-wall room.

Two approaches were tried. Severing the doorway "neck" with a morphological
opening does not work: adjoining rooms usually have collinear side walls, so
the two regions meet along a wide front rather than a neck, and the opening
never splits them. What does work is using the **walls themselves** as the
barrier — every fitted wall is rasterised across its full extent (including
across its own doorway, because the room *ends* at the doorway), the barrier is
subtracted from the occupied floor, and the result is flood-filled from where
the camera actually stood.

| | before | after |
|---|---|---|
| room_0 area error | +42.3% | +0.10% |
| room_1 area error | +26.6% | +0.01% |
| room_2 area error | +28.0% | −0.07% |
| walls found | 18–30 | 4/4 |
| doors found | 0 | all |

Guarded by `test_a_room_is_not_contaminated_by_what_it_sees_through_its_doorways`.

A separate bug in the synthetic house generator was masking this: exterior
geometry was built per room, 4 m beyond *that* room's bbox, which places it
inside the neighbouring rooms. Room 0 was measuring room 1's exterior slab
standing in the middle of its own floor. Now built once around the whole
property, guarded by `test_house_exterior_does_not_intrude_into_other_rooms`.

## 7b. End-to-end stitching now meets the drift target

**Six-room house, full integrated path** — render captures, reconstruct each
room, detect its doorways, align, solve the house graph:

| | |
|---|---|
| worst room drift | **12.6 cm** (target ≤15 cm) |
| RMS drift | 6.8 cm |
| end-to-end span error | 6.0 cm over 19.9 m |
| edges accepted | 5 of 5 |
| ICP RMSE per edge | 1.06 – 1.53 cm |
| doorway overlap per edge | 0.69 – 0.73 |
| total floor area | −0.12% |

Getting there took two real fixes and one correction to the *measurement*.

### The ICP gate was scoring geometry the rooms did not share

The §6.3 5 cm gate was failing correct alignments: point-to-plane RMSE at the
**ground-truth** transform between two adjacent reconstructed rooms was 5.71 cm.

The cause was the correspondence threshold, not the alignment. Near a doorway
each room sees a good deal of geometry the other never did, and those points
have no true counterpart. At the ground-truth transform the median
correspondence distance is **0.75 cm** — but the 90th percentile is **46 cm**.
A 30 cm correspondence cap lets that tail set the score:

| correspondence cap | kept | RMSE at ground truth |
|---|---|---|
| 30 cm (was) | 83% | 6.00 cm — fails |
| 10 cm (now) | 68% | **1.59 cm** |
| 5 cm | 63% | 0.71 cm |

The threshold is now annealed from 30 cm to 10 cm across the iterations — loose
early so a rough initial guess can pull in, tight at the end so the reported
RMSE measures the alignment rather than the non-overlap.

Scale was investigated first and ruled out: the two rooms differ by ~0.9%
residual scale after the §6.2 solve, but adding scale as a fourth ICP DoF moved
RMSE only 6.0 → 5.6 cm. It was reverted rather than kept as complexity that had
not paid for itself.

**Overlap fraction is now the primary outlier signal.** A wrong doorway pairing
can fit a handful of points well and post a small RMSE; it cannot make two rooms
broadly agree. Correct pairings run 0.69–0.73; the deliberately-corrupted test
case collapses to 0.19.

### Doorway pairing in a corridor of identical doors

A doorway may join exactly one pair of rooms. Reusing one let a room attach
through the door it had already arrived by — a clean 180° flip that neither ICP
nor an interpenetration test objects to, because the result does not intersect
anything. Also added: rooms may not interpenetrate, unverified alignments are
flagged rather than treated as having passed the gate, and rooms left with no
accepted edge are reported instead of floating wherever the initial guess put
them.

### The drift metric was wrong

It compared each room's **local origin** to truth. A rectangular room with
symmetrically placed doors maps onto itself under a half-turn, so an equally
valid solution places it at yaw 180° and its origin lands at the opposite
corner. That reported 5.6 m of "drift" for a room sitting exactly where it
belonged. Drift is now measured on the room centroid, after a rigid best-fit
alignment of the reconstructed house to truth — without that the anchor room
itself showed 8.8 cm, which is a global frame offset and not drift at all.

---

## 7c. SPZ is not verified byte-compatible

`recon/io/export.py` implements the SPZ *scheme* from plan §3.7 — quantise,
entropy-code, keep spherical harmonics — and it round-trips through its own
reader with the precision and size properties asserted in tests. It has **not**
been checked against `nianticlabs/spz` or its specification, and there is at
least one deliberate divergence:

Quaternions use **smallest-three** encoding (drop the largest component, store
the other three at 10 bits each) rather than storing xyz and recovering w from
the unit-norm constraint. The latter is ill-conditioned wherever w is small —
`dw/d|xyz|` grows without bound as w → 0 — and measured **up to 10.4° of
rotation error**, enough to visibly reorient a Gaussian. Smallest-three keeps it
under 0.3° for one extra byte per splat.

Before any client consumes these files, validate against the reference decoder
or generate them with the reference encoder.

## 8. Next

1. **Rent an L40S**, run the real Phase-0 grid: {DA3-BASE, DA3METRIC-LARGE,
   DA3-LARGE, MapAnything-Apache} × {gt, sam3} masks. Compare against the
   synthetic numbers above to isolate model error from pipeline error.
2. **Capture the real ground-truth set** (plan §8: 15 rooms × 4 property types)
   with the pose loggers. Run `recon validate` on the first bundle off the
   device before the rest of the day is spent.
3. **Clean-room PGSR on gsplat.**
4. Written clarification from ByteDance on DA3-LARGE weights.
