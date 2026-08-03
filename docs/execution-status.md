# Execution status

Companion to [`room-scanner-implementation-plan.md`](../room-scanner-implementation-plan.md),
which stays the canonical architecture. This file records what has been built,
what deviates from the plan and why, and what is blocked.

**Last updated:** 2026-08-03

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
| DA3 / MapAnything / SAM 3 adapters | written | ❌ needs GPU |
| PGSR clean-room trainer | **not written** | see §6 |
| iOS + Android pose loggers | done | needs device build |
| Licensing audit | done | re-runnable |
| Cloud service (Phase 2) | not started | |
| Room stitching (Phase 5) | not started | |

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

## 8. Next

1. **Rent an L40S**, run the real Phase-0 grid: {DA3-BASE, DA3METRIC-LARGE,
   DA3-LARGE, MapAnything-Apache} × {gt, sam3} masks. Compare against the
   synthetic numbers above to isolate model error from pipeline error.
2. **Capture the real ground-truth set** (plan §8: 15 rooms × 4 property types)
   with the pose loggers. Run `recon validate` on the first bundle off the
   device before the rest of the day is spent.
3. **Clean-room PGSR on gsplat.**
4. Written clarification from ByteDance on DA3-LARGE weights.
