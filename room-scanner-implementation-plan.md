# Room-to-House 3D Scanner — Implementation Plan

**Version:** 1.0 · **Date:** August 2026
**Audience:** implementing engineer / coding agent
**Status:** approved architecture, ready to build

---

## 0. TL;DR — The Architecture in One Paragraph

A cross-platform mobile app (iOS + Android, **no LiDAR required**) captures a room using the native AR stack for 6-DoF metric poses, while an on-device guidance system tells the user where to stand next until coverage is complete. Keyframes + poses + IMU gravity are uploaded to a GPU worker, which runs **Depth Anything 3 in pose-conditioned mode** for dense geometry, **PGSR** (planar-based Gaussian splatting) for a representation that is simultaneously photorealistic *and* metrically accurate, and **SAM 3** to mask mirrors and windows before fusion. Metric scale comes from VIO, corrected by user-supplied reference measurements and architectural priors solved in a pose graph. Rooms become nodes in a house graph joined at doorway edges; global optimization distributes drift. Output: an SPZ splat scene for walkthroughs, a watertight mesh, and a dimensioned 2D floor plan.

---

## 1. Scope

### 1.1 In scope (v1)
- Single-user capture of one room at a time, multiple rooms per property
- iOS 17+ (A14 or newer), Android 13+ with ARCore-supported device
- Photorealistic walkthrough of the whole property
- Dimensioned 2D floor plan with per-room area
- Export: SPZ (splat), GLB (mesh), SVG/PDF (floor plan), JSON (structured layout)

### 1.2 Explicitly out of scope (v1)
- Multi-floor stair linking (design the schema for it; don't build it)
- Furniture/object semantic labelling beyond what's needed for masking
- Real-time collaborative capture
- Outdoor/exterior capture
- On-device full reconstruction (cloud only in v1)

### 1.3 Success criteria
| Metric | Target |
|---|---|
| Wall-length error vs. laser measure | ≤ 1.5% or ≤ 3 cm, whichever is larger |
| Room floor-area error | ≤ 3% |
| End-to-end drift across 6-room house | ≤ 15 cm after global optimization |
| Novel-view PSNR (held-out keyframes) | ≥ 26 dB |
| Capture time per average bedroom | ≤ 90 s |
| Cloud reconstruction turnaround per room | ≤ 6 min |
| Capture abandonment rate | < 15% |

---

## 2. System Architecture

```
┌─────────────────────── MOBILE CLIENT ───────────────────────┐
│  Capture Layer                                              │
│    ARKit (iOS) / ARCore (Android)                           │
│    → 6-DoF pose @60Hz, intrinsics, gravity, sparse points   │
│  Guidance Layer  (shared Rust core)                         │
│    → coverage voxel grid, next-best-station solver,         │
│      quality gates (blur, exposure, tracking health)        │
│  On-device preview                                          │
│    → DA3-Small via ExecuTorch, coarse TSDF, live mesh       │
│  Viewer                                                     │
│    → embedded WebView, Babylon.js 8+, SPZ playback          │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTPS: keyframes (JPEG/HEIC),
                           │ poses, intrinsics, IMU, measurements
                           ▼
┌─────────────────────── API + QUEUE ─────────────────────────┐
│  FastAPI · Postgres (metadata, house graph) · S3 (blobs)    │
│  Redis/Celery job queue                                      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌────────────────── GPU RECONSTRUCTION WORKER ────────────────┐
│  1. Masking          SAM 3 (mirror/window/TV/person)        │
│  2. Geometry         Depth Anything 3, pose-conditioned      │
│  3. Pose refine      GTSAM pose graph + scale constraints    │
│  4. Render+surface   PGSR via gsplat backend                 │
│  5. Metric layer     TSDF fusion → Atlanta-world planes      │
│  6. Layout           wall/floor/ceiling → 2D floor plan      │
│  7. Package          SPZ + GLB + SVG + layout JSON           │
└─────────────────────────────────────────────────────────────┘
```

**Why cloud-first:** PGSR training needs a real GPU. Do not attempt on-device reconstruction in v1. The on-device DA3-Small pass exists purely to drive live coverage feedback, not to produce the deliverable.

---

## 3. Technology Selections

Every choice below is current as of August 2026. Where a well-known older option was rejected, the reason is stated.

### 3.1 Depth & geometry — **Depth Anything 3 (DA3)**

Repo: `ByteDance-Seed/Depth-Anything-3` (released Nov 2025; ICLR 2026)

DA3 predicts spatially consistent geometry from arbitrary visual inputs, with or without known camera poses. Its two design insights are that a plain transformer backbone (vanilla DINO encoder) suffices without architectural specialization, and that a single depth-ray representation removes the need for complex multi-task learning. It outperforms DA2 on monocular depth and beats VGGT on multi-view depth and pose estimation — reported at 44.3% better camera-pose accuracy than VGGT.

**This is the single most important upgrade over a DA2-based design.** DA3 offers four input configurations from one model:
- Monocular depth from a single RGB image
- Multi-view depth, producing consistent maps across images for high-quality fusion
- **Pose-conditioned depth — superior depth consistency when camera poses are provided as input**
- Camera pose estimation from one or more images

**We are in the pose-conditioned case.** ARKit/ARCore hand us poses and intrinsics for free, which puts us in DA3's strongest operating mode. Do *not* throw the AR poses away and re-solve them.

Use **DA3-Large** on the server. Use **DA3-Small** on device for live preview. For long captures, `DA3-Streaming` handles ultra-long video sequences with sliding-window inference under 12 GB of GPU memory — use it for whole-house passes.

*Rejected:* Depth Anything V2 (superseded), Depth Pro / UniDepth v2 / Metric3D v2 (metric-from-single-image models — unnecessary when VIO already provides scale, and they lack DA3's multi-view consistency), VGGT (DA3 outperforms it on both pose and multi-view depth).

### 3.2 Fallback / cross-check — **MapAnything**

Repo: `facebookresearch/map-anything` (Meta Reality Labs + CMU, 3DV 2026, Apache-licensed model)

MapAnything ingests one or more images along with optional geometric inputs such as camera intrinsics, poses, depth, or partial reconstructions, and directly regresses metric 3D scene geometry and cameras. It uses a factored representation — depth maps, local ray maps, camera poses, and a metric scale factor that upgrades local reconstructions into a globally consistent metric frame — and a single feed-forward model supports over 12 reconstruction tasks including SfM, multi-view stereo, monocular metric depth, registration, and depth completion.

Two reasons to keep this in the stack:
1. Its **dedicated scale token** gives an independent estimate of global metric scale — a free cross-check against VIO and the user's tape measure.
2. Its **registration** mode is a strong candidate for automatic room-to-room alignment (see §6.3).

**Action for the agent:** benchmark DA3-Large vs. MapAnything on the same captures in Phase 0 and pick the winner per-task. They may well split — DA3 for depth, MapAnything for registration.

### 3.3 Render + surface layer — **PGSR** (Planar-based Gaussian Splatting)

PGSR employs planar-aligned Gaussians that encode surface normals and signed distances, enabling analytic projection and multiview geometric regularization, achieving strong geometric accuracy and photometric fidelity compared with contemporary baselines. It introduces unbiased depth rendering and multi-view constraints for geometric and photometric consistency, and reaches accuracy comparable to Neuralangelo while training over a hundred times faster.

**Why this collapses our two-representation problem:** the original plan was 3DGS for looks + a separate TSDF pass for measurement, because vanilla 3DGS geometry is fuzzy. PGSR gives strong surface reconstruction quality *and* excellent novel view synthesis from one training run. And indoor rooms are overwhelmingly planar, which is exactly PGSR's inductive bias.

Keep the TSDF pass anyway (§6.2) as a robustness net, but PGSR is the primary.

*Benchmark against:* **2DGS** (one of the fastest surface reconstruction methods, collapses 3D Gaussians into view-consistent 2D oriented disks, meshes via TSDF fusion) and **GOF** (builds an opacity field and extracts meshes on a tetrahedral grid via marching tetrahedra, no Poisson or TSDF post-process needed, better completeness than 2DGS but costly). Practical guidance from teams running both: 2DGS trains fastest, PGSR produces better results.

*Rejected:* vanilla 3DGS (unusable geometry), SuGaR (fragmented in low-texture regions — fatal for white walls), MILo (expensive), NeRF-family (obsolete for this use case; GS is preferred in 2026 for faster training and real-time rendering).

### 3.4 Rasterizer backend — **gsplat**

Born out of Nerfstudio, gsplat packages CUDA-accelerated rasterization behind clean Python bindings and has become the standard open library for Gaussian splatting research and production. Versus the original Inria implementation it trains with up to 15% less time and up to 4× less GPU memory, and adds multi-GPU DDP, pinhole/fisheye camera models, batched rendering, TorchScript deployment, and NVIDIA 3DGUT integration. Apache 2.0, actively maintained.

Use gsplat as the rasterization layer under a PGSR implementation. Use Nerfstudio only as a Phase-0 comparison harness (`ns-train splatfacto`), not in production.

### 3.5 Masking — **SAM 3**

SAM 3 introduces Promptable Concept Segmentation: it detects, segments, and tracks all instances of a visual concept specified by short noun phrases, image exemplars, or both, returning masks and identities for every matching instance. A SAM3-based adaptation set a new state of the art on mirror detection benchmarks, improving best IoU from 0.918 to 0.943 on MSD and 0.728 to 0.804 on PMD.

This matters because earlier SAM generations were specifically bad here: SAM often identified the objects *behind* transparent surfaces and the objects *reflected in* a mirror, while failing to recognize the glass or the mirror itself. Since monocular depth models hallucinate straight through windows and into mirrors, un-masked glass produces phantom rooms.

Run SAM 3 with text prompts: `"mirror"`, `"window"`, `"glass door"`, `"television screen"`, `"person"`. Mask these regions out of TSDF fusion and down-weight them in PGSR's geometric loss (keep them in the *photometric* loss — you still want the mirror to look like a mirror in the walkthrough).

### 3.6 On-device inference — **ExecuTorch 1.0**

ExecuTorch hit 1.0 GA in October 2025 with a 50 KB base runtime, 12+ hardware backends (Apple, Qualcomm, Arm, MediaTek, Vulkan), and is used by Meta across Instagram, WhatsApp, Messenger and Facebook. It delegates to Core ML on iOS, QNN on Qualcomm, and XNNPACK as CPU fallback.

Choose ExecuTorch if you're already in PyTorch and want a native deployment path; choose **LiteRT** for maximum compatibility and more mature tooling, since it runs the same model on Android, iOS and the browser. DA3 is PyTorch-native → **ExecuTorch is the default**, with LiteRT as the fallback if a specific Android SoC misbehaves.

Target: DA3-Small at ≤50 ms/frame at 1–2 Hz on the guidance path. Do not run it every frame.

### 3.7 Delivery format — **SPZ**, migrating to glTF

SPZ is Niantic's open compressed 3DGS format (Apache 2.0 reference implementation, `nianticlabs/spz`), achieving roughly 10× reduction vs. PLY — a 500 MB PLY becomes about 50 MB — via quantization plus entropy coding, decompressing in under 100 ms on modern hardware. It preserves spherical harmonics, so view-dependent effects survive; SPLAT strips SH and looks flatter.

Khronos added Gaussian Splatting to glTF via `KHR_gaussian_splatting` and `KHR_gaussian_splatting_compression_spz`, with the release candidate published February 2026 and wide tool adoption expected to follow.

**Decision:** store PLY as the archival master, ship SPZ to clients, and emit glTF with `KHR_gaussian_splatting` behind a feature flag so we're ready when tooling lands. Note the practical warnings: verify SH support in your target viewer, and record source vs. target coordinate systems explicitly or you will ship flipped scenes.

### 3.8 Client framework

**Recommended: native shells + shared Rust core.**
- iOS: Swift + ARKit. Android: Kotlin + ARCore (actively maintained; latest updates April 2026, Unity 6 / AR Foundation 6 compatible, Swift Package Manager support on iOS).
- Shared core in Rust via UniFFI: coverage voxel grid, next-best-station solver, quality gates, session serialization, pose-graph client stubs. This is where the interesting math lives, it stays testable on desktop, and it runs unmodified on a Jetson for ground-truth work.
- Viewer: embedded WebView running Babylon.js 8+, which supports SPZ natively. Avoids writing and maintaining two native splat renderers.

**Alternative: Unity + AR Foundation 6.x.** One codebase across ARKit and ARCore. Faster to a demo, heavier app binary, more friction for custom CV. Watch the known coordinate-system trap: Apple's right-handed system versus Google's left-handed convention trips up even experienced developers.

Pick native if this is a product; pick Unity if you need a demo in three weeks.

### 3.9 Optimization / geometry libraries
- **GTSAM** — pose graph optimization, factor graph with custom scale/prior factors. (Alternative: Ceres.)
- **Open3D** — TSDF fusion, RANSAC plane fitting, ICP, point-cloud IO.
- **OpenCV** — image quality gates, homographies, misc.
- **COLMAP** — Phase-0 ground-truth baseline only, not production.

---

## 4. Build Phases

### Phase 0 — Offline validation harness *(2 weeks)*
**Goal:** prove the pipeline works on a laptop before writing a single line of app code.

1. Capture 5 rooms with a phone, recording video + ARKit/ARCore pose logs via a throwaway test app.
2. Capture the same rooms with the ZED 2i rig → ground-truth point clouds.
3. Measure every room with a laser distance meter → ground-truth dimensions.
4. Build a Python harness: frames + poses in → DA3 depth → PGSR → mesh + planes out.
5. Benchmark grid: {DA3-Large, MapAnything} × {PGSR, 2DGS, GOF}. Report Chamfer distance vs. ZED, wall-length error vs. laser, PSNR, wall-clock time, peak VRAM.

**Exit criterion:** at least one configuration hits ≤2% wall-length error on 4 of 5 rooms. If nothing does, stop and re-plan — do not build an app on top of a pipeline that can't measure.

### Phase 1 — Capture client *(3 weeks)*
- AR session with pose/intrinsics/gravity logging, keyframe selection (see §6.1)
- Quality gates: blur (variance of Laplacian), exposure clipping, tracking-state health
- Session bundle format + upload with resumability
- No guidance yet — just a "walk around and press stop" flow

**Exit criterion:** a session bundle from a real phone reconstructs in the Phase-0 harness at Phase-0 quality.

### Phase 2 — Cloud reconstruction service *(3 weeks)*
- FastAPI + Celery + GPU worker, containerized
- Pipeline steps 1–7 from §2, each independently retryable
- Artifact storage and signed download URLs
- Cost instrumentation per job

**Exit criterion:** ≤6 min per room end-to-end on an L40S/A100; results equal to the offline harness.

### Phase 3 — Capture guidance *(3 weeks)*
The differentiating feature. Full spec in §6.1.
- On-device DA3-Small + coarse TSDF for live coverage
- Coverage voxel grid + next-best-station solver
- AR ring markers on the floor, coverage heat overlay, live nags
- "Room complete" detection

**Exit criterion:** median capture time under 90 s for a bedroom with an untrained user; coverage ≥95%.

### Phase 4 — Metric layer & floor plan *(3 weeks)*
- Reference-measurement UX and scale solver (§6.2)
- Atlanta-world plane extraction, wall/floor/ceiling classification
- 2D floor plan generation, area computation
- Door and window detection from SAM 3 masks + plane holes

**Exit criterion:** hits the §1.3 accuracy targets on the Phase-0 ground-truth set.

### Phase 5 — Multi-room stitching *(3 weeks)*
- Doorway tagging UX
- 3-DoF alignment solver + ICP refinement
- House-level pose graph with GTSAM
- Whole-house viewer and floor plan

**Exit criterion:** ≤15 cm end-to-end drift across a 6-room test house.

### Phase 6 — Export, polish, beta *(3 weeks)*
- SPZ/GLB/SVG/JSON export
- Sharing links, web viewer
- Onboarding, error recovery, offline queueing
- 20-property closed beta

---

## 5. Data Schemas

```jsonc
// SessionBundle — client → server
{
  "session_id": "uuid",
  "device": { "model": "iPhone 16 Pro", "os": "iOS 18.2",
              "ar_platform": "arkit", "ar_version": "..." },
  "room_id": "uuid",
  "gravity": [0.0, -1.0, 0.0],          // world-frame down vector
  "keyframes": [{
    "id": "kf_0001",
    "image": "kf_0001.heic",
    "timestamp_ns": 123456789,
    "pose": { "R": [[9 floats]], "t": [3 floats] },   // world←camera
    "intrinsics": { "fx":0, "fy":0, "cx":0, "cy":0,
                    "width":0, "height":0, "distortion":[...] },
    "tracking_state": "normal|limited|relocalizing",
    "blur_score": 0.0,
    "exposure_clip_ratio": 0.0
  }],
  "sparse_points": [{ "xyz": [3 floats], "observations": ["kf_0001", ...] }],
  "reference_measurements": [{
    "type": "user_tape|door_prior|ceiling_prior|camera_height",
    "point_a": [3], "point_b": [3],
    "measured_value_m": 0.83, "sigma_m": 0.005
  }],
  "doorways": [{ "id":"dw_01", "plane": [4 floats], "centroid":[3],
                 "width_m": 0.0, "height_m": 0.0 }]
}
```

```jsonc
// HouseGraph — server-side, the canonical property model
{
  "property_id": "uuid",
  "floors": [{
    "level": 0,
    "rooms": ["room_uuid_a", "room_uuid_b"],
    "edges": [{
      "from_room": "room_uuid_a", "to_room": "room_uuid_b",
      "via_doorway": ["dw_01", "dw_07"],       // one ID per side
      "transform": { "yaw_rad": 0.0, "tx": 0.0, "tz": 0.0 },
      "covariance": [[3x3]],
      "source": "user_tagged|auto_registered|continuous_tracking"
    }]
  }]
}
```

---

## 6. Algorithm Specifications

### 6.1 Capture guidance

**Coverage representation.** Maintain a voxel grid over the room bounding box at 5 cm resolution. Each voxel stores an observation quality score:

```
q(v) = Σ_kf  w_angle(v,kf) · w_dist(v,kf) · w_blur(kf)
w_angle = max(0, cos θ)²      θ = angle between surface normal and view ray
w_dist  = clamp(1 - |d - d_opt| / d_range, 0, 1)     d_opt ≈ 2.0 m
w_blur  = sigmoid(blur_score)
```
A voxel is **covered** when `q(v) ≥ q_min` (tune in Phase 3; start at 2.0, which roughly means two decent views).

**Next-best-station.** Do not use RL-based next-best-view methods from the robotics literature — they solve a harder problem (autonomous agent, unconstrained 6-DoF placement) than we have. We have a human who must physically stand somewhere on the floor. Use a direct information-gain search:

1. Sample candidate positions on the detected floor plane, 40 cm grid, ≥60 cm from any wall, in already-observed free space (so we don't send a user into a wall).
2. For each candidate, at standing height (1.5 m), integrate expected coverage gain over a full yaw sweep, ray-casting against the current partial mesh.
3. Penalize walking distance from the user's current position: `score = gain - λ · distance`. Start `λ = 0.3` per metre.
4. Emit the top candidate as an AR floor ring. Recompute every 2 s or after any station is completed.

**Per-station capture.** At each ring, ask for a slow yaw sweep. Fire keyframe selection on: ≥15° rotation OR ≥30 cm translation since the last keyframe, AND blur below threshold, AND tracking state normal. Cap at ~150 keyframes per room.

**Live nags** (surface at most one at a time, with a 3 s cooldown):
- `TOO_CLOSE` — median depth < 0.8 m
- `TOO_FAST` — angular velocity > 60°/s
- `TEXTURELESS` — sparse-point density in view below threshold → *"Include a corner or doorway in the frame"*
- `CEILING_MISSING` — ceiling plane coverage < 40% → *"Tilt up"*
- `TRACKING_LOST` — guide back to the last known good station

**Room-complete.** Declare done when: floor coverage ≥95%, all four wall directions ≥90%, ceiling ≥70%, and no candidate station offers gain above 2% of total volume.

### 6.2 Metric scale & the reference measurement

VIO already provides metric scale from the IMU — typically within a few percent. The reference measurement is **correcting slow drift, not creating scale.** Frame it that way in the solver and in the UI copy.

**Constraint sources**, each entering as a factor in the pose graph:

| Source | Typical σ | Notes |
|---|---|---|
| VIO scale prior | 3% | the baseline; never discard it |
| User tape measure | 5 mm | AR two-point pick + numeric entry |
| Door height prior | 4 cm | interior doors cluster hard near 2.03 m; regional table needed |
| Ceiling height prior | 10 cm | weak, but free |
| Camera height above floor | 8 cm | ARKit/ARCore already estimate this |
| MapAnything scale token | TBD | measure in Phase 0 before trusting |

**Solver.** Single global scale factor `s` per room, estimated by weighted least squares over all constraints:

```
s* = argmin Σ_i  (1/σ_i²) · (s · d_i^est - d_i^obs)²
```

Then, at the house level, do **not** apply per-room scales independently — that creates discontinuities at doorways. Instead add scale as a variable in the house pose graph with doorway edges as equality constraints on shared door geometry, and let GTSAM distribute the residual. This is the difference between error piling up in the last room scanned and error spreading evenly across the property.

**UX.** Ask for exactly one measurement per room, phrased concretely: *"Measure any doorway width with a tape and enter it."* Doorways are chosen because they're planar, unambiguous, and detected automatically. Accept skips — the priors alone should get within ~5%.

### 6.3 Room stitching

**Path A — continuous tracking (preferred when it works).** If the user walks between rooms with the AR session live, everything is already in one coordinate frame. Detect room transitions via doorway crossing. Add a loop-closure prompt at the end ("return to where you started") and optimize.

**Path B — independent scans joined at doorways (must always work).** Because gravity is known from the IMU, alignment is only **3 DoF**: yaw plus 2D translation in the floor plane.

1. User taps the same doorway in both rooms (or auto-detect from SAM 3 `"door"` masks plus plane holes).
2. Solve yaw from the door-plane normals: `ψ = atan2(n_a × n_b · g, n_a · n_b)`.
3. Solve translation by matching door threshold centroids and floor planes.
4. Refine with point-to-plane ICP on geometry within 2 m of the doorway, initialized from step 2–3. Reject if final RMSE > 5 cm and fall back to asking the user to re-tag.
5. **Optional auto-path:** feed both rooms' partial reconstructions to MapAnything in registration mode — it accepts partial reconstructions as input and supports registration as one of its native tasks. Use its output as the ICP initialization, and as a sanity check on the user's tagging.

**Global optimization.** Build the factor graph:
- Nodes: one SE(2)+scale variable per room (3 DoF + 1, since gravity is fixed)
- Edges: doorway constraints with covariance from ICP residuals
- Priors: each room's internal scale constraints from §6.2
- Loop closures: any room pair with overlapping geometry
- Solve with Levenberg–Marquardt in GTSAM; robustify with Huber to survive one bad doorway tag

### 6.4 Metric layer & floor plan

1. **TSDF fusion** of the DA3 depth maps (SAM-3-masked) at 2 cm voxel size, using refined poses. This is the robustness net beneath PGSR's geometry.
2. **Plane extraction** — RANSAC over the fused cloud with normals.
3. **Atlanta-world regularization**, not Manhattan. Manhattan assumes three mutually orthogonal vanishing directions; Atlanta adds one degree of freedom, allowing walls at arbitrary yaw provided they remain vertical. Real houses have angled walls, bay windows, and non-square rooms, and Manhattan will actively destroy them. Enforce: all walls vertical (from gravity), floor and ceiling horizontal and parallel, wall yaw snapped only when within 3° of an existing wall direction.
4. **Wall/floor/ceiling classification** by normal direction and height.
5. **Openings** — door and window rectangles from SAM 3 masks intersected with wall planes.
6. **Floor plan** — intersect wall planes pairwise for corners, trace the room polygon, compute area, emit SVG with dimension annotations.

*Note on layout models:* the deep room-layout literature is dominated by 360° panorama methods (HorizonNet, LED²-Net, HoHoNet), because perspective images have limited FOV and rarely capture the ceiling. We sidestep this entirely — we have dense multi-view geometry, so geometric plane fitting beats single-image layout prediction. Do not bolt on a panorama layout network.

---

## 7. Failure Modes & Mitigations

| Failure | Cause | Mitigation |
|---|---|---|
| Phantom rooms behind glass | Depth models see through windows and into mirrors | SAM 3 masking before fusion (§3.5); mask *before* TSDF, not after |
| VIO drift / tracking loss | Textureless white walls starve feature tracking | Live `TEXTURELESS` nag; relocalization guidance; sparse-point density gate on keyframe acceptance |
| Scale error compounding across rooms | Per-room scale applied independently | House-level pose graph with scale as a shared variable (§6.2) |
| Splat geometry unusable for measurement | Vanilla 3DGS is fuzzy by construction | PGSR + independent TSDF cross-check; never measure the raw splats |
| Bad doorway tag corrupts whole house | Single outlier edge in pose graph | Huber robust kernel; ICP RMSE rejection gate; per-edge covariance |
| Flipped/mirrored scenes on export | Coordinate-system mismatches between capture and viewer; Apple right-handed vs. Google left-handed | Record source and target coordinate systems explicitly in metadata; assert on load; visual regression test per platform |
| Reconstruction cost blows up | Unbounded keyframe counts | Hard cap 150 keyframes/room; adaptive selection; monitor $/job |
| Dark rooms | Noise destroys both VIO and depth | Lux gate at session start; prompt to turn lights on |

---

## 8. Evaluation Harness

Build this in Phase 0 and never stop running it.

- **Ground-truth set:** 15 rooms across 4 property types (empty, furnished, cluttered, mirror-heavy). Each with ZED 2i point cloud + full laser-measured dimensions.
- **Automated regression:** every pipeline change runs the full set and reports Chamfer distance, wall-length error, area error, PSNR/SSIM/LPIPS on held-out views, runtime, VRAM.
- **Fail the build** if wall-length error regresses more than 0.5 percentage points.
- **Capture-UX metrics from beta:** time-to-complete, abandonment rate, number of nags fired, re-scan rate.

---

## 9. Cost Model

Indicative cloud training costs: a moderate 5,000-image capture at ~3M Gaussians runs 2–4 hours on an H100, roughly $4–8 per scene; spot instances on A100 cut 40–60% where available. Our per-room job is far smaller (~150 keyframes, ~500k Gaussians) — budget **$0.30–0.80 per room** on an L40S, so roughly **$2–5 per house**. Split high-VRAM training (H100/A100) from cost-efficient rendering (L40S/RTX PRO 6000) as separate instance classes.

Instrument cost per job from day one in Phase 2.

---

## 10. Licensing — Resolve Before Shipping

This is a real risk, not boilerplate. Several key components are research code with restrictive or unclear commercial terms.

| Component | Status | Action |
|---|---|---|
| gsplat | Apache 2.0 | ✅ clear |
| MapAnything model | Apache | ✅ verify code license separately |
| SPZ | Apache 2.0 / MIT reference impl | ✅ clear |
| DA3 | Trained exclusively on public academic datasets — check repo license terms | ⚠️ **verify before commercial launch** |
| PGSR | Research code | ⚠️ **verify; may need reimplementation** |
| SAM 3 | Meta license | ⚠️ **verify commercial terms** |
| 2DGS / GOF | Research code | ⚠️ verify if used |

Have the agent produce a licensing audit as a Phase-0 deliverable. If PGSR's license blocks commercial use, budget for a clean-room reimplementation on top of gsplat — the paper is detailed enough to make this feasible.

---

## 11. Open Decisions

1. **Native + Rust vs. Unity** — decide before Phase 1 starts. Affects everything downstream.
2. **DA3 vs. MapAnything per task** — decide from Phase 0 benchmarks, not from priors.
3. **PGSR vs. 2DGS** — quality/speed tradeoff; may differ for the preview tier vs. the final tier.
4. **Target vertical** — renovation quoting, insurance documentation, real-estate listings, and BIM/IFC export all imply different export priorities. Polycam, Canvas, magicplan and Apple's own RoomPlan already occupy the generic "make a mesh" space; the wedge is what comes *out* of the scan, not the scan itself. This should be decided by Phase 4.
5. **Multi-floor** — schema supports it; build after v1.

---

## 12. First Three Tasks for the Agent

1. Stand up the Phase-0 harness: clone Depth Anything 3 and MapAnything, get both running on a GPU box, and reconstruct one hand-captured room end-to-end with a PGSR (or 2DGS) pass on top. Report the benchmark table from §4/Phase 0.
2. Produce the licensing audit from §10.
3. Write the throwaway pose-logging test app for iOS and Android that emits the `SessionBundle` in §5 — this unblocks all real data collection.
