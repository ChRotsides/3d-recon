# Where the build stands

A one-page answer to "is it done?", against
[`room-scanner-implementation-plan.md`](../room-scanner-implementation-plan.md).
Detail and reasoning live in [`execution-status.md`](execution-status.md).

**188 tests green** — 168 Python, 20 Rust — plus 6 Android JVM tests.

---

## The short version

Everything in the plan that can be built and verified **without a GPU, a phone
or a Mac is built and verified**. What remains is not unwritten code; it is
hardware.

| | |
|---|---|
| Blocked on a GPU | DA3 / MapAnything / SAM 3 / PGSR actually running |
| Blocked on a device | on-device DA3-Small, capture-time metrics |
| Blocked on a Mac | the iOS Xcode project |


---

## §1.3 success criteria

| Criterion | Target | Measured |
|---|---|---|
| Wall-length error | ≤1.5% or 3 cm | **0.06 – 0.15%** ✅ |
| Floor-area error | ≤3% | **0.05 – 0.16%** ✅ |
| Whole-house drift | ≤15 cm over 6 rooms | **12.6 cm** worst, 6.8 cm RMS ✅ |
| Novel-view PSNR | ≥26 dB | needs the GPU splat stage |
| Capture time | ≤90 s | needs a device and users |
| Cloud turnaround | ≤6 min | needs an L40S |
| Abandonment | <15% | needs the beta |

The three that can be measured on this hardware are met with a wide margin,
against a synthetic oracle that is exact to machine precision — a stronger
reference than the ZED 2i rig the plan specifies, though it says nothing about
whether the *models* work on real rooms. That is what the GPU box is for.

## §4 phases

| Phase | State |
|---|---|
| 0 · offline harness | done — 5 scenes, exit criterion met |
| 1 · capture client | Android builds, tests, and calls the shared core; iOS needs a Mac |
| 2 · cloud service | done — 19 tests, upload → reconstruct → download |
| 3 · capture guidance | core done (20 tests), driving the Android client; coverage and the AR ring await on-device depth |
| 4 · metric layer & floor plan | done — meets the §1.3 accuracy targets |
| 5 · multi-room stitching | done — both §6.3 paths |
| 6 · export & polish | exports done; beta needs users |

## §7 failure modes

Every one has an implementation and a test.

| Failure | Mitigation |
|---|---|
| Phantom rooms behind glass | masking applied **before** fusion; measured worth 91 pp of wall error |
| VIO drift on textureless walls | `TEXTURELESS` nag, sparse-density gate |
| Scale compounding across rooms | scale is a variable in the house graph, not a per-room multiplier |
| Splat geometry unusable | independent TSDF; the raw splats are never measured |
| Bad doorway tag | Huber kernel, ICP RMSE gate, overlap gate, one-doorway-one-join |
| Flipped / mirrored scenes | four independent guards — schema, loader, Android pose, viewer |
| Cost blow-up | 150-keyframe cap enforced in client, core **and** server |
| Dark rooms | lux gate and `DarkRoom` nag |

## What the plan got wrong

Found by building it, all with the reasoning recorded in `execution-status.md`:

* **`w_dist` symmetric about `d_opt`** makes ceilings permanently uncoverable —
  a 2.45 m ceiling from a 1.5 m eye height is 0.95 m away and scores 0.53.
* **`q_min = 2.0` is unreachable**: every weight is strictly below 1, so no pair
  of views can ever sum to 2.0.
* **Room-complete never terminates** if gain is measured against *remaining*
  rather than total volume.
* **DA3-LARGE is CC BY-NC** in the upstream model zoo — the plan's headline
  server model is not commercially usable as it stands.
* **PGSR, 2DGS and GOF are all non-commercial**, so the clean-room
  reimplementation is the only path, not a contingency.
* **MapAnything's default weights are CC BY-NC**; only the `-apache` variant is
  usable.

## Known limitations

* **Automatic doorway pairing is unreliable in a symmetric block of rooms.** The
  plan's own answer — the user taps the doorway — works and is supported.
* **SPZ is not verified byte-compatible** with `nianticlabs/spz`.
* **The PGSR trainer raises `NotImplementedError`** on purpose: its loss weights
  are guesses until something has converged, and a benchmark row filled in by
  never-executed code is worse than a gap.
* **Job retry marks completed steps rather than genuinely resuming**; real
  resumption needs intermediate artifacts persisted.

## To unblock the rest

1. **Rent one L40S.** Run the Phase-0 grid: {DA3-BASE, DA3METRIC-LARGE,
   DA3-LARGE, MapAnything-Apache} × {gt, sam3}. Compare against the synthetic
   numbers above to separate model error from pipeline error.
2. **Capture the real ground-truth set** (§8: 15 rooms × 4 property types) with
   the Android logger. Run `recon validate` on the first bundle before spending
   the day.
3. **Get written clarification from ByteDance** on the DA3-LARGE weight licence.
4. **A Mac** for the iOS project.
