# Licensing audit

**Deliverable:** plan §10, agent task 2.
**Audited:** 2026-08-03. **Method:** GitHub licence API + full `LICENSE` text, and
HuggingFace model-card metadata, for every component named in the plan.
**Verdict scope:** commercial SaaS — we run inference on our own servers and ship
*outputs* (splats, meshes, floor plans) to paying customers. We do **not**
redistribute model weights.

> ⚠️ Not legal advice. Two items below need written clarification from the
> upstream owner before commercial launch, and one needs a purchased licence or
> a reimplementation.

---

## 1. Headline findings

Three findings change the plan, not just the paperwork.

### 1.1 🔴 Every splat backend in the plan's benchmark grid is non-commercial

PGSR, 2DGS and GOF are **all** research-only. The plan treats a clean-room PGSR
reimplementation as a contingency ("if PGSR's license blocks commercial use").
It is not a contingency — it is the **only** path to a commercial splat layer,
because the two named fallbacks are encumbered as well.

| Backend | Licence | Commercial |
|---|---|---|
| PGSR | ZJU 3DV custom | ❌ "educational, research and non-profit purposes only… prohibited for commercial use" |
| 2DGS | Inria/MPII Gaussian-Splatting Licence | ❌ research only |
| GOF | Inria/MPII Gaussian-Splatting Licence | ❌ research only |
| gsplat | Apache 2.0 | ✅ |

PGSR additionally imposes copyleft: *"Any modification based on this work must be
open-source"* — so it cannot be quietly vendored and patched either.

**Consequence.** gsplat (Apache 2.0) is the only licence-clean rasteriser, which
is exactly why the plan already puts it underneath. The planar-Gaussian *method*
from the PGSR paper is not itself protected — papers describe, licences cover
code — so a clean-room implementation of unbiased depth rendering and multi-view
geometric regularisation on top of gsplat is the intended route.

**Actions.**
1. Use PGSR/2DGS/GOF in Phase 0 **for benchmarking only**. That is squarely
   research use and is permitted. Do not let their code reach the product.
2. Budget the clean-room PGSR-on-gsplat implementation as **committed Phase-2
   work**, not a risk item. Written from the paper, by an engineer who has not
   copied the reference repo.
3. In parallel, email `zhangguofeng@zju.edu.cn` (the contact PGSR's licence
   names) for commercial terms. A cheap licence beats a reimplementation.

### 1.2 🔴 DA3-LARGE — the plan's server model — is contradictory, and probably non-commercial

The plan says "Use **DA3-Large** on the server" and describes DA3 as trained
exclusively on public academic datasets. The licensing does not line up:

| Checkpoint | GitHub model zoo | HuggingFace card | Params |
|---|---|---|---|
| DA3-SMALL | Apache 2.0 | Apache 2.0 | 0.08 B |
| DA3-BASE | Apache 2.0 | Apache 2.0 | 0.12 B |
| DA3METRIC-LARGE | Apache 2.0 | Apache 2.0 | — |
| DA3MONO-LARGE | Apache 2.0 | Apache 2.0 | — |
| **DA3-LARGE / -1.1** | **CC BY-NC 4.0** | **apache-2.0** | 0.35 B |
| DA3-GIANT / -1.1 | CC BY-NC 4.0 | cc-by-nc-4.0 | 1.15 B |
| DA3NESTED-GIANT-LARGE | CC BY-NC 4.0 | — | — |

The repository's own model-zoo table and the model card **disagree** for
DA3-LARGE. The GIANT row is consistently non-commercial in both places, which
suggests the model zoo table is the deliberate one and the Apache tag on the
LARGE card is an oversight — but that is an inference, not a licence.

The code repository is Apache 2.0. Only the *weights* are in question.

**Actions.**
1. Get written clarification from ByteDance-Seed before shipping DA3-LARGE.
   Until it arrives, treat DA3-LARGE as **CC BY-NC 4.0 — unusable**.
2. Benchmark **DA3-BASE** (0.12 B, unambiguously Apache 2.0) as the
   commercial-path server model in Phase 0, alongside DA3-LARGE. The plan's
   grid should compare BASE against MapAnything-Apache, not just LARGE.
3. Check whether **DA3METRIC-LARGE** (Apache 2.0) is the same 0.35 B backbone
   with a metric head. If so it may be the licence-clean way to get LARGE-class
   quality — a genuinely valuable Phase-0 result.
4. Good news for the client: **DA3-SMALL exists and is Apache 2.0**, so the
   plan's on-device guidance model (§3.6) is clean.

### 1.3 🟠 MapAnything's default weights are CC BY-NC — but an Apache variant exists

The plan states "Apache-licensed model". That is only true of one of the two
published checkpoints:

| Checkpoint | Licence | Use |
|---|---|---|
| `facebook/map-anything` | **CC BY-NC 4.0** | ❌ research only |
| `facebook/map-anything-apache` | **Apache 2.0** | ✅ the one to use |

`map-anything-apache` is the variant trained only on permissively licensed
data. The code repository is Apache 2.0.

**Action.** Pin `facebook/map-anything-apache` everywhere. The adapter's default
checkpoint has been changed accordingly, and Phase 0 must benchmark the *Apache*
variant, since that is the one that can ship — benchmarking the NC weights would
measure a model we cannot use.

---

## 2. Full component table

| Component | Licence (verified) | Commercial SaaS | Notes / obligations |
|---|---|---|---|
| **gsplat** | Apache 2.0 | ✅ | Preserve NOTICE. The only clean rasteriser. |
| **Depth Anything 3** (code) | Apache 2.0 | ✅ | Weights are the issue, not the code. |
| DA3-SMALL / BASE / METRIC-LARGE / MONO-LARGE | Apache 2.0 | ✅ | Attribution. |
| DA3-LARGE, DA3-GIANT | ⚠️ CC BY-NC 4.0 vs Apache (conflict) | ❌ until clarified | §1.2 |
| **MapAnything** (code) | Apache 2.0 | ✅ | |
| `map-anything-apache` weights | Apache 2.0 | ✅ | Pin this one. |
| `map-anything` weights | CC BY-NC 4.0 | ❌ | Phase-0 comparison only. |
| **SAM 3** | Meta "SAM License" (2025-11-19) | ✅ with conditions | §3 below. |
| **PGSR** | ZJU 3DV custom | ❌ | Non-commercial **and** copyleft. §1.1 |
| **2DGS** | Inria/MPII GS Licence | ❌ | Research only. |
| **GOF** | Inria/MPII GS Licence | ❌ | Research only. |
| **SPZ** | MIT | ✅ | Keep the copyright notice. |
| **Open3D** | MIT | ✅ | Bundles third-party code; check its own notices if we ever redistribute binaries. |
| **GTSAM** | BSD 3-clause | ✅ | Bundles CCOLAMD etc. — **not** all BSD, see §4. |
| **OpenCV** | Apache 2.0 (4.5+) | ✅ | Avoid `opencv-contrib` non-free modules (SIFT/SURF patents). |
| **COLMAP** | BSD | ✅ | Phase-0 ground truth only, per plan. |
| **ExecuTorch** | BSD 3-clause | ✅ | |
| **Babylon.js** | Apache 2.0 | ✅ | |
| **ARKit / ARCore** | Platform SDK terms | ✅ | Standard app-store terms. |

---

## 3. SAM 3 — permitted, with three obligations worth reading

Meta's SAM License grants a "non-exclusive, worldwide, non-transferable and
royalty-free" licence to use, reproduce, distribute, modify and create derivative
works. There is **no non-commercial clause and no monthly-active-user ceiling**
(unlike some Llama licences). Commercial server-side inference is fine.

Three obligations do apply:

1. **Agreement propagation.** Distributing SAM Materials or derivatives requires
   passing this agreement along with them. Shipping only *outputs* (masks, and
   downstream splats/meshes) does not trigger this — but bundling SAM weights
   into the mobile app would.
2. **No reverse engineering.** §1.b.iv bars using the materials to "reverse
   engineer, decompile or discover the underlying components". Unusual for an
   otherwise permissive licence; note it if anyone proposes distilling SAM 3
   into a smaller on-device masker. **That specific plan would need legal
   review.**
3. **Trade controls.** Standard OFAC/ITAR restrictions; no military, weapons or
   nuclear end uses. Fine for a property-scanning product, but it constrains
   which markets we may sell into.

Also check Meta's separate Acceptable Use Policy, which the agreement
incorporates by reference.

---

## 4. Secondary risks

- **GTSAM's third-party code.** GTSAM itself is BSD-3, but the repo bundles
  CCOLAMD and other SuiteSparse components. Parts of SuiteSparse are **LGPL or
  GPL**. If we ever statically link GTSAM into a distributed binary (a mobile
  client, not a server), audit which SuiteSparse modules are compiled in.
  Server-side use is unaffected. Ceres (Apache 2.0, per plan §3.9's alternative)
  sidesteps this entirely and is worth preferring if GTSAM ever needs to ship
  on-device.
- **Training data provenance.** An Apache-2.0 code licence says nothing about
  what the weights were trained on. For DA3 and SAM 3 we are relying on the
  publishers' representations. Acceptable risk for v1; revisit if the product
  moves into a regulated vertical (plan §11 open decision 4 lists insurance and
  BIM, both of which raise the bar).
- **`KHR_gaussian_splatting`** (plan §3.7) is a Khronos specification, royalty
  free under the glTF IP policy. No issue.
- **Patents.** Apache 2.0 grants an express patent licence; MIT and BSD do not.
  For MIT/BSD components (SPZ, Open3D, GTSAM) there is no patent grant. No known
  reads on this stack, but it is the reason to prefer Apache-licensed
  dependencies where there is a choice.

---

## 5. What to do next

| # | Action | Owner | Blocks |
|---|---|---|---|
| 1 | Written clarification from ByteDance-Seed on DA3-LARGE weights | eng lead | commercial launch |
| 2 | Benchmark DA3-BASE and DA3METRIC-LARGE alongside DA3-LARGE in Phase 0 | Phase 0 | plan §11 decision 2 |
| 3 | Pin `facebook/map-anything-apache` — **done in code** | — | — |
| 4 | Email ZJU for PGSR commercial terms | eng lead | Phase 2 scope |
| 5 | Scope clean-room PGSR-on-gsplat as committed Phase-2 work | planning | Phase 2 |
| 6 | Confirm SAM 3 Acceptable Use Policy covers our use | legal | launch |
| 7 | Re-run this audit before launch — model licences change | — | launch |

## 6. How to re-run

```bash
scripts/check_licenses.sh
```

Queries the GitHub licence API and HuggingFace model-card metadata for every
component and diffs against the table above.
