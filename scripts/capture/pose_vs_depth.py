"""Is the cross-frame disagreement a depth error or a pose error?

Overlapping frames of bundle_e4a5da81 disagree about where the room is by a
median 24.7 cm, which is what stops TSDF fusing them into one surface. That
number is measured by reprojecting one frame's depth into another, and it
measures poses and depth *jointly* — so on its own it cannot say which to fix.

This separates them by the **shape** of the residual rather than its size.
Reproject A into B and look at `pred - meas` as a function of what it varies
with:

* a per-frame **depth scale** error is multiplicative — a frame that thinks
  everything is 10% too far is wrong by 10 cm at 1 m and 50 cm at 5 m, so the
  residual grows with depth and passes through the origin;
* a **pose translation** error is additive — the whole frame sits 20 cm behind
  where it should, and that is 20 cm at every depth;
* a **pose rotation** error points the frame the wrong way, so its signature is
  a gradient across the *image*, not across depth.

Fitting all three and comparing what each removes says which is present. The
global check then asks whether the per-pair scales are mutually *consistent*:
if every pair's disagreement is one scale per frame, those scales must satisfy
a graph, and the residual after solving it is the part no per-frame scale can
explain.

    python scripts/capture/pose_vs_depth.py BUNDLE [depth.npz]

With no .npz it uses the ARCore depth the capture carries, which is the version
that matters — that is the measurement being trusted as the reference, and the
control mesh built from it layered exactly the same way.

**Caveat, stated because it bounds the conclusion:** `pred` is treated as
scaling with frame A's depth, which is exact only when the two cameras share a
centre. Here they are within 1.2 m while the room is 2.7 m deep, so parallax
makes this approximate rather than exact, and the pair set is restricted to
frames looking within 30 degrees of each other to keep it so.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from recon.io.bundle import BundleDir

BUNDLE = Path(sys.argv[1])
DEPTH_NPZ = Path(sys.argv[2]) if len(sys.argv) > 2 else None
#: Frames looking further apart than this share too little, and what they do
#: share is at a grazing angle where depth is least reliable.
MAX_VIEW_ANGLE_DEG = 30.0
MIN_PIXELS = 2000

bundle = BundleDir.load(BUNDLE)
kfs = bundle.bundle.keyframes

if DEPTH_NPZ:
    cache = np.load(DEPTH_NPZ)
    depths = {kf.id: np.asarray(cache[kf.id], dtype=np.float64) for kf in kfs if kf.id in cache}
    source = str(cache["model"])
else:
    depths = {kf.id: np.asarray(bundle.depth(kf), dtype=np.float64) for kf in kfs
              if bundle.depth(kf) is not None}
    source = "arcore"
print(f"{len(depths)} depth maps · source {source}\n")


def intrinsics(kf):
    intr = kf.intrinsics
    ih, iw = depths[kf.id].shape
    sx, sy = iw / intr.width, ih / intr.height
    return intr.fx * sx, intr.fy * sy, intr.cx * sx, intr.cy * sy, iw, ih


def forward(kf):
    return -np.asarray(kf.pose.R)[:, 2]


def overlap(a, c):
    """Reproject a's depth into c. Returns (pred, meas, u, v) in c's image."""
    fx, fy, cx, cy, iw, ih = intrinsics(a)
    da = depths[a.id]
    v, u = np.nonzero((da > 0.3) & (da < 6.0))
    if len(u) < 5000:
        return None
    step = max(1, len(u) // 25000)
    u, v = u[::step], v[::step]
    z = da[v, u]
    world = (np.stack([(u - cx) / fx * z, -(v - cy) / fy * z, -z], axis=1)
             @ np.asarray(a.pose.R).T + np.asarray(a.pose.t))

    Rb, tb = np.asarray(c.pose.R), np.asarray(c.pose.t)
    local = (world - tb) @ Rb
    zb = -local[:, 2]
    ok = zb > 0.3
    fx2, fy2, cx2, cy2, iw2, ih2 = intrinsics(c)
    uu = (local[ok, 0] / zb[ok]) * fx2 + cx2
    vv = (-local[ok, 1] / zb[ok]) * fy2 + cy2
    zb = zb[ok]
    inside = (uu >= 0) & (uu < iw2 - 1) & (vv >= 0) & (vv < ih2 - 1)
    if inside.sum() < MIN_PIXELS:
        return None
    uu, vv, zb = uu[inside], vv[inside], zb[inside]
    meas = depths[c.id][vv.astype(int), uu.astype(int)]
    good = (meas > 0.3) & (meas < 6.0)
    if good.sum() < MIN_PIXELS:
        return None
    return zb[good], meas[good], uu[good], vv[good]


def robust_median_abs(x):
    return float(np.median(np.abs(x)))


rows = []
for i, j in combinations(range(len(kfs)), 2):
    a, c = kfs[i], kfs[j]
    if a.id not in depths or c.id not in depths:
        continue
    if np.degrees(np.arccos(np.clip(forward(a) @ forward(c), -1, 1))) > MAX_VIEW_ANGLE_DEG:
        continue
    got = overlap(a, c)
    if got is None:
        continue
    pred, meas, uu, vv = got
    resid = pred - meas
    base = robust_median_abs(resid)

    # 1. Additive: one constant offset for the whole frame (pose translation).
    additive = robust_median_abs(resid - np.median(resid))

    # 2. Multiplicative: one scale on this frame's depth (depth scale error).
    #    Fit through the origin, which is what "scale" means.
    s = float(np.sum(pred * meas) / max(np.sum(pred * pred), 1e-9))
    multiplicative = robust_median_abs(s * pred - meas)

    # 3. Image-space plane: a gradient across the frame (pose rotation).
    A = np.stack([np.ones_like(uu), uu, vv], axis=1)
    coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
    rotational = robust_median_abs(resid - A @ coef)

    # 4. A full 6-DoF rigid correction — the actual meaning of "pose error",
    #    which the one- and three-parameter models above only glimpse.
    #
    #    Closed form (Kabsch) on the correspondences the reprojection already
    #    established, rather than ICP: the pairing is fixed here, so there is no
    #    iteration to slide along a wall and no local minimum to fall into. The
    #    pairing is approximate — it matches a point to whatever lies along the
    #    same ray in B, not to its true surface mate — so what this gives is an
    #    upper bound on how much any rigid fix could remove. That is exactly the
    #    quantity in question: if even the best rigid correction leaves most of
    #    the disagreement, the error is not rigid, and no pose graph will fix it.
    fx2, fy2, cx2, cy2, _, _ = intrinsics(c)
    ray = np.stack([(uu - cx2) / fx2, -(vv - cy2) / fy2, -np.ones_like(uu)], axis=1)
    src = ray * pred[:, None]     # where A says the surface is, in B's camera
    dst = ray * meas[:, None]     # where B measured it, along the same rays
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    aligned = (src - sc) @ R.T + dc
    # Score the same way as the others: disagreement in depth along B's rays.
    rigid = robust_median_abs(-aligned[:, 2] - meas)
    shift_cm = float(np.linalg.norm(dc - sc)) * 100
    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

    rows.append({
        "a": a.id, "c": c.id, "n": int(len(pred)),
        "base": base, "additive": additive,
        "multiplicative": multiplicative, "rotational": rotational,
        "rigid": rigid, "shift_cm": shift_cm, "rot_deg": rot_deg,
        "scale": s, "offset": float(np.median(resid)),
        "i": i, "j": j,
    })

if not rows:
    print("no overlapping pairs")
    raise SystemExit(1)


def col(key):
    return np.array([r[key] for r in rows])


print(f"{len(rows)} overlapping pairs (within {MAX_VIEW_ANGLE_DEG:.0f} deg)\n")
print("what each single-parameter model removes, median over pairs:")
base = col("base")
print(f"  raw disagreement                        {np.median(base) * 100:6.1f} cm")
for key, label in (
    ("additive", "one constant offset   (pose translation)"),
    ("multiplicative", "one depth scale       (depth error)     "),
    ("rotational", "an image-space plane  (pose rotation)   "),
    ("rigid", "a full 6-DoF rigid fit (best case for pose)"),
):
    v = col(key)
    print(f"  after {label}  {np.median(v) * 100:6.1f} cm"
          f"   ({100 * (1 - np.median(v) / np.median(base)):4.0f}% removed)")

print(f"\nfitted per-pair depth scale: median {np.median(col('scale')):.3f}"
      f"  min {col('scale').min():.3f}  max {col('scale').max():.3f}")
print(f"fitted per-pair offset:      median {np.median(col('offset')) * 100:+.1f} cm"
      f"  min {col('offset').min() * 100:+.1f}  max {col('offset').max() * 100:+.1f}")

# --- are the per-pair scales mutually consistent? ------------------------
# If every pair's disagreement is one scale per *frame*, then for pair (i, j)
# log s_j - log s_i = log(observed ratio), and a least-squares solve over the
# graph must reproduce every observation. What it cannot reproduce is the part
# that is not a per-frame depth scale at all.
idx = sorted({r["i"] for r in rows} | {r["j"] for r in rows})
pos = {k: n for n, k in enumerate(idx)}
A = np.zeros((len(rows), len(idx)))
b = np.zeros(len(rows))
for n, r in enumerate(rows):
    A[n, pos[r["i"]]] = -1.0
    A[n, pos[r["j"]]] = 1.0
    b[n] = np.log(max(r["scale"], 1e-6))
A = np.vstack([A, np.ones((1, len(idx)))])   # gauge: scales average to 1
b = np.append(b, 0.0)
sol, *_ = np.linalg.lstsq(A, b, rcond=None)
pred_log = A[:-1] @ sol
resid_log = b[:-1] - pred_log
rank = np.linalg.matrix_rank(A)
spare = len(rows) + 1 - rank
print(f"\nper-frame depth scales solved over the pair graph ({len(idx)} frames,"
      f" {len(rows)} constraints):")
print(f"  design matrix rank {rank} of {len(idx)} unknowns · "
      f"{spare} constraint(s) to spare")
if spare <= 0:
    # Said plainly because the residual below is otherwise the most persuasive
    # wrong number here: with no redundancy the solve reproduces every
    # observation by construction, and a ~0% residual means the graph is a
    # forest, not that per-frame scales explain anything.
    print("  UNDERDETERMINED — the residual below is an artifact, not evidence.")
print(f"  implied per-frame scale spread: {np.exp(sol).max() / np.exp(sol).min():.2f}x")
print(f"  residual after the solve: median {np.median(np.abs(resid_log)) * 100:.1f}%"
      f"  p90 {np.percentile(np.abs(resid_log), 90) * 100:.1f}%")

print(f"\n{'pair':22} {'n':>6} {'raw':>7} {'+off':>7} {'xscale':>7} {'plane':>7} {'scale':>6}")
for r in sorted(rows, key=lambda r: -r["base"])[:12]:
    print(f"{r['a']}->{r['c']:10} {r['n']:6d} {r['base'] * 100:6.1f} "
          f"{r['additive'] * 100:6.1f} {r['multiplicative'] * 100:6.1f} "
          f"{r['rotational'] * 100:6.1f} {r['scale']:6.3f}")
