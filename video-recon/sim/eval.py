"""Scoring a run: how much of the world was explored, how good the map is, how good the pose is.

Everything the robot produced lives in its own estimate frame; the ground truth lives in the
world frame. One Sim(3) fit (Umeyama, the same one video-recon/vio/export_walk2.py uses) between
the two camera paths relates them, and every map metric is computed after applying it.

usage: eval.py OUT_DIR [--plots]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    """Least-squares similarity src -> dst; returns (s, R, t) with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ d @ Vt
    s = (np.trace(np.diag(sig) @ d) / (S**2).sum() * len(src)) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def load_tum(path):
    v = np.loadtxt(path, ndmin=2)
    if v.ndim != 2 or v.shape[1] < 8:     # an empty file (VIO that never initialised)
        return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 4))
    return v[:, 0], v[:, 1:4], v[:, 4:8]


def match_times(ta, tb, tol=0.02):
    j = np.searchsorted(ta, tb)
    j = np.clip(j, 1, max(1, len(ta) - 1))
    j = np.where(np.abs(ta[j - 1] - tb) < np.abs(ta[j] - tb), j - 1, j)
    return j, np.abs(ta[j] - tb) < tol


def explored_fraction(world, state, origin_xy, res: float, s: float, R: np.ndarray, t: np.ndarray):
    """Fraction of the GT free cells whose position, mapped into the estimate frame, is
    marked known-free. (s, R, t) map estimate -> world, so the inverse is applied here."""
    from .navgrid import FREE
    from .world import GRID_RES

    iy, ix = np.nonzero(world.free)
    if not len(iy):
        return 0.0, np.zeros(0, bool)
    p = np.zeros((len(iy), 3))
    p[:, 0] = (ix + 0.5) * GRID_RES
    p[:, 1] = (iy + 0.5) * GRID_RES
    p[:, 2] = 0.5  # the camera plane, which is what the height band is referenced to
    q = (p - t) @ R / s  # world -> estimate
    gx = ((q[:, 0] - origin_xy[0]) / res).astype(int)
    gy = ((q[:, 1] - origin_xy[1]) / res).astype(int)
    ok = (gx >= 0) & (gx < state.shape[1]) & (gy >= 0) & (gy < state.shape[0])
    hit = np.zeros(len(iy), bool)
    hit[ok] = state[gy[ok], gx[ok]] == FREE
    return float(hit.mean()), hit


def evaluate(out_dir: Path, plots: bool = True) -> dict:
    import open3d as o3d

    from .navgrid import NavGrid
    from .world import World

    out_dir = Path(out_dir)
    cfg = json.loads((out_dir / "stats.json").read_text())
    world = World.generate(cfg["seed"], cfg["world"])
    tg, pg, qg = load_tum(out_dir / "gt.txt")
    te, pe, qe = load_tum(out_dir / "est.txt")
    rep = {"config": cfg["config"], "runtime": cfg["runtime"], "stage_ms": cfg["stage_ms"],
           "collisions": cfg["collisions"], "distance_m": cfg["distance_m"],
           "stuck_events": cfg["stuck_events"], "vio_init_s": cfg["vio_init_s"],
           "gt_free_area_m2": round(world.free_area, 2)}

    j, ok = match_times(tg, te)
    if ok.sum() >= 10:
        A, B = pe[ok], pg[j[ok]]
        s1, R1, t1 = umeyama(A, B, with_scale=False)
        err = np.linalg.norm((R1 @ A.T).T + t1 - B, axis=1)
        s3, _, _ = umeyama(A, B, with_scale=True)
        rep["ate_se3"] = {"rmse_cm": round(100 * float(np.sqrt((err**2).mean())), 1),
                          "median_cm": round(100 * float(np.median(err)), 1),
                          "max_cm": round(100 * float(err.max()), 1)}
        rep["scale_error"] = round(float(s3), 4)
        n = max(3, len(A) // 50)
        rep["start_end_drift_cm"] = round(100 * float(np.linalg.norm(
            (A[-n:].mean(0) - A[:n].mean(0)) - (B[-n:].mean(0) - B[:n].mean(0)))), 1)
        rep["ate_series"] = [[round(float(te[ok][i]), 2), round(100 * float(err[i]), 1)]
                             for i in range(0, len(err), max(1, len(err) // 200))]
        s, R, t = s1, R1, t1
    else:
        rep["ate_se3"] = None
        rep["scale_error"] = None
        s, R, t = 1.0, np.eye(3), np.zeros(3)
    rep["alignment"] = {"scale": round(float(s), 4), "R": np.round(R, 6).tolist(), "t": np.round(t, 4).tolist()}

    # -- exploration from the final map (a run whose pose source never initialised has none)
    grid_path = out_dir / "navgrid.npz"
    if grid_path.exists():
        d = np.load(grid_path)
        g = NavGrid(res=float(d["res"]))
        g.log, g.origin = d["log"], d["origin"]
        state, cost, org = g.costmap(cfg["floor_z"])
        frac, _ = explored_fraction(world, state, org, g.res, s, R, t)
    else:
        frac = 0.0
    rep["explored_final"] = round(frac, 4)
    rep["explored_area_m2"] = round(frac * world.free_area, 2)
    ex = out_dir / "explored.csv"
    rep["explored_series"] = ([[float(a), float(b)] for a, b in np.loadtxt(ex, delimiter=",", ndmin=2)]
                              if ex.exists() and ex.stat().st_size else [])

    # -- map accuracy and completeness
    ply = out_dir / "tsdf_surface.ply"
    if ply.exists():
        pcd = o3d.io.read_point_cloud(str(ply))
        P = np.asarray(pcd.points)
        if len(P):
            Pw = (s * (R @ P.T)).T + t
            scene = world.scene
            dist = scene.compute_distance(o3d.core.Tensor(Pw.astype(np.float32))).numpy()
            rep["map_accuracy_cm"] = {"median": round(100 * float(np.median(dist)), 2),
                                      "p90": round(100 * float(np.percentile(dist, 90)), 2),
                                      "points": int(len(P))}
            # the same distances after a similarity fit, which separates a wrong metric scale
            # (a property of the estimator) from a wrong map shape
            if ok.sum() >= 10 and rep.get("scale_error"):
                s2, R2, t2 = umeyama(pe[ok], pg[j[ok]], with_scale=True)
                P2 = (s2 * (R2 @ P.T)).T + t2
                d2 = scene.compute_distance(o3d.core.Tensor(P2.astype(np.float32))).numpy()
                rep["map_accuracy_sim3_cm"] = {"median": round(100 * float(np.median(d2)), 2),
                                               "p90": round(100 * float(np.percentile(d2, 90)), 2)}
            from scipy.spatial import cKDTree

            gt_s = world.sample_surface(150_000, seed=cfg["seed"])
            near = cKDTree(Pw).query(gt_s, k=1)[0]
            rep["completeness_5cm"] = round(float((near < 0.05).mean()), 4)
        else:
            rep["map_accuracy_cm"] = None
            rep["completeness_5cm"] = 0.0

    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    if plots:
        _plots(out_dir, rep, world, pg, pe, s, R, t)
    return rep


def _plots(out_dir: Path, rep, world, pg, pe, s, R, t):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    ser = np.asarray(rep.get("explored_series") or [[0, 0]])
    axs[0].plot(ser[:, 0], 100 * ser[:, 1], lw=2)
    axs[0].set_xlabel("sim time (s)")
    axs[0].set_ylabel("explored (% of GT free area)")
    axs[0].set_ylim(0, 100)
    axs[0].grid(alpha=0.3)
    axs[0].set_title(f"explored: {100 * rep['explored_final']:.0f} % of {rep['gt_free_area_m2']} m2")

    a = np.asarray(rep.get("ate_series") or [[0, 0]])
    axs[1].plot(a[:, 0], a[:, 1], lw=1.5, color="crimson")
    axs[1].set_xlabel("sim time (s)")
    axs[1].set_ylabel("ATE after SE(3) alignment (cm)")
    axs[1].grid(alpha=0.3)
    e = rep.get("ate_se3") or {}
    axs[1].set_title(f"ATE rmse {e.get('rmse_cm', '-')} cm, scale {rep.get('scale_error')}")

    axs[2].imshow(world.free.astype(float), origin="lower", cmap="Greys_r",
                  extent=[0, world.extent[0], 0, world.extent[1]], alpha=0.6)
    if len(pe):
        Pw = (s * (R @ pe.T)).T + t
        axs[2].plot(Pw[:, 0], Pw[:, 1], color="crimson", lw=1.5, label="estimate (aligned)")
    if len(pg):
        axs[2].plot(pg[:, 0], pg[:, 1], color="darkorange", lw=1.2, alpha=0.8, label="ground truth")
    axs[2].set_aspect("equal")
    axs[2].legend(fontsize=8)
    axs[2].set_title("paths over the GT free space")
    fig.suptitle(f"{out_dir.name}: pose={rep['config']['pose']} depth={rep['config']['depth']} "
                 f"world={rep['config']['world']} seed={rep['config']['seed']}")
    fig.tight_layout()
    fig.savefig(out_dir / "plots.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # run as a script, so re-enter through the package to keep the relative imports working
    import runpy
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if __package__ in (None, ""):
        sys.argv[0] = "sim.eval"
        runpy.run_module("sim.eval", run_name="__main__", alter_sys=True)
        raise SystemExit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    print(json.dumps(evaluate(Path(a.out_dir), plots=not a.no_plots), indent=1))
