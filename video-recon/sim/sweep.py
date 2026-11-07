"""Run the pose x depth x seed matrix sequentially and write docs/sim-results.md.

Each run has the machine to itself (the renderer, the ONNX depth model and OpenVINS all want
the CPU), so nothing here runs in parallel.

    python video-recon/sim/sweep.py --duration 180 --seeds 1 2 3 --out out/sim/sweep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable

POSES = ("gt", "gt_noisy", "vio")
DEPTHS = ("gt", "metric3d", "da3")


def run_one(pose, depth, seed, world, duration, out_root, dashboard=0, extra=(), threads=0):
    out = Path(out_root) / f"{world}_s{seed}_{pose}_{depth}"
    rep = out / "report.json"
    if rep.exists():
        return json.loads(rep.read_text())
    cmd = [PY, "-u", str(REPO / "video-recon/sim/run.py"), "--world", world, "--seed", str(seed),
           "--pose", pose, "--depth", depth, "--duration", str(duration), "--out", str(out), *extra]
    # runs are sequential, so one dashboard port serves the whole sweep, one run at a time
    cmd += ["--dashboard", str(dashboard)] if dashboard else ["--headless"]
    t0 = time.time()
    print(f"--- {out.name}", flush=True)
    log = out.parent / f"{out.name}.log"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if threads:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ORT_NUM_THREADS"):
            env[v] = str(threads)
    with open(log, "wb") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO, env=env)
    print(f"    exit {p.returncode} in {time.time() - t0:.0f} s", flush=True)
    if rep.exists():
        return json.loads(rep.read_text())
    reason = "no report"
    if log.exists():
        tail = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()][-1:]
        reason = tail[0][:110] if tail else reason
    return {"failed": True, "reason": reason, "log": str(log),
            "config": {"pose": pose, "depth": depth, "seed": seed, "world": world}}


def table(rows, duration, seeds):
    head = ("| pose | depth | seed | explored % | collisions | ATE rmse cm | scale | "
            "map median cm | map p90 cm | completeness % | dist m | VIO init s | RTF |")
    sep = "|---" * 13 + "|"
    lines = [head, sep]
    for r in rows:
        c = r.get("config", {})
        if r.get("failed"):
            why = str(r.get("reason", "run failed")).replace("|", "/")[:60]
            lines.append(f"| {c.get('pose','?')} | {c.get('depth','?')} | {c.get('seed','?')} | "
                         f"**failed**: {why} |" + " – |" * 10)
            continue
        a = r.get("ate_se3") or {}
        m = r.get("map_accuracy_cm") or {}
        lines.append(
            f"| {c['pose']} | {c['depth']} | {c['seed']} | {100 * r['explored_final']:.0f} | "
            f"{r['collisions']} | {a.get('rmse_cm', '–')} | {r.get('scale_error', '–')} | "
            f"{m.get('median', '–')} | {m.get('p90', '–')} | {100 * r.get('completeness_5cm', 0):.0f} | "
            f"{r['distance_m']} | {round(r['vio_init_s'], 2) if r.get('vio_init_s') is not None else '–'} | "
            f"{r['runtime']['rtf']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="apartment")
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--poses", nargs="+", default=list(POSES))
    ap.add_argument("--depths", nargs="+", default=list(DEPTHS))
    ap.add_argument("--out", default="out/sim/sweep")
    ap.add_argument("--doc", default="video-recon/docs/sim-results.md")
    ap.add_argument("--dashboard", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="cap BLAS/OMP/ORT threads per run")
    ap.add_argument("--vio-config", default=None, help="estimator config for the vio rows")
    ap.add_argument("--merge", action="store_true",
                    help="keep the rows in <out>/rows.json that this invocation does not rerun")
    a = ap.parse_args()

    rows_path = Path(a.out) / "rows.json"
    kept = []
    if a.merge and rows_path.exists():
        want = {(p, d, s) for p in a.poses for d in a.depths for s in a.seeds}
        for r in json.loads(rows_path.read_text()):
            c = r.get("config", {})
            if (c.get("pose"), c.get("depth"), c.get("seed")) not in want:
                kept.append(r)
        print(f"merging: keeping {len(kept)} existing rows", flush=True)
    rows = []
    t0 = time.time()
    total = len(a.poses) * len(a.depths) * len(a.seeds)
    for pose in a.poses:
        for depth in a.depths:
            for seed in a.seeds:
                extra = ["--vio-config", a.vio_config] if (a.vio_config and pose == "vio") else []
                rows.append(run_one(pose, depth, seed, a.world, a.duration, a.out, a.dashboard,
                                    extra, a.threads))
                done = len(rows)
                print(f"    [{done}/{total}] {100 * done / total:.0f} % of the sweep, "
                      f"{(time.time() - t0) / 60:.1f} min elapsed", flush=True)
                rows_path.write_text(json.dumps(kept + rows, indent=1))
    rows = kept + rows
    order = {p: i for i, p in enumerate(POSES)}
    dorder = {d: i for i, d in enumerate(DEPTHS)}
    rows.sort(key=lambda r: (order.get(r.get("config", {}).get("pose"), 9),
                             dorder.get(r.get("config", {}).get("depth"), 9),
                             r.get("config", {}).get("seed", 0)))

    doc = Path(REPO / a.doc)
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        f"# Simulated ground robot: pose x depth sweep\n\n"
        f"`video-recon/sim/sweep.py`, world `{a.world}`, seeds {a.seeds}, {a.duration:.0f} s of\n"
        f"simulated time per run, one run at a time on 12 CPU cores. Explored % is the fraction of\n"
        f"the ground-truth free floor area the robot's own map marks known-free, after aligning the\n"
        f"estimate frame to the world with a rigid Umeyama fit on the two camera paths. Map median /\n"
        f"p90 are the distances from the TSDF surface points to the ground-truth mesh; completeness\n"
        f"is the fraction of ground-truth surface samples within 5 cm of an estimated point.\n\n"
        f"Scale is the Sim(3) factor that maps the estimate onto the truth, so 0.52 means the\n"
        f"estimator's metric scale is 1/0.52 = 1.9x too large. A run ends early when the map has\n"
        f"no reachable frontier left for 25 s, so `dist m` and the real-time factor are over the\n"
        f"time the run actually took, not over {a.duration:.0f} s.\n\n"
        + table(rows, a.duration, a.seeds)
        + f"\n\nTotal sweep wall clock: {(time.time() - t0) / 60:.1f} min.\n")
    print(f"wrote {doc}", flush=True)


if __name__ == "__main__":
    main()
