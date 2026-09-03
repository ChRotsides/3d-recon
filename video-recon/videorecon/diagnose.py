# ruff: noqa: E501
"""Measure why a reconstruction looks wrong, instead of guessing.

Everything here reads the finished outputs of a run (bundle, depth cache,
cloud) and the recording, and writes numbers plus pictures into
`<out>/diagnostics/`. Each measurement targets one visible symptom:

* doubled walls  -> cross-view depth agreement, per-frame anchor shape error
* split floor    -> floor height per frame from its own depth map
* points outside -> far-depth fraction
* smearing along the walk -> sensor-vs-SfM residual over time (drift)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_harness() -> None:
    try:
        import recon  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(REPO_ROOT / "harness"))


@dataclass
class FrameGeometry:
    id: str
    depth: np.ndarray  # (h, w) metres
    k: np.ndarray  # 3x3 at depth resolution
    c2w_cv: np.ndarray  # 4x4 camera(OpenCV) -> world(Y up)
    w2c_cv: np.ndarray


def _load_frames(
    bundle_dir: Path, depth_npz: Path, limit: int | None = None
) -> list[FrameGeometry]:
    _ensure_harness()
    from recon.frames import world_to_camera_cv
    from recon.io.bundle import BundleDir

    bundle = BundleDir.load(bundle_dir)
    cache = np.load(depth_npz)
    frames = []
    for kf in bundle.bundle.keyframes:
        if kf.id not in cache:
            continue
        d = np.asarray(cache[kf.id], dtype=np.float32)
        h, w = d.shape
        intr = kf.intrinsics
        sx, sy = w / intr.width, h / intr.height
        k = np.array([[intr.fx * sx, 0, intr.cx * sx], [0, intr.fy * sy, intr.cy * sy], [0, 0, 1]])
        w2c = world_to_camera_cv(kf)
        frames.append(FrameGeometry(kf.id, d, k, np.linalg.inv(w2c), w2c))
        if limit and len(frames) >= limit:
            break
    return frames


def _backproject(f: FrameGeometry, step: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 3) and their depths for a subsampled pixel grid."""
    h, w = f.depth.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = f.depth[v, u]
    ok = z > 0
    u, v, z = u[ok].astype(np.float64), v[ok].astype(np.float64), z[ok].astype(np.float64)
    x = (u - f.k[0, 2]) / f.k[0, 0] * z
    y = (v - f.k[1, 2]) / f.k[1, 1] * z
    pc = np.stack([x, y, z], axis=1)
    pw = pc @ f.c2w_cv[:3, :3].T + f.c2w_cv[:3, 3]
    return pw, z


def floor_per_frame(frames: list[FrameGeometry]) -> dict:
    """Floor height each frame's own depth map implies, in the world frame.

    If depth were consistent across frames every value would be the same
    number: the floor is one plane. Spread here is per-frame depth error.
    """
    heights = []
    cam_y = []
    for f in frames:
        pw, _ = _backproject(f)
        if len(pw) < 500:
            continue
        cy = f.c2w_cv[1, 3]
        below = pw[:, 1][pw[:, 1] < cy - 0.4]
        if len(below) < 200:
            continue
        # the lowest dense band: 200 bins between p0.5 and p50 of below-camera points
        lo, hi = np.percentile(below, [0.5, 60])
        hist, edges = np.histogram(below, np.linspace(lo, hi, 120))
        smooth = np.convolve(hist, np.ones(3), mode="same")
        cands = np.flatnonzero(smooth >= 0.4 * smooth.max())
        k = int(cands.min())
        y0 = 0.5 * (edges[k] + edges[k + 1])
        band = (below > y0 - 0.06) & (below < y0 + 0.06)
        heights.append(float(np.median(below[band])))
        cam_y.append(cy)
    hts = np.asarray(heights)
    return {
        "frames_with_floor": int(len(hts)),
        "floor_y_median_m": float(np.median(hts)) if len(hts) else None,
        "floor_y_std_m": float(np.std(hts)) if len(hts) else None,
        "floor_y_p10_p90_spread_m": float(np.percentile(hts, 90) - np.percentile(hts, 10))
        if len(hts)
        else None,
        "camera_height_above_floor_median_m": float(np.median(np.asarray(cam_y) - hts))
        if len(hts)
        else None,
        "per_frame": hts.tolist(),
    }


def anchor_check(bundle_dir: Path, frames: list[FrameGeometry]) -> dict:
    """Depth vs COLMAP's points, per frame: residual scale and near/far shape error."""
    _ensure_harness()
    from recon.io.bundle import BundleDir

    bundle = BundleDir.load(bundle_dir)
    by_id = {f.id: f for f in frames}
    obs: dict[str, list[np.ndarray]] = {}
    for p in bundle.bundle.sparse_points:
        for i in p.observations:
            obs.setdefault(i, []).append(np.asarray(p.xyz))
    ratios, dispersions, slopes, counts = [], [], [], []
    for fid, pts in obs.items():
        f = by_id.get(fid)
        if f is None or len(pts) < 30:
            continue
        p = np.asarray(pts)
        pc = p @ f.w2c_cv[:3, :3].T + f.w2c_cv[:3, 3]
        z = pc[:, 2]
        ok = z > 0.2
        u = f.k[0, 0] * pc[:, 0] / z + f.k[0, 2]
        v = f.k[1, 1] * pc[:, 1] / z + f.k[1, 2]
        h, w = f.depth.shape
        ok &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if ok.sum() < 30:
            continue
        pred = f.depth[v[ok].astype(int), u[ok].astype(int)]
        good = pred > 0
        r = pred[good] / z[ok][good]
        if len(r) < 30:
            continue
        med = float(np.median(r))
        disp = float(np.median(np.abs(r - med)) / med)
        # near/far: how the ratio changes between the nearest and farthest third of
        # the anchor points; 0 means a pure scale error, negative means the model
        # compresses depth (far things predicted too near). Medians, so a few
        # wrong sparse points cannot set it.
        zz = z[ok][good]
        near, far = zz <= np.percentile(zz, 33), zz >= np.percentile(zz, 67)
        if near.sum() >= 8 and far.sum() >= 8:
            slope = float(
                np.log(np.median(r[far]) / np.median(r[near]))
                / np.log(np.median(zz[far]) / np.median(zz[near]))
            )
        else:
            slope = 0.0
        ratios.append(med)
        dispersions.append(disp)
        slopes.append(slope)
        counts.append(int(len(r)))
    return {
        "frames": len(ratios),
        "residual_scale_median": float(np.median(ratios)) if ratios else None,
        "residual_scale_p10_p90": [
            float(np.percentile(ratios, 10)),
            float(np.percentile(ratios, 90)),
        ]
        if ratios
        else None,
        "within_frame_dispersion_median": float(np.median(dispersions)) if dispersions else None,
        "near_far_slope_median": float(np.median(slopes)) if slopes else None,
        "near_far_slope_abs_p90": float(np.percentile(np.abs(slopes), 90)) if slopes else None,
        "points_per_frame_median": float(np.median(counts)) if counts else None,
    }


def cross_view(frames: list[FrameGeometry], offsets: tuple[int, ...] = (1, 5, 20)) -> dict:
    """Project frame A's depth into frame B and compare with B's own depth."""
    out = {}
    for off in offsets:
        rel = []
        for i in range(0, len(frames) - off, max(1, (len(frames) - off) // 60)):
            a, b = frames[i], frames[i + off]
            pw, _ = _backproject(a, step=6)
            pc = pw @ b.w2c_cv[:3, :3].T + b.w2c_cv[:3, 3]
            z = pc[:, 2]
            ok = z > 0.2
            u = b.k[0, 0] * pc[:, 0] / np.where(ok, z, 1) + b.k[0, 2]
            v = b.k[1, 1] * pc[:, 1] / np.where(ok, z, 1) + b.k[1, 2]
            h, w = b.depth.shape
            ok &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
            if ok.sum() < 200:
                continue
            zb = b.depth[v[ok].astype(int), u[ok].astype(int)]
            good = zb > 0
            # signed: positive means A's surface lies *behind* what B sees there
            rel.append((z[ok][good] - zb[good]) / zb[good])
        if not rel:
            continue
        r = np.concatenate(rel)
        out[f"offset_{off}"] = {
            "pairs": len(rel),
            "median_abs_rel": float(np.median(np.abs(r))),
            "within_5pct": float(np.mean(np.abs(r) < 0.05)),
            "within_10pct": float(np.mean(np.abs(r) < 0.10)),
            "beyond_25pct": float(np.mean(np.abs(r) > 0.25)),
        }
    return out


def far_depth(frames: list[FrameGeometry], limit_m: float = 5.0) -> dict:
    total = 0
    far = 0
    zero = 0
    for f in frames:
        d = f.depth
        total += d.size
        far += int((d > limit_m).sum())
        zero += int((d <= 0).sum())
    return {
        "fraction_beyond_limit": far / max(total, 1),
        "limit_m": limit_m,
        "fraction_no_return": zero / max(total, 1),
    }


def drift_over_time(out_dir: Path, recording_dir: Path) -> dict:
    """Sensor-vs-SfM orientation residual along the walk, from the bundle's poses."""
    from videorecon import align, imu
    from videorecon.recording import Recording

    b = json.loads((out_dir / "bundle" / "bundle.json").read_text())
    rec = Recording.load(recording_dir)
    pts_path = out_dir / "video_pts.json"
    if pts_path.exists():
        pts = np.asarray(json.loads(pts_path.read_text()), dtype=np.int64)
    else:
        from videorecon.recording import video_frame_pts_ns

        pts = video_frame_pts_ns(rec.video_path)
    from videorecon.recording import match_video_to_capture_results

    timing = match_video_to_capture_results(pts, rec.frames[:, 1])
    track = rec.orientation_track("gamerotv")
    r_dc = imu.device_to_camera(rec.sensor_orientation_deg)
    idx = np.asarray([int(k["id"][2:]) for k in b["keyframes"]])
    t_ns = timing.timestamps_ns[idx]
    r_imu = np.einsum("nij,jk->nik", track.rotation_at(t_ns), r_dc)  # cv camera -> ENU
    r_imu_gl = np.einsum("ij,njk,kl->nil", imu.ENU_TO_Y_UP, r_imu, imu.CV_TO_GL)
    r_sfm = np.asarray([k["pose"]["R"] for k in b["keyframes"]])
    rot = align.solve_rotation(r_imu_gl, r_sfm)
    res = rot.residual_deg
    t = (t_ns - t_ns[0]) * 1e-9
    thirds = np.array_split(np.arange(len(res)), 3)
    return {
        "median_deg": rot.median_deg,
        "p95_deg": rot.p95_deg,
        "median_by_third_deg": [float(np.median(res[i])) for i in thirds],
        "worst_10_frames": [(int(idx[i]), float(res[i])) for i in np.argsort(res)[-10:][::-1]],
        "time_s": t.tolist(),
        "residual_deg": res.tolist(),
    }


def render(
    out_dir: Path, diag_dir: Path, floor: dict, drift: dict | None, floor_y: float | None
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    meta = json.loads((out_dir / "cloud.json").read_text())
    n = meta["count"]
    pos = np.fromfile(out_dir / "cloud.bin", dtype=np.float32, count=n * 3).reshape(-1, 3)
    b = json.loads((out_dir / "bundle" / "bundle.json").read_text())
    cams = np.asarray([k["pose"]["t"] for k in b["keyframes"]])
    fy = floor_y if floor_y is not None else float(np.percentile(pos[:, 1], 3))

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    band = (pos[:, 1] > fy + 0.4) & (pos[:, 1] < fy + 1.6)
    ax[0].hexbin(pos[band, 0], -pos[band, 2], gridsize=260, bins="log", cmap="magma", mincnt=1)
    ax[0].plot(cams[:, 0], -cams[:, 2], "c-", lw=1, label="camera path")
    ax[0].set_aspect("equal")
    ax[0].set_title("Top-down slice 0.4–1.6 m above the floor (walls should be single thin lines)")
    ax[0].set_xlabel("x east (m)")
    ax[0].set_ylabel("north (m)")
    ax[0].legend(loc="upper right")
    ax[1].hexbin(pos[:, 0], pos[:, 1], gridsize=260, bins="log", cmap="magma", mincnt=1)
    ax[1].axhline(fy, color="c", lw=0.8, label="floor from sparse points")
    ax[1].plot(cams[:, 0], cams[:, 1], "w.", ms=1, label="cameras")
    ax[1].set_aspect("equal")
    ax[1].set_title("Side elevation, all points (floor and ceiling should be flat lines)")
    ax[1].set_xlabel("x east (m)")
    ax[1].set_ylabel("y up (m)")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    p = diag_dir / "slices.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(str(p))

    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    hts = np.asarray(floor.get("per_frame", []))
    ax[0].plot(hts, ".-", ms=3)
    ax[0].axhline(fy, color="c", label="floor from sparse points")
    ax[0].set_title(
        "Floor height implied by each keyframe's own depth map (should be one flat line)"
    )
    ax[0].set_xlabel("keyframe (in order)")
    ax[0].set_ylabel("floor y (m)")
    ax[0].legend()
    if drift:
        ax[1].plot(drift["time_s"], drift["residual_deg"], ".", ms=3)
        ax[1].set_title(
            "Sensor vs SfM orientation residual along the walk (a rising trend is drift)"
        )
        ax[1].set_xlabel("time (s)")
        ax[1].set_ylabel("degrees")
    fig.tight_layout()
    p = diag_dir / "per_frame.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    written.append(str(p))
    return written


def run(
    out_dir: Path, recording_dir: Path | None, *, depth_npz: Path | None = None, label: str = ""
) -> dict:
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    frames = _load_frames(out_dir / "bundle", depth_npz or out_dir / "depth.npz")
    report: dict = {"label": label, "frames": len(frames)}
    b = json.loads((out_dir / "bundle" / "bundle.json").read_text())
    k = b["keyframes"][0]["intrinsics"]
    sfm = out_dir / "sfm.json"
    if sfm.exists():
        s = json.loads(sfm.read_text())
        report["intrinsics"] = {
            "bundle_fx": k["fx"],
            "sfm_refined_fx": s["intrinsics"]["fx"],
            "mismatch_pct": 100 * (k["fx"] / s["intrinsics"]["fx"] - 1),
        }
    report["floor_per_frame"] = floor_per_frame(frames)
    report["anchor"] = anchor_check(out_dir / "bundle", frames)
    report["cross_view"] = cross_view(frames)
    report["far_depth"] = far_depth(frames)
    drift = None
    if recording_dir is not None and recording_dir.exists():
        try:
            drift = drift_over_time(out_dir, recording_dir)
            report["drift"] = {
                k2: v for k2, v in drift.items() if k2 not in ("time_s", "residual_deg")
            }
        except Exception as exc:  # noqa: BLE001 - diagnostics must not die on one missing input
            report["drift"] = {"error": str(exc)}
    floor_sparse = None
    if b.get("sparse_points"):
        pts = np.asarray([p["xyz"] for p in b["sparse_points"]])
        cams = np.asarray([kf["pose"]["t"] for kf in b["keyframes"]])
        from videorecon import align

        est = align.scale_from_floor(pts, cams)
        if est.usable:
            floor_sparse = float(np.median(cams[:, 1]) - 1.5)
    if (out_dir / "cloud.bin").exists():
        report["images"] = render(out_dir, diag_dir, report["floor_per_frame"], drift, floor_sparse)
    slim = {
        kk: (vv if kk != "floor_per_frame" else {a: c for a, c in vv.items() if a != "per_frame"})
        for kk, vv in report.items()
    }
    (diag_dir / f"report{('-' + label) if label else ''}.json").write_text(
        json.dumps(slim, indent=2)
    )
    return slim
