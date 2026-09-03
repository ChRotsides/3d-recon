# ruff: noqa: E501
"""Live dashboard for a reconstruction run. Read-only: it never touches the run.

Polls the output directory and COLMAP's SQLite database (opened read-only) and
serves one page that refreshes itself: stage, keypoint and match statistics,
the chain of verified matches between neighbouring keyframes, keypoint and
match overlays rendered on demand, and the viewers once fusion has written them.
"""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

PAIR_MOD = 2147483647
MIN_INLIERS = 15


def _db(root: Path) -> sqlite3.Connection | None:
    path = root / "colmap" / "database.db"
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.5)
        conn.execute("pragma query_only = 1")
        return conn
    except sqlite3.Error:
        return None


def _names(conn: sqlite3.Connection) -> dict[int, str]:
    return {int(i): n for i, n in conn.execute("select image_id, name from images")}


_STATE: dict = {}


def _display_rotation(root: Path) -> int:
    """Clockwise degrees that make the sensor-native frames upright on screen."""
    if "rotation" in _STATE:
        return _STATE["rotation"]
    rot = 90
    prov = root / "bundle" / "provenance.json"
    if prov.exists():
        with contextlib.suppress(KeyError, ValueError, json.JSONDecodeError):
            rot = int(
                json.loads(prov.read_text())["manifest_stages"]["recording"][
                    "sensor_orientation_deg"
                ]
            )
    _STATE["rotation"] = rot % 360
    return _STATE["rotation"]


def _upright(img: np.ndarray, xy: np.ndarray | None, rotation: int):
    """Rotate an image clockwise by `rotation` and map pixel coordinates along with it."""
    h, w = img.shape[:2]
    if rotation == 90:
        out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        pts = None if xy is None else np.stack([h - 1 - xy[:, 1], xy[:, 0]], axis=1)
    elif rotation == 180:
        out = cv2.rotate(img, cv2.ROTATE_180)
        pts = None if xy is None else np.stack([w - 1 - xy[:, 0], h - 1 - xy[:, 1]], axis=1)
    elif rotation == 270:
        out = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pts = None if xy is None else np.stack([xy[:, 1], w - 1 - xy[:, 0]], axis=1)
    else:
        out, pts = img, xy
    return out, pts


def _status(root: Path, log_path: Path | None) -> dict:
    out: dict = {"time": time.time(), "root": str(root), "stage": "waiting"}
    keyframes = root / "keyframes.json"
    if keyframes.exists():
        kf = json.loads(keyframes.read_text())
        out["keyframes"] = len(kf["indices"])
        out["stage"] = "keyframes"
    conn = _db(root)
    if conn is not None:
        try:
            names = _names(conn)
            kp_rows = conn.execute("select image_id, rows from keypoints").fetchall()
            counts = np.array([r for _, r in kp_rows], dtype=int)
            out["extracted"] = int(len(kp_rows))
            if len(counts):
                out["keypoints"] = {
                    "median": int(np.median(counts)),
                    "min": int(counts.min()),
                    "max": int(counts.max()),
                }
            out["stage"] = "extracting features"
            if len(kp_rows) >= out.get("keyframes", 10**9):
                out["stage"] = "matching"
            matched = conn.execute("select count(*) from matches").fetchone()[0]
            verified = conn.execute("select count(*) from two_view_geometries").fetchone()[0]
            out["matched_pairs"] = int(matched)
            out["verified_pairs"] = int(verified)
            if verified:
                ordered = sorted(names.items(), key=lambda kv: kv[1])
                pos = {i: k for k, (i, _) in enumerate(ordered)}
                chain = np.zeros(max(len(ordered) - 1, 0), dtype=int)
                degree = np.zeros(len(ordered), dtype=int)
                for pid, rows in conn.execute("select pair_id, rows from two_view_geometries"):
                    a, b = pos.get(pid // PAIR_MOD), pos.get(pid % PAIR_MOD)
                    if a is None or b is None or rows < MIN_INLIERS:
                        continue
                    degree[a] += 1
                    degree[b] += 1
                    lo, hi = min(a, b), max(a, b)
                    if hi - lo == 1:
                        chain[lo] = max(chain[lo], rows)
                out["chain"] = chain.tolist()
                out["weak_links"] = int((chain < MIN_INLIERS).sum())
                out["degree_median"] = int(np.median(degree)) if len(degree) else 0
            out["names"] = [n for _, n in sorted(names.items(), key=lambda kv: kv[1])]
        except sqlite3.Error as exc:
            out["db_error"] = str(exc)
        finally:
            conn.close()
    sparse = root / "colmap" / "sparse"
    if sparse.exists() and any(sparse.iterdir()):
        out["stage"] = "mapping"
    sfm = root / "sfm.json"
    if sfm.exists():
        s = json.loads(sfm.read_text())
        out["sfm"] = s.get("stats", {})
        out["stage"] = "aligning"
    manifest = root / "manifest.json"
    if (root / "bundle" / "bundle.json").exists():
        out["stage"] = "depth"
    if (root / "depth.npz").exists():
        out["stage"] = "fusing"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        out["manifest"] = {"warnings": m.get("warnings", []), "stages": m.get("stages", {})}
        out["stage"] = "done" if (root / "mesh.ply").exists() else "finished (no mesh)"
    got = _pair_matrix(root) if "extracted" in out else None
    if got is not None:
        m, new_pairs = got
        out["latest_pair"] = _STATE.get("latest_pair")
        out["new_pairs_since_last_poll"] = len(new_pairs)
        n = len(m)
        out["pairs_total"] = n * (n - 1) // 2
        out["strong_pairs"] = int((np.triu(m, 1) >= MIN_INLIERS).sum())
    out["sparse"] = _refresh_sparse(root, _STATE)
    for name in ("sparse", "cloud", "mesh"):
        f = root / f"{name}.json"
        out[f"{name}_mtime"] = f.stat().st_mtime if f.exists() else None
    cj = root / "cloud.json"
    if cj.exists():
        try:
            c = json.loads(cj.read_text())
            out["cloud"] = {
                "count": c.get("count"),
                "frames": c.get("keyframes"),
                "partial": c.get("partial"),
            }
        except json.JSONDecodeError:
            pass
    if log_path and log_path.exists():
        text = log_path.read_text(errors="replace")
        # COLMAP reports "Registering image #12 (num_reg_frames=196)" as it goes.
        reg = re.findall(r"num_reg_frames=(\d+)", text)
        if reg and "sfm" not in out:
            out["mapping_registered"] = int(reg[-1])
            if "mapping" in out["stage"] or out["stage"] == "matching":
                out["stage"] = "mapping"
        lines = [
            ln
            for ln in text.splitlines()
            if ln and not ln.startswith(("I2", "W2", "E2", " ", "[", "Warning"))
        ]
        out["log"] = lines[-12:]
    return out


def _keypoint_image(root: Path, name: str) -> bytes:
    conn = _db(root)
    if conn is None:
        raise HTTPException(404, "no database yet")
    try:
        row = conn.execute(
            "select k.rows, k.cols, k.data from keypoints k join images i on i.image_id=k.image_id "
            "where i.name=?",
            (name,),
        ).fetchone()
    finally:
        conn.close()
    img = cv2.imread(str(root / "images" / name))
    if img is None:
        raise HTTPException(404, "no such image")
    rotation = _display_rotation(root)
    if row is not None:
        n, cols, blob = row
        kp = np.frombuffer(blob, dtype=np.float32).reshape(n, cols)
        img, xy = _upright(img, kp[:, :2], rotation)
        for r, (x, y) in zip(kp, xy, strict=True):
            scale = np.sqrt(abs(np.linalg.det(r[2:6].reshape(2, 2)))) if cols >= 6 else float(r[2])
            cv2.circle(img, (int(x), int(y)), max(2, int(scale)), (0, 255, 0), 1, cv2.LINE_AA)
    else:
        img, _ = _upright(img, None, rotation)
    if row is not None:
        cv2.putText(
            img, f"{name}  {n} keypoints", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3
        )
    target_h = 720
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.resize(img, (int(target_h * img.shape[1] / img.shape[0]), target_h)),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    return buf.tobytes()


def _match_image(root: Path, index: int, other: int | None = None) -> bytes:
    conn = _db(root)
    if conn is None:
        raise HTTPException(404, "no database yet")
    try:
        names = sorted(_names(conn).items(), key=lambda kv: kv[1])
        j = index + 1 if other is None else other
        if index < 0 or j < 0 or index >= len(names) or j >= len(names) or index == j:
            raise HTTPException(404, "pair out of range")
        (ia, na), (ib, nb) = names[index], names[j]
        lo, hi = (ia, ib) if ia < ib else (ib, ia)
        pid = lo * PAIR_MOD + hi
        geom = conn.execute(
            "select rows, cols, data from two_view_geometries where pair_id=?", (pid,)
        ).fetchone()
        kps = {}
        for iid in (ia, ib):
            r = conn.execute(
                "select rows, cols, data from keypoints where image_id=?", (iid,)
            ).fetchone()
            kps[iid] = (
                np.frombuffer(r[2], dtype=np.float32).reshape(r[0], r[1])[:, :2] if r else None
            )
    finally:
        conn.close()
    a = cv2.imread(str(root / "images" / na))
    b = cv2.imread(str(root / "images" / nb))
    if a is None or b is None:
        raise HTTPException(404, "images missing")
    rotation = _display_rotation(root)
    a, kps[ia] = _upright(a, kps[ia], rotation)
    b, kps[ib] = _upright(b, kps[ib], rotation)
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), dtype=np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1] :] = b
    inliers = 0
    if geom is not None and geom[0] > 0 and kps[ia] is not None and kps[ib] is not None:
        m = np.frombuffer(geom[2], dtype=np.uint32).reshape(geom[0], geom[1])
        if ia > ib:
            m = m[:, ::-1]
        inliers = len(m)
        for i, j in m[:: max(1, len(m) // 400)]:
            p = kps[ia][i]
            q = kps[ib][j]
            cv2.line(
                canvas,
                (int(p[0]), int(p[1])),
                (int(q[0]) + a.shape[1], int(q[1])),
                (0, 220, 0),
                1,
                cv2.LINE_AA,
            )
    label = f"{na} -> {nb}: {inliers} verified inliers" + (
        "" if geom is not None else " (not matched yet)"
    )
    cv2.putText(canvas, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    target_h = 900
    small = cv2.resize(canvas, (int(canvas.shape[1] * target_h / canvas.shape[0]), target_h))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _pair_matrix(root: Path) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """(N, N) verified inliers between keyframes in capture order, plus pairs new since last poll."""
    conn = _db(root)
    if conn is None:
        return None
    try:
        names = _names(conn)
        ordered = sorted(names.items(), key=lambda kv: kv[1])
        pos = {i: k for k, (i, _) in enumerate(ordered)}
        n = len(ordered)
        m = np.zeros((n, n), dtype=np.int32)
        seen: set[int] = _STATE.setdefault("seen_pairs", set())
        new_pairs: list[tuple[int, int]] = []
        for pid, rows in conn.execute("select pair_id, rows from two_view_geometries"):
            a, b = pos.get(pid // PAIR_MOD), pos.get(pid % PAIR_MOD)
            if a is None or b is None:
                continue
            m[a, b] = m[b, a] = rows
            if pid not in seen:
                seen.add(pid)
                if rows >= MIN_INLIERS:
                    new_pairs.append((min(a, b), max(a, b)))
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if new_pairs:
        _STATE["latest_pair"] = new_pairs[-1]
    return m, new_pairs


def _matrix_png(root: Path) -> bytes:
    got = _pair_matrix(root)
    if got is None:
        raise HTTPException(404, "no database yet")
    m, _ = got
    n = len(m)
    if n == 0:
        raise HTTPException(404, "no images yet")
    # log scale: 0 inliers black, 15 dim, hundreds bright; below the threshold shown dark red
    v = np.log1p(np.maximum(m, 0)) / np.log1p(max(int(m.max()), 200))
    img = cv2.applyColorMap((np.clip(v, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    img[m == 0] = (12, 12, 12)
    weak = (m > 0) & (m < MIN_INLIERS)
    img[weak] = (40, 30, 120)
    size = 640
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    step = max(1, n // 8)
    for k in range(0, n, step):
        y = int(k * size / n)
        cv2.putText(
            img, str(k), (2, max(12, y + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWER = REPO_ROOT / "viewer" / "cloud.html"


def _ensure_viewer(root: Path) -> None:
    if VIEWER.exists() and root.exists() and not (root / "index.html").exists():
        (root / "index.html").write_bytes(VIEWER.read_bytes())


def _latest_snapshot(root: Path) -> Path | None:
    snaps = root / "colmap" / "snapshots"
    if not snaps.exists():
        return None
    dirs = [d for d in snaps.iterdir() if d.is_dir() and (d / "images.bin").exists()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _write_sparse(
    root: Path, pts: np.ndarray, cols: np.ndarray, cams: np.ndarray, source: str
) -> None:
    """Sparse points plus the camera path (bright red) in the cloud viewer's format."""
    if len(cams):
        pts = np.concatenate([pts, cams]) if len(pts) else cams
        cols = (
            np.concatenate([cols, np.tile([255, 40, 40], (len(cams), 1))])
            if len(cols)
            else np.tile([255, 40, 40], (len(cams), 1))
        )
    pos = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    col = np.asarray(cols, dtype=np.uint8).reshape(-1, 3)
    tmp = root / "sparse.bin.tmp"
    with open(tmp, "wb") as f:
        f.write(pos.tobytes())
        f.write(col.tobytes())
    tmp.replace(root / "sparse.bin")
    if len(pos):
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        vlo, vhi = np.percentile(pos, [2.0, 98.0], axis=0)
    else:
        lo = hi = vlo = vhi = np.zeros(3)
    extent = float(np.max(vhi - vlo)) or 1.0
    meta = {
        "count": int(len(pos)),
        "min": lo.tolist(),
        "max": hi.tolist(),
        "centre": ((lo + hi) / 2).tolist(),
        "extent": (hi - lo).tolist(),
        "view_min": vlo.tolist(),
        "view_max": vhi.tolist(),
        "view_centre": ((vlo + vhi) / 2).tolist(),
        "view_extent": (vhi - vlo).tolist(),
        "model": source,
        "keyframes": int(len(cams)),
        "voxel_m": extent / 150.0,
        "units": "m" if source == "sfm.json (metric)" else "relative",
    }
    tmp = root / "sparse.json.tmp"
    tmp.write_text(json.dumps(meta))
    tmp.replace(root / "sparse.json")


def _refresh_sparse(root: Path, state: dict) -> dict | None:
    """Rebuild sparse.bin when its source changed. Returns a small summary."""
    bundle = root / "bundle" / "bundle.json"
    sfm = root / "sfm.json"
    snap = _latest_snapshot(root)
    if bundle.exists():
        src, stamp = bundle, bundle.stat().st_mtime
    elif sfm.exists():
        src, stamp = sfm, sfm.stat().st_mtime
    elif snap is not None:
        src, stamp = snap, (snap / "images.bin").stat().st_mtime
    else:
        return None
    key = (str(src), stamp)
    if state.get("sparse_key") == key:
        return state.get("sparse_summary")
    try:
        if src == bundle:
            b = json.loads(bundle.read_text())
            pts = np.asarray(
                [p["xyz"] for p in b.get("sparse_points", [])], dtype=np.float64
            ).reshape(-1, 3)
            cols = np.full((len(pts), 3), 200, dtype=np.uint8)
            cams = np.asarray([k["pose"]["t"] for k in b["keyframes"]], dtype=np.float64).reshape(
                -1, 3
            )
            summary = {
                "source": "bundle (metric, Y up)",
                "points": int(len(pts)),
                "cameras": int(len(cams)),
            }
            _write_sparse(root, pts, cols, cams, "sfm.json (metric)")
        elif src == sfm:
            d = json.loads(sfm.read_text())
            pts = np.asarray(d["points"], dtype=np.float64).reshape(-1, 3)
            cols = np.full((len(pts), 3), 200, dtype=np.uint8)
            cams = []
            for m in d["images"].values():
                w2c = np.asarray(m)
                cams.append(-w2c[:3, :3].T @ w2c[:3, 3])
            cams = np.asarray(cams).reshape(-1, 3)
            summary = {
                "source": "sfm.json (COLMAP frame)",
                "points": int(len(pts)),
                "cameras": int(len(cams)),
            }
            _write_sparse(root, pts, cols, cams, "COLMAP")
        else:
            import pycolmap

            rec = pycolmap.Reconstruction(str(snap))
            pts = np.asarray([p.xyz for p in rec.points3D.values()], dtype=np.float64).reshape(
                -1, 3
            )
            cols = np.asarray([p.color for p in rec.points3D.values()], dtype=np.uint8).reshape(
                -1, 3
            )
            cams = np.asarray(
                [im.projection_center() for im in rec.images.values() if im.has_pose],
                dtype=np.float64,
            ).reshape(-1, 3)
            summary = {
                "source": f"mapper snapshot {snap.name}",
                "points": int(len(pts)),
                "cameras": int(len(cams)),
            }
            _write_sparse(root, pts, cols, cams, "COLMAP snapshot")
    except Exception as exc:  # a snapshot mid-write, a truncated json: try again next poll
        return {"error": str(exc)}
    state["sparse_key"] = key
    state["sparse_summary"] = summary
    return summary


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Video reconstruction</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#ddd}
header{padding:12px 18px;background:#1b1b1b;display:flex;gap:24px;align-items:baseline}
h1{font-size:18px;margin:0}.stage{font-size:22px;color:#ffd54f}
main{padding:14px 18px;display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.card{background:#1b1b1b;border-radius:8px;padding:12px}.card h2{font-size:13px;color:#999;margin:0 0 8px;text-transform:uppercase}
.big{font-size:28px}.muted{color:#888;font-size:12px}
.wide{grid-column:1/-1}svg{width:100%;height:120px;background:#161616;border-radius:6px}
img{max-width:100%;border-radius:6px}pre{font-size:11px;white-space:pre-wrap;color:#aaa;margin:0}
.warn{color:#ff8a65}a{color:#80cbc4}input{width:70px}
</style></head><body>
<header><h1>Video reconstruction</h1><span class="stage" id="stage">…</span><span class="muted" id="root"></span></header>
<main>
<div class="card"><h2>Keyframes</h2><div class="big" id="kf">–</div><div class="muted">chosen from the video by rotation / time</div></div>
<div class="card"><h2>SIFT</h2><div class="big" id="ext">–</div><div class="muted" id="kpstats"></div></div>
<div class="card"><h2>Matching</h2><div class="big" id="pairs">–</div><div class="muted" id="verified"></div></div>
<div class="card"><h2>Registration</h2><div class="big" id="reg">–</div><div class="muted" id="regstats"></div></div>
<div class="card wide"><h2>Verified inliers to the next keyframe (red = under 15, a break in the chain)</h2><svg id="chain" preserveAspectRatio="none" viewBox="0 0 1000 120"></svg><div class="muted" id="chainstats"></div></div>
<div class="card"><h2>Match matrix: keyframe × keyframe, verified inliers (live)</h2><div class="muted" id="matrixinfo"></div><img id="matrix" style="image-rendering:pixelated"><div class="muted">bright = many inliers · dark purple = under 15 (rejected) · black = not matched. The diagonal band is the video order; blocks away from it are loop closures.</div></div>
<div class="card"><h2>Most recently verified pair (live)</h2><label><input type="checkbox" id="livepair" checked> follow</label> <span class="muted" id="pairinfo"></span><img id="livepairimg"></div>
<div class="card wide"><h2>Sparse model: points and camera path (red)</h2><div class="muted" id="sparseinfo">waiting for COLMAP…</div><iframe id="sparseview" src="about:blank" style="width:100%;height:520px;border:0;border-radius:6px;background:#000"></iframe></div>
<div class="card wide"><h2>Dense cloud as it is fused</h2><div class="muted" id="cloudinfo">waiting for fusion…</div><iframe id="cloudview" src="about:blank" style="width:100%;height:520px;border:0;border-radius:6px;background:#000"></iframe></div>
<div class="card wide"><h2>Sensor alignment &amp; scale</h2><pre id="align">–</pre></div>
<div class="card"><h2>Keypoints on a keyframe</h2><div>index <input id="kpi" type="number" value="0" min="0"> <button onclick="showKp()">show</button> <button onclick="latestKp()">latest</button></div><img id="kpimg"></div>
<div class="card"><h2>Matches between keyframe i and i+1</h2><div>i <input id="mi" type="number" value="0" min="0"> <button onclick="showMatch()">show</button> <button onclick="weakest()">weakest link</button></div><img id="mimg"></div>
<div class="card wide"><h2>Log</h2><pre id="log"></pre></div>
<div class="card wide" id="result" hidden><h2>Result</h2><p><a href="/walk" target="_blank"><b>Walk: play the video and move through the 3D room in sync</b></a> · <a href="/out/index.html" target="_blank">Point cloud viewer</a> · <a href="/out/mesh.html" target="_blank">Mesh viewer</a> · <a href="/out/manifest.json" target="_blank">manifest.json</a></p><pre id="fusion"></pre></div>
<div class="card wide" id="warnings" hidden><h2>Warnings</h2><pre class="warn" id="warntext"></pre></div>
</main>
<script>
let S={};
function showKp(){const i=+document.getElementById('kpi').value;if(!S.names||!S.names[i])return;document.getElementById('kpimg').src='/keypoints/'+S.names[i]+'?t='+Date.now();}
function latestKp(){if(!S.extracted)return;document.getElementById('kpi').value=S.extracted-1;showKp();}
function showMatch(){const i=+document.getElementById('mi').value;document.getElementById('mimg').src='/matches/'+i+'?t='+Date.now();}
function weakest(){if(!S.chain)return;let m=0;for(let i=0;i<S.chain.length;i++)if(S.chain[i]<S.chain[m])m=i;document.getElementById('mi').value=m;showMatch();}
function drawChain(c){const svg=document.getElementById('chain');if(!c||!c.length){svg.innerHTML='';return;}
 const max=Math.max(60,...c);const w=1000/c.length;let s='';
 for(let i=0;i<c.length;i++){const h=Math.min(1,c[i]/max)*110;s+=`<rect x="${i*w}" y="${115-h}" width="${Math.max(w-0.5,0.5)}" height="${h}" fill="${c[i]<15?'#ef5350':'#66bb6a'}"/>`;}
 svg.innerHTML=s;}
let tick=async function(){try{const r=await fetch('/status');S=await r.json();}catch(e){return;}
 document.getElementById('stage').textContent=S.stage;document.getElementById('root').textContent=S.root;
 document.getElementById('kf').textContent=S.keyframes??'–';
 document.getElementById('ext').textContent=(S.extracted??0)+' / '+(S.keyframes??'?');
 document.getElementById('kpstats').textContent=S.keypoints?`keypoints per image: median ${S.keypoints.median}, min ${S.keypoints.min}, max ${S.keypoints.max}`:'';
 document.getElementById('pairs').textContent=S.matched_pairs??'–';
 document.getElementById('verified').textContent=S.verified_pairs!=null?`${S.verified_pairs} pairs verified geometrically; median ${S.degree_median??0} partners per image`:'';
 if(!S.sfm&&S.mapping_registered!=null){document.getElementById('reg').textContent=`${S.mapping_registered} / ${S.keyframes??'?'}`;document.getElementById('regstats').textContent=`mapping in progress: ${Math.round(100*S.mapping_registered/(S.keyframes||1))}% of keyframes registered so far`;}
 if(S.sfm){document.getElementById('reg').textContent=`${S.sfm.images_registered} / ${S.sfm.images_total}`;
  document.getElementById('regstats').textContent=`${S.sfm.models} model(s), ${S.sfm.points} points, reprojection ${(+S.sfm.mean_reprojection_error_px).toFixed(2)} px, focal ${(+S.sfm.focal_refined_px[0]).toFixed(0)} px (prior ${(+S.sfm.focal_prior_px[0]).toFixed(0)})`;}
 drawChain(S.chain);document.getElementById('chainstats').textContent=S.chain?`${S.weak_links} weak links of ${S.chain.length}`:'';
 const a=S.manifest&&S.manifest.stages&&S.manifest.stages.align;document.getElementById('align').textContent=a?JSON.stringify(a,null,1):'–';
 document.getElementById('log').textContent=(S.log||[]).join('\\n');
 const w=S.manifest&&S.manifest.warnings||[];document.getElementById('warnings').hidden=!w.length;document.getElementById('warntext').textContent=w.join('\\n');
 const f=S.manifest&&S.manifest.stages&&S.manifest.stages.fusion;document.getElementById('result').hidden=!(S.stage==='done');if(f)document.getElementById('fusion').textContent=JSON.stringify(f,null,1);
}
let seen={sparse:null,cloud:null};
function refreshViewer(name,mtime,frameId){if(mtime&&mtime!==seen[name]){seen[name]=mtime;document.getElementById(frameId).src='/out/index.html?base='+name+'&t='+Math.floor(mtime);}}
function viewers(){refreshViewer('sparse',S.sparse_mtime,'sparseview');refreshViewer('cloud',S.cloud_mtime,'cloudview');
 document.getElementById('sparseinfo').textContent=S.sparse?(S.sparse.error?('sparse: '+S.sparse.error):`${S.sparse.points} points, ${S.sparse.cameras} cameras · ${S.sparse.source} · reloads when the model changes`):'waiting for COLMAP…';
 document.getElementById('cloudinfo').textContent=S.cloud?`${S.cloud.count} points from ${S.cloud.frames} fused frames${S.cloud.partial?' (partial, still fusing)':' (final)'}`:'waiting for fusion…';}
let lastPair=null,lastMatrix=0;
function matching(){if(S.pairs_total!=null){document.getElementById('matrixinfo').textContent=`${S.verified_pairs??0} of ${S.pairs_total} pairs matched · ${S.strong_pairs??0} with ≥15 inliers`;
  if(Date.now()-lastMatrix>6000&&(S.new_pairs_since_last_poll||!lastMatrix)){lastMatrix=Date.now();document.getElementById('matrix').src='/matrix.png?t='+lastMatrix;}}
 const p=S.latest_pair;if(p&&document.getElementById('livepair').checked&&(''+p)!==(''+lastPair)){lastPair=p;document.getElementById('livepairimg').src=`/matches/${p[0]}/${p[1]}?t=${Date.now()}`;document.getElementById('pairinfo').textContent=`keyframes ${p[0]} ↔ ${p[1]}`;}}
const _tick=tick;tick=async function(){await _tick();viewers();matching();};
tick();setInterval(tick,3000);
</script></body></html>"""


def create_app(root: Path, log_path: Path | None = None, recording: Path | None = None) -> FastAPI:
    app = FastAPI(title="Video reconstruction monitor")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/status")
    def status() -> JSONResponse:
        return JSONResponse(_status(root, log_path))

    @app.get("/keypoints/{name}")
    def keypoints(name: str) -> Response:
        return Response(_keypoint_image(root, name), media_type="image/jpeg")

    @app.get("/matches/{index}")
    def matches(index: int) -> Response:
        return Response(_match_image(root, index), media_type="image/jpeg")

    @app.get("/matches/{index}/{other}")
    def matches_pair(index: int, other: int) -> Response:
        return Response(_match_image(root, index, other), media_type="image/jpeg")

    @app.get("/matrix.png")
    def matrix() -> Response:
        return Response(_matrix_png(root), media_type="image/png")

    from videorecon import walk

    @app.get("/walk", response_class=HTMLResponse)
    def walk_page() -> str:
        return walk.WALK_PAGE

    @app.get("/walk/status")
    def walk_status() -> JSONResponse:
        return JSONResponse(walk.preview_status(root, recording or walk.recording_dir(root)))

    @app.get("/walk/trajectory")
    def walk_trajectory() -> JSONResponse:
        return JSONResponse(walk.trajectory(root, recording or walk.recording_dir(root)))

    if root.exists():
        _ensure_viewer(root)
        app.mount("/out", StaticFiles(directory=str(root)), name="out")
    return app
