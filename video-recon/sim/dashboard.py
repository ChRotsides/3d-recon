"""Live view of a run: the camera, the estimated depth beside the ground truth, the top-down
costmap the robot is actually planning on, the TSDF surface in three.js, and the stats panel.

The server runs in a daemon thread and only ever reads the snapshot the simulation thread
publishes, so it cannot slow the loop down.
"""

from __future__ import annotations

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
WWW = _DIR / "www"
THREE_SRC = Path("/home/charalambos/ai/vio/www")
ZUP_TO_YUP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]], dtype=np.float32)
MAX_VIEW_POINTS = 400_000


def _ensure_assets():
    WWW.mkdir(exist_ok=True)
    for n in ("three.min.js", "OrbitControls.js", "PLYLoader.js"):
        dst = WWW / n
        if not dst.exists() and (THREE_SRC / n).exists():
            shutil.copy(THREE_SRC / n, dst)


def _yup(p: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=np.float32) @ ZUP_TO_YUP.T)


def serve(sim, port: int, host: str = "0.0.0.0"):
    _ensure_assets()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path in ("/", "/index.html"):
                    return self._send((_DIR / "dashboard.html").read_bytes(), "text/html")
                if path.lstrip("/") in ("three.min.js", "OrbitControls.js", "PLYLoader.js"):
                    return self._send((WWW / path.lstrip("/")).read_bytes(), "application/javascript")
                with sim.lock:
                    snap = dict(sim.snap)
                if path == "/state":
                    st = snap.get("stats", {})
                    s, R, t = sim.align
                    pe = np.array([r[1:4] for r in sim.est_log[-4000:]], dtype=np.float32).reshape(-1, 3)
                    pg = np.array([r[1:4] for r in sim.gt_log[-4000:]], dtype=np.float32).reshape(-1, 3)
                    if len(pg):
                        pg = ((pg - t) @ R / s).astype(np.float32)  # GT into the estimate frame
                    st = dict(st)
                    st["est_path"] = np.round(_yup(pe[::2]), 3).tolist()
                    st["gt_path"] = np.round(_yup(pg[::2]), 3).tolist()
                    st["points"] = int(len(snap.get("surface", [])))
                    st["has"] = {k: (k in snap) for k in ("rgb", "depth", "gtdepth", "costmap")}
                    return self._send(json.dumps(st, default=float).encode(), "application/json")
                if path == "/points.bin":
                    S = snap.get("surface", np.zeros((0, 6), np.float32))
                    if len(S) > MAX_VIEW_POINTS:
                        S = S[np.linspace(0, len(S) - 1, MAX_VIEW_POINTS).astype(int)]
                    out = np.empty((len(S), 6), np.float32)
                    out[:, :3] = _yup(S[:, :3])
                    out[:, 3:] = S[:, 3:]
                    return self._send(out.tobytes(), "application/octet-stream")
                for name, ctype in (("rgb", "image/jpeg"), ("depth", "image/jpeg"),
                                    ("gtdepth", "image/jpeg"), ("costmap", "image/png")):
                    if path.startswith(f"/{name}."):
                        b = snap.get(name)
                        if b is None:
                            self.send_response(404)
                            self.send_header("Content-Length", "0")
                            self.end_headers()
                            return
                        return self._send(b, ctype)
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # a dashboard bug must never take the run down
                body = str(e).encode()
                try:
                    self._send(body, "text/plain")
                except Exception:
                    pass

    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
