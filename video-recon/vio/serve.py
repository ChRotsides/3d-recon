"""Static file server with HTTP Range support (python's http.server has none; video seeking needs it)."""
import os, sys, re, json, struct
from urllib.parse import urlparse, parse_qs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_depth import LIVE
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
class H(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        rng = self.headers.get("Range")
        if os.path.isfile(path) and rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            size = os.path.getsize(path)
            start = int(m.group(1)) if m.group(1) else 0
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            f = open(path, "rb"); f.seek(start)
            self._range = (f, end - start + 1)
            return None
        self._range = None
        return super().send_head()
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/depth":
            q = parse_qs(u.query)
            try:
                res, err = LIVE.infer(q["rec"][0], q["run"][0], float(q["t"][0]), int(q.get("res", ["362"])[0]))
            except Exception as e:
                res, err = None, f"{type(e).__name__}: {e}"
            if err:
                body = err.encode(); self.send_response(503); self.send_header("Content-Type", "text/plain"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            hdr, depth = res; h = json.dumps(hdr).encode(); body = struct.pack("<I", len(h)) + h + depth
            self.send_response(200); self.send_header("Content-Type", "application/octet-stream"); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body); return
        f = self.send_head()
        if getattr(self, "_range", None):
            fh, n = self._range
            try:
                while n > 0:
                    chunk = fh.read(min(1 << 16, n))
                    if not chunk: break
                    self.wfile.write(chunk); n -= len(chunk)
            except BrokenPipeError: pass
            finally: fh.close()
        elif f:
            try: self.copyfile(f, self.wfile)
            except BrokenPipeError: pass
            finally: f.close()
    def log_message(self, *a): pass
os.chdir(sys.argv[2]); ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1])), H).serve_forever()
