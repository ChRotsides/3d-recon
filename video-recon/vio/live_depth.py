"""Live depth for the walk viewer: DA3-Small runs when the page asks, on the frame at the
video's current time, conditioned on the VIO poses of a sliding window of recent requests.
Nothing is computed ahead of playback and nothing is cached across requests.

GET /depth?rec=<rec>&run=<run name>&t=<video seconds>  ->  binary: 4-byte header length,
JSON header {w,h,fx,fy,cx,cy,t,c,q,ms_decode,ms_model,window}, then uint16 depth (mm) w*h.
"""
import json, sys, time, threading
from pathlib import Path
import numpy as np, cv2
from scipy.spatial.transform import Rotation
REPO = Path("/home/charalambos/Documents/Projects/3d-recon")
sys.path.insert(0, str(REPO / "harness"))
RECS = {"rec0045": REPO / "data/recordings/rec_20260903-070045", "rec2445": REPO / "data/recordings/rec_20260903-232445"}
WWW = Path(__file__).resolve().parent / "www"
SHORT = 362; WINDOW = 4
CV_TO_GL = np.diag([1.0, -1.0, -1.0])

class LiveDepth:
    def __init__(self):
        self.lock = threading.Lock(); self.model = None; self.caps = {}; self.walks = {}; self.windows = {}; self.intr = {}; self.last_idx = {}
    def _model(self):
        if self.model is None:
            import torch
            from recon.depth.da3 import DA3Config, DA3Depth   # the repo adapter: Kepler FP32 forward, legacy-CUDA setup
            self.torch = torch
            self.adapter = DA3Depth(DA3Config(variant="small", enforce_commercial_licence=False, anchor_to_sparse_points=False))
            self.model = self.adapter._load()
        return self.model
    def _cap(self, rec, proxy=False):
        """Low resolutions decode from the 960x540 streaming proxy (4x cheaper per frame) when it exists."""
        ck = (rec, proxy)
        if ck not in self.caps:
            src = WWW / rec / "video_proxy.mp4" if proxy and (WWW / rec / "video_proxy.mp4").exists() else RECS[rec] / "video.mp4"
            self.caps[ck] = cv2.VideoCapture(str(src))
        if rec not in self.intr:
            meta = json.loads((RECS[rec] / "meta.json").read_text())
            fx, fy, cx, cy, _ = meta["camera"]["intrinsic_calibration"]
            # video is a centre crop of the pre-correction array to 16:9 then scaled to 1920x1080 (videorecon.recording)
            aw, ah = meta["camera"]["pre_correction_active_array"]["width"], meta["camera"]["pre_correction_active_array"]["height"]
            ch = aw * 9 / 16; s = 1920 / aw
            self.intr[rec] = (fx * s, fy * s, cx * s, (cy - (ah - ch) / 2) * s)
            self.walks[rec] = json.loads((WWW / rec / "walk.json").read_text())
            self.windows[rec] = []
        return self.caps[ck]
    def pose_at(self, rec, run, t):
        P = np.asarray(self.walks[rec]["runs"][run]); k = int(np.argmin(np.abs(P[:, 0] - t)))
        if abs(P[k, 0] - t) > 0.05: return None
        return P[k]
    def frame_at(self, rec, t, short=SHORT):
        """Decode the frame at video time t. Requests arrive in playback order, so read
        forward from the last decoded frame when the target is a little ahead; seeking
        in H.264 means decoding from the previous keyframe and costs 150-300 ms."""
        proxy = short <= 270; cap = self._cap(rec, proxy); lk = (rec, proxy); idx = int(round(t * 30.0)); last = self.last_idx.get(lk, -10**9)
        if 0 < idx - last <= 90:
            for _ in range(idx - last - 1): cap.grab()
        elif idx != last + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read(); self.last_idx[lk] = idx if ok else -10**9
        return None if not ok else bgr
    def infer(self, rec, run, t, short=SHORT):
        with self.lock:
            t0 = time.time(); bgr = self.frame_at(rec, t, short)
            if bgr is None: return None, "no frame"
            p = self.pose_at(rec, run, t)
            if p is None: return None, "no VIO pose at this time"
            s = short / min(bgr.shape[:2]); W, H = int(round(bgr.shape[1] * s)), int(round(bgr.shape[0] * s))
            rgb = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            fx, fy, cx, cy = [v * (W / 1920.0) for v in self.intr[rec]]   # intrinsics are in 1920x1080 pixels, whichever file was decoded
            R_gl = Rotation.from_quat(p[4:8]).as_matrix(); R_wc = R_gl @ CV_TO_GL   # world<-camera (OpenCV axes)
            w2c = np.eye(4); w2c[:3, :3] = R_wc.T; w2c[:3, 3] = -R_wc.T @ p[1:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            win = self.windows[rec]; win = [e for e in win if e[0] < t and e[1].shape[0] == H][-(WINDOW - 1):]; win.append((t, rgb, w2c, K)); self.windows[rec] = win
            t1 = time.time(); model = self._model(); mode = "posed"
            if len(win) < 3:
                return None, f"warming up: {len(win)}/3 frames in the pose window"
            with self.torch.inference_mode():
                try:
                    out = model.inference(image=[e[1] for e in win], extrinsics=np.stack([e[2] for e in win]), intrinsics=np.stack([e[3] for e in win]),
                                          align_to_input_ext_scale=True, process_res=max(W, H))
                except Exception as ex:   # e.g. collinear camera centres: DA3's Umeyama scale fit is degenerate
                    if "Umeyama" not in str(ex) and "Degenerate" not in str(ex): raise
                    return None, f"pose window degenerate ({len(win)} nearly collinear centres); move on"
            d = np.asarray(out.depth[-1], dtype=np.float32)
            if d.shape != (H, W): d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
            t2 = time.time()
            hdr = {"w": W, "h": H, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "t": t, "c": p[1:4].tolist(), "q": p[4:8].tolist(),
                   "ms_decode": round(1e3 * (t1 - t0)), "ms_model": round(1e3 * (t2 - t1)), "window": len(win), "mode": mode}
            mm = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
            return (hdr, mm.tobytes()), None
LIVE = LiveDepth()
