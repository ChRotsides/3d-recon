"""Depth for the fusion stage: ground truth, or one of the four ONNX models the phone runs.

The preprocessing is the phone's, copied from DepthRunner.kt: nearest-neighbour resample of the
camera frame to the model's input grid, ImageNet normalisation, NCHW float32 (with a leading
view axis for DA3), and for the intrinsics-aware models a K scaled from the source size to the
model size. Metric3D's canonical output is multiplied by fx_model/1000; UniDepth and Metric3D
are used as they come; DA3 (relative depth) and DAv2 (relative inverse depth) need a scale.

The phone gets that scale from the VIO landmarks. A ground robot has something better: the
camera is bolted 0.50 m above the floor, so the floor plane's distance is known exactly. Each
frame back-projects the lower third, uses gravity from the pose to say where the floor should
be, and fits the one (or two) unknowns to put it there.

Everything comes out as uint16 millimetres on a 322x181 grid with K scaled to match, which is
what tsdf.cpp wants.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path("/home/charalambos/ai/phone/model")
FUSE_W, FUSE_H = 322, 181
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# file, input grid (h, w), kind
MODELS = {
    "da3": ("da3_small.onnx", "relative"),
    "dav2": ("dav2_small_model_quantized.onnx", "relative_inverse"),
    "metric3d": ("metric3d_vits_182x322.onnx", "metric_canonical"),
    "unidepth": ("unidepth_vits_182x322.onnx", "metric_k"),
}


def fusion_K(K_cam: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    sx, sy = FUSE_W / src_w, FUSE_H / src_h
    return np.array([K_cam[0] * sx, K_cam[1] * sy, K_cam[2] * sx, K_cam[3] * sy], dtype=np.float32)


def _to_mm(z_m: np.ndarray) -> np.ndarray:
    if z_m.shape != (FUSE_H, FUSE_W):
        z_m = cv2.resize(z_m, (FUSE_W, FUSE_H), interpolation=cv2.INTER_NEAREST)
    return np.clip(z_m * 1000.0, 0, 65535).astype(np.uint16)


class GtDepth:
    name = "gt"

    def __init__(self, K_cam, src_w, src_h, noisy: bool = False, seed: int = 0, **_):
        self.K = fusion_K(K_cam, src_w, src_h)
        self.noisy = noisy
        self.rng = np.random.default_rng(seed + 5150)
        self.scale_log: list[float] = []

    def compute(self, rgb, gt_depth, R_cw):
        t0 = time.time()
        z = cv2.resize(gt_depth, (FUSE_W, FUSE_H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if self.noisy:
            z = z * (1.0 + 0.03 * self.rng.standard_normal(z.shape).astype(np.float32))
            gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
            edge = np.hypot(gx, gy) > 0.15 * np.maximum(z, 0.3)
            z[edge] *= 1.0 + 0.08 * self.rng.standard_normal(int(edge.sum())).astype(np.float32)
        return _to_mm(z), {"ms": 1000 * (time.time() - t0), "scale": 1.0, "inliers": 1.0}


class OnnxDepth:
    def __init__(self, name: str, K_cam, src_w: int, src_h: int, cam_height: float = 0.50,
                 threads: int = 6, **_):
        import onnxruntime as ort

        self.name = name
        fn, self.kind = MODELS[name]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(MODEL_DIR / fn), so, providers=["CPUExecutionProvider"])
        info = {i.name: i.shape for i in self.sess.get_inputs()}
        self.k_name = next((n for n in info if n in ("K", "intrinsics")), None)
        self.in_name = next(n for n in info if n != self.k_name)
        shape = info[self.in_name]
        self.five_d = len(shape) == 5
        self.ih = shape[-2] if isinstance(shape[-2], int) else 182
        self.iw = shape[-1] if isinstance(shape[-1], int) else 322
        self.src_w, self.src_h = src_w, src_h
        self.K_cam = np.asarray(K_cam, dtype=np.float32)
        self.K = fusion_K(K_cam, src_w, src_h)
        # the phone's nearest-neighbour resample: sy = y * sh / ih, sx = x * sw / iw
        self.map_y = (np.arange(self.ih) * src_h // self.ih).astype(np.int32)
        self.map_x = (np.arange(self.iw) * src_w // self.iw).astype(np.int32)
        self.km = np.array([self.K_cam[0] * self.iw / src_w, self.K_cam[1] * self.ih / src_h,
                            self.K_cam[2] * self.iw / src_w, self.K_cam[3] * self.ih / src_h], dtype=np.float32)
        self.k_feed = np.array([[[self.km[0], 0, self.km[2]], [0, self.km[1], self.km[3]], [0, 0, 1.0]]], dtype=np.float32)
        self.cam_height = cam_height
        self._fit = None  # (a, b) for inverse models, or (s,) for relative
        self._out_rays = None
        self.scale_log: list[float] = []

    # -- the phone's preprocessing, verbatim
    def _pre(self, rgb: np.ndarray) -> np.ndarray:
        small = rgb[self.map_y][:, self.map_x].astype(np.float32) * np.float32(1.0 / 255.0)
        small = (small - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.ascontiguousarray(small.transpose(2, 0, 1))[None]
        return chw[None] if self.five_d else chw

    def _rays(self, h: int, w: int) -> np.ndarray:
        if self._out_rays is None or self._out_rays.shape[:2] != (h, w):
            fx = self.K_cam[0] * w / self.src_w
            fy = self.K_cam[1] * h / self.src_h
            cx = self.K_cam[2] * w / self.src_w
            cy = self.K_cam[3] * h / self.src_h
            u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._out_rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
        return self._out_rays

    def _floor_targets(self, pred: np.ndarray, R_cw: np.ndarray):
        """z the floor pixels of the lower third *should* have, given gravity and the 0.50 m mount."""
        h, w = pred.shape
        rays = self._rays(h, w)
        up_c = np.asarray(R_cw, dtype=np.float32).T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        y0 = int(h * 2 / 3)
        r = rays[y0:].reshape(-1, 3)
        d = pred[y0:].reshape(-1)
        proj = r @ up_c  # < 0 for rays pointing below the horizon
        m = (proj < -0.05) & np.isfinite(d)
        return d[m], -self.cam_height / proj[m]

    def _scale(self, pred: np.ndarray, R_cw: np.ndarray):
        d, z_t = self._floor_targets(pred, R_cw)
        if len(d) < 200:
            return self._fit, 0.0
        if self.kind == "relative":
            good = d > 1e-6
            ratio = z_t[good] / d[good]
            for _ in range(3):  # RANSAC-flavoured: keep what agrees with the current median
                s = float(np.median(ratio))
                keep = np.abs(ratio - s) < 0.10 * s
                if keep.sum() < 100:
                    break
                ratio = ratio[keep]
            inl = len(ratio) / max(1, len(d))
            return (float(np.median(ratio)),), inl
        # inverse: 1/z = a * d + b, refit after dropping the pixels that are not floor
        y = 1.0 / np.maximum(z_t, 1e-3)
        x = d.astype(np.float64)
        keep = np.ones(len(x), bool)
        for _ in range(3):
            if keep.sum() < 100:
                break
            A = np.stack([x[keep], np.ones(keep.sum())], 1)
            sol, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
            res = np.abs(x * sol[0] + sol[1] - y)
            keep = res < max(1e-3, 2.0 * np.median(res[keep]))
        if keep.sum() < 100 or sol[0] <= 0:
            return self._fit, 0.0
        return (float(sol[0]), float(sol[1])), keep.sum() / len(x)

    def compute(self, rgb, gt_depth, R_cw):
        t0 = time.time()
        feed = {self.in_name: self._pre(rgb)}
        if self.k_name:
            feed[self.k_name] = self.k_feed
        pred = np.asarray(self.sess.run(None, feed)[0], dtype=np.float32)
        pred = pred.reshape(pred.shape[-2], pred.shape[-1])
        info = {"inliers": 1.0}
        if self.kind == "metric_canonical":
            # the exported network predicts depth for a canonical camera with fx = 1000 px
            pred = pred * (self.km[0] / 1000.0)
            z = pred
        elif self.kind == "metric_k":
            z = pred
        else:
            fit, inl = self._scale(pred, R_cw)
            info["inliers"] = float(inl)
            if fit is None:
                return None, {"ms": 1000 * (time.time() - t0), "scale": 0.0, "inliers": 0.0}
            if inl > 0.15:
                self._fit = fit
            fit = self._fit
            if self.kind == "relative":
                z = pred * fit[0]
                info["scale"] = fit[0]
            else:
                inv = fit[0] * pred + fit[1]
                z = np.where(inv > 0.05, 1.0 / np.maximum(inv, 1e-6), 0.0)
                info["scale"] = fit[0]
            self.scale_log.append(float(info["scale"]))
        info.setdefault("scale", 1.0)
        info["ms"] = 1000 * (time.time() - t0)
        return _to_mm(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)), info


def make(kind: str, K_cam, src_w: int, src_h: int, **kw):
    if kind in ("gt", "gt_noisy"):
        return GtDepth(K_cam, src_w, src_h, noisy=(kind == "gt_noisy"), **kw)
    return OnnxDepth(kind, K_cam, src_w, src_h, **kw)
