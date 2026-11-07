"""ctypes binding to the phone's voxel-hash TSDF (android/app/src/main/cpp/tsdf.cpp, unchanged).

Same parameters as LiveActivity: 3 cm voxels, a 4-voxel truncation band, weights capped at 16,
surfaces shown from 2 observations, 6 m maximum depth. Depth is uint16 millimetres, K is
(fx, fy, cx, cy), R is camera->world (OpenCV axes) row-major and c is the camera centre.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
_LIB = _DIR / "build" / "libtsdf.so"


def _load():
    if not _LIB.exists():
        subprocess.run([str(_DIR / "build.sh")], check=True, capture_output=True)
    lib = ctypes.CDLL(str(_LIB))
    lib.tsdf_create.restype = ctypes.c_void_p
    lib.tsdf_create.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    for fn in (lib.tsdf_destroy, lib.tsdf_clear):
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = None
    lib.tsdf_blocks.argtypes = [ctypes.c_void_p]
    lib.tsdf_blocks.restype = ctypes.c_longlong
    p = ctypes.POINTER
    lib.tsdf_integrate.argtypes = [ctypes.c_void_p, p(ctypes.c_uint16), ctypes.c_int, ctypes.c_int,
                                   p(ctypes.c_float), p(ctypes.c_float), p(ctypes.c_float),
                                   p(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int, p(ctypes.c_double)]
    lib.tsdf_integrate.restype = None
    lib.tsdf_surface.argtypes = [ctypes.c_void_p, p(ctypes.c_float), ctypes.c_longlong]
    lib.tsdf_surface.restype = ctypes.c_longlong
    return lib


_libc = None


class Tsdf:
    def __init__(self, voxel: float = 0.03, trunc_voxels: int = 4, max_weight: float = 16.0,
                 min_weight: float = 2.0, max_depth: float = 6.0):
        global _libc
        if _libc is None:
            _libc = _load()
        self.lib = _libc
        self.h = self.lib.tsdf_create(voxel, trunc_voxels, max_weight, min_weight, max_depth)
        self.voxel = voxel
        self.stats = np.zeros(5, dtype=np.float64)
        self._buf = np.empty(0, dtype=np.float32)

    def __del__(self):
        if getattr(self, "h", None):
            self.lib.tsdf_destroy(ctypes.c_void_p(self.h))
            self.h = None

    def clear(self) -> None:
        self.lib.tsdf_clear(ctypes.c_void_p(self.h))

    @property
    def blocks(self) -> int:
        return int(self.lib.tsdf_blocks(ctypes.c_void_p(self.h)))

    def integrate(self, depth_mm: np.ndarray, K, R_cw: np.ndarray, c_w: np.ndarray,
                  rgb: np.ndarray | None = None, stride: int = 2) -> dict:
        d = np.ascontiguousarray(depth_mm, dtype=np.uint16)
        h, w = d.shape
        kk = np.ascontiguousarray(K, dtype=np.float32)
        rr = np.ascontiguousarray(np.asarray(R_cw, dtype=np.float32).reshape(9))
        cc = np.ascontiguousarray(np.asarray(c_w, dtype=np.float32).reshape(3))
        p = ctypes.POINTER
        if rgb is None:
            rgba_p, rw, rh = None, 0, 0
        else:
            rgba = np.ascontiguousarray(rgb, dtype=np.uint8)
            if rgba.shape[2] == 3:
                rgba = np.concatenate([rgba, np.full(rgba.shape[:2] + (1,), 255, np.uint8)], axis=2)
                rgba = np.ascontiguousarray(rgba)
            rh, rw = rgba.shape[:2]
            self._rgba = rgba  # keep alive for the call
            rgba_p = rgba.ctypes.data_as(p(ctypes.c_uint8))
        self.lib.tsdf_integrate(ctypes.c_void_p(self.h), d.ctypes.data_as(p(ctypes.c_uint16)), w, h,
                                kk.ctypes.data_as(p(ctypes.c_float)), rr.ctypes.data_as(p(ctypes.c_float)),
                                cc.ctypes.data_as(p(ctypes.c_float)), rgba_p, rw, rh, stride,
                                self.stats.ctypes.data_as(p(ctypes.c_double)))
        s = self.stats
        return {"integrate_ms": s[0], "extract_ms": s[1], "blocks_touched": int(s[2]),
                "blocks": int(s[3]), "points": int(s[4])}

    def surface(self, max_points: int = 2_000_000) -> np.ndarray:
        """(N, 6) float32: x, y, z, r, g, b with rgb in 0..1, in world (Z-up) coordinates."""
        if len(self._buf) < max_points * 6:
            self._buf = np.empty(max_points * 6, dtype=np.float32)
        n = int(self.lib.tsdf_surface(ctypes.c_void_p(self.h),
                                      self._buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), max_points))
        return self._buf[: n * 6].reshape(n, 6).copy()
