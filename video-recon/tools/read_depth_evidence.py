#!/usr/bin/env python3
"""Decode phone depth evidence into named NumPy arrays and a JSON manifest.

Usage: python tools/read_depth_evidence.py depth-0.bin.gz --output /tmp/frame
Array pixels are rectified at depth_width x depth_height; gray is the exact
distorted tracking image. Pose is camera-to-map, Y-up, OpenGL camera axes.
"""
import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np


def read(path):
    with gzip.open(path, "rb") as f:
        def scalar(fmt):
            size = struct.calcsize(fmt)
            data = f.read(size)
            if len(data) != size:
                raise ValueError("truncated evidence")
            return struct.unpack(fmt, data)[0]

        def text():
            # Model names and reason strings are ASCII in version 1. Java uses
            # modified UTF-8 for writeUTF; decode null encoding defensively.
            return f.read(scalar(">H")).replace(b"\xc0\x80", b"\x00").decode("utf-8")

        def array(dtype):
            count = scalar(">i")
            if not 0 <= count <= 16_000_000:
                raise ValueError("invalid array length")
            dt = np.dtype(dtype)
            data = f.read(count * dt.itemsize)
            if len(data) != count * dt.itemsize:
                raise ValueError("truncated array")
            return np.frombuffer(data, dt).astype(dt.newbyteorder("="))

        if scalar(">i") != 0x44505448 or scalar(">i") != 1:
            raise ValueError("unsupported evidence format")
        meta = dict(timestamp=scalar(">d"), generation=scalar(">q"), map_revision=scalar(">q"),
                    model=text(), reason=text(), fused=bool(scalar(">?")),
                    depth_width=scalar(">i"), depth_height=scalar(">i"),
                    source_width=scalar(">i"), source_height=scalar(">i"))
        arrays = dict(pose=array(">f4"), intrinsics=array(">f4"), source_calibration=array(">f8"),
                      landmarks=array(">f4"), prediction=array(">f4"), depth_mm=array(">u2"),
                      confidence=array("u1"), rgba=array("u1"), gray=array("u1"))
        if f.read(1):
            raise ValueError("unexpected trailing bytes")
    w, h = meta["depth_width"], meta["depth_height"]
    for name in ("prediction", "depth_mm", "confidence"):
        if arrays[name].size:
            arrays[name] = arrays[name].reshape(h, w)
    arrays["rgba"] = arrays["rgba"].reshape(h, w, 4)
    arrays["gray"] = arrays["gray"].reshape(meta["source_height"], meta["source_width"])
    arrays["landmarks"] = arrays["landmarks"].reshape(-1, 3)
    return meta, arrays


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata, data = read(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **data)
    print(json.dumps(metadata, indent=2))
