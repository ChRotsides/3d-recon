#!/usr/bin/env python3
"""Assert that the prepared camera + main graphs reproduce the source UniDepth model on real captures.

Runs everything on the ONNX Runtime CPU provider: source(image, K) versus
main(image, camera(K)) for every depth-evidence capture, each with its own per-frame K, and exits
with status 1 if any frame differs by more than the tolerances. Read-only: writes nothing.

  python -I verify_unidepth_qnn.py --source unidepth_vits_182x322.onnx --prepared-dir OUT_DIR \
      --evidence-dir /path/to/depth-evidence/map-XXXX

Captures are the depth-*.bin.gz files written by DepthEvidenceRecorder (decoded with
tools/read_depth_evidence.py). Their RGBA is the rectified model-input image, 322x182.
"""
import argparse
import glob
import importlib.util
import os
import sys
import time

import numpy as np
import onnxruntime as ort

CAMERA_NAME = "qnn_camera.onnx"
MAIN_NAME = "qnn_main_static_gelu.onnx"
MEAN = np.array([.485, .456, .406])
STD = np.array([.229, .224, .225])


def load_decoder(path):
    spec = importlib.util.spec_from_file_location("read_depth_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    default_decoder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "read_depth_evidence.py")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="fp32 UniDepth ONNX the prepared files were made from")
    parser.add_argument("--prepared-dir", required=True, help=f"directory holding {CAMERA_NAME} and {MAIN_NAME}")
    parser.add_argument("--evidence-dir", required=True, help="directory with depth-*.bin.gz captures")
    parser.add_argument("--decoder", default=default_decoder, help="path to read_depth_evidence.py")
    parser.add_argument("--max-abs", type=float, default=1e-4, help="largest allowed |difference| in metres")
    parser.add_argument("--max-rel", type=float, default=1e-4, help="largest allowed relative difference")
    parser.add_argument("--min-frames", type=int, default=10, help="fail if fewer captures are compared")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    decoder = load_decoder(args.decoder)
    files = sorted(glob.glob(os.path.join(args.evidence_dir, "depth-*.bin.gz")),
                   key=lambda s: int(os.path.basename(s).split("-")[1].split(".")[0]))
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = args.threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    reference = ort.InferenceSession(args.source, options, providers=["CPUExecutionProvider"])
    camera = ort.InferenceSession(os.path.join(args.prepared_dir, CAMERA_NAME), options, providers=["CPUExecutionProvider"])
    main_graph = ort.InferenceSession(os.path.join(args.prepared_dir, MAIN_NAME), options, providers=["CPUExecutionProvider"])
    camera_names = [o.name for o in camera.get_outputs()]
    if set(camera_names) != {i.name for i in main_graph.get_inputs()} - {"image"}:
        sys.exit("camera outputs do not match the main graph inputs")

    worst_abs = worst_rel = 0.0
    compared, failures, intrinsics, timings = 0, [], [], {"source": [], "camera": [], "main": []}
    for path in files:
        _, arrays = decoder.read(path)
        rgba = arrays["rgba"]
        if rgba.shape != (182, 322, 4):
            print(f"skip {os.path.basename(path)}: rgba {rgba.shape}, not the 322x182 model grid")
            continue
        k = arrays["intrinsics"].astype(np.float32)
        intrinsics.append(k)
        K = np.array([[[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]]], np.float32)
        image = ((rgba[..., :3] / 255. - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        started = time.perf_counter(); expected = reference.run(None, {"image": image, "K": K})[0]
        timings["source"].append(time.perf_counter() - started)
        started = time.perf_counter(); conditioning = camera.run(None, {"K": K})
        timings["camera"].append(time.perf_counter() - started)
        started = time.perf_counter(); actual = main_graph.run(None, {"image": image, **dict(zip(camera_names, conditioning))})[0]
        timings["main"].append(time.perf_counter() - started)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            failures.append(f"{os.path.basename(path)}: shape {actual.shape} or non-finite output")
            continue
        difference = float(np.abs(actual - expected).max())
        relative = float((np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-6)).max())
        worst_abs, worst_rel = max(worst_abs, difference), max(worst_rel, relative)
        if difference > args.max_abs or relative > args.max_rel:
            failures.append(f"{os.path.basename(path)}: max|diff| {difference:.3e} m, max rel {relative:.3e}")
        compared += 1

    if intrinsics:
        k = np.array(intrinsics)
        print("per-frame K range: " + ", ".join(f"{n} {k[:, i].min():.2f}-{k[:, i].max():.2f}" for i, n in enumerate(("fx", "fy", "cx", "cy"))))
    print(f"compared {compared} captures: worst max|diff| {worst_abs:.3e} m, worst max rel {worst_rel:.3e}")
    if compared > 2:
        print("CPU median ms: " + ", ".join(f"{name} {np.median(v[2:]) * 1000:.1f}" for name, v in timings.items()))
    if failures or compared < args.min_frames:
        for failure in failures:
            print("FAIL", failure)
        sys.exit(f"verification failed: {len(failures)} frame(s) out of tolerance, {compared} compared (minimum {args.min_frames})")
    print("PASS: prepared graphs reproduce the source model on the CPU provider")


if __name__ == "__main__":
    main()
