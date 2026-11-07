#!/usr/bin/env python3
"""Quantize a user's existing UniDepth ONNX export for the optional live model.

The source weights retain UniDepth's CC BY-NC 4.0 license:
https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE
No weights are downloaded or included in the app by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper

from onnxruntime.quantization import QuantType, quantize_dynamic


def live_grid(model):
    """Reduce the known export's internal grid, preserving dynamic camera input.

    Deliberately reject other export layouts instead of patching arbitrary node
    numbers. Image and depth grids remain 182x322; only internal work is reduced.
    Camera coordinates follow UniDepth's existing resize convention, with a
    separate horizontal scale so the rounded patch grid cannot stretch rays.
    """
    shapes = {x.name: [d.dim_value for d in x.type.tensor_type.shape.dim] for x in model.graph.input}
    if shapes != {"image": [1, 3, 182, 322], "K": [1, 3, 3]}:
        raise ValueError("live grid requires the fixed 182x322 image + K export")
    nodes = {n.name: n for n in model.graph.node}
    size, scale = nodes["/Constant_12"], nodes["/Constant_74"]
    if (numpy_helper.to_array(size.attribute[0].t).tolist() != [336, 602]
            or abs(float(numpy_helper.to_array(scale.attribute[0].t)) - 1.8473593) >= 1e-6):
        raise ValueError("unrecognized internal image/camera resize constants")
    consumers = [n for n in model.graph.node if "K" in n.input]
    if sorted(n.name for n in consumers) != ["/Gather", "/ScatterND", "/Shape_2"]:
        raise ValueError("unrecognized camera input consumers")
    sy, sx = 280 / 182, 490 / 322
    size.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array([280, 490], np.int64)))
    scale.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array(sy, np.float32)))
    axes = np.ones((1, 3, 3), np.float32)
    axes[0, 0, :] = sx / sy
    model.graph.initializer.append(numpy_helper.from_array(axes, "K_axis_scale"))
    for node in consumers:
        for i, name in enumerate(node.input):
            if name == "K":
                node.input[i] = "K_axis_scaled"
    model.graph.node.insert(0, helper.make_node("Mul", ["K", "K_axis_scale"],
        ["K_axis_scaled"], name="/K_axis_scale_Mul"))
    onnx.checker.check_model(model)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--live-grid", action="store_true", help="reproduce the experimental 490x280 internal grid (failed map-quality comparison)")
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error("source ONNX export does not exist")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unidepth-export-") as directory:
        source = args.source
        if args.live_grid:
            source = Path(directory) / "live.onnx"
            onnx.save(live_grid(onnx.load(args.source)), source)
        quantize_dynamic(
            str(source), str(args.output), op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8, per_channel=True,
            extra_options={"MatMulConstBOnly": True},
        )
    manifest = {"internal_grid": [490, 280] if args.live_grid else [602, 336],
                "dynamic_intrinsics": True, "per_channel": True, "license": "CC-BY-NC-4.0"}
    for name, path in (("source", args.source), ("quantized", args.output)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[f"{name}_sha256"] = digest
        print(f"{name}: {path.stat().st_size} bytes sha256={digest}")
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
