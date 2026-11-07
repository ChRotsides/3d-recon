#!/usr/bin/env python3
"""Prepare UniDepth v2 ViT-S (182x322 input, original internal geometry) for QnnDepthSession.

Input: the fp32 ONNX written by video-recon/vio/export_unidepth.py 182 322 OUT (resolution level 0),
with inputs image [1,3,182,322] and K [1,3,3] and output depth [1,182,322].

Writes three new files into --out-dir (existing files are never overwritten):
  qnn_camera.onnx            K -> camera-conditioning tensors. Run on the CPU for every frame's K.
  qnn_main_static_gelu.onnx  image + those tensors -> depth. Fully static shapes, ops registered by the
                             ONNX Runtime 1.22 QNN execution provider. Intended for QNN HTP.
  qnn_prepare_manifest.json  SHA-256 of the source and outputs, tool versions and graph statistics.

Steps: ONNX Runtime basic constant folding; Shape nodes replaced by constants from symbolic shape
inference and folded again; split at the tensors that depend only on K's values; exact-GELU
expansions x*0.5*(1+Erf(x/sqrt2)) fused into ONNX Gelu (opset 20, approximate='none'), because the
QNN EP has no Erf builder. The arithmetic is unchanged; verify_unidepth_qnn.py checks CPU equivalence.

No network access. No model weights are bundled or downloaded; obtain them under their own licence.
Run with an interpreter that has onnx and onnxruntime, e.g.  python -I prepare_unidepth_qnn.py --help
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

CAMERA_NAME = "qnn_camera.onnx"
MAIN_NAME = "qnn_main_static_gelu.onnx"
MANIFEST_NAME = "qnn_prepare_manifest.json"
IMAGE_SHAPE = [1, 3, 182, 322]
K_SHAPE = [1, 3, 3]
DEPTH_SHAPE = [1, 182, 322]
# Op types registered by onnxruntime/core/providers/qnn/builder/op_builder_factory.cc at v1.22.0.
# Registration is necessary, not sufficient: attribute limits are checked on device with
# session.disable_cpu_ep_fallback=1, which QnnDepthSession always sets.
QNN_1_22_OPS = {
    "Add", "Asin", "Atan", "Mul", "Abs", "And", "Ceil", "Cos", "Sign", "Div", "Equal", "Exp", "Floor", "Greater",
    "GreaterOrEqual", "LeakyRelu", "Less", "LessOrEqual", "Log", "Max", "Min", "Neg", "Not", "Or", "Pow", "PRelu",
    "Relu", "Gelu", "Elu", "Round", "Where", "ScatterND", "Sigmoid", "Sin", "Sqrt", "Sub", "Sum", "Tanh", "Concat",
    "QuantizeLinear", "DequantizeLinear", "HardSwish", "HardSigmoid", "DepthToSpace", "SpaceToDepth", "GridSample",
    "LpNormalization", "Softmax", "LogSoftmax", "Cast", "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd",
    "ReduceSum", "ReduceL2", "Conv", "ConvTranspose", "GlobalAveragePool", "AveragePool", "MaxPool", "GlobalMaxPool",
    "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Gemm", "Gather", "GatherElements", "ArgMax", "ArgMin", "Clip",
    "Slice", "Split", "Resize", "TopK", "Tile", "InstanceNormalization", "BatchNormalization", "LayerNormalization",
    "LRN", "Transpose", "Pad", "Expand", "MatMul",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dims(value_info):
    return [d.dim_value if d.HasField("dim_value") else None for d in value_info.type.tensor_type.shape.dim]


def fold(path_in, path_out):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = path_out
    options.log_severity_level = 3
    ort.InferenceSession(path_in, options, providers=["CPUExecutionProvider"])


def pin_shapes(model):
    inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=False)
    shapes = {v.name: dims(v) for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)}
    pinned = 0
    for index, node in enumerate(list(inferred.graph.node)):
        if node.op_type != "Shape":
            continue
        shape = shapes.get(node.input[0])
        if shape is None or any(d is None for d in shape):
            continue
        start = next((a.i for a in node.attribute if a.name == "start"), 0)
        end = next((a.i for a in node.attribute if a.name == "end"), None)
        value = numpy_helper.from_array(np.array(shape[start:end], np.int64))
        inferred.graph.node[index].CopyFrom(helper.make_node("Constant", [], [node.output[0]], name=node.name + "_static", value=value))
        pinned += 1
    return inferred, pinned


def split(model):
    sources = {}
    for node in model.graph.node:
        origin = set()
        for name in node.input:
            if name in ("K", "image"):
                origin.add(name)
            elif name in sources:
                origin |= sources[name]
        for output in node.output:
            sources[output] = origin
    k_only = {o for n in model.graph.node if sources[n.output[0]] == {"K"} for o in n.output}
    image_nodes = [n for n in model.graph.node if "image" in sources[n.output[0]]]
    frontier = sorted({x for n in image_nodes for x in n.input if x in k_only})
    if not frontier:
        raise SystemExit("no K-only frontier found: unexpected source graph")
    extractor = onnx.utils.Extractor(model)
    return extractor.extract_model(["K"], frontier), extractor.extract_model(["image"] + frontier, ["depth"]), frontier


def fuse_gelu(model):
    graph = model.graph
    constants = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            constants[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
    producer = {o: n for n in graph.node for o in n.output}
    consumers = collections.defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node)

    def is_constant(name, value):
        return name in constants and constants[name].size == 1 and abs(float(constants[name].ravel()[0]) - value) < 1e-4

    def other(node, name):
        return [x for x in node.input if x != name][0]

    order = {id(n): i for i, n in enumerate(graph.node)}
    remove, insert_at = set(), {}
    for erf in [n for n in graph.node if n.op_type == "Erf"]:
        pre = producer.get(erf.input[0])
        if pre is None or len(consumers[pre.output[0]]) != 1:
            continue
        if pre.op_type == "Div" and is_constant(pre.input[1], 2 ** .5):
            x = pre.input[0]
        elif pre.op_type == "Mul" and any(is_constant(i, 2 ** -.5) for i in pre.input):
            x = [i for i in pre.input if not is_constant(i, 2 ** -.5)][0]
        else:
            continue
        plus = consumers[erf.output[0]]
        if len(plus) != 1 or plus[0].op_type != "Add" or not is_constant(other(plus[0], erf.output[0]), 1.0):
            continue
        plus = plus[0]
        times = consumers[plus.output[0]]
        if len(times) != 1 or times[0].op_type != "Mul":
            continue
        times = times[0]
        partner = other(times, plus.output[0])
        if partner == x:  # x * (1 + erf) * 0.5
            half = consumers[times.output[0]]
            if len(half) != 1 or half[0].op_type != "Mul" or not is_constant(other(half[0], times.output[0]), .5):
                continue
            nodes, y = [pre, erf, plus, times, half[0]], half[0].output[0]
        else:             # (x * 0.5) * (1 + erf)
            half = producer.get(partner)
            if half is None or half.op_type != "Mul" or x not in half.input or not is_constant(other(half, x), .5):
                continue
            if len(consumers[half.output[0]]) != 1:
                continue
            nodes, y = [pre, erf, plus, times, half], times.output[0]
        last = max(nodes, key=lambda n: order[id(n)])
        remove |= {id(n) for n in nodes}
        insert_at[id(last)] = helper.make_node("Gelu", [x], [y], name=erf.name + "_Gelu", approximate="none")
    kept = []
    for node in graph.node:
        if id(node) in insert_at:
            kept.append(insert_at[id(node)])
        if id(node) not in remove:
            kept.append(node)
    del graph.node[:]
    graph.node.extend(kept)
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = 20
    return len(insert_at)


def save_new(model, path):
    """Write next to the target, then hard-link into place: fails instead of replacing an existing file."""
    handle, temporary = tempfile.mkstemp(prefix=".partial-", suffix=".onnx", dir=os.path.dirname(path))
    os.close(handle)
    try:
        onnx.save(model, temporary)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="fp32 UniDepth ONNX with inputs image [1,3,182,322] and K [1,3,3]")
    parser.add_argument("--out-dir", required=True, help="directory for the three new files (created if missing)")
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    targets = {name: os.path.join(out_dir, name) for name in (CAMERA_NAME, MAIN_NAME, MANIFEST_NAME)}
    existing = [p for p in targets.values() if os.path.exists(p)]
    if existing:
        raise SystemExit(f"refusing to overwrite existing files: {existing}")

    original = onnx.load(source, load_external_data=False)
    inputs = {i.name: dims(i) for i in original.graph.input}
    outputs = {o.name: dims(o) for o in original.graph.output}
    if inputs != {"image": IMAGE_SHAPE, "K": K_SHAPE} or list(outputs) != ["depth"]:
        raise SystemExit(f"unexpected source signature: inputs {inputs}, outputs {outputs}")

    with tempfile.TemporaryDirectory(prefix=".qnn-prepare-", dir=out_dir) as work:
        first, second = os.path.join(work, "fold1.onnx"), os.path.join(work, "fold2.onnx")
        fold(source, first)
        pinned_model, pinned = pin_shapes(onnx.load(first))
        onnx.save(pinned_model, second)
        fold(second, first)
        camera, main_graph, frontier = split(onnx.load(first))

    # Static-shape check before the Gelu fusion (which changes no shapes); older shape inference
    # releases may not know opset-20 Gelu.
    inferred = SymbolicShapeInference.infer_shapes(main_graph, auto_merge=False)
    dynamic = [v.name for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)
               if any(d is None for d in dims(v))]
    if dynamic:
        raise SystemExit(f"main graph still has tensors without static shapes: {dynamic[:10]}")
    fused = fuse_gelu(main_graph)
    onnx.checker.check_model(camera)
    onnx.checker.check_model(main_graph)
    ops = collections.Counter(n.op_type for n in main_graph.graph.node)
    unsupported = sorted(set(ops) - QNN_1_22_OPS)
    if unsupported or ops.get("Erf"):
        raise SystemExit(f"main graph has ops without a QNN 1.22 builder: {unsupported}")
    main_inputs = {i.name: dims(i) for i in main_graph.graph.input}
    camera_outputs = {o.name: dims(o) for o in camera.graph.output}
    if main_inputs.get("image") != IMAGE_SHAPE or {k: v for k, v in main_inputs.items() if k != "image"} != camera_outputs:
        raise SystemExit("camera outputs do not match the main graph inputs")
    if [dims(o) for o in main_graph.graph.output] != [DEPTH_SHAPE]:
        raise SystemExit("main graph output is not depth [1,182,322]")

    save_new(camera, targets[CAMERA_NAME])
    save_new(main_graph, targets[MAIN_NAME])
    manifest = {
        "source": {"path": source, "sha256": sha256(source)},
        "camera": {"file": CAMERA_NAME, "sha256": sha256(targets[CAMERA_NAME]), "nodes": len(camera.graph.node),
                   "outputs": camera_outputs},
        "main": {"file": MAIN_NAME, "sha256": sha256(targets[MAIN_NAME]), "nodes": len(main_graph.graph.node),
                 "op_types": dict(sorted(ops.items())), "gelu_fused": fused},
        "shape_nodes_pinned": pinned,
        "frontier": frontier,
        "versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0]},
        "licence_note": "Weights are not bundled. UniDepth weights and the Qualcomm QNN runtime are distributed "
                        "under their own licences; review them before shipping.",
    }
    with open(targets[MANIFEST_NAME], "x") as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps({k: manifest[k] for k in ("camera", "main", "shape_nodes_pinned")}, indent=2))
    print(f"wrote {targets[CAMERA_NAME]}, {targets[MAIN_NAME]}, {targets[MANIFEST_NAME]}")


if __name__ == "__main__":
    main()
