#!/usr/bin/env python3
"""Export the pinned official pretrained XFeat network for Android inference."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import torch


UPSTREAM_URL = "https://github.com/verlab/accelerated_features"
UPSTREAM_COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
WEIGHTS_SHA256 = "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="official accelerated_features checkout")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    revision = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise SystemExit(f"expected XFeat {UPSTREAM_COMMIT}, found {revision}")
    weights = args.source / "weights" / "xfeat.pt"
    actual = sha256(weights)
    if actual != WEIGHTS_SHA256:
        raise SystemExit(f"unexpected XFeat weights sha256: {actual}")

    sys.path.insert(0, str(args.source))
    from modules.model import XFeatModel  # pylint: disable=import-error,import-outside-toplevel

    network = XFeatModel().eval()
    network.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))

    class AndroidXFeat(torch.nn.Module):
        """Full official forward pass plus its sparse detector ranking."""

        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, image: torch.Tensor):
            descriptors, logits, reliability = self.model(image)
            probability = torch.softmax(logits, dim=1)[:, :64]
            batch, _, coarse_height, coarse_width = probability.shape
            probability = probability.permute(0, 2, 3, 1).reshape(
                batch, coarse_height, coarse_width, 8, 8
            ).permute(0, 1, 3, 2, 4).reshape(batch, 1, coarse_height * 8, coarse_width * 8)
            keep = (probability == torch.nn.functional.max_pool2d(probability, 5, 1, 2)) & (probability > 0.05)
            height, width = probability.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
            grid = torch.stack((2.0 * xx / (width - 1) - 1.0, 2.0 * yy / (height - 1) - 1.0), dim=-1)
            grid = grid.to(image).unsqueeze(0).expand(batch, -1, -1, -1)
            reliability_full = torch.nn.functional.grid_sample(reliability, grid, mode="bilinear", align_corners=False)
            scores = torch.where(keep, probability * reliability_full, torch.zeros_like(probability)).flatten(1)
            top_scores, top_indices = torch.topk(scores, 1000, dim=1, sorted=True)
            return descriptors, logits, reliability, top_scores, top_indices

    model = AndroidXFeat(network).eval()
    sample = torch.zeros(1, 1, 192, 320)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        args.output,
        input_names=["image"],
        output_names=["descriptors", "keypoint_logits", "reliability", "top_scores", "top_indices"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx  # pylint: disable=import-error,import-outside-toplevel

    exported = onnx.load(args.output)
    onnx.checker.check_model(exported)
    metadata = {
        "source": UPSTREAM_URL,
        "source_commit": UPSTREAM_COMMIT,
        "weights_sha256": WEIGHTS_SHA256,
        "license": "Apache-2.0",
        "input_geometry": "1x1x192x320 grayscale; 320x180 content with replicated vertical padding",
    }
    del exported.metadata_props[:]
    for key, value in metadata.items():
        entry = exported.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(exported, args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes, sha256 {sha256(args.output)})")


if __name__ == "__main__":
    main()
