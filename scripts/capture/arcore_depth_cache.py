#!/usr/bin/env python3
"""Write a bundle's captured ARCore depth maps in reconstruction cache format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from recon.io.bundle import BundleDir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    bundle = BundleDir.load(args.bundle)
    arrays = {
        keyframe.id: depth
        for keyframe in bundle.bundle.keyframes
        if (depth := bundle.depth(keyframe)) is not None
    }
    if not arrays:
        raise SystemExit("bundle contains no captured depth maps")
    arrays["model"] = np.asarray("arcore-captured")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"wrote {args.output}: {len(arrays) - 1} metric depth maps")


if __name__ == "__main__":
    main()
