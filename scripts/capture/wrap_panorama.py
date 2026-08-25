#!/usr/bin/env python3
"""Fold a transparent spherical stitch into a closed 2:1 panorama."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from recon.image_stitch import wrap_equirectangular


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
    if source is None or source.ndim != 3 or source.shape[2] != 4:
        raise SystemExit("input must be a readable BGRA panorama")
    wrapped = wrap_equirectangular(source[:, :, :3], source[:, :, 3])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), wrapped):
        raise SystemExit(f"could not write {args.output}")
    print(f"wrote {args.output} {wrapped.shape[1]}x{wrapped.shape[0]}")


if __name__ == "__main__":
    main()
