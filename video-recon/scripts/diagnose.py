#!/usr/bin/env python3
"""Measure a finished reconstruction.

python video-recon/scripts/diagnose.py out/video/<stamp> [--depth other.npz] [--label name]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "harness"))

from videorecon.diagnose import run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--recording", type=Path, default=None)
    p.add_argument("--depth", type=Path, default=None)
    p.add_argument("--label", default="")
    a = p.parse_args()
    rec = a.recording
    if rec is None and (a.out / "manifest.json").exists():
        rec = Path(json.loads((a.out / "manifest.json").read_text())["recording"])
    print(json.dumps(run(a.out, rec, depth_npz=a.depth, label=a.label), indent=2))


if __name__ == "__main__":
    main()
