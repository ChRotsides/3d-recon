#!/usr/bin/env python3
"""Serve a live dashboard for a reconstruction output directory.

python video-recon/scripts/monitor.py out/video/<stamp> [--log run.log] [--port 8778]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import uvicorn  # noqa: E402
from videorecon.monitor import create_app  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument(
        "--recording",
        type=Path,
        default=None,
        help="recording dir (defaults to the one in manifest.json)",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8778)
    a = p.parse_args()
    uvicorn.run(
        create_app(a.out.resolve(), a.log, a.recording),
        host=a.host,
        port=a.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
