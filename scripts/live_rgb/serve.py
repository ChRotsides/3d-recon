#!/usr/bin/env python3
"""Run the Phase-1 final and Phase-2 incremental RGB-only service."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from recon.live_rgb.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=Path, default=Path("out/live-rgb-service"))
    parser.add_argument("--depth-backend", choices=("stub", "da3-small"), default="da3-small")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.storage, depth_backend=args.depth_backend), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
