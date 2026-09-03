"""Convert a legacy capture bundle to the deliberately tiny RGB protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReplayFrame:
    frame_id: str
    timestamp_ns: int
    jpeg_path: Path


def frames_from_bundle(bundle: Path) -> list[ReplayFrame]:
    """Read only frame identity, ordering timestamp and JPEG path."""

    wire = json.loads((bundle / "bundle.json").read_text())
    return [
        ReplayFrame(
            frame_id=str(frame["id"]),
            timestamp_ns=int(frame["timestamp_ns"]),
            jpeg_path=bundle / str(frame["image"]),
        )
        for frame in wire["keyframes"]
    ]


def rgb_input_digest(frames: list[ReplayFrame]) -> str:
    digest = hashlib.sha256()
    for frame in sorted(frames, key=lambda item: (item.timestamp_ns, item.frame_id)):
        digest.update(frame.frame_id.encode())
        digest.update(frame.timestamp_ns.to_bytes(8, "big", signed=False))
        digest.update(frame.jpeg_path.read_bytes())
    return digest.hexdigest()
