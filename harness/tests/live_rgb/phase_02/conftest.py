from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image


def jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(
        stream, format="JPEG", quality=quality
    )
    return stream.getvalue()


@pytest.fixture
def textured_images() -> list[bytes]:
    random = np.random.default_rng(20260902)
    base = random.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    cv2.putText(base, "RGB PHASE TWO", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    output = []
    for shift in (0, 8, 16, 24):
        matrix = np.float32([[1, 0, shift], [0, 1, 0]])
        output.append(jpeg(cv2.warpAffine(base, matrix, (320, 240), borderMode=cv2.BORDER_WRAP)))
    return output


def make_station(root: Path, images: list[bytes]) -> Path:
    import hashlib
    import json

    station = root / "sessions/s/stations/t"
    station.mkdir(parents=True, exist_ok=True)
    (station / "station.json").write_text(
        json.dumps({"schema": "live-rgb-station-v1", "session_id": "s", "station_id": "t"})
    )
    for index, wire in enumerate(images):
        frame = station / "frames" / f"f-{index:03d}"
        frame.mkdir(parents=True, exist_ok=True)
        (frame / "frame.jpg").write_bytes(wire)
        (frame / "frame.json").write_text(
            json.dumps(
                {
                    "schema": "live-rgb-frame-v1",
                    "session_id": "s",
                    "station_id": "t",
                    "frame_id": f"f-{index:03d}",
                    "timestamp_ns": index,
                    "jpeg_sha256": hashlib.sha256(wire).hexdigest(),
                    "jpeg_bytes": len(wire),
                }
            )
        )
    return station
