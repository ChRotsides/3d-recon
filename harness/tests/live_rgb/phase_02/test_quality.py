from __future__ import annotations

import cv2
import numpy as np
from recon.live_rgb.incremental import classify_frame_quality


def test_blur_exposure_and_textureless_reasons_are_separate():
    random = np.random.default_rng(4)
    sharp = random.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (61, 61), 20)
    black = np.zeros_like(sharp)
    flat = np.full_like(sharp, 127)
    assert "blur" in classify_frame_quality(blurred)["reasons"]
    assert "exposure" in classify_frame_quality(black)["reasons"]
    assert "textureless" in classify_frame_quality(flat)["reasons"]


def test_parallax_reason_threshold_is_a_named_channel(tmp_path, textured_images):
    from recon.live_rgb.incremental import IncrementalGraphEngine

    from .conftest import make_station

    station = make_station(tmp_path, textured_images[:1])
    guidance = IncrementalGraphEngine(tmp_path).process_station(station)
    assert all("parallax_risk" in cell["channels"] for cell in guidance["coverage"]["cells"])
