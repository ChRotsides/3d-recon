from __future__ import annotations

import json
from pathlib import Path

import pytest

# These check a real capture's outputs, which live in out/ and are not in the repository.
pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[4] / "out/capture-065a").exists(),
    reason="needs the local out/capture-065a reference outputs",
)


def test_established_reference_metrics_are_present() -> None:
    root = Path(__file__).resolve().parents[4]
    diagnostics = json.loads((root / "out/capture-065a/feature-stitch-v4.json").read_text())
    assert len(diagnostics["selected_frames"]) == 30
    assert len(diagnostics["accepted_edges"]) == 65
    assert diagnostics["median_cycle_error_deg"] <= 1.30


def test_established_real_da3_output_uses_exact_cube_and_800k_points() -> None:
    root = Path(__file__).resolve().parents[4]
    output = root / "out/capture-065a/panorama-depth-anything-cubemap"
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    cloud = json.loads((output / "cloud.json").read_text())
    assert diagnostics["faces"] == ["front", "right", "back", "left", "ceiling", "floor"]
    assert diagnostics["virtual_camera"]["shared_optical_center"] is True
    assert diagnostics["face_size"] == [512, 512]
    assert diagnostics["prediction_size"] == [504, 504]
    assert cloud["count"] == 800_000
