import numpy as np
import pytest
from recon.live_rgb.draft import (
    MAX_POINTS,
    MAX_TRANSPORT_BYTES,
    RECORD_DTYPE,
    validate_draft_asset,
    write_draft_asset,
)


def records(count=100, version=7):
    result = np.zeros(count, dtype=RECORD_DTYPE)
    result["position"] = np.random.default_rng(8).normal(size=(count, 3))
    result["rgb"] = np.random.default_rng(9).integers(0, 256, (count, 3), dtype=np.uint8)
    result["confidence"] = np.linspace(0, 1, count)
    result["panorama_support"] = np.linspace(1, 0, count)
    result["source_version"] = version
    return result


def test_transport_round_trip_is_bounded_checksummed_and_version_pure(tmp_path):
    cloud = write_draft_asset(tmp_path / "draft", records(), {"version": 7})
    decoded = validate_draft_asset(tmp_path / "draft.json", tmp_path / "draft.bin.gz")
    assert np.array_equal(decoded, records())
    assert cloud["count"] <= MAX_POINTS
    assert cloud["asset_bytes"] <= MAX_TRANSPORT_BYTES
    assert cloud["record_bytes"] == 28


def test_transport_rejects_corruption_mixed_versions_and_invalid_bounds(tmp_path):
    mixed = records(3)
    mixed["source_version"][-1] = 8
    with pytest.raises(ValueError, match="mixed"):
        write_draft_asset(tmp_path / "mixed", mixed, {"version": 7})
    write_draft_asset(tmp_path / "draft", records(), {"version": 7})
    path = tmp_path / "draft.bin.gz"
    wire = bytearray(path.read_bytes())
    wire[-1] ^= 1
    path.write_bytes(wire)
    with pytest.raises(ValueError, match="checksum"):
        validate_draft_asset(tmp_path / "draft.json", path)


def test_every_finite_positive_depth_is_eligible_before_point_budget():
    from recon.live_rgb.draft import backproject_evidence, cubemap_cameras

    cameras = cubemap_cameras(face_size=4, process_res=4)
    views = [np.full((4, 4, 3), 127, np.uint8) for _ in cameras]
    supports = [np.ones((4, 4), np.float32) for _ in cameras]
    depths = np.ones((6, 4, 4), np.float32)
    depths[0, 0, 0] = np.nan
    depths[1, 0, 0] = -1
    confidence = np.zeros_like(depths)  # zero confidence must not erase geometry
    result = backproject_evidence(views, supports, depths, confidence, cameras, 2, 1_000)
    assert len(result) == 6 * 16 - 2
    assert np.all(result["confidence"] == 0)
