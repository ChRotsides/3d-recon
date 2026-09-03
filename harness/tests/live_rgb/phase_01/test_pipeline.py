from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from recon.live_rgb.pipeline import (
    _backproject,
    _prepare_stitch_images,
    _stub_depth,
    _write_cloud,
    cubemap_cameras,
    render_face,
)
from recon.live_rgb.replay import frames_from_bundle, rgb_input_digest
from recon.image_stitch import _capture_order_pairs


def test_virtual_cameras_tile_sphere_from_one_center() -> None:
    cameras = cubemap_cameras()
    assert [name for name, _ in cameras] == ["front", "right", "back", "left", "ceiling", "floor"]
    for _, basis in cameras:
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-7)
        assert np.linalg.det(basis) == pytest.approx(1.0)


def test_stub_is_deterministic_positive_and_backprojection_keeps_all_depths() -> None:
    panorama = np.full((64, 128, 3), 127, dtype=np.uint8)
    cameras = cubemap_cameras()
    rendered = [render_face(panorama, basis, 32) for _, basis in cameras]
    views = [item[0] for item in rendered]
    intrinsics = [item[1] for item in rendered]
    first, _, revision = _stub_depth(views, 24)
    second, _, _ = _stub_depth(views, 24)
    np.testing.assert_array_equal(first, second)
    assert revision == "deterministic-depth-stub-v1"
    points, _ = _backproject(views, first, intrinsics, cameras, 1_000_000)
    assert len(points) == int(np.count_nonzero(np.isfinite(first) & (first > 0)))


def test_live_phone_images_are_bounded_before_stitching() -> None:
    images = [np.zeros((3000, 4000, 3), dtype=np.uint8) for _ in range(2)]
    prepared = _prepare_stitch_images(images)
    assert prepared[0].shape == (768, 1024, 3)
    assert prepared[1].shape == prepared[0].shape


def test_capture_order_pair_window_includes_neighbors_and_loop_closure() -> None:
    pairs = _capture_order_pairs(35, 5)
    assert len(pairs) == 175
    assert (0, 1) in pairs
    assert (0, 34) in pairs
    assert (0, 6) not in pairs


def test_cloud_contract_checksum_finiteness_and_bounds(tmp_path: Path) -> None:
    points = np.array([[-1, 2, 3], [4, 5, 6]], dtype=np.float32)
    colours = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    metadata = _write_cloud(tmp_path / "cloud", points, colours, {"model_revision": "test"})
    payload = (tmp_path / "cloud.bin").read_bytes()
    assert metadata["asset_sha256"] == hashlib.sha256(payload).hexdigest()
    assert metadata["asset_bytes"] == len(payload) == 30
    assert np.isfinite(points).all()
    assert np.all(points >= metadata["min"])
    assert np.all(points <= metadata["max"])


def test_replay_adapter_ignores_poisoned_non_rgb_fields(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "images").mkdir(parents=True)
    cv2.imwrite(str(bundle / "images/a.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    clean = {
        "keyframes": [{"id": "a", "timestamp_ns": 1, "image": "images/a.jpg", "pose": {"x": 1}}]
    }
    (bundle / "bundle.json").write_text(json.dumps(clean))
    before = rgb_input_digest(frames_from_bundle(bundle))
    clean["keyframes"][0].update(
        pose="invalid",
        depth="invalid",
        intrinsics="invalid",
        heading_deg="invalid",
        tracking_state="invalid",
    )
    clean["sparse_points"] = "invalid"
    (bundle / "bundle.json").write_text(json.dumps(clean))
    assert rgb_input_digest(frames_from_bundle(bundle)) == before


@pytest.mark.real_da3
def test_real_da3_checkpoint_is_available() -> None:
    pytest.importorskip("depth_anything_3")
    from depth_anything_3.api import DepthAnything3

    DepthAnything3.from_pretrained("depth-anything/DA3-SMALL", local_files_only=True)
