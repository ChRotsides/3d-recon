import numpy as np
import pytest


@pytest.mark.real_da3
def test_real_da3_small_draft_inference_uses_six_unposed_rgb_faces():
    pytest.importorskip("depth_anything_3")
    from recon.live_rgb.pipeline import _da3_depth

    views = [np.full((256, 256, 3), 127 + index, dtype=np.uint8) for index in range(6)]
    depth, confidence, revision, device = _da3_depth(views, 252)
    assert depth.shape == confidence.shape
    assert depth.shape[0] == 6
    assert np.count_nonzero(np.isfinite(depth) & (depth > 0)) > 0
    assert revision == "depth-anything/DA3-SMALL"
    assert device
