"""End-to-end accuracy tests against the synthetic oracle.

These are the tests that would have caught every bug found while building the
metric layer: the missing floor, the flipped floor-plane basis, the bow-tie
footprint, and the outward TSDF bias.
"""

from __future__ import annotations

import numpy as np
import pytest
from recon import synth
from recon.eval.metrics import layout_metrics, match_walls
from recon.io.bundle import BundleDir
from recon.masking.base import GroundTruthMasker, NullMasker
from recon.pipeline import Pipeline, PipelineConfig

pytestmark = pytest.mark.slow

#: Small enough to run in CI, large enough to be representative.
FAST = dict(width=160, height=120, n_stations=2, yaw_step_deg=40, max_keyframes=54)


def _scene(tmp_path, preset: str, **kw):
    cfg = synth.CaptureConfig(**{**FAST, **kw})
    spec = synth.preset(preset)
    synth.generate(spec, tmp_path / preset, cfg)
    return BundleDir.load(tmp_path / preset)


@pytest.fixture(scope="module")
def rect_scene(tmp_path_factory):
    return _scene(tmp_path_factory.mktemp("rect"), "empty_rect", vio_scale_error=1.025)


def test_rectangular_room_hits_accuracy_targets(rect_scene):
    """PLAN.md §1.3: wall length <=1.5% or <=3 cm, area <=3%."""
    res = Pipeline(masker=GroundTruthMasker()).run(rect_scene)
    m = layout_metrics(res.layout, rect_scene.ground_truth())
    assert m.n_walls_pred == 4
    assert m.passes(), (
        f"wall {m.wall_error_max_abs_pct:.2f}% / {m.wall_error_max_abs_m*100:.1f} cm, "
        f"area {m.area_error_pct:+.2f}%"
    )


def test_scale_correction_is_applied(rect_scene):
    """The reconstruction must come out in *corrected* metres, not VIO metres."""
    gt = rect_scene.ground_truth()
    corrected = Pipeline(masker=GroundTruthMasker()).run(rect_scene)
    uncorrected = Pipeline(
        masker=GroundTruthMasker(), config=PipelineConfig(apply_scale=False)
    ).run(rect_scene)

    assert corrected.scale.scale == pytest.approx(1.0 / gt["applied_scale"], rel=5e-3)
    assert abs(corrected.layout.area_m2 / gt["area_m2"] - 1.0) < 0.03
    # Without the correction the room is systematically ~5% too big in area.
    assert uncorrected.layout.area_m2 > corrected.layout.area_m2 * 1.02


def test_footprint_is_a_simple_polygon(rect_scene):
    """A bow-tie polygon still reports a sane perimeter but zero area — this is
    the exact failure the floor-plane basis bug produced."""
    res = Pipeline(masker=GroundTruthMasker()).run(rect_scene)
    poly = np.asarray(res.layout.footprint)
    assert len(poly) >= 3
    assert res.layout.area_m2 > 0.5 * res.layout.perimeter_m ** 2 / 100
    # No repeated vertices.
    d = np.linalg.norm(poly[:, None, :] - poly[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 0.2


def test_floor_and_ceiling_are_found(rect_scene):
    """Missing floor points made the whole layout stage degenerate."""
    res = Pipeline(masker=GroundTruthMasker()).run(rect_scene)
    assert res.planes.floor is not None, "no floor plane"
    assert res.planes.ceiling is not None, "no ceiling plane"
    gt = rect_scene.ground_truth()
    assert res.layout.ceiling_height_m == pytest.approx(gt["ceiling_height_m"], abs=0.05)


def test_angled_wall_survives_regularization(tmp_path):
    """Atlanta, not Manhattan. A 3 deg snap tolerance must not straighten a
    wall that is genuinely off-axis."""
    scene = _scene(tmp_path, "angled")
    res = Pipeline(masker=GroundTruthMasker()).run(scene)
    gt = scene.ground_truth()
    m = layout_metrics(res.layout, gt)

    gt_poly = np.asarray(gt["footprint"])
    angles = []
    for poly in (np.asarray(res.layout.footprint), gt_poly):
        d = np.roll(poly, -1, axis=0) - poly
        angles.append(np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0)
    pred_dirs, gt_dirs = angles

    def circ_diff(a, b):
        """Wall directions live modulo 180 deg: 179.8 and 0.0 are the same wall."""
        return np.abs((a - b + 90.0) % 180.0 - 90.0)

    # Every ground-truth direction — including the ~139 deg angled wall — must
    # survive regularization.
    for g in gt_dirs:
        assert np.min(circ_diff(pred_dirs, g)) < 4.0, (
            f"direction {g:.1f} deg lost; predicted {np.round(pred_dirs, 1)}"
        )
    assert m.passes()


def test_l_shaped_room_keeps_its_notch(tmp_path):
    scene = _scene(tmp_path, "l_shaped")
    res = Pipeline(masker=GroundTruthMasker()).run(scene)
    gt = scene.ground_truth()
    m = layout_metrics(res.layout, gt)
    assert m.n_walls_pred >= 6, "the L-shape collapsed to a rectangle"
    assert m.passes()


def test_masking_before_fusion_prevents_phantom_geometry(tmp_path):
    """PLAN.md §7: unmasked mirrors and windows must visibly wreck the room, and
    masking must fix it. If this test ever passes for the *unmasked* case the
    synthetic artifacts have stopped reproducing the real failure mode."""
    scene = _scene(tmp_path, "mirror_heavy")
    gt = scene.ground_truth()

    masked = Pipeline(masker=GroundTruthMasker()).run(scene)
    m_masked = layout_metrics(masked.layout, gt)
    assert m_masked.passes()

    unmasked = Pipeline(masker=NullMasker()).run(scene)
    m_unmasked = layout_metrics(unmasked.layout, gt)
    assert not m_unmasked.passes(), "the phantom-room failure mode stopped reproducing"
    assert m_unmasked.wall_error_max_abs_pct > 10 * max(m_masked.wall_error_max_abs_pct, 0.05)


def test_door_is_detected(rect_scene):
    res = Pipeline(masker=GroundTruthMasker()).run(rect_scene)
    doors = [o for o in res.layout.openings if o.kind == "door"]
    assert len(doors) == 1
    d = doors[0]
    gt_door = rect_scene.ground_truth()["openings"][0]
    width = d.u_end_m - d.u_start_m
    assert width == pytest.approx(gt_door["width_m"], abs=0.10)
    assert d.height_min_m < 0.15
    assert d.height_max_m == pytest.approx(gt_door["height_m"], abs=0.15)


def test_match_walls_is_rotation_invariant():
    poly = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
    rotated = np.roll(poly, 2, axis=0)
    pairs = match_walls(rotated, poly)
    assert len(pairs) == 4
    lengths_p = np.linalg.norm(np.roll(rotated, -1, axis=0) - rotated, axis=1)
    lengths_g = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    for i, j in pairs:
        assert lengths_p[i] == pytest.approx(lengths_g[j])
