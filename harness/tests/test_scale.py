import numpy as np
import pytest
from recon.metric.scale import (
    ScaleConstraint,
    build_constraints,
    solve_scale,
)
from recon.schemas import MeasurementType, ReferenceMeasurement


def _tape(d_est: float, d_obs: float, sigma: float = 0.005) -> ScaleConstraint:
    return ScaleConstraint(MeasurementType.USER_TAPE, d_est, d_obs, sigma)


def _vio(prior: float = 1.0, sigma: float = 0.03) -> ScaleConstraint:
    return ScaleConstraint(MeasurementType.VIO_SCALE, 1.0, prior, sigma)


def test_tape_dominates_vio_prior():
    """A 5 mm tape measure should overwhelm a 3% VIO prior."""
    sol = solve_scale([_vio(), _tape(0.8508, 0.83)])
    assert sol.scale == pytest.approx(0.83 / 0.8508, rel=2e-3)


def test_vio_prior_alone_is_identity():
    sol = solve_scale([_vio()])
    assert sol.scale == pytest.approx(1.0)


def test_weak_priors_land_within_five_percent():
    """PLAN.md §6.2 UX: skipping the tape must still get within ~5%."""
    true_scale = 1.0 / 1.025
    cs = [
        _vio(),
        ScaleConstraint(MeasurementType.CEILING_PRIOR, 2.45 * 1.025, 2.45, 0.10),
        ScaleConstraint(MeasurementType.CAMERA_HEIGHT, 1.5 * 1.025, 1.5, 0.08),
        ScaleConstraint(MeasurementType.DOOR_PRIOR, 2.03 * 1.025, 2.03, 0.04),
    ]
    sol = solve_scale(cs)
    assert abs(sol.scale / true_scale - 1.0) < 0.05


def test_fat_fingered_tape_entry_is_rejected():
    """A user typing 83 instead of 0.83 must not shrink the house."""
    good = [_vio(), _tape(0.8508, 0.83)]
    sol_good = solve_scale(good)
    sol_bad = solve_scale([*good, _tape(0.8508, 83.0)])
    assert sol_bad.rejected, "the absurd entry should have been rejected"
    assert sol_bad.scale == pytest.approx(sol_good.scale, rel=1e-2)


def test_vio_prior_is_never_rejected():
    """The anchor stays even when every other constraint disagrees with it."""
    cs = [_vio(), _tape(1.0, 3.0), _tape(1.0, 3.01)]
    sol = solve_scale(cs)
    kinds = {c.kind for c in sol.constraints}
    assert MeasurementType.VIO_SCALE in kinds


def test_solution_sigma_shrinks_with_more_constraints():
    one = solve_scale([_vio(), _tape(2.0, 2.0)])
    many = solve_scale([_vio(), _tape(2.0, 2.0), _tape(2.0, 2.0), _tape(2.0, 2.0)])
    assert many.sigma < one.sigma


def test_build_constraints_from_measurements():
    ms = [
        ReferenceMeasurement(type=MeasurementType.VIO_SCALE, measured_value_m=1.0,
                             sigma_m=0.03),
        ReferenceMeasurement(
            type=MeasurementType.USER_TAPE,
            point_a=[0.0, 0.0, 0.0], point_b=[1.0, 0.0, 0.0],
            measured_value_m=0.98, sigma_m=0.005,
        ),
    ]
    cs = build_constraints(ms, mapanything_scale=0.99, mapanything_sigma=0.05)
    kinds = [c.kind for c in cs]
    assert MeasurementType.MAPANYTHING_SCALE in kinds
    tape = next(c for c in cs if c.kind is MeasurementType.USER_TAPE)
    assert tape.d_est == pytest.approx(1.0)
    assert tape.d_obs == pytest.approx(0.98)


def test_degenerate_input_raises():
    with pytest.raises(ValueError):
        solve_scale([])


def test_zero_length_segments_are_skipped():
    ms = [
        ReferenceMeasurement(type=MeasurementType.VIO_SCALE, measured_value_m=1.0,
                             sigma_m=0.03),
        ReferenceMeasurement(
            type=MeasurementType.USER_TAPE,
            point_a=[1.0, 1.0, 1.0], point_b=[1.0, 1.0, 1.0],
            measured_value_m=0.83, sigma_m=0.005,
        ),
    ]
    cs = build_constraints(ms)
    assert all(c.d_est > 0 for c in cs)
    assert solve_scale(cs).scale == pytest.approx(1.0)


def test_scale_recovers_known_vio_error():
    """End-to-end intent: a 2.5% oversize reconstruction is corrected to truth."""
    err = 1.025
    cs = [
        _vio(),
        _tape(0.83 * err, 0.83),
        ScaleConstraint(MeasurementType.DOOR_PRIOR, 2.03 * err, 2.03, 0.04),
        ScaleConstraint(MeasurementType.CEILING_PRIOR, 2.45 * err, 2.45, 0.10),
    ]
    sol = solve_scale(cs)
    corrected = 4.20 * err * sol.scale
    assert abs(corrected - 4.20) < 0.03, f"wall off by {abs(corrected-4.20)*100:.1f} cm"
    assert np.isfinite(sol.sigma)
