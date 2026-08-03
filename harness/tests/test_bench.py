"""Tests for the Phase-0 exit criterion and the §8 regression gate.

A build gate that has never been seen to fail is not a gate.
"""

from __future__ import annotations

import pytest
from recon.eval.bench import REGRESSION_TOLERANCE_PP, BenchReport, BenchRow


def _row(scene: str, wall: float, area: float = 0.1, depth: str = "gt",
         mask: str = "gt") -> BenchRow:
    return BenchRow(
        scene=scene, depth_model=depth, mask_model=mask,
        layout={
            "wall_error_max_abs_pct": wall,
            "wall_error_max_abs_m": wall / 100.0 * 4.0,
            "area_error_pct": area,
        },
    )


def test_row_passes_when_inside_the_plan_targets():
    assert _row("a", 1.0, 1.0).passed
    assert not _row("a", 4.0, 1.0).passed
    assert not _row("a", 1.0, 5.0).passed


def test_a_small_room_passes_on_the_absolute_bound():
    """Plan §1.3 is 1.5% *or* 3 cm, whichever is larger — a 2% error on a short
    wall is 2 cm and must pass."""
    row = BenchRow(
        scene="s", depth_model="gt", mask_model="gt",
        layout={
            "wall_error_max_abs_pct": 2.0,
            "wall_error_max_abs_m": 0.02,
            "area_error_pct": 1.0,
        },
    )
    assert row.passed


def test_a_failed_config_is_a_result_not_a_crash():
    row = BenchRow(scene="s", depth_model="gt", mask_model="none",
                   error="MemoryError: volume too large")
    assert not row.passed
    report = BenchReport(rows=[row])
    assert "ERROR" in report.to_markdown()


def test_exit_criterion_needs_four_of_five_scenes():
    """Plan §4/Phase 0: >=1 config at <=2% wall error on 4 of 5 scenes."""
    scenes = ["empty_rect", "l_shaped", "angled", "furnished", "mirror_heavy"]
    good = BenchReport(rows=[_row(s, 0.5) for s in scenes])
    assert good.exit_criterion()["any_config_meets_exit"]

    three = BenchReport(
        rows=[_row(s, 0.5) for s in scenes[:3]] + [_row(s, 9.0) for s in scenes[3:]]
    )
    assert not three.exit_criterion()["any_config_meets_exit"]

    four = BenchReport(
        rows=[_row(s, 0.5) for s in scenes[:4]] + [_row(scenes[4], 9.0)]
    )
    assert four.exit_criterion()["any_config_meets_exit"]


def test_exit_criterion_is_per_config_not_pooled():
    """One config passing three scenes and another passing two must not add up
    to a pass."""
    scenes = ["a", "b", "c", "d"]
    rows = [_row(s, 0.5, mask="gt") for s in scenes[:2]]
    rows += [_row(s, 9.0, mask="gt") for s in scenes[2:]]
    rows += [_row(s, 9.0, mask="none") for s in scenes[:2]]
    rows += [_row(s, 0.5, mask="none") for s in scenes[2:]]
    assert not BenchReport(rows=rows).exit_criterion()["any_config_meets_exit"]


def test_regression_gate_fails_on_a_real_regression():
    base = BenchReport(rows=[_row("empty_rect", 0.10)])
    worse = BenchReport(rows=[_row("empty_rect", 0.10 + REGRESSION_TOLERANCE_PP + 0.01)])
    check = worse.regression_check(base)
    assert not check["passed"]
    assert check["failures"][0]["scene"] == "empty_rect"
    assert check["failures"][0]["delta_pp"] > REGRESSION_TOLERANCE_PP


def test_regression_gate_tolerates_noise_below_the_threshold():
    base = BenchReport(rows=[_row("empty_rect", 0.10)])
    jittery = BenchReport(rows=[_row("empty_rect", 0.10 + REGRESSION_TOLERANCE_PP - 0.01)])
    assert jittery.regression_check(base)["passed"]


def test_regression_gate_welcomes_improvement():
    base = BenchReport(rows=[_row("empty_rect", 2.0)])
    better = BenchReport(rows=[_row("empty_rect", 0.1)])
    assert better.regression_check(base)["passed"]


def test_regression_gate_ignores_scenes_absent_from_the_baseline():
    """Adding a new scene must not fail the build on its first run."""
    base = BenchReport(rows=[_row("empty_rect", 0.1)])
    added = BenchReport(rows=[_row("empty_rect", 0.1), _row("brand_new", 12.0)])
    assert added.regression_check(base)["passed"]


def test_report_round_trips_through_json():
    r = BenchReport(rows=[_row("a", 0.5), _row("b", 1.2)], host={"gpu": "none"},
                    created_utc="2026-08-03T00:00:00")
    again = BenchReport.from_json(r.to_json())
    assert [x.scene for x in again.rows] == ["a", "b"]
    assert again.rows[0].wall_err == pytest.approx(0.5)
    assert again.host["gpu"] == "none"
