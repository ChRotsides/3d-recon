from __future__ import annotations

import json
import shutil

import numpy as np

from recon.live_rgb.incremental import (
    IncrementalGraphEngine,
    _nearest_coverage_target,
    accept_newer,
)

from .conftest import make_station


def test_feature_and_pair_caches_survive_restart(tmp_path, textured_images):
    station = make_station(tmp_path, textured_images[:3])
    first = IncrementalGraphEngine(tmp_path)
    result = first.process_station(station)
    assert first.feature_calls == 3
    assert first.match_calls == 3
    restarted = IncrementalGraphEngine(tmp_path)
    assert restarted.process_station(station) == result
    assert restarted.feature_calls == 0
    assert restarted.match_calls == 0


def test_only_neighbours_and_start_loop_closures_are_candidates():
    assert IncrementalGraphEngine._candidate_pairs(12) == [
        (0, 11), (1, 11), (7, 11), (8, 11), (9, 11), (10, 11),
    ]


def test_restart_and_uninterrupted_graphs_are_identical(tmp_path, textured_images):
    first_root = tmp_path / "first"
    station = make_station(first_root, textured_images[:3])
    IncrementalGraphEngine(first_root).process_station(station)
    second_root = tmp_path / "second"
    shutil.copytree(first_root, second_root)
    make_station(first_root, textured_images)
    make_station(second_root, textured_images)
    uninterrupted = IncrementalGraphEngine(first_root).process_station(
        first_root / "sessions/s/stations/t"
    )
    restarted = IncrementalGraphEngine(second_root).process_station(
        second_root / "sessions/s/stations/t"
    )
    assert restarted["graph"]["selected"] == uninterrupted["graph"]["selected"]
    assert {
        (edge["first"], edge["second"]) for edge in restarted["graph"]["edges"]
    } == {
        (edge["first"], edge["second"]) for edge in uninterrupted["graph"]["edges"]
    }
    assert restarted["graph"]["rotations"].keys() == uninterrupted["graph"]["rotations"].keys()
    for key in restarted["graph"]["rotations"]:
        assert np.allclose(
            restarted["graph"]["rotations"][key],
            uninterrupted["graph"]["rotations"][key],
            atol=1e-4,
        )
    assert restarted["coverage"] == uninterrupted["coverage"]
    assert restarted["target"] == uninterrupted["target"]


def test_atomic_versions_and_stale_guard(tmp_path, textured_images):
    station = make_station(tmp_path, textured_images[:1])
    engine = IncrementalGraphEngine(tmp_path)
    one = engine.process_station(station)
    make_station(tmp_path, textured_images[:2])
    two = engine.process_station(station)
    assert two["version"] > one["version"]
    assert accept_newer(two, one) == two
    assert accept_newer(one, two) == two
    assert json.loads((station / "incremental/current.json").read_text()) == two
    assert not list((station / "incremental").rglob(".*.json.*"))


def test_every_coverage_cell_has_state_channels_and_reasons(tmp_path, textured_images):
    station = make_station(tmp_path, textured_images[:2])
    guidance = IncrementalGraphEngine(tmp_path).process_station(station)
    cells = guidance["coverage"]["cells"]
    assert len(cells) == guidance["coverage"]["width"] * guidance["coverage"]["height"]
    expected = {
        "photographed", "independent_views", "edge_strength", "feature_spread",
        "reprojection", "cycle_consistency", "sharpness", "exposure",
        "parallax_risk", "seam_support",
    }
    assert all(cell["state"] in {"red", "amber", "green"} for cell in cells)
    assert all(set(cell["channels"]) == expected for cell in cells)
    assert all(isinstance(cell["reason_mask"], list) for cell in cells)
    observed = [cell for cell in cells if cell["channels"]["independent_views"] == 1]
    assert observed
    assert all(cell["channels"]["photographed"] == 1.0 for cell in observed)
    assert all("single_view" not in cell["reason_mask"] for cell in observed)


def test_target_is_nearest_to_rgb_localized_view_not_first_grid_cell():
    cells = []
    for index in range(8):
        cells.append(
            {
                "index": index,
                "row": 0,
                "column": index,
                "state": "red",
                "channels": {"independent_views": 0},
                "reason_mask": ["missing"],
            }
        )
    coverage = {"width": 8, "height": 1, "cells": cells}
    target = _nearest_coverage_target(coverage, yaw_degrees=40, pitch_degrees=0, version=3)
    assert target is not None
    assert target["yaw_deg"] == 22.5
    assert target["target_id"] == "cell-00-04"



def test_amber_repairs_do_not_block_a_fast_scan():
    cells = [
        {
            "index": index,
            "row": row,
            "column": column,
            "state": "amber",
            "channels": {"independent_views": 1},
            "reason_mask": ["blur"],
        }
        for index, (row, column) in enumerate(
            (row, column) for row in range(12) for column in range(24)
        )
    ]
    coverage = {"width": 24, "height": 12, "cells": cells}
    assert _nearest_coverage_target(coverage, 0, 0, 9) is None


def test_pole_detail_gaps_do_not_block_a_fast_scan():
    cells = []
    for row in range(12):
        for column in range(24):
            cells.append(
                {
                    "index": row * 24 + column,
                    "row": row,
                    "column": column,
                    "state": "green" if 4 <= row < 8 else "red",
                    "channels": {"independent_views": 1 if 4 <= row < 8 else 0},
                    "reason_mask": [] if 4 <= row < 8 else ["missing"],
                }
            )
    coverage = {"width": 24, "height": 12, "cells": cells}
    assert _nearest_coverage_target(coverage, 0, 0, 11) is None
