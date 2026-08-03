"""Viewer scene manifest tests (PLAN.md §1.1, §3.8)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from recon.io.viewer import (
    BABYLON_CONVERSION,
    ViewerScene,
    build_scene,
    spawn_point,
    write_scene,
)
from recon.schemas import RoomLayout

VIEWER_DIR = Path(__file__).resolve().parents[2] / "viewer"


def _layout(room_id="r0", footprint=None, **kw) -> RoomLayout:
    return RoomLayout(
        room_id=room_id,
        footprint=footprint or [[0.0, 0.0], [4.2, 0.0], [4.2, 3.35], [0.0, 3.35]],
        floor_height_m=kw.get("floor", 0.0),
        ceiling_height_m=kw.get("ceiling", 2.45),
        area_m2=kw.get("area", 14.07),
        perimeter_m=15.1,
        wall_lengths_m=[4.2, 3.35, 4.2, 3.35],
    )


def test_scene_states_its_frame_and_the_required_conversion():
    """Babylon is left-handed by default and our scenes are right-handed. A
    viewer that guesses renders a mirrored house that looks entirely plausible
    (plan §7), so the manifest has to say."""
    scene = build_scene("p", {"r0": {"layout": _layout(), "artifacts": {}}})
    assert scene.frame["handedness"] == "right"
    assert scene.frame["up_axis"] == "y_up"
    assert scene.frame["viewer_conversion"] == BABYLON_CONVERSION


def test_scene_round_trips_through_json():
    scene = build_scene(
        "p", {"r0": {"layout": _layout(), "artifacts": {"spz": "r0/room.spz"}}}
    )
    again = ViewerScene.from_json(scene.to_json())
    assert again.property_id == "p"
    assert again.rooms[0].artifacts["spz"] == "r0/room.spz"
    assert again.total_area_m2 == pytest.approx(scene.total_area_m2)


def test_spawn_point_is_inside_an_l_shaped_room():
    """The centroid of a non-convex footprint can fall outside it — an L-shaped
    room would drop the camera inside a wall."""
    from recon import geom

    l_shape = [[0.0, 0.0], [5.0, 0.0], [5.0, 2.0], [2.0, 2.0], [2.0, 4.5], [0.0, 4.5]]
    layout = _layout(footprint=l_shape)
    x, y, z = spawn_point(layout)
    assert geom.point_in_polygon(np.array([[x, z]]), np.asarray(l_shape))[0]
    assert y == pytest.approx(1.6, abs=0.01)

    centroid = np.asarray(l_shape).mean(axis=0)
    assert not geom.point_in_polygon(np.array([centroid]), np.asarray(l_shape))[0], (
        "this footprint was chosen because its centroid is outside it"
    )


def test_spawn_point_respects_floor_height():
    layout = _layout(floor=0.4)
    assert spawn_point(layout)[1] == pytest.approx(2.0, abs=0.01)


def test_total_area_sums_the_rooms():
    scene = build_scene("p", {
        "a": {"layout": _layout("a", area=10.0), "artifacts": {}},
        "b": {"layout": _layout("b", area=12.5), "artifacts": {}},
    })
    assert scene.total_area_m2 == pytest.approx(22.5)


def test_rooms_default_to_the_origin_without_a_house(tmp_path):
    """A single-room scan has no house graph, and that is not an error."""
    scene = build_scene("p", {"r0": {"layout": _layout(), "artifacts": {}}})
    r = scene.rooms[0]
    assert (r.yaw_rad, r.tx, r.tz, r.scale) == (0.0, 0.0, 0.0, 1.0)
    assert not scene.portals
    written = write_scene(tmp_path / "scene.json", scene)
    assert json.loads(written.read_text())["schema_version"] == 1


@pytest.mark.slow
def test_scene_places_rooms_from_a_house_solve(tmp_path):
    from recon import synth, synth_house
    from recon.masking.base import GroundTruthMasker
    from recon.pipeline import Pipeline
    from recon.stitch.house import reconstruct_rooms, stitch

    house = synth_house.corridor_house(n_rooms=2, seed=1)
    cfg = synth.CaptureConfig(width=110, height=83, n_stations=3, yaw_step_deg=36,
                              max_keyframes=48)
    gt = synth_house.generate_house(house, tmp_path, cfg)
    rooms = reconstruct_rooms([tmp_path / r["room_id"] for r in gt.rooms],
                              Pipeline(masker=GroundTruthMasker()))
    conns = [(gt.rooms[0]["room_id"], gt.rooms[1]["room_id"])]
    result = stitch(rooms, conns, np.array([0.0, 1.0, 0.0]), property_id="p")

    scene = build_scene(
        "p",
        {rid: {"layout": r.layout, "artifacts": {"spz": f"{rid}/room.spz"}}
         for rid, r in rooms.items()},
        house_result=result,
    )
    assert len(scene.rooms) == 2
    # The second room must actually be placed away from the first.
    offsets = sorted(abs(r.tx) for r in scene.rooms)
    assert offsets[1] > 2.0, offsets
    assert scene.portals and scene.portals[0]["from_room"]


# -- the viewer itself ------------------------------------------------------


def test_viewer_files_exist():
    assert (VIEWER_DIR / "index.html").exists()
    assert (VIEWER_DIR / "walkthrough.js").exists()


def test_viewer_refuses_to_guess_the_coordinate_frame():
    """applyFrame throwing on an unlabelled scene is the guard; if someone
    replaces it with a default, this catches it."""
    js = (VIEWER_DIR / "walkthrough.js").read_text()
    assert "useRightHandedSystem = true" in js
    assert re.search(r"refusing to guess", js), "the no-frame branch must throw"
    assert "unsupported viewer_conversion" in js


def test_viewer_loads_babylon_locally_not_from_a_cdn():
    """A capture review in someone's basement cannot depend on signal."""
    html = (VIEWER_DIR / "index.html").read_text()
    assert "./vendor/babylon.js" in html
    assert "http://" not in html and "https://" not in html.split("<!--")[0]


def test_viewer_consumes_the_manifest_this_module_writes():
    """The contract between the two: field names the JS reads must exist."""
    js = (VIEWER_DIR / "walkthrough.js").read_text()
    scene = build_scene(
        "p", {"r0": {"layout": _layout(), "artifacts": {"spz": "a.spz"}}}
    )
    payload = json.loads(scene.to_json())
    for field in ("rooms", "frame", "total_area_m2"):
        assert field in payload, field
        assert field in js, f"viewer never reads scene.{field}"
    for field in ("room_id", "footprint", "spawn", "area_m2", "yaw_rad", "tx", "tz"):
        assert field in payload["rooms"][0], field
        assert field in js, f"viewer never reads room.{field}"
