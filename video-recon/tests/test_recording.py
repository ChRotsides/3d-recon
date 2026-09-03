import json

import numpy as np
import pytest
from videorecon.recording import Recording, intrinsics_from_meta


def _meta(**camera):
    base = {
        "device": {"model": "SM-S918B", "os": "android 15"},
        "video": {"width": 1920, "height": 1080, "fps": 30},
        "camera": {
            "id": "0",
            "sensor_orientation_deg": 90,
            "timestamp_source": "REALTIME",
            "active_array": {"width": 4000, "height": 3000},
            "pre_correction_active_array": {"width": 4000, "height": 3000},
            "intrinsic_calibration": [3000.0, 3000.0, 2000.0, 1500.0, 0.0],
            "focal_length_mm": 6.0,
            "physical_size_mm": {"width": 8.0, "height": 6.0},
            "distortion": [],
        },
    }
    base["camera"].update(camera)
    return base


def test_intrinsics_centre_crop_and_scale_from_calibration():
    intr = intrinsics_from_meta(_meta())
    # 4:3 array cropped to 16:9 -> 4000x2250 with a 375 px vertical offset, then x0.48.
    assert intr.source == "LENS_INTRINSIC_CALIBRATION"
    np.testing.assert_allclose([intr.fx, intr.fy, intr.cx, intr.cy], [1440, 1440, 960, 540])
    assert (intr.width, intr.height) == (1920, 1080)


def test_intrinsics_fallback_from_focal_length_mm():
    intr = intrinsics_from_meta(_meta(intrinsic_calibration=None))
    assert intr.source.startswith("focal_length_mm")
    np.testing.assert_allclose(intr.fx, 6.0 / 8.0 * 4000 * 0.48)
    np.testing.assert_allclose([intr.cx, intr.cy], [960, 540])


def _write_recording(root, timestamp_source="REALTIME"):
    root.mkdir()
    (root / "meta.json").write_text(json.dumps(_meta(timestamp_source=timestamp_source)))
    (root / "frames.csv").write_text(
        "frame_number,sensor_timestamp_ns,exposure_ns,focus_diopters,iso\n"
        + "".join(f"{i},{1_000_000_000 + i * 33_333_333},10000000,0.4,400\n" for i in range(10))
    )
    lines = ["timestamp_ns,sensor,x,y,z,w"]
    for i in range(50):
        t = 990_000_000 + i * 10_000_000
        lines.append(f"{t},acc,0.0,0.0,9.81,")
        lines.append(f"{t},gamerotv,0.0,0.0,0.0,1.0")
        lines.append(f"{t},mag,20.0,0.0,-40.0,")
    (root / "imu.csv").write_text("\n".join(lines) + "\n")


def test_recording_loads_frames_and_sensors(tmp_path):
    root = tmp_path / "rec_test"
    _write_recording(root)
    rec = Recording.load(root)
    assert rec.frames.shape == (10, 2)
    assert rec.sensor_orientation_deg == 90
    assert rec.has_sensor("acc") and rec.has_sensor("gamerotv") and not rec.has_sensor("rotv")
    assert abs(rec.sensor_rates_hz()["acc"] - 100.0) < 1e-6
    track = rec.orientation_track("gamerotv")
    np.testing.assert_allclose(track.rotation_at(np.array([1_000_000_000]))[0], np.eye(3))
    with pytest.raises(KeyError):
        rec.orientation_track("rotv")


def test_recording_refuses_unknown_clock(tmp_path):
    root = tmp_path / "rec_bad"
    _write_recording(root, timestamp_source="UNKNOWN")
    with pytest.raises(ValueError):
        Recording.load(root)


def test_every_cli_flag_reaches_options():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "reconstruct.py"
    spec = importlib.util.spec_from_file_location("reconstruct_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = mod.build_parser().parse_args(
        [
            "rec",
            "out",
            "--camera-height",
            "1.3",
            "--scale-source",
            "floor",
            "--variant",
            "small",
            "--exhaustive-below",
            "99",
            "--min-rotation",
            "5",
            "--no-depth",
        ]
    )
    o = mod.options_from_args(a)
    assert o.camera_height_m == 1.3 and o.scale_source == "floor" and o.depth_variant == "small"
    assert o.exhaustive_below == 99 and o.min_rotation_deg == 5 and o.run_depth is False
    # Every parsed flag except the positionals maps to an Options field or a run switch.
    dests = {
        act.dest
        for act in mod.build_parser()._actions
        if act.dest not in ("help", "recording", "out")
    }
    covered = {
        "orientation_sensor",
        "min_rotation",
        "max_interval",
        "max_long_edge",
        "max_keyframes",
        "exhaustive_below",
        "variant",
        "no_depth",
        "no_fusion",
        "voxel",
        "depth_trunc",
        "force",
        "no_magnetic_heading",
        "camera_height",
        "scale_source",
    }
    assert dests == covered, dests ^ covered
