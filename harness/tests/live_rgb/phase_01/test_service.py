from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from recon.live_rgb.pipeline import _write_cloud
from recon.live_rgb.schema import CreateSession, CreateStation, FinalizeStation, FrameMetadata
from recon.live_rgb.service import ConflictError, LiveRgbService


def jpeg_bytes(colour: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(output, format="JPEG")
    return output.getvalue()


def fake_runner(frames, output: Path, **_) -> dict:
    output.mkdir(parents=True)
    points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    colours = np.array([[1, 2, 3]], dtype=np.uint8)
    _write_cloud(output / "cloud", points, colours, {"model_revision": "fake"})
    (output / "diagnostics.json").write_text("{}\n")
    return {}


def service(tmp_path: Path) -> LiveRgbService:
    result = LiveRgbService(tmp_path, runner=fake_runner)
    result.create_session(CreateSession(session_id="session-1"))
    result.create_station(CreateStation(session_id="session-1", station_id="station-1"))
    return result


def test_duplicate_upload_is_one_logical_frame(tmp_path: Path) -> None:
    target = service(tmp_path)
    metadata = FrameMetadata(
        session_id="session-1", station_id="station-1", frame_id="frame-1", timestamp_ns=4
    )
    first = target.upload_frame(metadata, jpeg_bytes())
    second = target.upload_frame(metadata, jpeg_bytes())
    assert first == second
    assert len(target.frames("session-1", "station-1")) == 1


def test_reusing_frame_id_for_different_jpeg_conflicts(tmp_path: Path) -> None:
    target = service(tmp_path)
    metadata = FrameMetadata(
        session_id="session-1", station_id="station-1", frame_id="frame-1", timestamp_ns=4
    )
    target.upload_frame(metadata, jpeg_bytes())
    with pytest.raises(ConflictError):
        target.upload_frame(metadata, jpeg_bytes((200, 20, 30)))


def test_duplicate_finish_creates_one_job_and_immutable_version(tmp_path: Path) -> None:
    target = service(tmp_path)
    metadata = FrameMetadata(
        session_id="session-1", station_id="station-1", frame_id="frame-1", timestamp_ns=4
    )
    target.upload_frame(metadata, jpeg_bytes())
    request = FinalizeStation(session_id="session-1", station_id="station-1")
    first = target.finalize(request)
    second = target.finalize(request)
    assert first == second
    assert first.result_version == 1
    assert len(list((tmp_path / "sessions/session-1/stations/station-1/jobs").glob("*.json"))) == 1
    assert target.result_path("session-1", "station-1", 1, "cloud.bin").is_file()


def test_failed_finish_can_retry_without_reuploading_frames(tmp_path: Path) -> None:
    attempts = 0

    def flaky_runner(frames, output: Path, **_) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            output.mkdir(parents=True)
            (output / "partial").write_text("failed\n")
            raise RuntimeError("first attempt failed")
        return fake_runner(frames, output)

    target = LiveRgbService(tmp_path, runner=flaky_runner)
    target.create_session(CreateSession(session_id="session-1"))
    target.create_station(CreateStation(session_id="session-1", station_id="station-1"))
    target.upload_frame(
        FrameMetadata(
            session_id="session-1",
            station_id="station-1",
            frame_id="frame-1",
            timestamp_ns=4,
        ),
        jpeg_bytes(),
    )
    request = FinalizeStation(session_id="session-1", station_id="station-1")
    with pytest.raises(RuntimeError, match="first attempt failed"):
        target.finalize(request)
    result = target.finalize(request)
    assert result.state == "complete"
    assert attempts == 2
    assert not (
        tmp_path
        / "sessions/session-1/stations/station-1/results/v000001/partial"
    ).exists()


def test_service_restart_makes_interrupted_job_retryable(tmp_path: Path) -> None:
    target = service(tmp_path)
    metadata = FrameMetadata(
        session_id="session-1",
        station_id="station-1",
        frame_id="frame-1",
        timestamp_ns=4,
    )
    target.upload_frame(metadata, jpeg_bytes())
    station = tmp_path / "sessions/session-1/stations/station-1"
    running = {
        "job_id": "job-station-1-v000001",
        "state": "running",
        "result_version": None,
        "error": None,
    }
    (station / "finalize.json").write_text(json.dumps(running))
    (station / "jobs").mkdir(exist_ok=True)
    (station / "jobs/job-station-1-v000001.json").write_text(
        json.dumps(running)
    )
    restarted = LiveRgbService(tmp_path, runner=fake_runner)
    result = restarted.finalize(
        FinalizeStation(session_id="session-1", station_id="station-1")
    )
    assert result.state == "complete"
