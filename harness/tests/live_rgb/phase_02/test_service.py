from __future__ import annotations

from recon.live_rgb.schema import CreateSession, CreateStation, FrameMetadata, PreviewMetadata
from recon.live_rgb.service import LiveRgbService


def test_duplicate_frame_has_one_version_edge_and_target_event(tmp_path, textured_images):
    service = LiveRgbService(tmp_path)
    service.create_session(CreateSession(session_id="s"))
    service.create_station(CreateStation(session_id="s", station_id="t"))
    metadata = FrameMetadata(session_id="s", station_id="t", frame_id="f", timestamp_ns=1)
    first = service.upload_frame(metadata, textured_images[0])
    second = service.upload_frame(metadata, textured_images[0])
    assert second == first
    station = tmp_path / "sessions/s/stations/t/incremental"
    assert len(list((station / "versions").glob("*.json"))) == 1
    assert len(list((station / "events").glob("*.json"))) == 1


def test_textureless_preview_is_unlocalized_idempotent_and_does_not_mutate_graph(
    tmp_path, textured_images
):
    import io
    from PIL import Image

    service = LiveRgbService(tmp_path)
    service.create_session(CreateSession(session_id="s"))
    service.create_station(CreateStation(session_id="s", station_id="t"))
    service.upload_frame(
        FrameMetadata(session_id="s", station_id="t", frame_id="f", timestamp_ns=1),
        textured_images[0],
    )
    before = service.guidance("s", "t")
    stream = io.BytesIO()
    Image.new("RGB", (320, 240), (127, 127, 127)).save(stream, "JPEG")
    metadata = PreviewMetadata(
        session_id="s", station_id="t", preview_id="p", timestamp_ns=2
    )
    first = service.localize_preview(metadata, stream.getvalue())
    second = service.localize_preview(metadata, stream.getvalue())
    assert first == second
    assert first["localization"]["state"] == "unlocalized"
    assert first["target"] is None
    assert first["graph"] == before["graph"]
    assert first["graph_version"] == before["graph_version"]
    assert first["version"] > before["version"]
