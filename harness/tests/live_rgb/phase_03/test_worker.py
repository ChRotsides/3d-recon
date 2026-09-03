import threading

from recon.live_rgb.draft import DraftWorker
from recon.live_rgb.replay import ReplayFrame


def frame(tmp_path, index):
    path = tmp_path / f"{index}.jpg"
    path.write_bytes(b"frame")
    return ReplayFrame(f"frame-{index}", index, path)


def test_pending_versions_coalesce_and_running_stale_work_never_publishes(tmp_path):
    started = threading.Event()
    release = threading.Event()
    built = []

    def builder(frames, output, *, source_version, depth_backend):
        built.append(source_version)
        output.mkdir(parents=True)
        if source_version == 1:
            started.set()
            assert release.wait(3)
        (output / "draft.json").write_text("{}")
        (output / "draft.bin.gz").write_bytes(b"x")
        return {"cloud": {"asset_sha256": str(source_version), "asset_bytes": 1, "count": 1}}

    station = tmp_path / "station"
    worker = DraftWorker(tmp_path, builder=builder)
    try:
        worker.submit(station, 1, [frame(tmp_path, 1)])
        assert started.wait(2)
        worker.submit(station, 2, [frame(tmp_path, 2)])
        worker.submit(station, 3, [frame(tmp_path, 3)])
        release.set()
        assert worker.wait_idle(5)
        assert built == [1, 3]
        assert not (station / "drafts/versions/v000001").exists()
        assert not (station / "drafts/versions/v000002").exists()
        assert (station / "drafts/versions/v000003").exists()
        assert worker.status(station)["version"] == 3
    finally:
        worker.close()
