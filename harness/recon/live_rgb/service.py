"""Durable, idempotent storage for RGB-only sessions and reconstruction jobs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from recon.live_rgb.draft import DraftWorker
from recon.live_rgb.incremental import IncrementalGraphEngine
from recon.live_rgb.pipeline import reconstruct
from recon.live_rgb.replay import ReplayFrame
from recon.live_rgb.schema import (
    CreateSession,
    CreateStation,
    FinalizeStation,
    FrameMetadata,
    JobStatus,
    PreviewMetadata,
)


class ConflictError(ValueError):
    """An idempotency key was reused for different content."""


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class LiveRgbService:
    def __init__(
        self,
        root: Path,
        *,
        depth_backend: str = "stub",
        runner: Callable[..., dict] = reconstruct,
        draft_worker: DraftWorker | None = None,
    ) -> None:
        self.root = root
        self.depth_backend = depth_backend
        self.runner = runner
        self.root.mkdir(parents=True, exist_ok=True)
        self.incremental = IncrementalGraphEngine(root)
        self.drafts = draft_worker or DraftWorker(root, depth_backend=depth_backend)
        self._recover_interrupted_jobs()

    def _recover_interrupted_jobs(self) -> None:
        """Turn process-local running markers into retryable failures on restart."""

        for marker in self.root.glob("sessions/*/stations/*/finalize.json"):
            status = JobStatus.model_validate(json.loads(marker.read_text()))
            if status.state != "running":
                continue
            failed = JobStatus(
                job_id=status.job_id,
                state="failed",
                error="reconstruction service restarted while this job was running",
            )
            _atomic_json(marker, failed.model_dump())
            _atomic_json(marker.parent / "jobs" / f"{status.job_id}.json", failed.model_dump())

    def _session(self, session_id: str) -> Path:
        return self.root / "sessions" / session_id

    def _station(self, session_id: str, station_id: str) -> Path:
        return self._session(session_id) / "stations" / station_id

    def create_session(self, request: CreateSession) -> dict:
        path = self._session(request.session_id) / "session.json"
        payload = {"schema": "live-rgb-session-v1", "session_id": request.session_id}
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ConflictError("session id already has different content")
        if not path.exists():
            _atomic_json(path, payload)
        return payload

    def create_station(self, request: CreateStation) -> dict:
        session = self._session(request.session_id) / "session.json"
        if not session.exists():
            raise FileNotFoundError("session does not exist")
        path = self._station(request.session_id, request.station_id) / "station.json"
        payload = {
            "schema": "live-rgb-station-v1",
            "session_id": request.session_id,
            "station_id": request.station_id,
        }
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ConflictError("station id already has different content")
        if not path.exists():
            _atomic_json(path, payload)
        return payload

    def upload_frame(self, metadata: FrameMetadata, jpeg: bytes) -> dict:
        station = self._station(metadata.session_id, metadata.station_id)
        if not (station / "station.json").exists():
            raise FileNotFoundError("station does not exist")
        if (station / "finalize.json").exists():
            raise ConflictError("station is already finalized")
        if len(jpeg) < 4 or jpeg[:2] != b"\xff\xd8" or jpeg[-2:] != b"\xff\xd9":
            raise ValueError("body is not a complete JPEG")
        frame_dir = station / "frames" / metadata.frame_id
        image_path = frame_dir / "frame.jpg"
        meta_path = frame_dir / "frame.json"
        digest = hashlib.sha256(jpeg).hexdigest()
        payload = {
            **metadata.model_dump(),
            "schema": "live-rgb-frame-v1",
            "jpeg_sha256": digest,
            "jpeg_bytes": len(jpeg),
        }
        if meta_path.exists():
            if json.loads(meta_path.read_text()) != payload or image_path.read_bytes() != jpeg:
                raise ConflictError("frame id already has different content")
            guidance = self.incremental.guidance(station)
            return {**payload, "guidance": guidance, "draft": self.drafts.status(station)}
        frame_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".frame.", dir=frame_dir)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(jpeg)
                output.flush()
                os.fsync(output.fileno())
            with Image.open(temporary) as image:
                if image.format != "JPEG":
                    raise ValueError("body is not a JPEG")
                image.verify()
            os.replace(temporary, image_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        _atomic_json(meta_path, payload)
        guidance = self.incremental.process_station(station)
        self._maybe_queue_draft(metadata.session_id, metadata.station_id, guidance)
        return {**payload, "guidance": guidance, "draft": self.drafts.status(station)}

    def _maybe_queue_draft(
        self,
        session_id: str,
        station_id: str,
        guidance: dict,
        *,
        force: bool = False,
    ) -> bool:
        """Queue only connected, periodically stable RGB panorama snapshots."""

        selected = len(guidance["graph"]["selected"])
        edges = len(guidance["graph"]["edges"])
        if selected < 6 or edges < 5:
            return False
        station = self._station(session_id, station_id)
        source_version = int(guidance["graph_version"])
        status = self.drafts.status(station)
        previous = int(status.get("latest_requested_version", status.get("version", 0)))
        if not force and previous and source_version - previous < 4:
            return False
        return self.drafts.submit(
            station,
            source_version,
            self.frames(session_id, station_id),
        )

    def guidance(self, session_id: str, station_id: str) -> dict:
        station = self._station(session_id, station_id)
        if not (station / "station.json").exists():
            raise FileNotFoundError("station does not exist")
        return self.incremental.guidance(station)

    def draft_status(self, session_id: str, station_id: str) -> dict:
        station = self._station(session_id, station_id)
        if not (station / "station.json").exists():
            raise FileNotFoundError("station does not exist")
        return self.drafts.status(station)

    def draft_path(self, session_id: str, station_id: str, version: int, name: str) -> Path:
        if name not in {"draft.bin.gz", "draft.json", "diagnostics.json", "panorama.png"}:
            raise ValueError("unknown draft asset")
        path = (
            self._station(session_id, station_id) / "drafts" / "versions" / f"v{version:06d}" / name
        )
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def localize_preview(self, metadata: PreviewMetadata, jpeg: bytes) -> dict:
        station = self._station(metadata.session_id, metadata.station_id)
        if not (station / "station.json").exists():
            raise FileNotFoundError("station does not exist")
        if len(jpeg) < 4 or jpeg[:2] != b"\xff\xd8" or jpeg[-2:] != b"\xff\xd9":
            raise ValueError("body is not a complete JPEG")
        return self.incremental.localize_preview(station, metadata.preview_id, jpeg)

    def frames(self, session_id: str, station_id: str) -> list[ReplayFrame]:
        station = self._station(session_id, station_id)
        frames = []
        for path in sorted((station / "frames").glob("*/frame.json")):
            metadata = json.loads(path.read_text())
            frames.append(
                ReplayFrame(
                    frame_id=metadata["frame_id"],
                    timestamp_ns=metadata["timestamp_ns"],
                    jpeg_path=path.with_name("frame.jpg"),
                )
            )
        return sorted(frames, key=lambda frame: (frame.timestamp_ns, frame.frame_id))

    def finalize(self, request: FinalizeStation) -> JobStatus:
        station = self._station(request.session_id, request.station_id)
        if not (station / "station.json").exists():
            raise FileNotFoundError("station does not exist")
        marker = station / "finalize.json"
        if marker.exists():
            previous = JobStatus.model_validate(json.loads(marker.read_text()))
            if previous.state != "failed":
                return previous
        frame_list = self.frames(request.session_id, request.station_id)
        if not frame_list:
            raise ValueError("cannot finalize an empty station")
        # Loop closure may make the final live graph materially more stable. It
        # is queued independently and never blocks this existing final export.
        self._maybe_queue_draft(
            request.session_id,
            request.station_id,
            self.incremental.guidance(station),
            force=True,
        )
        job_id = f"job-{request.station_id}-v000001"
        running = JobStatus(job_id=job_id, state="running")
        _atomic_json(marker, running.model_dump())
        _atomic_json(station / "jobs" / f"{job_id}.json", running.model_dump())
        result = station / "results" / "v000001"
        if result.exists():
            shutil.rmtree(result)
        try:
            self.runner(frame_list, result, depth_backend=self.depth_backend)
            complete = JobStatus(job_id=job_id, state="complete", result_version=1)
            _atomic_json(marker, complete.model_dump())
            _atomic_json(station / "jobs" / f"{job_id}.json", complete.model_dump())
            return complete
        except Exception as error:
            failed = JobStatus(job_id=job_id, state="failed", error=str(error))
            _atomic_json(marker, failed.model_dump())
            _atomic_json(station / "jobs" / f"{job_id}.json", failed.model_dump())
            raise

    def result_path(self, session_id: str, station_id: str, version: int, name: str) -> Path:
        if name not in {"cloud.bin", "cloud.json", "diagnostics.json", "panorama.png"}:
            raise ValueError("unknown result asset")
        path = self._station(session_id, station_id) / "results" / f"v{version:06d}" / name
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def close(self) -> None:
        self.drafts.close()
