"""FastAPI transport for the strict RGB-only service contract."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError

from recon.live_rgb.schema import (
    CreateSession,
    CreateStation,
    FinalizeStation,
    FrameMetadata,
    PreviewMetadata,
)
from recon.live_rgb.service import ConflictError, LiveRgbService


def create_app(root: Path, *, depth_backend: str = "stub") -> FastAPI:
    service = LiveRgbService(root, depth_backend=depth_backend)
    app = FastAPI(title="Live RGB Reconstruction", version="1.0.0")

    def execute(operation):
        try:
            return operation()
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except ConflictError as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/v1/sessions")
    def create_session(request: CreateSession):
        return execute(lambda: service.create_session(request))

    @app.post("/v1/stations")
    def create_station(request: CreateStation):
        return execute(lambda: service.create_station(request))

    @app.put("/v1/frames", status_code=201)
    def upload_frame(
        jpeg: bytes = Body(media_type="image/jpeg"),
        x_rgb_frame: str = Header(alias="X-RGB-Frame"),
    ):
        try:
            metadata = FrameMetadata.model_validate(json.loads(x_rgb_frame))
        except (json.JSONDecodeError, ValidationError) as error:
            raise HTTPException(422, f"invalid X-RGB-Frame: {error}") from error
        return execute(lambda: service.upload_frame(metadata, jpeg))

    @app.get("/v1/guidance/{session_id}/{station_id}")
    def guidance(session_id: str, station_id: str):
        return execute(lambda: service.guidance(session_id, station_id))

    @app.get("/v1/drafts/{session_id}/{station_id}/latest")
    def latest_draft(session_id: str, station_id: str):
        return execute(lambda: service.draft_status(session_id, station_id))

    @app.get("/v1/drafts/{session_id}/{station_id}/{version}/{name}")
    def draft_asset(session_id: str, station_id: str, version: int, name: str):
        return execute(
            lambda: FileResponse(service.draft_path(session_id, station_id, version, name))
        )

    @app.put("/v1/previews")
    def preview(
        jpeg: bytes = Body(media_type="image/jpeg"),
        x_rgb_preview: str = Header(alias="X-RGB-Preview"),
    ):
        try:
            metadata = PreviewMetadata.model_validate(json.loads(x_rgb_preview))
        except (json.JSONDecodeError, ValidationError) as error:
            raise HTTPException(422, f"invalid X-RGB-Preview: {error}") from error
        return execute(lambda: service.localize_preview(metadata, jpeg))

    @app.post("/v1/stations/finalize")
    def finalize(request: FinalizeStation):
        return execute(lambda: service.finalize(request).model_dump())

    @app.get("/v1/results/{session_id}/{station_id}/{version}/{name}")
    def result(session_id: str, station_id: str, version: int, name: str):
        return execute(
            lambda: FileResponse(service.result_path(session_id, station_id, version, name))
        )

    @app.on_event("shutdown")
    def shutdown() -> None:
        service.close()

    return app
