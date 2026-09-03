"""Strict wire contracts for the RGB-only capture protocol."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CreateSession(StrictModel):
    session_id: Identifier


class CreateStation(StrictModel):
    session_id: Identifier
    station_id: Identifier


class FrameMetadata(StrictModel):
    session_id: Identifier
    station_id: Identifier
    frame_id: Identifier
    timestamp_ns: Annotated[int, Field(ge=0)]


class PreviewMetadata(StrictModel):
    session_id: Identifier
    station_id: Identifier
    preview_id: Identifier
    timestamp_ns: Annotated[int, Field(ge=0)]


class FinalizeStation(StrictModel):
    session_id: Identifier
    station_id: Identifier


class JobStatus(StrictModel):
    job_id: Identifier
    state: Literal["queued", "running", "complete", "failed"]
    result_version: int | None = None
    error: str | None = None


FORBIDDEN_GEOMETRY_FIELDS = frozenset(
    {
        "pose",
        "depth",
        "intrinsics",
        "extrinsics",
        "sparse_points",
        "planes",
        "anchors",
        "tracking_state",
        "heading",
        "heading_deg",
        "imu",
        "gravity",
        "translation",
        "rotation",
    }
)
