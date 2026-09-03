from __future__ import annotations

import pytest
from pydantic import ValidationError
from recon.live_rgb.schema import CreateSession, FrameMetadata


@pytest.mark.parametrize(
    "field",
    ["pose", "depth", "intrinsics", "sparse_points", "tracking_state", "heading_deg", "imu"],
)
def test_frame_contract_rejects_every_geometric_field(field: str) -> None:
    payload = {
        "session_id": "session-1",
        "station_id": "station-1",
        "frame_id": "frame-1",
        "timestamp_ns": 1,
        field: {},
    }
    with pytest.raises(ValidationError):
        FrameMetadata.model_validate(payload)


def test_all_contracts_reject_undeclared_fields() -> None:
    with pytest.raises(ValidationError):
        CreateSession.model_validate({"session_id": "session-1", "device": "phone"})


def test_openapi_models_declare_closed_objects() -> None:
    assert CreateSession.model_json_schema()["additionalProperties"] is False
    assert FrameMetadata.model_json_schema()["additionalProperties"] is False
