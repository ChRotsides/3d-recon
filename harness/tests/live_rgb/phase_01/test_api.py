from __future__ import annotations

from fastapi.testclient import TestClient
from recon.live_rgb.api import create_app


def test_json_endpoint_rejects_undeclared_geometry(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/v1/sessions",
        json={"session_id": "session-1", "pose": {"translation": [1, 2, 3]}},
    )
    assert response.status_code == 422


def test_session_and_station_creation_are_idempotent(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))
    session = {"session_id": "session-1"}
    station = {"session_id": "session-1", "station_id": "station-1"}
    assert client.post("/v1/sessions", json=session).status_code == 200
    assert client.post("/v1/sessions", json=session).status_code == 200
    assert client.post("/v1/stations", json=station).status_code == 200
    assert client.post("/v1/stations", json=station).status_code == 200
