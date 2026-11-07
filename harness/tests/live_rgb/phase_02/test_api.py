from __future__ import annotations

import json

from fastapi.testclient import TestClient
from recon.live_rgb.api import create_app
from recon.live_rgb.schema import FORBIDDEN_GEOMETRY_FIELDS


def test_guidance_and_preview_contract_are_rgb_only(tmp_path, textured_images):
    client = TestClient(create_app(tmp_path))
    assert client.post("/v1/sessions", json={"session_id": "s"}).status_code == 200
    assert (
        client.post("/v1/stations", json={"session_id": "s", "station_id": "t"}).status_code == 200
    )
    initial = client.get("/v1/guidance/s/t")
    assert initial.status_code == 200
    assert initial.json()["localization"]["state"] == "not_requested"
    header = {"session_id": "s", "station_id": "t", "preview_id": "p", "timestamp_ns": 1}
    preview = client.put(
        "/v1/previews",
        content=textured_images[0],
        headers={"content-type": "image/jpeg", "X-RGB-Preview": json.dumps(header)},
    )
    assert preview.status_code == 200
    poisoned = {**header, "preview_id": "poison", "pose": [1, 2, 3]}
    rejected = client.put(
        "/v1/previews",
        content=textured_images[0],
        headers={"content-type": "image/jpeg", "X-RGB-Preview": json.dumps(poisoned)},
    )
    assert rejected.status_code == 422
    wire = json.dumps(preview.json()).lower()
    assert not any(f'"{field}"' in wire for field in FORBIDDEN_GEOMETRY_FIELDS)


def test_delayed_versions_cannot_replace_newer_state():
    from recon.live_rgb.incremental import accept_newer

    newer = {"version": 9, "value": "new"}
    assert accept_newer(newer, {"version": 8, "value": "old"}) is newer
