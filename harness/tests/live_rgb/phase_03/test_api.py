from recon.live_rgb.api import create_app


def test_versioned_draft_event_and_asset_routes_are_declared(tmp_path):
    app = create_app(tmp_path)
    try:
        paths = app.openapi()["paths"]
        assert "/v1/drafts/{session_id}/{station_id}/latest" in paths
        assert "/v1/drafts/{session_id}/{station_id}/{version}/{name}" in paths
        assert set(paths["/v1/drafts/{session_id}/{station_id}/latest"]) == {"get"}
    finally:
        # create_app owns a daemon worker; close it explicitly in this unit fixture.
        for route in app.router.on_shutdown:
            route()
