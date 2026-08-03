"""Cloud service tests (PLAN.md §2, §4/Phase 2).

Runs the real app against local blob/queue backends — no S3, no Redis, no
Postgres — so the whole upload → reconstruct → download path is exercised on
every commit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from app.main import Repository, Settings, create_app
from app.storage import LocalBlobStore, UrlSigner
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """One small synthetic room, rendered once and reused."""
    from recon import synth

    d = tmp_path_factory.mktemp("srv")
    cfg = synth.CaptureConfig(width=96, height=72, n_stations=2, yaw_step_deg=60,
                              pitches_deg=(0.0, -20.0), max_keyframes=18)
    synth.generate(synth.preset("empty_rect"), d / "room", cfg)
    return d / "room"


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(blob_root=tmp_path / "blobs", signing_secret="test-secret"))
    with TestClient(app) as c:
        yield c


def _upload(client, scene: Path, *, skip: set[str] | None = None, aux: bool = False) -> str:
    """Upload a bundle. With `aux`, also send the depth and mask side channels —
    which is what an RGBD rig (plan §8's ZED 2i) provides and what lets the
    worker run without a GPU."""
    bundle = json.loads((scene / "bundle.json").read_text())
    sid = client.post("/v1/sessions", json={"room_id": bundle["room_id"]}).json()["session_id"]
    assert client.put(f"/v1/sessions/{sid}/bundle", json=bundle).status_code == 200
    for kf in bundle["keyframes"]:
        if skip and kf["id"] in skip:
            continue
        img = scene / kf["image"]
        with img.open("rb") as fh:
            r = client.put(
                f"/v1/sessions/{sid}/keyframes/{kf['id']}",
                files={"file": (img.name, fh, "image/png")},
            )
        assert r.status_code == 200, r.text
        if not aux:
            continue
        for kind, rel in (("depth", kf.get("depth")), ("mask", f"masks/{kf['id']}.png")):
            src = scene / rel if rel else None
            if src is None or not src.exists():
                continue
            with src.open("rb") as fh:
                assert client.put(
                    f"/v1/sessions/{sid}/aux/{kind}/{kf['id']}",
                    files={"file": (src.name, fh, "image/png")},
                ).status_code == 200
    return sid


# -- upload -----------------------------------------------------------------


def test_health(client):
    assert client.get("/v1/healthz").json()["ok"] is True


def test_bundle_is_validated_at_the_boundary(client):
    """A bundle that cannot be parsed must fail *before* 150 images are
    uploaded, not on the worker afterwards."""
    sid = client.post("/v1/sessions", json={}).json()["session_id"]
    r = client.put(f"/v1/sessions/{sid}/bundle", json={"nonsense": True})
    assert r.status_code == 422
    assert "validation" in r.json()["error"]


def test_keyframe_cap_is_enforced(client, scene):
    """PLAN §7 cost control. The server cannot trust the client to respect it."""
    bundle = json.loads((scene / "bundle.json").read_text())
    kf = bundle["keyframes"][0]
    bundle["keyframes"] = [dict(kf, id=f"kf_{i:04d}") for i in range(151)]
    sid = client.post("/v1/sessions", json={}).json()["session_id"]
    r = client.put(f"/v1/sessions/{sid}/bundle", json=bundle)
    assert r.status_code == 413


def test_upload_is_resumable(client, scene):
    """A capture is 150 keyframes over a phone connection in someone's house.
    Losing signal at keyframe 140 must not mean starting again."""
    bundle = json.loads((scene / "bundle.json").read_text())
    dropped = {bundle["keyframes"][3]["id"], bundle["keyframes"][7]["id"]}
    sid = _upload(client, scene, skip=dropped)

    status = client.get(f"/v1/sessions/{sid}").json()
    assert set(status["keyframes_missing"]) == dropped

    # Completing with gaps is refused, and the response says exactly what to send.
    r = client.post(f"/v1/sessions/{sid}/complete")
    assert r.status_code == 409
    assert set(r.json()["missing"]) == dropped

    for kf_id in dropped:
        kf = next(k for k in bundle["keyframes"] if k["id"] == kf_id)
        with (scene / kf["image"]).open("rb") as fh:
            client.put(f"/v1/sessions/{sid}/keyframes/{kf_id}",
                       files={"file": ("x.png", fh, "image/png")})
    assert client.get(f"/v1/sessions/{sid}").json()["keyframes_missing"] == []
    assert client.post(f"/v1/sessions/{sid}/complete").status_code == 202


def test_empty_keyframe_is_rejected(client, scene):
    sid = client.post("/v1/sessions", json={}).json()["session_id"]
    r = client.put(f"/v1/sessions/{sid}/keyframes/kf_0000",
                   files={"file": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_unknown_session_is_404(client):
    assert client.get("/v1/sessions/nope").status_code == 404


# -- reconstruction ---------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_upload_reconstruct_download(client, scene):
    sid = _upload(client, scene, aux=True)
    job_id = client.post(f"/v1/sessions/{sid}/complete").json()["job_id"]

    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["state"] == "succeeded", job.get("error")
    assert [s["name"] for s in job["steps"]] == [
        "mask", "depth", "pose", "splat", "fuse", "layout", "package"
    ]
    assert {s["state"] for s in job["steps"]} <= {"done", "skipped"}
    # The splat step is GPU-bound and off by default; it must be *skipped*, not
    # silently reported as done.
    assert job["steps"][3]["state"] == "skipped"

    for name in ("layout.json", "floorplan.svg", "floorplan.pdf", "summary.json"):
        assert name in job["artifacts"], job["artifacts"]

    url = job["artifact_urls"]["layout.json"]
    r = client.get(url)
    assert r.status_code == 200
    layout = r.json()
    assert layout["area_m2"] > 5.0
    assert r.headers["content-type"].startswith("application/json")


@pytest.mark.slow
def test_cost_is_instrumented(client, scene):
    """PLAN §9: 'Instrument cost per job from day one.'"""
    sid = _upload(client, scene, aux=True)
    client.post(f"/v1/sessions/{sid}/complete")
    costs = client.get("/v1/costs").json()
    assert costs["jobs"] >= 1
    assert costs["by_job"]
    one = next(iter(costs["by_job"].values()))
    assert one["seconds"] > 0
    assert one["usd"] >= 0
    assert costs["budget_usd_per_room"] == [0.30, 0.80]


def test_a_failed_job_is_reported_not_crashed(client, tmp_path):
    """A bundle whose images were never uploaded reaches the worker only if the
    completion check is bypassed; the worker must still fail cleanly."""
    from app.jobs import new_job, run_job

    blobs = LocalBlobStore(tmp_path / "b")
    job = new_job("s1", "r1")
    run_job(job, tmp_path / "does-not-exist", blobs)
    assert job.state.value == "failed"
    assert job.error
    assert any(s.state.value == "failed" for s in job.steps)
    assert job.cost.seconds >= 0


def test_retry_resumes_from_the_first_unfinished_step(client, tmp_path):
    from app.jobs import Job, JobState, StepState

    repo = Repository()
    submitted: list[Job] = []

    class Recording:
        def submit(self, job: Job) -> None:
            submitted.append(job)

    app = create_app(
        Settings(blob_root=tmp_path / "b", signing_secret="s"), repo=repo, queue=Recording()
    )
    with TestClient(app) as c:
        job = Job(id="j1", session_id="s1", room_id="r1", state=JobState.FAILED)
        job.step("mask").state = StepState.DONE
        job.step("depth").state = StepState.DONE
        job.step("pose").state = StepState.FAILED
        job.step("pose").error = "boom"
        repo.jobs["j1"] = job

        r = c.post("/v1/jobs/j1/retry")
        assert r.status_code == 202
        assert r.json()["resumed_from"] == "pose"
        assert job.step("pose").error is None, "the failed step should be reset"
        assert job.step("mask").state == StepState.DONE, "done steps must not be reset"
        assert submitted and submitted[0] is job


def test_retry_refuses_a_succeeded_job(client, tmp_path):
    from app.jobs import Job, JobState

    repo = Repository()
    app = create_app(Settings(blob_root=tmp_path / "b", signing_secret="s"), repo=repo)
    with TestClient(app) as c:
        repo.jobs["ok"] = Job(id="ok", session_id="s", room_id="r",
                              state=JobState.SUCCEEDED)
        assert c.post("/v1/jobs/ok/retry").status_code == 409


# -- signed URLs ------------------------------------------------------------


def test_artifact_urls_are_signed_and_expiring(tmp_path):
    signer = UrlSigner("secret", default_ttl_s=60)
    signed = signer.sign("artifacts/j1/layout.json")
    assert "sig=" in signed.url and "expires=" in signed.url
    expires = int(signed.url.split("expires=")[1].split("&")[0])
    sig = signed.url.split("sig=")[1]
    assert signer.verify("artifacts/j1/layout.json", expires, sig)
    assert not signer.verify("artifacts/OTHER/layout.json", expires, sig)
    assert not signer.verify("artifacts/j1/layout.json", int(time.time()) - 1, sig)


def test_a_forged_signature_is_refused(client, tmp_path):
    r = client.get("/v1/artifacts/artifacts/x/layout.json?expires=99999999999&sig=deadbeef")
    assert r.status_code == 403


def test_refuses_to_sign_with_a_default_secret():
    """An unsigned or predictably-signed artifact endpoint exposes every
    customer's scan of their own home."""
    with pytest.raises(ValueError):
        UrlSigner("change-me")
    with pytest.raises(ValueError):
        UrlSigner("")


def test_blob_keys_cannot_escape_the_store(tmp_path):
    store = LocalBlobStore(tmp_path / "root")
    with pytest.raises(ValueError):
        store.put("../../etc/evil", b"x")


def test_blob_round_trip_and_delete(tmp_path):
    store = LocalBlobStore(tmp_path / "root")
    store.put("a/b.txt", b"hello")
    assert store.exists("a/b.txt") and store.get("a/b.txt") == b"hello"
    assert store.size("a/b.txt") == 5
    assert store.delete_prefix("a") == 1
    assert not store.exists("a/b.txt")


# -- house graph ------------------------------------------------------------


def test_stitch_needs_at_least_two_rooms(client):
    r = client.post("/v1/properties/p1/stitch", json={"job_ids": ["only-one"]})
    assert r.status_code == 400


def test_unknown_property_is_404(client):
    assert client.get("/v1/properties/nope").status_code == 404


@pytest.mark.slow
def test_stitch_two_rooms_into_a_house(client, tmp_path):
    from recon import synth, synth_house

    house = synth_house.corridor_house(n_rooms=2, seed=1)
    cfg = synth.CaptureConfig(width=110, height=83, n_stations=3, yaw_step_deg=36,
                              max_keyframes=48)
    src = tmp_path / "house"
    gt = synth_house.generate_house(house, src, cfg)

    job_ids = []
    for room in gt.rooms:
        sid = _upload(client, src / room["room_id"], aux=True)
        job_ids.append(client.post(f"/v1/sessions/{sid}/complete").json()["job_id"])

    r = client.post(
        "/v1/properties/p1/stitch",
        json={
            "job_ids": job_ids,
            "connections": [[gt.rooms[0]["room_id"], gt.rooms[1]["room_id"]]],
            "use_gt_masks": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["rooms"] == 2
    assert body["summary"]["edges"] == 1, body["rejected"]

    graph = client.get("/v1/properties/p1").json()
    assert graph["property_id"] == "p1"
    assert len(graph["floors"][0]["edges"]) == 1
    assert client.get(body["house_graph_url"]).status_code == 200
