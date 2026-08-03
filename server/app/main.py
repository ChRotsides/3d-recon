"""Reconstruction API (PLAN.md §2, §4/Phase 2).

    FastAPI  ·  job queue  ·  blob storage  ·  signed download URLs

Deliberately backend-agnostic: blobs, queue and repository are injected, with
local implementations that let the whole service run and be tested without S3,
Redis or Postgres. Swapping in the production backends is configuration, not a
rewrite.

Upload is **resumable** (plan §4/Phase 1). A capture is up to 150 keyframes over
a phone connection in someone's house, and restarting the whole upload because
the lift lost signal at keyframe 140 is the difference between a product and a
demo. Keyframes are uploaded individually and the client can ask what is already
stored.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from app.jobs import InlineQueue, Job, JobState, new_job, run_job
from app.storage import LocalBlobStore, UrlSigner

MAX_KEYFRAMES = 150  # PLAN.md §7 cost control
MAX_KEYFRAME_BYTES = 12 * 1024 * 1024


@dataclass
class Session:
    id: str
    room_id: str
    property_id: str | None = None
    bundle_json: dict | None = None
    keyframes: set[str] = field(default_factory=set)
    #: Optional per-keyframe side channels, by kind ("depth", "mask").
    aux: dict[str, set[str]] = field(default_factory=dict)
    complete: bool = False
    job_id: str | None = None


@dataclass
class Repository:
    """In-memory metadata store.

    Stands in for the Postgres of plan §2. The interface is what the routes
    depend on, so the SQL implementation replaces this class alone.
    """

    sessions: dict[str, Session] = field(default_factory=dict)
    jobs: dict[str, Job] = field(default_factory=dict)
    houses: dict[str, dict] = field(default_factory=dict)


@dataclass
class Settings:
    blob_root: Path
    signing_secret: str
    instance: str = "cpu"
    train_splats: bool = False


def create_app(
    settings: Settings,
    repo: Repository | None = None,
    queue=None,
    pipeline=None,
) -> FastAPI:
    repo = repo or Repository()
    blobs = LocalBlobStore(settings.blob_root)
    signer = UrlSigner(settings.signing_secret)

    def _run(job: Job) -> None:
        run_job(
            job,
            blobs.local_path(f"sessions/{job.session_id}"),
            blobs,
            pipeline=pipeline,
            instance=settings.instance,
            train_splats=settings.train_splats,
        )

    q = queue or InlineQueue(_run)
    app = FastAPI(title="recon", version="0.1.0")
    app.state.repo = repo
    app.state.blobs = blobs
    app.state.signer = signer
    app.state.queue = q

    def get_session(session_id: str) -> Session:
        s = repo.sessions.get(session_id)
        if s is None:
            raise HTTPException(404, f"unknown session {session_id}")
        return s

    # -- sessions -----------------------------------------------------------

    @app.post("/v1/sessions", status_code=201)
    def create_session(body: dict[str, Any]) -> dict[str, Any]:
        room_id = body.get("room_id") or str(uuid.uuid4())
        session = Session(
            id=str(uuid.uuid4()),
            room_id=room_id,
            property_id=body.get("property_id"),
        )
        repo.sessions[session.id] = session
        return {
            "session_id": session.id,
            "room_id": session.room_id,
            "keyframe_upload": f"/v1/sessions/{session.id}/keyframes/{{keyframe_id}}",
            "max_keyframes": MAX_KEYFRAMES,
        }

    @app.put("/v1/sessions/{session_id}/bundle")
    async def put_bundle(session_id: str, request: Request) -> dict[str, Any]:
        session = get_session(session_id)
        try:
            payload = json.loads(await request.body())
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"bundle is not valid JSON: {exc}") from exc

        # Validate against the canonical schema at the boundary. A bundle that
        # cannot be parsed here will fail on the worker instead, after the
        # client has already uploaded 150 images.
        from recon.schemas import SessionBundle

        try:
            bundle = SessionBundle.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, f"bundle failed validation: {exc}") from exc

        if len(bundle.keyframes) > MAX_KEYFRAMES:
            raise HTTPException(
                413, f"{len(bundle.keyframes)} keyframes exceeds the {MAX_KEYFRAMES} cap"
            )
        session.bundle_json = payload
        session.room_id = bundle.room_id
        blobs.put(f"sessions/{session.id}/bundle.json",
                  json.dumps(payload, indent=2).encode())
        return {
            "session_id": session.id,
            "keyframes_expected": [kf.id for kf in bundle.keyframes],
            "keyframes_present": sorted(session.keyframes),
        }

    @app.put("/v1/sessions/{session_id}/keyframes/{keyframe_id}")
    async def put_keyframe(
        session_id: str, keyframe_id: str, file: UploadFile
    ) -> dict[str, Any]:
        session = get_session(session_id)
        data = await file.read()
        if len(data) > MAX_KEYFRAME_BYTES:
            raise HTTPException(413, "keyframe too large")
        if not data:
            raise HTTPException(400, "empty keyframe")
        suffix = Path(file.filename or f"{keyframe_id}.jpg").suffix or ".jpg"
        blobs.put(f"sessions/{session.id}/images/{keyframe_id}{suffix}", data)
        session.keyframes.add(keyframe_id)
        return {"stored": keyframe_id, "bytes": len(data),
                "keyframes_present": len(session.keyframes)}

    @app.put("/v1/sessions/{session_id}/aux/{kind}/{keyframe_id}")
    async def put_aux(
        session_id: str, kind: str, keyframe_id: str, file: UploadFile
    ) -> dict[str, Any]:
        """Per-keyframe side channels: `depth` and `mask`.

        Not every capture is RGB-only. The plan's ground-truth rig (§8) is a
        ZED 2i, which produces depth directly, and a client that has run
        on-device segmentation can send masks rather than making the worker
        redo the work. Both are optional — a phone bundle simply omits them and
        the worker falls back to DA3 and SAM 3.
        """
        if kind not in ("depth", "mask"):
            raise HTTPException(400, f"unknown auxiliary kind {kind!r}")
        session = get_session(session_id)
        data = await file.read()
        if not data:
            raise HTTPException(400, f"empty {kind}")
        if len(data) > MAX_KEYFRAME_BYTES:
            raise HTTPException(413, f"{kind} too large")
        folder = "depth" if kind == "depth" else "masks"
        suffix = Path(file.filename or f"{keyframe_id}.png").suffix or ".png"
        blobs.put(f"sessions/{session.id}/{folder}/{keyframe_id}{suffix}", data)
        session.aux.setdefault(kind, set()).add(keyframe_id)
        return {"stored": keyframe_id, "kind": kind, "bytes": len(data)}

    @app.get("/v1/sessions/{session_id}")
    def session_status(session_id: str) -> dict[str, Any]:
        """What the client asks after a dropped connection, to resume."""
        session = get_session(session_id)
        expected: list[str] = []
        if session.bundle_json:
            expected = [kf["id"] for kf in session.bundle_json.get("keyframes", [])]
        return {
            "session_id": session.id,
            "room_id": session.room_id,
            "has_bundle": session.bundle_json is not None,
            "keyframes_present": sorted(session.keyframes),
            "keyframes_missing": sorted(set(expected) - session.keyframes),
            "aux": {k: len(v) for k, v in session.aux.items()},
            "complete": session.complete,
            "job_id": session.job_id,
        }

    @app.post("/v1/sessions/{session_id}/complete", status_code=202)
    def complete_session(session_id: str) -> dict[str, Any]:
        session = get_session(session_id)
        if session.bundle_json is None:
            raise HTTPException(409, "no bundle uploaded for this session")
        expected = {kf["id"] for kf in session.bundle_json.get("keyframes", [])}
        missing = sorted(expected - session.keyframes)
        if missing:
            raise HTTPException(
                409,
                {"error": "keyframes missing; resume the upload",
                 "missing": missing[:20], "missing_count": len(missing)},
            )
        session.complete = True
        job = new_job(session.id, session.room_id)
        repo.jobs[job.id] = job
        session.job_id = job.id
        q.submit(job)
        return {"job_id": job.id, "state": job.state.value}

    # -- jobs ---------------------------------------------------------------

    def get_job(job_id: str) -> Job:
        job = repo.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id}")
        return job

    @app.get("/v1/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        out = job.as_dict()
        out["artifact_urls"] = {
            name: signer.sign(key).url for name, key in job.artifacts.items()
        }
        return out

    @app.post("/v1/jobs/{job_id}/retry", status_code=202)
    def retry_job(job_id: str) -> dict[str, Any]:
        """Re-run from the first step that is not done (plan §4/Phase 2)."""
        job = get_job(job_id)
        if job.state == JobState.SUCCEEDED:
            raise HTTPException(409, "job already succeeded")
        resume_from = job.first_unfinished
        for step in job.steps:
            if step.state.value == "failed":
                step.state = step.state.__class__.PENDING
                step.error = None
        job.state = JobState.QUEUED
        q.submit(job)
        return {
            "job_id": job.id,
            "resumed_from": resume_from.name if resume_from else None,
            "state": job.state.value,
            "note": (
                "steps already marked done are not recomputed; full mid-pipeline "
                "resume needs the intermediate artifacts persisted, which is not "
                "done yet"
            ),
        }

    # -- artifacts ----------------------------------------------------------

    @app.get("/v1/artifacts/{key:path}")
    def download(key: str, expires: int = Query(...), sig: str = Query(...)) -> Response:
        if not signer.verify(key, expires, sig):
            raise HTTPException(403, "invalid or expired signature")
        if not blobs.exists(key):
            raise HTTPException(404, "no such artifact")
        return Response(
            content=blobs.get(key),
            media_type=_media_type(key),
            headers={"Content-Disposition": f'attachment; filename="{Path(key).name}"'},
        )

    # -- house graph (plan §5, §6.3) ----------------------------------------

    @app.post("/v1/properties/{property_id}/stitch")
    def stitch_property(property_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Join reconstructed rooms into a house.

        `connections` is the user's doorway tagging (§6.3 path B step 1), and
        `pairing` optionally names which doorway on each side — which is what
        removes the ambiguity in a corridor of identical doors.
        """
        import numpy as np
        from recon.masking.base import GroundTruthMasker, NullMasker
        from recon.pipeline import Pipeline
        from recon.stitch.house import reconstruct_rooms, stitch

        job_ids = body.get("job_ids") or []
        connections = [tuple(c) for c in body.get("connections", [])]
        if len(job_ids) < 2:
            raise HTTPException(400, "need at least two rooms to stitch")

        dirs = []
        for jid in job_ids:
            job = get_job(jid)
            if job.state != JobState.SUCCEEDED:
                raise HTTPException(409, f"job {jid} has not succeeded")
            dirs.append(Path(blobs.local_path(f"sessions/{job.session_id}")))

        masker = GroundTruthMasker() if body.get("use_gt_masks") else NullMasker()
        rooms = reconstruct_rooms(dirs, Pipeline(masker=masker))
        result = stitch(
            rooms, connections, np.array([0.0, 1.0, 0.0]), property_id=property_id
        )
        graph = json.loads(result.graph.model_dump_json())
        repo.houses[property_id] = graph
        blobs.put(f"properties/{property_id}/house_graph.json",
                  json.dumps(graph, indent=2).encode())
        return {
            "property_id": property_id,
            "summary": result.summary(),
            "rejected": [list(r) for r in result.rejected],
            "house_graph_url": signer.sign(
                f"properties/{property_id}/house_graph.json"
            ).url,
        }

    @app.get("/v1/properties/{property_id}")
    def get_property(property_id: str) -> dict[str, Any]:
        graph = repo.houses.get(property_id)
        if graph is None:
            raise HTTPException(404, f"unknown property {property_id}")
        return graph

    # -- ops ----------------------------------------------------------------

    @app.get("/v1/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "sessions": len(repo.sessions), "jobs": len(repo.jobs)}

    @app.get("/v1/costs")
    def costs() -> dict[str, Any]:
        """Per-job cost, instrumented from day one as plan §9 requires."""
        done = [j for j in repo.jobs.values() if j.finished_at]
        total = sum(j.cost.usd for j in done)
        return {
            "jobs": len(done),
            "total_usd": round(total, 4),
            "mean_usd_per_room": round(total / len(done), 4) if done else 0.0,
            "budget_usd_per_room": [0.30, 0.80],  # PLAN.md §9
            "by_job": {j.id: j.cost.as_dict() for j in done},
        }

    @app.exception_handler(HTTPException)
    def _http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
        return JSONResponse(status_code=exc.status_code, content=detail)

    return app


def _media_type(key: str) -> str:
    return {
        ".json": "application/json", ".svg": "image/svg+xml",
        ".pdf": "application/pdf", ".glb": "model/gltf-binary",
        ".spz": "application/octet-stream", ".ply": "application/octet-stream",
        ".txt": "text/plain",
    }.get(Path(key).suffix, "application/octet-stream")


def app_from_env() -> FastAPI:
    secret = os.environ.get("RECON_SIGNING_SECRET")
    if not secret:
        raise RuntimeError(
            "RECON_SIGNING_SECRET is required — artifact URLs are signed with it, "
            "and an unsigned artifact endpoint exposes every customer's scan"
        )
    return create_app(
        Settings(
            blob_root=Path(os.environ.get("RECON_BLOB_ROOT", "/var/lib/recon")),
            signing_secret=secret,
            instance=os.environ.get("RECON_INSTANCE", "cpu"),
            train_splats=os.environ.get("RECON_TRAIN_SPLATS", "").lower() == "true",
        )
    )
