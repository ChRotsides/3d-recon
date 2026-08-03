# Reconstruction service

Plan §2 and §4/Phase 2: FastAPI, a job queue, blob storage and signed artifact
URLs.

## Run it

```bash
export RECON_SIGNING_SECRET="$(openssl rand -hex 32)"   # required, no default
export RECON_BLOB_ROOT=/var/lib/recon
uvicorn app.main:app_from_env --factory --reload
```

Backends are injected. The defaults are local — filesystem blobs, an in-process
queue, an in-memory repository — so the whole service runs and is tested without
S3, Redis or Postgres. Swapping in the production ones replaces `LocalBlobStore`,
`InlineQueue` and `Repository` and touches nothing else.

## API

| | |
|---|---|
| `POST /v1/sessions` | start a capture session |
| `PUT  /v1/sessions/{id}/bundle` | upload the `SessionBundle`; validated here |
| `PUT  /v1/sessions/{id}/keyframes/{kf}` | upload one keyframe image |
| `PUT  /v1/sessions/{id}/aux/{depth\|mask}/{kf}` | optional RGBD depth / precomputed masks |
| `GET  /v1/sessions/{id}` | what is already stored — **use this to resume** |
| `POST /v1/sessions/{id}/complete` | enqueue reconstruction |
| `GET  /v1/jobs/{id}` | per-step state, cost, signed artifact URLs |
| `POST /v1/jobs/{id}/retry` | re-run from the first unfinished step |
| `GET  /v1/artifacts/{key}` | signed, expiring download |
| `POST /v1/properties/{id}/stitch` | join rooms into a house (§6.3) |
| `GET  /v1/properties/{id}` | the `HouseGraph` |
| `GET  /v1/costs` | per-job cost against the §9 budget |

## Two things that are load-bearing

**Upload is resumable.** A capture is up to 150 keyframes sent over a phone
connection from inside someone's house. `GET /v1/sessions/{id}` returns
`keyframes_missing`, and `complete` refuses with the same list rather than
starting a job on a partial bundle.

**Artifact URLs are signed and expire.** These files are a photoreal scan of
where a customer lives. Serving them from a guessable path would make every
scan world-readable to anyone who learns a job id. `RECON_SIGNING_SECRET` has no
default and the service refuses to start without one.

## Known gap

Steps are recorded and retried per step, but the pipeline still runs as a single
call, so a retry does not genuinely resume mid-pipeline — it re-runs and marks
the already-done steps. Real resumption needs the intermediate artifacts (masks,
depth, TSDF volume) persisted to blobs between steps. The `retry` response says
so rather than implying otherwise.
