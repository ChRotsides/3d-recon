# Reconstruction worker + API (PLAN.md §4/Phase 2).
#
# One image runs both roles: `uvicorn` for the API, or the Celery worker for the
# GPU queue. Splitting them into separate images would mean keeping two
# dependency sets in step for no benefit — the code is identical, only the
# entrypoint differs. Plan §9 does split the *instance classes* (H100/A100 for
# training, L40S for rendering); that is a deployment concern, set by
# RECON_INSTANCE, not an image concern.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RECON_BLOB_ROOT=/var/lib/recon

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY pyproject.toml README.md ./
COPY harness/ harness/
COPY server/ server/

RUN pip install --upgrade pip && pip install -e '.[geometry,server]'

# The GPU stages are a separate layer so a CPU-only deployment does not pull
# torch. See docs/execution-status.md for why they are unvalidated.
FROM base AS gpu
RUN pip install -e '.[gpu]'

FROM base AS api
RUN useradd -m -u 10001 recon && mkdir -p /var/lib/recon && chown recon /var/lib/recon
USER recon
EXPOSE 8000
# RECON_SIGNING_SECRET is required and has no default — an unsigned artifact
# endpoint would expose every customer's scan of their own home.
CMD ["uvicorn", "app.main:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
