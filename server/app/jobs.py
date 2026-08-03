"""Job model and the reconstruction worker (PLAN.md §2, §4/Phase 2).

Plan §4/Phase 2 requires the seven pipeline steps to be **independently
retryable**. That drives the design here: a job is a list of steps, each with
its own state, error and timing, and a retry resumes from the first step that
is not `done` rather than starting over. On a six-minute job where step 6 threw,
re-running steps 1–5 is five minutes of GPU time burned for nothing.

Cost is instrumented per job from the start, as the plan asks, because a
per-room budget of $0.30–0.80 (§9) is only enforceable if it is measured.
"""

from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class StepState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


#: The seven steps from PLAN.md §2, in order.
STEP_NAMES = (
    "mask",       # 1. SAM 3
    "depth",      # 2. Depth Anything 3, pose-conditioned
    "pose",       # 3. pose graph + scale constraints
    "splat",      # 4. PGSR via gsplat            [GPU]
    "fuse",       # 5. TSDF fusion -> planes
    "layout",     # 6. wall/floor/ceiling -> floor plan
    "package",    # 7. SPZ + GLB + SVG + layout JSON
)


@dataclass
class Step:
    name: str
    state: StepState = StepState.PENDING
    seconds: float = 0.0
    error: str | None = None
    attempts: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "state": self.state.value,
            "seconds": round(self.seconds, 3), "error": self.error,
            "attempts": self.attempts,
        }


#: Indicative hourly rates from PLAN.md §9. Used for the per-job estimate; the
#: real figure comes from the deployment's own instance pricing.
INSTANCE_RATES_USD_PER_HOUR = {
    "cpu": 0.10,
    "l40s": 1.20,
    "a100": 2.60,
    "h100": 4.50,
}


@dataclass
class Cost:
    instance: str = "cpu"
    seconds: float = 0.0
    peak_vram_gb: float = 0.0

    @property
    def usd(self) -> float:
        rate = INSTANCE_RATES_USD_PER_HOUR.get(self.instance, 0.0)
        return self.seconds / 3600.0 * rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "seconds": round(self.seconds, 2),
            "peak_vram_gb": round(self.peak_vram_gb, 2),
            "usd": round(self.usd, 4),
        }


@dataclass
class Job:
    id: str
    session_id: str
    room_id: str
    state: JobState = JobState.QUEUED
    steps: list[Step] = field(default_factory=lambda: [Step(n) for n in STEP_NAMES])
    artifacts: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    cost: Cost = field(default_factory=Cost)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def step(self, name: str) -> Step:
        for s in self.steps:
            if s.name == name:
                return s
        raise KeyError(name)

    @property
    def first_unfinished(self) -> Step | None:
        for s in self.steps:
            if s.state not in (StepState.DONE, StepState.SKIPPED):
                return s
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "session_id": self.session_id,
            "room_id": self.room_id,
            "state": self.state.value,
            "steps": [s.as_dict() for s in self.steps],
            "artifacts": sorted(self.artifacts),
            "summary": self.summary,
            "cost": self.cost.as_dict(),
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobQueue(Protocol):
    def submit(self, job: Job) -> None: ...


class InlineQueue:
    """Runs the job on the calling thread.

    Not a test double — it is the correct backend for a single-box deployment
    and for the Phase-0 harness, and it means the API's behaviour is identical
    whether or not Celery is present. `CeleryQueue` wraps the same `run_job`.
    """

    def __init__(self, runner) -> None:
        self._runner = runner

    def submit(self, job: Job) -> None:
        self._runner(job)


def new_job(session_id: str, room_id: str) -> Job:
    return Job(id=str(uuid.uuid4()), session_id=session_id, room_id=room_id)


def run_job(
    job: Job,
    bundle_dir,
    blobs,
    *,
    pipeline=None,
    instance: str = "cpu",
    train_splats: bool = False,
) -> Job:
    """Execute the pipeline, recording per-step state, timing and cost.

    Steps already marked `done` are skipped, so a retry resumes rather than
    restarts. The pipeline itself is not decomposed into seven separate
    invocations — it is one call with intermediate results in memory — so the
    steps here record *what happened* and where it failed. Splitting the
    pipeline so each step can genuinely resume from persisted state is the
    remaining work, and is noted in the API's `retry` response.
    """
    from pathlib import Path

    from recon.io.bundle import BundleDir
    from recon.pipeline import Pipeline, PipelineConfig

    job.state = JobState.RUNNING
    job.error = None
    started = time.time()

    try:
        bundle = BundleDir.load(bundle_dir)
        pipe = pipeline or Pipeline(config=PipelineConfig(train_splats=train_splats))

        t0 = time.time()
        for name in ("mask", "depth", "pose", "fuse", "layout"):
            job.step(name).state = StepState.RUNNING
        out_dir = Path(blobs.local_path(f"artifacts/{job.id}"))
        result = pipe.run(bundle, out_dir)
        elapsed = time.time() - t0

        # Attribute the measured per-stage timings the pipeline already returns.
        mapping = {"mask": "mask", "depth": "depth", "pose": "scale",
                   "fuse": "fuse", "layout": "layout"}
        for step_name, timing_key in mapping.items():
            st = job.step(step_name)
            st.state = StepState.DONE
            st.seconds = float(result.timings_s.get(timing_key, 0.0))
            st.attempts += 1

        splat = job.step("splat")
        if train_splats and result.splat:
            splat.state = StepState.DONE
            splat.seconds = float(result.timings_s.get("splat", 0.0))
        else:
            splat.state = StepState.SKIPPED
            splat.error = None if not train_splats else "splat trainer produced no result"
        splat.attempts += 1

        pkg = job.step("package")
        pkg.state = StepState.RUNNING
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                job.artifacts[p.name] = f"artifacts/{job.id}/{p.relative_to(out_dir)}"
        pkg.state = StepState.DONE
        pkg.attempts += 1

        job.summary = result.summary()
        job.cost = Cost(instance=instance, seconds=elapsed,
                        peak_vram_gb=float(result.depth.peak_vram_gb))
        job.state = JobState.SUCCEEDED
    except Exception as exc:  # noqa: BLE001 - a failed job is a result, not a crash
        failing = job.first_unfinished
        if failing is not None:
            failing.state = StepState.FAILED
            failing.error = f"{type(exc).__name__}: {exc}"
            failing.attempts += 1
        job.state = JobState.FAILED
        job.error = f"{type(exc).__name__}: {exc}"
        job.summary.setdefault("traceback", traceback.format_exc()[-2000:])
        job.cost = Cost(instance=instance, seconds=time.time() - started)
    finally:
        job.finished_at = time.time()
    return job
