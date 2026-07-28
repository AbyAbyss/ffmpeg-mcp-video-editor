"""Background worker pool and the handler registry it dispatches to.

Tools do not execute their own work. They validate arguments, enqueue a job, and
return a job id immediately; a worker in this pool picks the job up and runs the
registered handler. That keeps MCP calls short even when the underlying render
takes minutes.

Workers claim jobs through :meth:`JobStore.claim_next`, which is atomic, so
running a pool in the MCP server process *and* another in the UI process is safe
-- they simply share the queue.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..config import Settings, get_settings
from ..errors import FFmpegMCPError, JobCancelledError
from ..models import JobError, JobRecord
from .store import JobStore, get_store

log = logging.getLogger(__name__)

_IDLE_POLL_SECONDS = 0.2
_CLEANUP_INTERVAL_SECONDS = 900.0


@dataclass
class JobOutcome:
    """What a handler returns when it succeeds."""

    outputs: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    command: str | None = None


@dataclass
class JobContext:
    """Everything a handler needs: its job, the store, and progress reporting."""

    job: JobRecord
    store: JobStore
    settings: Settings
    _last_reported: float = field(default=0.0, init=False)
    _last_percent: float = field(default=-1.0, init=False)

    @property
    def job_id(self) -> str:
        return self.job.job_id

    @property
    def params(self) -> dict[str, Any]:
        return self.job.params

    @property
    def workdir(self) -> Path:
        """The job's own directory in the workspace, covered by the retention policy."""
        return self.store.job_dir(self.job.job_id)

    async def report(self, percent: float | None = None, message: str | None = None) -> None:
        """Publish progress, throttled so a fast render does not hammer SQLite."""
        now = time.monotonic()
        significant = percent is None or abs(percent - self._last_percent) >= 1.0
        if message is None and not significant and now - self._last_reported < 0.5:
            return
        self._last_reported = now
        if percent is not None:
            self._last_percent = percent
        await asyncio.to_thread(self.store.update_progress, self.job.job_id, percent, message, None)

    def set_command(self, command: str) -> None:
        """Record the resolved ffmpeg command line for auditability."""
        self.store.update_progress(self.job.job_id, None, None, command)

    def cancelled(self) -> bool:
        """Whether cancellation has been requested, possibly by another process."""
        return self.store.is_cancel_requested(self.job.job_id)

    def make_progress_hook(self, floor: float = 0.0, ceiling: float = 100.0) -> Any:
        """Build an ffmpeg progress hook that maps 0-100 into a sub-range.

        Multi-pass handlers use this so, for example, the transcription stage
        occupies 0-60% and the caption burn 60-100%.
        """
        span = ceiling - floor

        async def hook(percent: float, _update: Any) -> None:
            await self.report(floor + percent / 100.0 * span)

        return hook


JobHandler = Callable[[JobContext], Awaitable[JobOutcome]]

_HANDLERS: dict[str, JobHandler] = {}


def register_handler(tool: str, handler: JobHandler) -> None:
    """Register the function that executes jobs for a tool."""
    if tool in _HANDLERS:
        raise ValueError(f"Duplicate job handler for tool {tool!r}")
    _HANDLERS[tool] = handler


def handler(tool: str) -> Callable[[JobHandler], JobHandler]:
    """Decorator form of :func:`register_handler`."""

    def decorate(fn: JobHandler) -> JobHandler:
        register_handler(tool, fn)
        return fn

    return decorate


def get_handler(tool: str) -> JobHandler | None:
    """Look up a registered handler."""
    return _HANDLERS.get(tool)


def registered_tools() -> list[str]:
    """Every tool name that has a job handler."""
    return sorted(_HANDLERS)


async def execute_job(record: JobRecord, store: JobStore, settings: Settings) -> None:
    """Run one claimed job to completion, recording the outcome in the store."""
    context = JobContext(job=record, store=store, settings=settings)
    fn = get_handler(record.tool)
    if fn is None:
        await asyncio.to_thread(
            store.fail,
            record.job_id,
            JobError(code="unknown_tool", message=f"No handler registered for {record.tool!r}."),
        )
        return
    try:
        outcome = await fn(context)
    except ValidationError as exc:
        # The tool layer already validated these arguments at enqueue time, so
        # reaching here means the job was written by a build whose schema
        # differs from this one. Say that, rather than dumping pydantic at the
        # user.
        log.warning("Job %s was written by an incompatible build", record.job_id)
        await asyncio.to_thread(
            store.fail,
            record.job_id,
            JobError(
                code="incompatible_build",
                message=(
                    f"This job was queued by a different build of ffmpeg-mcp than the "
                    f"worker that ran it, so its arguments for {record.tool!r} no longer "
                    f"validate. Restart every ffmpeg-mcp server and UI process so they "
                    f"all run the same code, then queue it again."
                ),
                details={"tool": record.tool, "validation_error": str(exc)},
            ),
        )
    except JobCancelledError:
        await asyncio.to_thread(store.mark_cancelled, record.job_id)
    except FFmpegMCPError as exc:
        await asyncio.to_thread(
            store.fail,
            record.job_id,
            JobError(code=exc.code, message=exc.message, details=exc.details),
            getattr(exc, "command", None),
        )
    except asyncio.CancelledError:
        await asyncio.to_thread(store.mark_cancelled, record.job_id)
        raise
    except Exception as exc:
        log.exception("Unhandled error in job %s (%s)", record.job_id, record.tool)
        await asyncio.to_thread(
            store.fail,
            record.job_id,
            JobError(code="internal_error", message=f"{type(exc).__name__}: {exc}"),
        )
    else:
        if store.is_cancel_requested(record.job_id):
            await asyncio.to_thread(store.mark_cancelled, record.job_id)
            return
        await asyncio.to_thread(
            store.finish, record.job_id, outcome.outputs, outcome.result, outcome.command
        )


class WorkerPool:
    """A fixed number of asyncio workers draining the shared queue."""

    def __init__(self, settings: Settings | None = None, store: JobStore | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store(self.settings)
        self.concurrency = max(1, self.settings.worker_concurrency)
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        """Start the workers and the periodic maintenance loop."""
        if self._tasks:
            return
        await asyncio.to_thread(self.store.reclaim_stale)
        self._stopping.clear()
        self._tasks = [
            asyncio.create_task(self._run_worker(i), name=f"ffmpeg-mcp-worker-{i}")
            for i in range(self.concurrency)
        ]
        self._tasks.append(asyncio.create_task(self._run_maintenance(), name="ffmpeg-mcp-janitor"))
        log.info("Worker pool started with %d workers", self.concurrency)

    async def stop(self) -> None:
        """Signal the workers to finish and wait for them."""
        self._stopping.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def _known_fingerprints(self) -> dict[str, str]:
        """Schema fingerprints this build can run, keyed by tool name."""
        from ..tools.registry import all_tools, load_all_tools

        specs = all_tools() or load_all_tools()
        return {spec.name: spec.schema_fingerprint() for spec in specs}

    async def _run_worker(self, index: int) -> None:
        pid = os.getpid()
        known = self._known_fingerprints()
        while not self._stopping.is_set():
            try:
                record = await asyncio.to_thread(self.store.claim_next, pid, known)
            except Exception:
                log.exception("Worker %d failed to claim a job", index)
                await asyncio.sleep(1.0)
                continue
            if record is None:
                await asyncio.sleep(_IDLE_POLL_SECONDS)
                continue
            log.info("Worker %d running job %s (%s)", index, record.job_id, record.tool)
            await execute_job(record, self.store, self.settings)

    async def _run_maintenance(self) -> None:
        while not self._stopping.is_set():
            try:
                await asyncio.to_thread(self.store.cleanup)
                await asyncio.to_thread(self.store.reclaim_stale)
            except Exception:
                log.exception("Maintenance pass failed")
            await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)


async def run_job_inline(record: JobRecord, settings: Settings | None = None) -> JobRecord:
    """Execute a job immediately in the current task. Used by tests."""
    settings = settings or get_settings()
    store = get_store(settings)
    await execute_job(record, store, settings)
    return store.get(record.job_id)


def submit(tool: str, params: dict[str, Any], settings: Settings | None = None) -> JobRecord:
    """Enqueue a job for a tool. Called from the synchronous tool layer."""
    settings = settings or get_settings()
    store = get_store(settings)
    if get_handler(tool) is None:
        raise ValueError(f"No handler registered for tool {tool!r}")
    from ..projects import active_project
    from ..tools.registry import get_tool

    spec = get_tool(tool)
    return store.create(
        tool,
        params,
        spec.schema_fingerprint() if spec else None,
        project=active_project(),
    )


def status_of(job_id: str, settings: Settings | None = None) -> JobRecord:
    """Fetch a job record by id."""
    return get_store(settings or get_settings()).get(job_id)


def ensure_finished(record: JobRecord) -> None:
    """Raise a descriptive error if a job has not reached a terminal state."""
    from ..errors import JobNotFinishedError

    if not record.status.is_terminal:
        raise JobNotFinishedError(
            "Job has not finished yet.",
            job_id=record.job_id,
            status=record.status.value,
            progress=record.progress,
        )
