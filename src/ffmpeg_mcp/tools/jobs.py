"""Phase 1 job-control tools: status, result, cancel, list.

These are the other half of the async job model. Every editing tool returns a
job id; these tools are how a client follows it to completion.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ..errors import JobNotFinishedError
from ..jobs.store import get_store
from ..models import JobError, JobRecord, JobStatus, StrictModel
from .registry import tool


class JobIdArgs(StrictModel):
    """Arguments identifying a single job."""

    job_id: str = Field(description="Job id returned when the work was queued.")


class JobStatusResult(StrictModel):
    """Current state of one job."""

    job_id: str
    tool: str
    status: JobStatus
    progress: float = Field(description="Percentage complete, 0-100.")
    message: str | None = None
    elapsed_seconds: float | None = None
    error: JobError | None = None
    command: str | None = Field(
        default=None, description="Resolved ffmpeg command line, once the job starts."
    )


class JobResultPayload(StrictModel):
    """Outputs of a finished job."""

    job_id: str
    tool: str
    status: JobStatus
    outputs: list[str] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)


class JobListArgs(StrictModel):
    """Arguments for listing jobs."""

    status: JobStatus | None = Field(default=None, description="Filter to one status.")
    project: str | None = Field(
        default=None,
        description=(
            "Which project to list. Defaults to the active one; pass 'all' to see "
            "every project at once."
        ),
    )
    limit: int = Field(default=25, ge=1, le=200, description="Page size.")
    offset: int = Field(default=0, ge=0, description="How many jobs to skip, for paging.")


class JobSummary(StrictModel):
    """One row in a job listing."""

    job_id: str
    tool: str
    status: JobStatus
    progress: float
    created_at: float
    message: str | None = None
    project: str | None = None


class JobListResult(StrictModel):
    """A page of jobs plus counts for the scope being listed."""

    jobs: list[JobSummary]
    counts: dict[str, int]
    project: str = Field(description="The scope these results cover; 'all' spans projects.")
    total: int = Field(description="Jobs matching the filter, ignoring this page.")
    offset: int
    limit: int
    has_more: bool


class CancelResult(StrictModel):
    """Outcome of a cancellation request."""

    job_id: str
    status: JobStatus
    cancelled: bool
    message: str


def _elapsed(record: JobRecord) -> float | None:
    if record.started_at is None:
        return None
    end = record.finished_at
    if end is None:
        import time

        end = time.time()
    return round(end - record.started_at, 3)


@tool("job_status", title="Job status", phase=1, read_only=True)
async def job_status(args: JobIdArgs) -> JobStatusResult:
    """Check how a queued job is progressing.

    Returns one of queued, running, done, failed or cancelled, with a progress
    percentage parsed from ffmpeg's own output. Poll this after queueing work,
    then call job_result once the status is 'done'.
    """
    record = get_store().get(args.job_id)
    return JobStatusResult(
        job_id=record.job_id,
        tool=record.tool,
        status=record.status,
        progress=record.progress,
        message=record.message,
        elapsed_seconds=_elapsed(record),
        error=record.error,
        command=record.command,
    )


@tool("job_result", title="Job result", phase=1, read_only=True)
async def job_result(args: JobIdArgs) -> JobResultPayload:
    """Fetch the output paths of a completed job.

    Errors if the job has not finished yet — poll job_status first. For a failed
    job this raises with the structured failure reason; for a successful one the
    'result' field carries the output path and a probe of the rendered file.
    """
    record = get_store().get(args.job_id)
    if not record.status.is_terminal:
        raise JobNotFinishedError(
            "Job has not finished yet; poll job_status.",
            job_id=record.job_id,
            status=record.status.value,
            progress=record.progress,
        )
    return JobResultPayload(
        job_id=record.job_id,
        tool=record.tool,
        status=record.status,
        outputs=record.outputs,
        result=record.result if record.error is None else {"error": record.error.model_dump()},
    )


@tool("cancel_job", title="Cancel a job", phase=1)
async def cancel_job(args: JobIdArgs) -> CancelResult:
    """Stop a queued or running job.

    A queued job is cancelled immediately. A running one is asked to stop: the
    worker that owns it terminates the ffmpeg subprocess within a second or so,
    which works even when the job was started by a different process such as the
    local UI. Jobs that already finished are left alone.
    """
    store = get_store()
    record = store.request_cancel(args.job_id)
    if record.status is JobStatus.CANCELLED:
        return CancelResult(
            job_id=record.job_id, status=record.status, cancelled=True, message="Job cancelled."
        )
    if record.status.is_terminal:
        return CancelResult(
            job_id=record.job_id,
            status=record.status,
            cancelled=False,
            message=f"Job already finished with status {record.status.value}.",
        )
    return CancelResult(
        job_id=record.job_id,
        status=record.status,
        cancelled=True,
        message="Cancellation requested; the worker will stop the job shortly.",
    )


@tool("list_jobs", title="List jobs", phase=1, read_only=True)
async def list_jobs(args: JobListArgs) -> JobListResult:
    """List jobs newest first, scoped to a project and returned a page at a time.

    Defaults to the active project, so a long-running server shared by several
    sessions does not bury your work in everyone else's. Pass project='all' to
    see everything, and use offset with limit to page through a busy queue —
    'total' and 'has_more' say whether there is another page.
    """
    from ..projects import active_project

    store = get_store()
    scope = args.project or active_project()
    selector = None if scope == "all" else scope
    records = store.list_jobs(
        status=args.status, project=selector, limit=args.limit, offset=args.offset
    )
    total = store.count_jobs(args.status, selector)
    return JobListResult(
        jobs=[
            JobSummary(
                job_id=r.job_id,
                tool=r.tool,
                status=r.status,
                progress=r.progress,
                created_at=r.created_at,
                message=r.message,
                project=r.project or "default",
            )
            for r in records
        ],
        counts=store.counts_by_status(selector),
        project=scope,
        total=total,
        offset=args.offset,
        limit=args.limit,
        has_more=args.offset + len(records) < total,
    )
