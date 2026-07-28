"""Shared plumbing for tool modules: argument bases, job submission, output naming."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import get_settings
from ..errors import InvalidParameterError
from ..ffmpeg.probe import probe
from ..jobs.worker import JobContext, JobOutcome, submit
from ..models import JobSubmission, MediaInfo, StrictModel
from ..paths import validate_input_file, validate_output_path


class MediaJobArgs(StrictModel):
    """Base arguments for a tool that reads one file and writes one file."""

    input_path: str = Field(description="Path to the source media file.")
    output_path: str | None = Field(
        default=None,
        description=(
            "Destination file. If omitted, the output is written into the job's "
            "workspace directory and its path is returned by job_result."
        ),
    )


class JobResult(StrictModel):
    """The standard payload a finished editing job reports."""

    output_path: str
    media: MediaInfo | None = None
    notes: list[str] = Field(default_factory=list)


def suggest_output_name(source: Path, operation: str, suffix: str | None = None) -> str:
    """Build a descriptive default filename for a job output."""
    extension = suffix or source.suffix or ".mp4"
    if not extension.startswith("."):
        extension = "." + extension
    return f"{source.stem}_{operation}{extension}"


def resolve_io(
    args: MediaJobArgs,
    *,
    operation: str,
    job_id: str | None = None,
    suffix: str | None = None,
) -> tuple[Path, Path]:
    """Validate the input path and resolve the output path for a job."""
    source = validate_input_file(args.input_path)
    destination = validate_output_path(
        args.output_path,
        suggested_name=suggest_output_name(source, operation, suffix),
        job_id=job_id,
    )
    if destination == source:
        raise InvalidParameterError(
            "Output path must differ from the input path.", path=str(source)
        )
    return source, destination


def precheck_inputs(*paths: str | None) -> None:
    """Validate paths at submission time so obvious mistakes fail before queueing."""
    for raw in paths:
        if raw:
            validate_input_file(raw)


def queue(tool: str, args: BaseModel, *, message: str | None = None) -> JobSubmission:
    """Enqueue a job for a tool and build the immediate response."""
    params: dict[str, Any] = args.model_dump(mode="json")
    record = submit(tool, params)
    return JobSubmission(
        job_id=record.job_id,
        tool=tool,
        message=message
        or "Job queued. Poll job_status for progress, then job_result for the output path.",
    )


async def finish_media_job(
    ctx: JobContext, output: Path, *, notes: list[str] | None = None, command: str | None = None
) -> JobOutcome:
    """Probe a rendered output and package it as the job's result."""
    media: MediaInfo | None = None
    try:
        media = await probe(output, ctx.settings)
    except Exception:
        media = None
    payload = JobResult(output_path=str(output), media=media, notes=notes or [])
    return JobOutcome(
        outputs=[str(output)], result=payload.model_dump(mode="json"), command=command
    )


def workspace_note() -> str:
    """Describe where unspecified outputs land, for tool descriptions."""
    return str(get_settings().jobs_dir)


__all__ = [
    "JobResult",
    "MediaJobArgs",
    "finish_media_job",
    "precheck_inputs",
    "queue",
    "resolve_io",
    "suggest_output_name",
    "validate_input_file",
    "validate_output_path",
    "workspace_note",
]
