"""Helpers for driving real tools through the real job pipeline in tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ffmpeg_mcp.config import Settings
from ffmpeg_mcp.ffmpeg.probe import probe
from ffmpeg_mcp.jobs.store import get_store
from ffmpeg_mcp.jobs.worker import execute_job
from ffmpeg_mcp.models import JobRecord, JobStatus, MediaInfo
from ffmpeg_mcp.tools.registry import get_tool, load_all_tools

load_all_tools()


async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Invoke a tool exactly as the MCP server would, including model validation."""
    spec = get_tool(name)
    assert spec is not None, f"tool {name!r} is not registered"
    return await spec.call(arguments)


async def run_job(name: str, arguments: dict[str, Any], settings: Settings) -> JobRecord:
    """Queue a tool's job and run it to completion in this task.

    Exercises the same path the worker pool uses -- claim, execute, record --
    without needing a background pool in the test.
    """
    submission = await call_tool(name, arguments)
    job_id = submission["job_id"]
    store = get_store(settings)
    claimed = store.claim_next(os.getpid())
    assert claimed is not None and claimed.job_id == job_id
    await execute_job(claimed, store, settings)
    return store.get(job_id)


async def run_job_ok(name: str, arguments: dict[str, Any], settings: Settings) -> JobRecord:
    """Run a job and assert it succeeded, surfacing the failure reason if not."""
    record = await run_job(name, arguments, settings)
    if record.status is not JobStatus.DONE:
        detail = record.error.model_dump() if record.error else {}
        raise AssertionError(f"{name} job ended {record.status.value}: {detail}")
    return record


def output_path(record: JobRecord) -> Path:
    """The single output file a media job produced."""
    assert record.outputs, "job produced no outputs"
    path = Path(record.outputs[0])
    assert path.exists(), f"reported output {path} does not exist"
    return path


async def probe_output(record: JobRecord, settings: Settings) -> MediaInfo:
    """Probe a finished job's output, which is how phase acceptance is checked."""
    return await probe(output_path(record), settings)
