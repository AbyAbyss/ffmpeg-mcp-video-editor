"""The single execution path for every ffmpeg and ffprobe subprocess.

Timeout handling, progress parsing, cancellation, and error wrapping live here
so no tool has to reimplement them. Arguments are always a list; ``shell=True``
appears nowhere in this project.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shlex
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..binaries import ffmpeg_env, get_binaries
from ..config import Settings, get_settings
from ..errors import FFmpegExecutionError, JobCancelledError, JobTimeoutError, ProbeError

log = logging.getLogger(__name__)

ProgressHook = Callable[[float, "ProgressUpdate"], Awaitable[None] | None]
CancelCheck = Callable[[], Awaitable[bool] | bool]

STDERR_TAIL_CHARS = 4000
_PROGRESS_POLL_SECONDS = 0.25


@dataclass
class ProgressUpdate:
    """One ``-progress pipe:1`` report block from ffmpeg."""

    out_time_seconds: float | None = None
    frame: int | None = None
    fps: float | None = None
    speed: float | None = None
    total_size: int | None = None
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class RunResult:
    """Outcome of a completed ffmpeg invocation."""

    command: str
    stderr: str
    duration_seconds: float


def format_command(argv: Sequence[str]) -> str:
    """Render an argv list as a copy-pasteable shell command, for logs and audit."""
    return " ".join(shlex.quote(str(a)) for a in argv)


def parse_progress_time(value: str) -> float | None:
    """Parse an ffmpeg ``out_time``/``out_time_ms`` style value into seconds.

    ``out_time`` looks like ``00:01:23.450000``. ffmpeg emits ``N/A`` before the
    first frame is written, and negative microseconds during some seeks.
    """
    value = value.strip()
    if not value or value == "N/A":
        return None
    if ":" in value:
        parts = value.split(":")
        try:
            hours, minutes, seconds = (float(p) for p in parts)
        except ValueError:
            return None
        return hours * 3600 + minutes * 60 + seconds
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _to_float(value: str) -> float | None:
    try:
        return float(value.rstrip("x"))
    except ValueError:
        return None


def build_progress_update(block: dict[str, str]) -> ProgressUpdate:
    """Turn one key=value block from ``-progress`` into a typed update."""
    out_time = None
    if "out_time_us" in block:
        micros = _to_int(block["out_time_us"])
        out_time = micros / 1_000_000 if micros is not None and micros >= 0 else None
    if out_time is None and "out_time_ms" in block:
        # Despite the name, ffmpeg reports microseconds here too.
        millis = _to_int(block["out_time_ms"])
        out_time = millis / 1_000_000 if millis is not None and millis >= 0 else None
    if out_time is None and "out_time" in block:
        out_time = parse_progress_time(block["out_time"])
    return ProgressUpdate(
        out_time_seconds=out_time,
        frame=_to_int(block.get("frame", "")) if "frame" in block else None,
        fps=_to_float(block.get("fps", "")) if "fps" in block else None,
        speed=_to_float(block.get("speed", "")) if "speed" in block else None,
        total_size=_to_int(block.get("total_size", "")) if "total_size" in block else None,
        raw=dict(block),
    )


async def _maybe_await(value: Awaitable[None] | None) -> None:
    if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
        await value


async def _check_cancel(cancel_check: CancelCheck | None) -> bool:
    if cancel_check is None:
        return False
    result = cancel_check()
    if asyncio.iscoroutine(result):
        return bool(await result)
    return bool(result)


async def _pump_progress(
    stream: asyncio.StreamReader,
    total_duration: float | None,
    on_progress: ProgressHook | None,
) -> None:
    """Read ``-progress`` output, emitting a percentage per completed block."""
    block: dict[str, str] = {}
    while True:
        raw_line = await stream.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", "replace").strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        block[key] = value
        if key != "progress":
            continue
        update = build_progress_update(block)
        block = {}
        if on_progress is None:
            continue
        percent = 0.0
        if total_duration and total_duration > 0 and update.out_time_seconds is not None:
            percent = min(99.0, max(0.0, update.out_time_seconds / total_duration * 100.0))
        await _maybe_await(on_progress(percent, update))


async def _drain(stream: asyncio.StreamReader, sink: list[str]) -> None:
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        sink.append(chunk.decode("utf-8", "replace"))


async def run_ffmpeg(
    args: Sequence[str],
    *,
    total_duration: float | None = None,
    on_progress: ProgressHook | None = None,
    cancel_check: CancelCheck | None = None,
    timeout: float | None = None,
    loglevel: str = "error",
    settings: Settings | None = None,
) -> RunResult:
    """Run one ffmpeg invocation to completion.

    Args:
        args: Arguments *after* the ffmpeg binary itself. Never shell-quoted,
            never joined into a string.
        total_duration: Expected output duration in seconds, used to turn
            ffmpeg's ``out_time`` into a percentage.
        on_progress: Called with ``(percent, update)`` on each progress block.
        cancel_check: Polled periodically; returning True terminates the process.
        timeout: Wall-clock limit; defaults to the configured max job duration.
        loglevel: ffmpeg log verbosity. Raise it to 'info' when a filter such as
            ``showinfo`` reports its findings through stderr.
        settings: Settings override, mainly for tests.

    Returns:
        A :class:`RunResult` with the resolved command line and stderr.

    Raises:
        FFmpegExecutionError: The process exited non-zero.
        JobTimeoutError: The process outlived ``timeout``.
        JobCancelledError: ``cancel_check`` asked for termination.
    """
    settings = settings or get_settings()
    binaries = get_binaries(settings)
    argv = [
        str(binaries.ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        loglevel,
        "-progress",
        "pipe:1",
        "-stats_period",
        "0.5",
        *[str(a) for a in args],
    ]
    command = format_command(argv)
    log.info("ffmpeg exec: %s", command)

    loop = asyncio.get_running_loop()
    started = loop.time()
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=ffmpeg_env(),
    )
    assert process.stdout is not None and process.stderr is not None

    stderr_parts: list[str] = []
    progress_task = asyncio.create_task(_pump_progress(process.stdout, total_duration, on_progress))
    stderr_task = asyncio.create_task(_drain(process.stderr, stderr_parts))

    limit = timeout if timeout is not None else float(settings.max_job_seconds)
    cancelled = False
    timed_out = False

    async def _supervise() -> None:
        nonlocal cancelled, timed_out
        while process.returncode is None:
            await asyncio.sleep(_PROGRESS_POLL_SECONDS)
            if process.returncode is not None:
                return
            if await _check_cancel(cancel_check):
                cancelled = True
                await _stop(process)
                return
            if loop.time() - started > limit:
                timed_out = True
                await _stop(process)
                return

    supervisor = asyncio.create_task(_supervise())
    try:
        await process.wait()
    finally:
        supervisor.cancel()
        for task in (progress_task, stderr_task):
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        await asyncio.gather(supervisor, return_exceptions=True)

    stderr = "".join(stderr_parts)
    elapsed = loop.time() - started

    if cancelled:
        raise JobCancelledError("Job cancelled.", command=command)
    if timed_out:
        raise JobTimeoutError(
            "ffmpeg exceeded the maximum allowed job duration.",
            command=command,
            timeout_seconds=limit,
        )
    if process.returncode != 0:
        raise FFmpegExecutionError(
            _summarise_stderr(stderr) or "ffmpeg exited with a non-zero status.",
            exit_code=process.returncode or -1,
            stderr_tail=stderr[-STDERR_TAIL_CHARS:],
            command=command,
        )
    return RunResult(command=command, stderr=stderr, duration_seconds=elapsed)


def _terminate(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        process.terminate()


async def _stop(process: asyncio.subprocess.Process, grace_seconds: float = 5.0) -> None:
    """SIGTERM, then SIGKILL if ffmpeg does not wind down within the grace period."""
    _terminate(process)
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=grace_seconds)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()


def _summarise_stderr(stderr: str) -> str | None:
    """Pick the most useful line out of ffmpeg's stderr for the error message."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return None
    for line in reversed(lines):
        lowered = line.lower()
        if "error" in lowered or "invalid" in lowered or "no such file" in lowered:
            return line
    return lines[-1]


async def run_ffprobe(
    args: Sequence[str], *, timeout: float = 120.0, settings: Settings | None = None
) -> dict[str, object]:
    """Run ffprobe with JSON output and return the parsed document."""
    settings = settings or get_settings()
    binaries = get_binaries(settings)
    argv = [
        str(binaries.ffprobe),
        "-hide_banner",
        "-loglevel",
        "error",
        "-print_format",
        "json",
        *[str(a) for a in args],
    ]
    log.debug("ffprobe exec: %s", format_command(argv))
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=ffmpeg_env(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError as exc:
        _terminate(process)
        raise JobTimeoutError("ffprobe timed out.", command=format_command(argv)) from exc

    if process.returncode != 0:
        raise ProbeError(
            _summarise_stderr(stderr.decode("utf-8", "replace")) or "ffprobe failed.",
            command=format_command(argv),
            exit_code=process.returncode,
        )
    try:
        parsed = json.loads(stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError("ffprobe returned output that is not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ProbeError("ffprobe returned an unexpected JSON shape.")
    return parsed


async def run_ffmpeg_simple(
    args: Sequence[str], *, timeout: float = 60.0, settings: Settings | None = None
) -> str:
    """Run a short informational ffmpeg command (``-encoders``, ``-filters``) and return stdout."""
    settings = settings or get_settings()
    binaries = get_binaries(settings)
    argv = [str(binaries.ffmpeg), "-hide_banner", *[str(a) for a in args]]
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=ffmpeg_env(),
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    if process.returncode != 0:
        raise FFmpegExecutionError(
            "ffmpeg informational command failed.",
            exit_code=process.returncode or -1,
            stderr_tail=stderr.decode("utf-8", "replace")[-STDERR_TAIL_CHARS:],
            command=format_command(argv),
        )
    return stdout.decode("utf-8", "replace")


async def stream_raw_frames(
    path: Path | str,
    *,
    fps: float,
    width: int,
    height: int,
    max_frames: int | None = None,
    settings: Settings | None = None,
) -> AsyncIterator[tuple[int, bytes]]:
    """Decode a file to RGB24 frames at a fixed rate, yielding them one at a time.

    Used by the vision tools. Frames are resampled and scaled by ffmpeg rather
    than in Python, and streamed through a pipe so a long video never has to be
    held in memory or written out as thousands of image files.

    Yields:
        ``(index, frame_bytes)`` where the frame is ``width * height * 3`` bytes
        and its timestamp is ``index / fps``.
    """
    settings = settings or get_settings()
    binaries = get_binaries(settings)
    frame_bytes = width * height * 3
    argv = [
        str(binaries.ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={fps},scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    log.debug("ffmpeg frame stream: %s", format_command(argv))
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=ffmpeg_env(),
    )
    assert process.stdout is not None and process.stderr is not None
    stderr_parts: list[str] = []
    stderr_task = asyncio.create_task(_drain(process.stderr, stderr_parts))
    index = 0
    try:
        while True:
            if max_frames is not None and index >= max_frames:
                break
            try:
                chunk = await process.stdout.readexactly(frame_bytes)
            except asyncio.IncompleteReadError:
                break
            yield index, chunk
            index += 1
    finally:
        await _stop(process, grace_seconds=2.0)
        # Drain whatever ffmpeg had already buffered, then reap it: leaving the
        # pipe transport unclosed surfaces later as 'Event loop is closed'.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.stdout.read(), timeout=2)
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(stderr_task, timeout=5)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=5)
        if process.returncode not in (0, None) and index == 0:
            raise FFmpegExecutionError(
                _summarise_stderr("".join(stderr_parts)) or "Frame extraction failed.",
                exit_code=process.returncode or -1,
                stderr_tail="".join(stderr_parts)[-STDERR_TAIL_CHARS:],
                command=format_command(argv),
            )
