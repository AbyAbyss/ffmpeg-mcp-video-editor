"""Exception hierarchy for the ffmpeg MCP server.

Every error raised across a tool boundary is one of these, so the server can
return a structured ``{code, message, details}`` payload instead of a traceback.
"""

from __future__ import annotations

from typing import Any


class FFmpegMCPError(Exception):
    """Base class for every error this server raises deliberately."""

    code = "internal_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        """Serialise the error for an MCP tool response."""
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class BinaryNotFoundError(FFmpegMCPError):
    """ffmpeg/ffprobe could not be located, downloaded, or executed."""

    code = "binary_not_found"


class UnsupportedPlatformError(FFmpegMCPError):
    """No static ffmpeg build is published for this OS/architecture."""

    code = "unsupported_platform"


class ChecksumMismatchError(FFmpegMCPError):
    """A downloaded archive did not match its published checksum."""

    code = "checksum_mismatch"


class InvalidPathError(FFmpegMCPError):
    """A path resolved outside the configured allowlist, or does not exist."""

    code = "invalid_path"


class FileTooLargeError(FFmpegMCPError):
    """An input file exceeds the configured maximum size."""

    code = "file_too_large"


class InvalidParameterError(FFmpegMCPError):
    """A tool argument was structurally valid but semantically wrong."""

    code = "invalid_parameter"


class UnsupportedCodecError(FFmpegMCPError):
    """The requested codec/encoder is not available in the resolved ffmpeg build."""

    code = "unsupported_codec"


class ProbeError(FFmpegMCPError):
    """ffprobe failed or returned output that could not be interpreted."""

    code = "probe_failed"


class FFmpegExecutionError(FFmpegMCPError):
    """An ffmpeg invocation exited non-zero."""

    code = "ffmpeg_failed"

    def __init__(self, message: str, *, exit_code: int, stderr_tail: str, command: str) -> None:
        super().__init__(message, exit_code=exit_code, stderr_tail=stderr_tail, command=command)
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail
        self.command = command


class JobTimeoutError(FFmpegMCPError):
    """A job exceeded the configured maximum wall-clock duration."""

    code = "job_timeout"


class JobNotFoundError(FFmpegMCPError):
    """No job exists with the given id."""

    code = "job_not_found"


class JobNotFinishedError(FFmpegMCPError):
    """A result was requested for a job that has not completed."""

    code = "job_not_finished"


class JobCancelledError(FFmpegMCPError):
    """A job was cancelled before it completed."""

    code = "job_cancelled"


class MissingDependencyError(FFmpegMCPError):
    """An optional Python dependency (whisper/vision/ui extra) is not installed."""

    code = "missing_dependency"
