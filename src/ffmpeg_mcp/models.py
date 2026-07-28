"""Pydantic models shared across tools.

Anything that crosses the MCP tool boundary is defined here or in a tool module
and reused; no ad hoc dicts.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model: reject unknown fields so typos surface as errors, not silence."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #


class VideoStreamInfo(StrictModel):
    """One video stream as reported by ffprobe."""

    index: int
    codec: str | None = None
    profile: str | None = None
    pix_fmt: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = Field(default=None, description="Average frame rate in frames per second.")
    bit_rate: int | None = None
    duration: float | None = None
    rotation: int | None = Field(
        default=None, description="Display rotation in degrees from stream side data."
    )


class AudioStreamInfo(StrictModel):
    """One audio stream as reported by ffprobe."""

    index: int
    codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None
    bit_rate: int | None = None
    duration: float | None = None
    language: str | None = None


class SubtitleStreamInfo(StrictModel):
    """One subtitle stream as reported by ffprobe."""

    index: int
    codec: str | None = None
    language: str | None = None


class MediaInfo(StrictModel):
    """Everything the probe tool reports about one media file."""

    path: str
    format_name: str | None = None
    duration: float | None = Field(default=None, description="Container duration in seconds.")
    size_bytes: int | None = None
    bit_rate: int | None = None
    video_streams: list[VideoStreamInfo] = Field(default_factory=list)
    audio_streams: list[AudioStreamInfo] = Field(default_factory=list)
    subtitle_streams: list[SubtitleStreamInfo] = Field(default_factory=list)

    @property
    def primary_video(self) -> VideoStreamInfo | None:
        return self.video_streams[0] if self.video_streams else None

    @property
    def primary_audio(self) -> AudioStreamInfo | None:
        return self.audio_streams[0] if self.audio_streams else None

    @property
    def has_video(self) -> bool:
        return bool(self.video_streams)

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_streams)


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #


class JobStatus(StrEnum):
    """Lifecycle states a job moves through."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}


class JobError(StrictModel):
    """Structured failure detail attached to a failed job."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class JobRecord(StrictModel):
    """A unit of work in the shared job store."""

    job_id: str
    tool: str
    status: JobStatus
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    error: JobError | None = None
    command: str | None = Field(
        default=None, description="Resolved ffmpeg command line, for auditability."
    )
    message: str | None = Field(default=None, description="Human-readable current step.")
    cancel_requested: bool = False
    worker_pid: int | None = None


class JobSubmission(StrictModel):
    """What every work-performing tool returns immediately."""

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    tool: str
    message: str = "Job queued. Poll job_status for progress, then job_result."


# --------------------------------------------------------------------------- #
# Common tool argument fragments
# --------------------------------------------------------------------------- #

TimeSeconds = Annotated[float, Field(ge=0, description="Time in seconds from the start of media.")]


class TimeRange(StrictModel):
    """An in/out point pair, in seconds."""

    start: TimeSeconds = 0.0
    end: float | None = Field(default=None, description="End time in seconds; null means EOF.")

    @model_validator(mode="after")
    def _check_order(self) -> TimeRange:
        if self.end is not None and self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self

    @property
    def duration(self) -> float | None:
        return None if self.end is None else self.end - self.start


class Resolution(StrictModel):
    """A pixel size."""

    width: int = Field(gt=0, le=16384)
    height: int = Field(gt=0, le=16384)


class Rect(StrictModel):
    """An axis-aligned rectangle in pixels."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class NormalizedRect(StrictModel):
    """A rectangle in 0..1 coordinates, relative to frame size."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


VideoCodec = Literal[
    "libx264",
    "libx265",
    "libvpx-vp9",
    "libaom-av1",
    "prores_ks",
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "h264_nvenc",
    "hevc_nvenc",
    "copy",
]
AudioCodec = Literal["aac", "libmp3lame", "libopus", "flac", "pcm_s16le", "copy"]


class EncodeOptions(StrictModel):
    """Encoder settings shared by every tool that re-encodes."""

    video_codec: VideoCodec = "libx264"
    audio_codec: AudioCodec = "aac"
    crf: int | None = Field(default=20, ge=0, le=63, description="Quality; lower is better.")
    preset: str | None = Field(default="medium", description="x264/x265 speed preset.")
    video_bitrate: str | None = Field(default=None, description="e.g. '5M'. Overrides crf.")
    audio_bitrate: str | None = Field(default="192k")
    pix_fmt: str | None = "yuv420p"
    fps: float | None = None
    extra_args: list[str] = Field(
        default_factory=list, description="Escape hatch: raw ffmpeg output args."
    )


class Segment(StrictModel):
    """A timed piece of text, used for transcripts, captions and SRT generation."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _check_order(self) -> Segment:
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class WordTiming(StrictModel):
    """A single word with its timestamps and confidence."""

    start: float
    end: float
    word: str
    probability: float | None = None
