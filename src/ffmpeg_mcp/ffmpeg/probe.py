"""ffprobe wrapper: raw JSON in, typed :class:`MediaInfo` out."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from ..config import Settings
from ..errors import ProbeError
from ..models import AudioStreamInfo, MediaInfo, SubtitleStreamInfo, VideoStreamInfo
from .runner import run_ffprobe


def _opt_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _opt_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_frame_rate(value: Any) -> float | None:
    """Parse an ffprobe rational frame rate such as ``30000/1001``."""
    if not value or not isinstance(value, str):
        return None
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if fraction.denominator == 0 or fraction == 0:
        return None
    return round(float(fraction), 6)


def _rotation(stream: dict[str, Any]) -> int | None:
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        return _opt_int(tags["rotate"])
    for entry in stream.get("side_data_list") or []:
        if "rotation" in entry:
            return _opt_int(entry["rotation"])
    return None


def parse_probe_document(document: dict[str, Any], path: str) -> MediaInfo:
    """Convert an ffprobe ``-show_format -show_streams`` document into a MediaInfo.

    Kept pure and separate from the subprocess call so it can be unit tested
    without ffmpeg present.
    """
    fmt = document.get("format") or {}
    info = MediaInfo(
        path=path,
        format_name=fmt.get("format_name"),
        duration=_opt_float(fmt.get("duration")),
        size_bytes=_opt_int(fmt.get("size")),
        bit_rate=_opt_int(fmt.get("bit_rate")),
    )
    for stream in document.get("streams") or []:
        kind = stream.get("codec_type")
        index = _opt_int(stream.get("index")) or 0
        if kind == "video":
            # Cover art in an audio file shows up as a video stream; ignore it.
            if stream.get("disposition", {}).get("attached_pic"):
                continue
            info.video_streams.append(
                VideoStreamInfo(
                    index=index,
                    codec=stream.get("codec_name"),
                    profile=stream.get("profile"),
                    pix_fmt=stream.get("pix_fmt"),
                    width=_opt_int(stream.get("width")),
                    height=_opt_int(stream.get("height")),
                    fps=parse_frame_rate(stream.get("avg_frame_rate"))
                    or parse_frame_rate(stream.get("r_frame_rate")),
                    bit_rate=_opt_int(stream.get("bit_rate")),
                    duration=_opt_float(stream.get("duration")),
                    rotation=_rotation(stream),
                )
            )
        elif kind == "audio":
            tags = stream.get("tags") or {}
            info.audio_streams.append(
                AudioStreamInfo(
                    index=index,
                    codec=stream.get("codec_name"),
                    sample_rate=_opt_int(stream.get("sample_rate")),
                    channels=_opt_int(stream.get("channels")),
                    channel_layout=stream.get("channel_layout"),
                    bit_rate=_opt_int(stream.get("bit_rate")),
                    duration=_opt_float(stream.get("duration")),
                    language=tags.get("language"),
                )
            )
        elif kind == "subtitle":
            tags = stream.get("tags") or {}
            info.subtitle_streams.append(
                SubtitleStreamInfo(
                    index=index,
                    codec=stream.get("codec_name"),
                    language=tags.get("language"),
                )
            )
    if info.duration is None and info.video_streams:
        info.duration = info.video_streams[0].duration
    if info.duration is None and info.audio_streams:
        info.duration = info.audio_streams[0].duration
    return info


async def probe(path: Path, settings: Settings | None = None) -> MediaInfo:
    """Probe a media file and return its typed description."""
    document = await run_ffprobe(["-show_format", "-show_streams", str(path)], settings=settings)
    info = parse_probe_document(document, str(path))
    if not info.video_streams and not info.audio_streams:
        raise ProbeError("File contains no video or audio streams.", path=str(path))
    return info


async def probe_duration(path: Path, settings: Settings | None = None) -> float | None:
    """Return just the duration in seconds, for progress percentage calculations."""
    info = await probe(path, settings)
    return info.duration
