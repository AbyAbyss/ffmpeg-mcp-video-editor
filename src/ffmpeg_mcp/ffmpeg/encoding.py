"""Translating :class:`EncodeOptions` into ffmpeg output arguments."""

from __future__ import annotations

from pathlib import Path

from ..errors import InvalidParameterError
from ..models import EncodeOptions

# Encoders that take -crf; everything else uses a bitrate.
_CRF_ENCODERS = {"libx264", "libx265", "libvpx-vp9", "libaom-av1"}
_PRESET_ENCODERS = {"libx264", "libx265"}

_CONTAINER_DEFAULTS: dict[str, tuple[str, str]] = {
    ".mp4": ("libx264", "aac"),
    ".mov": ("libx264", "aac"),
    ".mkv": ("libx264", "aac"),
    ".webm": ("libvpx-vp9", "libopus"),
    ".m4v": ("libx264", "aac"),
    ".avi": ("libx264", "libmp3lame"),
    ".gif": ("gif", "none"),
    ".mp3": ("none", "libmp3lame"),
    ".m4a": ("none", "aac"),
    ".wav": ("none", "pcm_s16le"),
    ".flac": ("none", "flac"),
    ".opus": ("none", "libopus"),
}

AUDIO_ONLY_SUFFIXES = {".mp3", ".m4a", ".wav", ".flac", ".opus", ".aac", ".ogg"}


def defaults_for_container(path: Path) -> tuple[str, str]:
    """Return the (video, audio) codec pair that suits a container extension."""
    return _CONTAINER_DEFAULTS.get(path.suffix.lower(), ("libx264", "aac"))


def video_encode_args(options: EncodeOptions) -> list[str]:
    """Build the video half of the output arguments."""
    if options.video_codec == "copy":
        return ["-c:v", "copy"]
    args = ["-c:v", options.video_codec]
    if options.video_bitrate:
        args += ["-b:v", options.video_bitrate]
    elif options.crf is not None and options.video_codec in _CRF_ENCODERS:
        args += ["-crf", str(options.crf)]
        if options.video_codec == "libvpx-vp9":
            # VP9 needs an explicit zero bitrate for constant-quality mode.
            args += ["-b:v", "0"]
    elif options.crf is not None and "videotoolbox" in options.video_codec:
        # VideoToolbox has no CRF; -q:v 1..100 is its quality scale, inverted
        # relative to CRF.
        args += ["-q:v", str(max(1, min(100, int((63 - options.crf) / 63 * 100))))]
    if options.preset and options.video_codec in _PRESET_ENCODERS:
        args += ["-preset", options.preset]
    if options.pix_fmt:
        args += ["-pix_fmt", options.pix_fmt]
    if options.fps:
        args += ["-r", str(options.fps)]
    return args


def audio_encode_args(options: EncodeOptions) -> list[str]:
    """Build the audio half of the output arguments."""
    if options.audio_codec == "copy":
        return ["-c:a", "copy"]
    args = ["-c:a", options.audio_codec]
    if options.audio_bitrate and options.audio_codec not in {"flac", "pcm_s16le"}:
        args += ["-b:a", options.audio_bitrate]
    return args


def output_args(
    options: EncodeOptions,
    output: Path,
    *,
    has_video: bool = True,
    has_audio: bool = True,
) -> list[str]:
    """Build every output argument for one destination file.

    Args:
        options: Requested encoder settings.
        output: Destination path; its extension decides container defaults.
        has_video: Whether a video stream will be mapped.
        has_audio: Whether an audio stream will be mapped.
    """
    if not has_video and not has_audio:
        raise InvalidParameterError("Output must contain at least one stream.")
    args: list[str] = []
    audio_only = output.suffix.lower() in AUDIO_ONLY_SUFFIXES
    if has_video and not audio_only:
        args += video_encode_args(options)
    if has_audio:
        args += audio_encode_args(options)
    if output.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        args += ["-movflags", "+faststart"]
    args += list(options.extra_args)
    return args


def stream_copy_args(*, has_video: bool = True, has_audio: bool = True) -> list[str]:
    """Arguments for a pure remux with no re-encoding."""
    args: list[str] = []
    if has_video:
        args += ["-c:v", "copy"]
    if has_audio:
        args += ["-c:a", "copy"]
    return args
