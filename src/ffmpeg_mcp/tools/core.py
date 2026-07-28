"""Phase 1 tools: probing, trimming, concatenation, conversion, transforms, speed.

``probe_media`` and ``list_capabilities`` answer synchronously rather than
returning a job id. They shell out to ffprobe/ffmpeg, but only for metadata, and
finish in well under a second; forcing three round trips to read a file's
duration would make every other tool harder to use. Everything that encodes
frames goes through the job queue.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..binaries import get_binaries
from ..errors import InvalidParameterError
from ..ffmpeg.encoding import defaults_for_container, output_args, stream_copy_args
from ..ffmpeg.filters import (
    Filter,
    FilterChain,
    FilterGraph,
    asetpts_reset,
    atempo_chain,
    crop_filter,
    flip_filters,
    fps_filter,
    pad_to_filter,
    rotate_filter,
    scale_filter,
    setpts_filter,
)
from ..ffmpeg.probe import probe
from ..ffmpeg.runner import run_ffmpeg, run_ffmpeg_simple
from ..jobs.worker import JobContext, JobOutcome, handler
from ..models import EncodeOptions, JobSubmission, MediaInfo, StrictModel
from ..paths import validate_input_file
from .common import JobResult, MediaJobArgs, finish_media_job, queue, resolve_io
from .registry import tool

# Codecs that survive a stream copy into an MP4/MOV/MKV container.
_COPY_SAFE_VIDEO = {"h264", "hevc", "av1", "vp9", "mpeg4", "prores"}
_COPY_SAFE_AUDIO = {"aac", "mp3", "opus", "flac", "alac", "ac3", "vorbis"}


# --------------------------------------------------------------------------- #
# probe_media
# --------------------------------------------------------------------------- #


class ProbeArgs(StrictModel):
    """Arguments for probing a media file."""

    input_path: str = Field(description="Path to the media file to inspect.")


@tool("probe_media", title="Probe media", phase=1, read_only=True)
async def probe_media(args: ProbeArgs) -> MediaInfo:
    """Read a media file's technical metadata.

    Returns container format, duration in seconds, per-stream codec, resolution,
    frame rate, bit rate, audio channels and sample rate, and subtitle streams.

    This answers immediately rather than returning a job id. Call it before any
    edit that depends on the source dimensions or duration, and call it again on
    an output to verify a render.
    """
    source = validate_input_file(args.input_path)
    return await probe(source)


# --------------------------------------------------------------------------- #
# list_capabilities
# --------------------------------------------------------------------------- #


class CapabilitiesArgs(StrictModel):
    """Arguments for the capability query."""

    check_encoders: list[str] = Field(
        default_factory=list,
        description="Encoder names to test for, e.g. ['h264_videotoolbox', 'libx265'].",
    )
    check_filters: list[str] = Field(
        default_factory=list, description="Filter names to test for, e.g. ['xfade', 'libvmaf']."
    )


class Capabilities(StrictModel):
    """What the resolved ffmpeg build can do."""

    ffmpeg_path: str
    ffprobe_path: str
    version: str
    source: str = Field(
        description="How the binary was found: configured, cached, system, or downloaded."
    )
    encoders_available: dict[str, bool] = Field(default_factory=dict)
    filters_available: dict[str, bool] = Field(default_factory=dict)
    hardware_encoders: list[str] = Field(default_factory=list)
    optional_features: dict[str, bool] = Field(
        default_factory=dict,
        description="Availability of this server's Python extras: whisper, vision.",
    )


_HARDWARE_ENCODER_NAMES = [
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "h264_nvenc",
    "hevc_nvenc",
    "av1_nvenc",
    "h264_qsv",
    "hevc_qsv",
    "h264_vaapi",
    "hevc_vaapi",
    "h264_amf",
]


def _parse_listing(text: str) -> set[str]:
    """Pull names out of ``ffmpeg -encoders``/``-filters`` tabular output."""
    names: set[str] = set()
    started = False
    for line in text.splitlines():
        if not started:
            if set(line.strip()) <= {"-"} and line.strip():
                started = True
            continue
        parts = line.split()
        if len(parts) >= 2:
            names.add(parts[1])
    return names


def _optional_features() -> dict[str, bool]:
    import importlib.util

    return {
        "whisper": importlib.util.find_spec("faster_whisper") is not None,
        "vision": importlib.util.find_spec("mediapipe") is not None
        and importlib.util.find_spec("cv2") is not None,
    }


@tool("list_capabilities", title="List ffmpeg capabilities", phase=1, read_only=True)
async def list_capabilities(args: CapabilitiesArgs) -> Capabilities:
    """Report the resolved ffmpeg build and which encoders and filters it has.

    Use this before assuming something is available — hardware encoding
    (videotoolbox, nvenc, qsv), newer filters like xfade, or this server's
    optional Whisper and vision extras. Answers immediately.
    """
    binaries = get_binaries()
    encoders_text = await run_ffmpeg_simple(["-encoders"])
    filters_text = await run_ffmpeg_simple(["-filters"])
    encoders = _parse_listing(encoders_text)
    filters = _parse_listing(filters_text)
    return Capabilities(
        ffmpeg_path=str(binaries.ffmpeg),
        ffprobe_path=str(binaries.ffprobe),
        version=binaries.version,
        source=binaries.source,
        encoders_available={name: name in encoders for name in args.check_encoders},
        filters_available={name: name in filters for name in args.check_filters},
        hardware_encoders=[name for name in _HARDWARE_ENCODER_NAMES if name in encoders],
        optional_features=_optional_features(),
    )


# --------------------------------------------------------------------------- #
# trim
# --------------------------------------------------------------------------- #


class TrimArgs(MediaJobArgs):
    """Arguments for cutting a segment out of a file."""

    start: float = Field(default=0.0, ge=0, description="In point, seconds from file start.")
    end: float | None = Field(default=None, description="Out point in seconds; null means EOF.")
    mode: Literal["auto", "copy", "reencode"] = Field(
        default="auto",
        description=(
            "'copy' remuxes without re-encoding: near-instant, but cuts snap to the "
            "nearest keyframe. 'reencode' is frame-accurate but slower. 'auto' uses "
            "copy when the codecs allow it."
        ),
    )
    encode: EncodeOptions = Field(default_factory=EncodeOptions)

    @model_validator(mode="after")
    def _check_range(self) -> TrimArgs:
        if self.end is not None and self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


@tool("trim", title="Trim a clip", phase=1)
async def trim(args: TrimArgs) -> JobSubmission:
    """Cut a segment out of a video or audio file.

    Stream-copies when the source codecs allow it, which is near-instant but
    snaps the cut to the nearest keyframe; pass mode='reencode' when the in and
    out points must be frame-accurate. The finished job reports the output's
    actual duration so you can confirm what you got.
    """
    resolve_io(args, operation="trim")
    return queue("trim", args)


def _can_stream_copy(info: MediaInfo, destination: Path) -> bool:
    """Whether the source streams can be remuxed into the destination as-is."""
    if destination.suffix.lower() != Path(info.path).suffix.lower():
        return False
    video = info.primary_video
    audio = info.primary_audio
    if video and (video.codec or "") not in _COPY_SAFE_VIDEO:
        return False
    if audio and (audio.codec or "") not in _COPY_SAFE_AUDIO:
        return False
    return bool(video or audio)


@handler("trim")
async def _run_trim(ctx: JobContext) -> JobOutcome:
    args = TrimArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="trim", job_id=ctx.job_id)
    info = await probe(source, ctx.settings)
    await ctx.report(1.0, "Probed source")

    end = args.end if args.end is not None else info.duration
    if end is not None and info.duration is not None and args.start >= info.duration:
        raise InvalidParameterError(
            "Trim start is at or past the end of the file.",
            start=args.start,
            duration=info.duration,
        )
    duration = None if end is None else max(0.0, end - args.start)

    use_copy = args.mode == "copy" or (args.mode == "auto" and _can_stream_copy(info, destination))
    argv: list[str] = ["-ss", f"{args.start:.6f}", "-i", str(source)]
    if duration is not None:
        argv += ["-t", f"{duration:.6f}"]
    if use_copy:
        argv += stream_copy_args(has_video=info.has_video, has_audio=info.has_audio)
        argv += ["-avoid_negative_ts", "make_zero"]
    else:
        argv += output_args(
            args.encode, destination, has_video=info.has_video, has_audio=info.has_audio
        )
    argv += ["-y", str(destination)]

    result = await run_ffmpeg(
        argv,
        total_duration=duration,
        on_progress=ctx.make_progress_hook(2.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    notes = []
    if use_copy:
        notes.append(
            "Stream-copied: the cut may have snapped to the nearest keyframe. "
            "Compare the reported duration against the requested range."
        )
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


# --------------------------------------------------------------------------- #
# concat
# --------------------------------------------------------------------------- #


class ConcatArgs(StrictModel):
    """Arguments for joining clips end to end."""

    input_paths: list[str] = Field(min_length=2, description="Clips to join, in order.")
    output_path: str | None = Field(
        default=None, description="Destination file; defaults to the job workspace."
    )
    target_resolution: str | None = Field(
        default=None,
        description=(
            "Force all clips to this size, e.g. '1920x1080'. Defaults to the first "
            "clip's resolution when the clips differ."
        ),
    )
    target_fps: float | None = Field(default=None, gt=0, description="Force a common frame rate.")
    encode: EncodeOptions = Field(default_factory=EncodeOptions)


@tool("concat", title="Concatenate clips", phase=1)
async def concat(args: ConcatArgs) -> JobSubmission:
    """Join two or more clips end to end into a single file.

    When every clip already shares a codec, resolution and frame rate, this
    remuxes with no re-encoding. Otherwise the clips are normalised first —
    scaled and padded to a common resolution, resampled to a common frame rate
    and audio layout — and then concatenated, which requires a re-encode.
    """
    for path in args.input_paths:
        validate_input_file(path)
    if args.target_resolution:
        _parse_resolution(args.target_resolution)
    return queue("concat", args)


def _parse_resolution(value: str) -> tuple[int, int]:
    """Parse a ``WIDTHxHEIGHT`` string into a pair of ints."""
    separator = "x" if "x" in value.lower() else ":"
    parts = value.lower().split(separator)
    if len(parts) != 2:
        raise InvalidParameterError("Resolution must look like '1920x1080'.", resolution=value)
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise InvalidParameterError(
            "Resolution must look like '1920x1080'.", resolution=value
        ) from exc
    if width <= 0 or height <= 0 or width > 16384 or height > 16384:
        raise InvalidParameterError("Resolution out of range.", resolution=value)
    return width, height


def _streams_match(infos: list[MediaInfo]) -> bool:
    """Whether every clip shares the properties the concat demuxer requires."""
    first = infos[0]
    for info in infos[1:]:
        fv, iv = first.primary_video, info.primary_video
        if bool(fv) != bool(iv):
            return False
        if fv and iv and (fv.codec, fv.width, fv.height) != (iv.codec, iv.width, iv.height):
            return False
        fa, ia = first.primary_audio, info.primary_audio
        if bool(fa) != bool(ia):
            return False
        if (
            fa
            and ia
            and (fa.codec, fa.sample_rate, fa.channels)
            != (
                ia.codec,
                ia.sample_rate,
                ia.channels,
            )
        ):
            return False
    return True


def build_concat_graph(
    infos: list[MediaInfo], width: int, height: int, fps: float, sample_rate: int
) -> tuple[str, bool]:
    """Build the filter_complex that normalises then concatenates several clips.

    Returns the graph string and whether the result carries audio. Pure, so the
    graph shape is unit tested without running ffmpeg.
    """
    graph = FilterGraph()
    with_audio = all(info.has_audio for info in infos)
    labels: list[str] = []
    for index in range(len(infos)):
        video_label = f"v{index}"
        chain = graph.chain(inputs=[f"{index}:v:0"], outputs=[video_label])
        chain.add(
            scale_filter(width, height, keep_aspect=True),
            pad_to_filter(width, height),
            Filter("setsar", {}, raw_options="1"),
            fps_filter(fps),
            Filter("format", {}, raw_options="yuv420p"),
        )
        labels.append(video_label)
        if with_audio:
            audio_label = f"a{index}"
            achain = graph.chain(inputs=[f"{index}:a:0"], outputs=[audio_label])
            achain.add(
                Filter(
                    "aformat",
                    {},
                    raw_options=f"sample_fmts=fltp:sample_rates={sample_rate}:channel_layouts=stereo",
                ),
                asetpts_reset(),
            )

    concat_inputs: list[str] = []
    for index in range(len(infos)):
        concat_inputs.append(f"v{index}")
        if with_audio:
            concat_inputs.append(f"a{index}")
    outputs = ["vout", "aout"] if with_audio else ["vout"]
    final = graph.chain(inputs=concat_inputs, outputs=outputs)
    final.add(Filter("concat", {"n": len(infos), "v": 1, "a": 1 if with_audio else 0}))
    return graph.render(), with_audio


@handler("concat")
async def _run_concat(ctx: JobContext) -> JobOutcome:
    args = ConcatArgs.model_validate(ctx.params)
    sources = [validate_input_file(p, ctx.settings) for p in args.input_paths]
    infos = [await probe(path, ctx.settings) for path in sources]
    await ctx.report(3.0, f"Probed {len(sources)} clips")

    from ..paths import validate_output_path

    suffix = sources[0].suffix or ".mp4"
    destination = validate_output_path(
        args.output_path,
        suggested_name=f"{sources[0].stem}_concat{suffix}",
        job_id=ctx.job_id,
        settings=ctx.settings,
    )
    total = sum(info.duration or 0.0 for info in infos) or None
    notes: list[str] = []

    identical = _streams_match(infos) and not args.target_resolution and not args.target_fps
    if identical:
        listing = ctx.workdir / "concat_list.txt"
        listing.write_text(
            "\n".join(f"file '{str(p).replace(chr(39), chr(39) * 3)}'" for p in sources) + "\n",
            encoding="utf-8",
        )
        argv = [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            *stream_copy_args(has_video=infos[0].has_video, has_audio=infos[0].has_audio),
            "-y",
            str(destination),
        ]
        notes.append("Clips matched; concatenated without re-encoding.")
    else:
        first_video = infos[0].primary_video
        if args.target_resolution:
            width, height = _parse_resolution(args.target_resolution)
        elif first_video and first_video.width and first_video.height:
            width, height = first_video.width, first_video.height
        else:
            raise InvalidParameterError(
                "Cannot determine a target resolution; pass target_resolution."
            )
        fps = args.target_fps or (first_video.fps if first_video and first_video.fps else 30.0)
        sample_rate = next(
            (
                i.primary_audio.sample_rate
                for i in infos
                if i.primary_audio and i.primary_audio.sample_rate
            ),
            48000,
        )
        graph, with_audio = build_concat_graph(infos, width, height, fps, sample_rate)
        argv = []
        for path in sources:
            argv += ["-i", str(path)]
        argv += ["-filter_complex", graph, "-map", "[vout]"]
        if with_audio:
            argv += ["-map", "[aout]"]
        argv += output_args(args.encode, destination, has_video=True, has_audio=with_audio)
        argv += ["-y", str(destination)]
        notes.append(f"Clips differed; normalised to {width}x{height} @ {fps}fps and re-encoded.")
        if not with_audio:
            notes.append("At least one clip had no audio, so the result is video-only.")

    result = await run_ffmpeg(
        argv,
        total_duration=total,
        on_progress=ctx.make_progress_hook(4.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


# --------------------------------------------------------------------------- #
# convert_format
# --------------------------------------------------------------------------- #


class ConvertArgs(MediaJobArgs):
    """Arguments for transcoding between containers and codecs."""

    container: str | None = Field(
        default=None,
        description=(
            "Target container extension, e.g. 'mp4', 'webm', 'mp3'. "
            "Inferred from output_path when given."
        ),
    )
    encode: EncodeOptions | None = Field(
        default=None,
        description="Encoder settings. Defaults are chosen to suit the target container.",
    )
    audio_only: bool = Field(default=False, description="Drop video and keep only audio.")


@tool("convert_format", title="Convert format", phase=1)
async def convert_format(args: ConvertArgs) -> JobSubmission:
    """Transcode a file to a different container or codec.

    Sensible codec defaults are chosen per container (H.264/AAC for mp4 and mov,
    VP9/Opus for webm, MP3 or AAC for audio-only targets); override them through
    'encode'. Set audio_only to extract the audio track.
    """
    validate_input_file(args.input_path)
    return queue("convert_format", args)


@handler("convert_format")
async def _run_convert(ctx: JobContext) -> JobOutcome:
    args = ConvertArgs.model_validate(ctx.params)
    source = validate_input_file(args.input_path, ctx.settings)
    info = await probe(source, ctx.settings)

    suffix = None
    if args.output_path:
        suffix = Path(args.output_path).suffix or None
    if suffix is None and args.container:
        suffix = args.container if args.container.startswith(".") else f".{args.container}"
    if suffix is None:
        suffix = ".m4a" if args.audio_only else ".mp4"

    from ..paths import validate_output_path

    destination = validate_output_path(
        args.output_path,
        suggested_name=f"{source.stem}_converted{suffix}",
        job_id=ctx.job_id,
        settings=ctx.settings,
    )
    video_default, audio_default = defaults_for_container(destination)
    encode = args.encode or EncodeOptions(
        video_codec=video_default if video_default != "none" else "libx264",  # type: ignore[arg-type]
        audio_codec=audio_default if audio_default != "none" else "aac",  # type: ignore[arg-type]
    )
    keep_video = info.has_video and not args.audio_only and video_default != "none"
    keep_audio = info.has_audio and audio_default != "none"
    if not keep_video and not keep_audio:
        raise InvalidParameterError(
            "The requested container cannot carry any of the source's streams.",
            container=destination.suffix,
        )

    argv = ["-i", str(source)]
    if not keep_video:
        argv += ["-vn"]
    if not keep_audio:
        argv += ["-an"]
    argv += output_args(encode, destination, has_video=keep_video, has_audio=keep_audio)
    argv += ["-y", str(destination)]

    result = await run_ffmpeg(
        argv,
        total_duration=info.duration,
        on_progress=ctx.make_progress_hook(1.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    return await finish_media_job(ctx, destination, command=result.command)


# --------------------------------------------------------------------------- #
# transform (crop / scale / rotate / flip)
# --------------------------------------------------------------------------- #


class TransformArgs(MediaJobArgs):
    """Arguments for geometric transforms, applied in one filter chain."""

    crop: str | None = Field(
        default=None,
        description="Crop rectangle as 'x,y,width,height' in pixels, applied first.",
    )
    scale_width: int | None = Field(default=None, gt=0, le=16384)
    scale_height: int | None = Field(default=None, gt=0, le=16384)
    pad_to_fit: bool = Field(
        default=False,
        description="When both scale dimensions are given, letterbox instead of stretching.",
    )
    rotate: int = Field(default=0, description="Clockwise rotation; must be 0, 90, 180 or 270.")
    flip_horizontal: bool = False
    flip_vertical: bool = False
    encode: EncodeOptions = Field(default_factory=EncodeOptions)

    @model_validator(mode="after")
    def _check_any(self) -> TransformArgs:
        if not any(
            [
                self.crop,
                self.scale_width,
                self.scale_height,
                self.rotate,
                self.flip_horizontal,
                self.flip_vertical,
            ]
        ):
            raise ValueError("transform needs at least one of crop, scale, rotate or flip")
        return self


def parse_crop(value: str) -> tuple[int, int, int, int]:
    """Parse an ``x,y,width,height`` crop string."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise InvalidParameterError("Crop must be 'x,y,width,height'.", crop=value)
    try:
        x, y, width, height = (int(p) for p in parts)
    except ValueError as exc:
        raise InvalidParameterError("Crop values must be integers.", crop=value) from exc
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        raise InvalidParameterError("Crop values out of range.", crop=value)
    return x, y, width, height


def build_transform_chain(args: TransformArgs) -> list[Filter]:
    """Build the crop/scale/pad/rotate/flip chain in a deterministic order.

    Order matters: cropping before scaling keeps the requested pixel rectangle
    meaningful, and rotating last keeps the scale dimensions in source
    orientation.
    """
    filters: list[Filter] = []
    if args.crop:
        filters.append(crop_filter(*parse_crop(args.crop)))
    if args.scale_width or args.scale_height:
        both = args.scale_width is not None and args.scale_height is not None
        filters.append(
            scale_filter(args.scale_width, args.scale_height, keep_aspect=both and args.pad_to_fit)
        )
        if both and args.pad_to_fit:
            assert args.scale_width is not None and args.scale_height is not None
            filters.append(pad_to_filter(args.scale_width, args.scale_height))
    filters.extend(rotate_filter(args.rotate))
    filters.extend(flip_filters(args.flip_horizontal, args.flip_vertical))
    return filters


@tool("transform", title="Crop, scale, rotate", phase=1)
async def transform(args: TransformArgs) -> JobSubmission:
    """Apply geometric transforms — crop, scale, rotate, flip — in one pass.

    Operations compose into a single filter chain in the order crop, scale, pad,
    rotate, flip, so the whole change costs one re-encode. Give only one of
    scale_width/scale_height to preserve the aspect ratio; give both with
    pad_to_fit to letterbox rather than stretch.
    """
    resolve_io(args, operation="transform")
    build_transform_chain(args)  # validate crop/rotate values before queueing
    return queue("transform", args)


@handler("transform")
async def _run_transform(ctx: JobContext) -> JobOutcome:
    args = TransformArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="transform", job_id=ctx.job_id)
    info = await probe(source, ctx.settings)
    if not info.has_video:
        raise InvalidParameterError("transform needs a video stream.", path=str(source))

    chain = FilterChain(filters=build_transform_chain(args))
    argv = ["-i", str(source), "-vf", chain.render()]
    argv += output_args(args.encode, destination, has_video=True, has_audio=info.has_audio)
    argv += ["-y", str(destination)]

    result = await run_ffmpeg(
        argv,
        total_duration=info.duration,
        on_progress=ctx.make_progress_hook(1.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    return await finish_media_job(ctx, destination, command=result.command)


# --------------------------------------------------------------------------- #
# speed_ramp
# --------------------------------------------------------------------------- #


class SpeedArgs(MediaJobArgs):
    """Arguments for changing playback speed."""

    speed: float = Field(
        gt=0.01, le=100.0, description="Playback multiplier: 2.0 is twice as fast, 0.5 is half."
    )
    audio_speed: float | None = Field(
        default=None,
        gt=0.01,
        le=100.0,
        description="Independent audio multiplier. Defaults to matching 'speed'.",
    )
    keep_pitch: bool = Field(
        default=True,
        description=(
            "Preserve pitch by time-stretching the audio. When false the audio is "
            "resampled instead, so it changes pitch like a tape speed-up."
        ),
    )
    drop_audio: bool = Field(default=False, description="Discard the audio track entirely.")
    encode: EncodeOptions = Field(default_factory=EncodeOptions)


def build_speed_filters(
    speed: float, audio_speed: float, keep_pitch: bool, sample_rate: int
) -> tuple[list[Filter], list[Filter]]:
    """Build the video and audio filter lists for a speed change.

    With ``keep_pitch`` the audio goes through a chain of ``atempo`` steps, each
    kept inside the 0.5x-2.0x range older builds accept. Without it, the sample
    rate is reinterpreted and then resampled back, which shifts pitch.
    """
    video = [setpts_filter(speed)]
    if math.isclose(audio_speed, 1.0, rel_tol=1e-9):
        return video, []
    if keep_pitch:
        return video, atempo_chain(audio_speed)
    return video, [
        Filter("asetrate", {}, raw_options=f"{round(sample_rate * audio_speed)}"),
        Filter("aresample", {}, raw_options=str(sample_rate)),
        asetpts_reset(),
    ]


@tool("speed_ramp", title="Change playback speed", phase=1)
async def speed_ramp(args: SpeedArgs) -> JobSubmission:
    """Speed a clip up or slow it down, video and audio independently.

    By default the audio is time-stretched so pitch is preserved, chained through
    as many atempo stages as the factor needs. Set keep_pitch false for a tape
    speed-up sound, or drop_audio to discard the track. Set audio_speed to hold
    audio at a different rate from the video.
    """
    resolve_io(args, operation="speed")
    return queue("speed_ramp", args)


@handler("speed_ramp")
async def _run_speed(ctx: JobContext) -> JobOutcome:
    args = SpeedArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="speed", job_id=ctx.job_id)
    info = await probe(source, ctx.settings)
    audio_speed = args.audio_speed if args.audio_speed is not None else args.speed
    sample_rate = (info.primary_audio.sample_rate if info.primary_audio else None) or 48000
    keep_audio = info.has_audio and not args.drop_audio

    video_filters, audio_filters = build_speed_filters(
        args.speed, audio_speed, args.keep_pitch, sample_rate
    )
    argv = ["-i", str(source)]
    graph = FilterGraph()
    if info.has_video:
        graph.chain(inputs=["0:v:0"], outputs=["vout"]).extend(video_filters)
    if keep_audio:
        achain = graph.chain(inputs=["0:a:0"], outputs=["aout"])
        achain.extend(audio_filters or [Filter("anull")])
    argv += ["-filter_complex", graph.render()]
    if info.has_video:
        argv += ["-map", "[vout]"]
    if keep_audio:
        argv += ["-map", "[aout]"]
    else:
        argv += ["-an"]
    argv += output_args(args.encode, destination, has_video=info.has_video, has_audio=keep_audio)
    argv += ["-y", str(destination)]

    expected = (info.duration / args.speed) if info.duration else None
    result = await run_ffmpeg(
        argv,
        total_duration=expected,
        on_progress=ctx.make_progress_hook(1.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    notes = []
    if not math.isclose(audio_speed, args.speed, rel_tol=1e-9) and keep_audio:
        notes.append("Audio and video run at different speeds, so they will drift out of sync.")
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


__all__ = [
    "Capabilities",
    "ConcatArgs",
    "ConvertArgs",
    "JobResult",
    "ProbeArgs",
    "SpeedArgs",
    "TransformArgs",
    "TrimArgs",
    "build_concat_graph",
    "build_speed_filters",
    "build_transform_chain",
    "parse_crop",
]
