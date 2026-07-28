"""Phase 5 tools: transitions, compositing, audio mixing, and full timeline rendering."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from ..errors import InvalidParameterError
from ..ffmpeg.encoding import output_args
from ..ffmpeg.filters import (
    Filter,
    FilterChain,
    FilterGraph,
    acrossfade_filter,
    afade_filter,
    chromakey_filter,
    loudnorm_filter,
    pad_to_filter,
    scale_filter,
    sidechain_duck_filter,
    volume_filter,
    xfade_filter,
)
from ..ffmpeg.probe import probe
from ..ffmpeg.runner import run_ffmpeg
from ..jobs.worker import JobContext, JobOutcome, handler
from ..models import EncodeOptions, JobSubmission, MediaInfo, StrictModel
from ..paths import validate_input_file, validate_output_path
from ..timeline import (
    TRANSITIONS,
    Timeline,
    compile_timeline,
    overlay_eof_action,
    overlay_position_expressions,
)
from .common import MediaJobArgs, finish_media_job, queue, resolve_io
from .registry import tool

SAMPLE_RATE = 48000


def _require_video(info: MediaInfo, path: Path, tool_name: str) -> tuple[int, int]:
    video = info.primary_video
    if video is None or not video.width or not video.height:
        raise InvalidParameterError(f"{tool_name} needs a video stream.", path=str(path))
    return video.width, video.height


# --------------------------------------------------------------------------- #
# add_transition
# --------------------------------------------------------------------------- #


class AddTransitionArgs(StrictModel):
    """Arguments for joining two clips with a transition."""

    first_path: str = Field(description="Clip that plays first.")
    second_path: str = Field(description="Clip that plays second.")
    output_path: str | None = None
    transition: str = Field(default="fade", description=f"One of: {', '.join(TRANSITIONS)}.")
    duration: float = Field(
        default=1.0, gt=0, le=30, description="How long the two clips overlap, in seconds."
    )
    crossfade_audio: bool = Field(
        default=True, description="Cross-fade the audio across the same overlap."
    )
    encode: EncodeOptions = Field(default_factory=EncodeOptions)

    @model_validator(mode="after")
    def _check_transition(self) -> AddTransitionArgs:
        if self.transition not in TRANSITIONS:
            raise ValueError(f"unknown transition {self.transition!r}")
        return self


@tool("add_transition", title="Add a transition", phase=5)
async def add_transition(args: AddTransitionArgs) -> JobSubmission:
    """Join two clips with a cross-fade or wipe-style transition.

    The clips overlap by 'duration' seconds, so the result is shorter than the
    two clips added together by exactly that much. The second clip is conformed
    to the first one's resolution and frame rate first, since xfade requires
    both sides to match.

    Available transitions include fade, fadeblack, dissolve, the wipe family
    (wipeleft/right/up/down), the slide family, and circleopen/circleclose.
    """
    validate_input_file(args.first_path)
    validate_input_file(args.second_path)
    return queue("add_transition", args)


@handler("add_transition")
async def _run_add_transition(ctx: JobContext) -> JobOutcome:
    args = AddTransitionArgs.model_validate(ctx.params)
    first = validate_input_file(args.first_path, ctx.settings)
    second = validate_input_file(args.second_path, ctx.settings)
    first_info = await probe(first, ctx.settings)
    second_info = await probe(second, ctx.settings)
    width, height = _require_video(first_info, first, "add_transition")
    _require_video(second_info, second, "add_transition")

    first_duration = first_info.duration or 0.0
    second_duration = second_info.duration or 0.0
    overlap = min(args.duration, first_duration, second_duration)
    if overlap <= 0:
        raise InvalidParameterError(
            "The transition is longer than one of the clips.",
            duration=args.duration,
            first_duration=first_duration,
            second_duration=second_duration,
        )
    offset = max(0.0, first_duration - overlap)

    destination = validate_output_path(
        args.output_path,
        suggested_name=f"{first.stem}_transition{first.suffix or '.mp4'}",
        job_id=ctx.job_id,
        settings=ctx.settings,
    )
    fps = (first_info.primary_video.fps if first_info.primary_video else None) or 30.0
    with_audio = first_info.has_audio and second_info.has_audio

    graph = FilterGraph()
    for index in (0, 1):
        graph.chain([f"{index}:v:0"], [f"v{index}"]).add(
            scale_filter(width, height, keep_aspect=True),
            pad_to_filter(width, height),
            Filter("setsar", {}, raw_options="1"),
            Filter("fps", {"fps": fps}),
            Filter("format", {}, raw_options="yuv420p"),
        )
    graph.chain(["v0", "v1"], ["vout"]).add(xfade_filter(args.transition, overlap, offset))
    if with_audio:
        for index in (0, 1):
            graph.chain([f"{index}:a:0"], [f"a{index}"]).add(
                Filter(
                    "aformat",
                    {},
                    raw_options=(
                        f"sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo"
                    ),
                )
            )
        if args.crossfade_audio:
            graph.chain(["a0", "a1"], ["aout"]).add(acrossfade_filter(overlap))
        else:
            graph.chain(["a0", "a1"], ["aout"]).add(Filter("concat", {"n": 2, "v": 0, "a": 1}))

    argv = [
        "-i",
        str(first),
        "-i",
        str(second),
        "-filter_complex",
        graph.render(),
        "-map",
        "[vout]",
    ]
    if with_audio:
        argv += ["-map", "[aout]"]
    argv += output_args(args.encode, destination, has_video=True, has_audio=with_audio)
    argv += ["-y", str(destination)]

    expected = first_duration + second_duration - overlap
    result = await run_ffmpeg(
        argv,
        total_duration=expected,
        on_progress=ctx.make_progress_hook(2.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    notes = [f"Joined with a {overlap:.2f}s '{args.transition}' transition."]
    if not with_audio:
        notes.append("One clip had no audio, so the result is video-only.")
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


# --------------------------------------------------------------------------- #
# overlay_media
# --------------------------------------------------------------------------- #


class OverlayMediaArgs(MediaJobArgs):
    """Arguments for compositing an image or video on top of another."""

    overlay_path: str = Field(description="Image or video to composite on top.")
    position: str = Field(
        default="top-right",
        description="Named position, e.g. 'top-right', 'center', 'bottom-left'.",
    )
    x: int | None = Field(default=None, description="Explicit x in pixels, overriding position.")
    y: int | None = Field(default=None, description="Explicit y in pixels, overriding position.")
    margin: int = Field(default=24, ge=0, le=2000)
    width: int | None = Field(
        default=None, gt=0, le=16384, description="Scale the overlay to this width."
    )
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    start: float = Field(default=0.0, ge=0)
    end: float | None = Field(default=None, description="Null means to the end of the base clip.")
    chroma_key: str | None = Field(
        default=None,
        description="Key out this colour from the overlay, e.g. '#00FF00' for a green screen.",
    )
    chroma_similarity: float = Field(default=0.3, gt=0.0, le=1.0)
    chroma_blend: float = Field(default=0.1, ge=0.0, le=1.0)
    encode: EncodeOptions = Field(default_factory=EncodeOptions)

    @model_validator(mode="after")
    def _check(self) -> OverlayMediaArgs:
        if (self.x is None) != (self.y is None):
            raise ValueError("give both x and y, or neither")
        if self.end is not None and self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


@tool("overlay_media", title="Overlay media", phase=5)
async def overlay_media(args: OverlayMediaArgs) -> JobSubmission:
    """Composite an image or video on top of another — picture-in-picture,
    watermark, or a chroma-keyed composite.

    Position with a named corner or explicit x/y, scale with 'width', and fade
    it in and out of existence with 'start'/'end'. Set chroma_key to a hex
    colour to key out a green or blue screen from the overlay before compositing.
    """
    resolve_io(args, operation="composited")
    validate_input_file(args.overlay_path)
    if args.x is None:
        overlay_position_expressions(args.position, args.margin)
    return queue("overlay_media", args)


@handler("overlay_media")
async def _run_overlay_media(ctx: JobContext) -> JobOutcome:
    args = OverlayMediaArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="composited", job_id=ctx.job_id)
    overlay_source = validate_input_file(args.overlay_path, ctx.settings)
    base_info = await probe(source, ctx.settings)
    _require_video(base_info, source, "overlay_media")
    try:
        overlay_info: MediaInfo | None = await probe(overlay_source, ctx.settings)
    except Exception:
        overlay_info = None

    graph = FilterGraph()
    prepared = graph.chain(["1:v:0"], ["ovl"])
    if args.width:
        prepared.add(scale_filter(args.width, None))
    if args.chroma_key:
        prepared.add(
            chromakey_filter(
                _to_ffmpeg_colour(args.chroma_key), args.chroma_similarity, args.chroma_blend
            )
        )
    if args.opacity < 1.0:
        prepared.add(
            Filter("format", {}, raw_options="rgba"),
            Filter("colorchannelmixer", {}, raw_options=f"aa={args.opacity:.4f}"),
        )

    if args.x is not None and args.y is not None:
        x_expr, y_expr = str(args.x), str(args.y)
    else:
        x_expr, y_expr = overlay_position_expressions(args.position, args.margin)

    end = args.end if args.end is not None else (base_info.duration or 0.0)
    eof_action = overlay_eof_action(str(overlay_source), overlay_info)
    overlay_options = f"x={x_expr}:y={y_expr}:eof_action={eof_action}"
    if args.start > 0 or args.end is not None:
        from ..ffmpeg.filters import between_expr

        overlay_options += f":enable='{between_expr(args.start, max(end, args.start + 0.01))}'"
    graph.chain(["0:v:0", "ovl"], ["vout"]).add(
        Filter("overlay", {}, raw_options=overlay_options),
        Filter("format", {}, raw_options="yuv420p"),
    )

    argv = [
        "-i",
        str(source),
        "-i",
        str(overlay_source),
        "-filter_complex",
        graph.render(),
        "-map",
        "[vout]",
    ]
    if base_info.has_audio:
        argv += ["-map", "0:a:0"]
    argv += output_args(args.encode, destination, has_video=True, has_audio=base_info.has_audio)
    argv += ["-y", str(destination)]

    result = await run_ffmpeg(
        argv,
        total_duration=base_info.duration,
        on_progress=ctx.make_progress_hook(2.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    notes = [
        "Composited overlay."
        if eof_action == "pass"
        else "Composited still overlay, held for the whole clip."
    ]
    if args.chroma_key:
        notes.append(f"Keyed out {args.chroma_key} from the overlay.")
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


def _to_ffmpeg_colour(colour: str) -> str:
    """Convert ``#RRGGBB`` to ffmpeg's ``0xRRGGBB``, passing named colours through."""
    match = _HEX.match(colour.strip())
    if match:
        return f"0x{match.group(1).upper()}"
    if re.fullmatch(r"[A-Za-z]+", colour.strip()):
        return colour.strip().lower()
    raise InvalidParameterError(
        "Colour must be a hex value like '#00FF00' or a colour name.", colour=colour
    )


# --------------------------------------------------------------------------- #
# mix_audio
# --------------------------------------------------------------------------- #


class AudioSource(StrictModel):
    """One input to the mix."""

    path: str
    gain_db: float = Field(default=0.0, ge=-60, le=30)
    start: float = Field(default=0.0, ge=0, description="Delay before this track begins.")
    duck: bool = Field(
        default=False, description="Lower this track whenever the voice track is loud."
    )


class MixAudioArgs(StrictModel):
    """Arguments for mixing audio tracks together."""

    voice_path: str = Field(
        description="The primary track — usually dialogue, or a video whose audio leads the mix."
    )
    tracks: list[AudioSource] = Field(
        min_length=1, max_length=16, description="Additional tracks to mix in."
    )
    output_path: str | None = None
    keep_video_from_voice: bool = Field(
        default=True,
        description="If the primary input is a video, carry its picture through unchanged.",
    )
    duration: str = Field(
        default="first",
        description="'first' ends with the primary track, 'longest' with the longest input.",
    )
    encode: EncodeOptions = Field(default_factory=EncodeOptions)

    @model_validator(mode="after")
    def _check_duration(self) -> MixAudioArgs:
        if self.duration not in {"first", "longest", "shortest"}:
            raise ValueError("duration must be 'first', 'longest' or 'shortest'")
        return self


@tool("mix_audio", title="Mix audio tracks", phase=5)
async def mix_audio(args: MixAudioArgs) -> JobSubmission:
    """Mix background music or effects under a primary voice track.

    Set 'duck' on a track to have it automatically drop in level whenever the
    voice track is loud, which is what makes music sit under narration without
    manual level automation. Per-track gain and start offsets are applied before
    the mix.

    If the primary input is a video, its picture is carried through untouched.
    """
    validate_input_file(args.voice_path)
    for track in args.tracks:
        validate_input_file(track.path)
    return queue("mix_audio", args)


@handler("mix_audio")
async def _run_mix_audio(ctx: JobContext) -> JobOutcome:
    args = MixAudioArgs.model_validate(ctx.params)
    voice = validate_input_file(args.voice_path, ctx.settings)
    voice_info = await probe(voice, ctx.settings)
    if not voice_info.has_audio:
        raise InvalidParameterError("The primary input has no audio.", path=str(voice))

    sources = [validate_input_file(t.path, ctx.settings) for t in args.tracks]
    infos = [await probe(p, ctx.settings) for p in sources]
    for path, info in zip(sources, infos, strict=True):
        if not info.has_audio:
            raise InvalidParameterError("Track has no audio.", path=str(path))

    keep_video = args.keep_video_from_voice and voice_info.has_video
    suffix = voice.suffix if keep_video else ".m4a"
    destination = validate_output_path(
        args.output_path,
        suggested_name=f"{voice.stem}_mixed{suffix}",
        job_id=ctx.job_id,
        settings=ctx.settings,
    )

    graph = FilterGraph()
    fmt = f"sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo"
    ducking = any(t.duck for t in args.tracks)

    graph.chain(["0:a:0"], ["voicemix", "voicekey"] if ducking else ["voice"]).add(
        Filter("aformat", {}, raw_options=fmt),
        *([Filter("asplit", {}, raw_options="2")] if ducking else []),
    )

    prepared: list[str] = []
    key = "voicekey"
    duckable = [i for i, t in enumerate(args.tracks) if t.duck]
    for index, track in enumerate(args.tracks):
        label = f"t{index}"
        chain = graph.chain([f"{index + 1}:a:0"], [label])
        chain.add(Filter("aformat", {}, raw_options=fmt))
        if track.gain_db != 0.0:
            chain.add(volume_filter(track.gain_db))
        if track.start > 0:
            chain.add(Filter("adelay", {}, raw_options=f"{int(track.start * 1000)}:all=1"))
        if track.duck:
            out_label = f"d{index}"
            if index != duckable[-1]:
                graph.chain([key], [f"k{index}", f"kr{index}"]).add(
                    Filter("asplit", {}, raw_options="2")
                )
                this_key, key = f"k{index}", f"kr{index}"
            else:
                this_key = key
            graph.chain([label, this_key], [out_label]).add(
                sidechain_duck_filter(0.03, 8.0, 20.0, 400.0)
            )
            prepared.append(out_label)
        else:
            prepared.append(label)

    voice_label = "voicemix" if ducking else "voice"
    graph.chain([voice_label, *prepared], ["aout"]).add(
        Filter(
            "amix",
            {},
            raw_options=(
                f"inputs={1 + len(prepared)}:duration={args.duration}:"
                "dropout_transition=0:normalize=0"
            ),
        ),
        Filter("aformat", {}, raw_options=fmt),
    )

    argv = ["-i", str(voice)]
    for path in sources:
        argv += ["-i", str(path)]
    argv += ["-filter_complex", graph.render(), "-map", "[aout]"]
    if keep_video:
        argv += ["-map", "0:v:0", "-c:v", "copy"]
    argv += output_args(args.encode, destination, has_video=False, has_audio=True)
    argv += ["-y", str(destination)]

    result = await run_ffmpeg(
        argv,
        total_duration=voice_info.duration,
        on_progress=ctx.make_progress_hook(2.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    notes = [f"Mixed {len(args.tracks)} track(s) under the primary audio."]
    if ducking:
        notes.append("Ducking applied to the marked track(s).")
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


# --------------------------------------------------------------------------- #
# normalize_audio
# --------------------------------------------------------------------------- #


class NormalizeAudioArgs(MediaJobArgs):
    """Arguments for loudness normalisation."""

    target_lufs: float = Field(
        default=-16.0,
        ge=-70,
        le=-5,
        description="Integrated loudness target. -16 suits online video, -23 broadcast.",
    )
    true_peak: float = Field(default=-1.5, ge=-9.0, le=0.0)
    loudness_range: float = Field(default=11.0, gt=0, le=20)
    two_pass: bool = Field(
        default=True,
        description=(
            "Measure first, then correct with those measurements. More accurate "
            "than a single streaming pass, at the cost of reading the file twice."
        ),
    )
    encode: EncodeOptions = Field(default_factory=EncodeOptions)


@tool("normalize_audio", title="Normalise loudness", phase=5)
async def normalize_audio(args: NormalizeAudioArgs) -> JobSubmission:
    """Normalise a file's loudness to a target level (EBU R128).

    By default this runs two passes: the first measures the actual loudness, the
    second corrects to the target using those measurements. That is noticeably
    more accurate than the single streaming pass, which has to guess as it goes.

    -16 LUFS suits online video, -23 LUFS is the broadcast standard.
    """
    resolve_io(args, operation="normalized")
    return queue("normalize_audio", args)


def parse_loudnorm_json(stderr: str) -> dict[str, Any] | None:
    """Extract the measurement JSON that loudnorm prints on its analysis pass."""
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(stderr[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def measured_loudnorm_filter(
    measurements: dict[str, Any], target_lufs: float, true_peak: float, lra: float
) -> Filter:
    """Build the second-pass loudnorm filter from the first pass's measurements."""
    return Filter(
        "loudnorm",
        {},
        raw_options=(
            f"I={target_lufs}:TP={true_peak}:LRA={lra}:"
            f"measured_I={measurements['input_i']}:"
            f"measured_TP={measurements['input_tp']}:"
            f"measured_LRA={measurements['input_lra']}:"
            f"measured_thresh={measurements['input_thresh']}:"
            f"offset={measurements.get('target_offset', 0)}:"
            "linear=true:print_format=summary"
        ),
    )


@handler("normalize_audio")
async def _run_normalize_audio(ctx: JobContext) -> JobOutcome:
    args = NormalizeAudioArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="normalized", job_id=ctx.job_id)
    info = await probe(source, ctx.settings)
    if not info.has_audio:
        raise InvalidParameterError("File has no audio to normalise.", path=str(source))

    notes: list[str] = []
    normalise: Filter = loudnorm_filter(args.target_lufs, args.true_peak, args.loudness_range)

    if args.two_pass:
        await ctx.report(2.0, "Measuring loudness")
        analysis = FilterChain(
            filters=[
                Filter(
                    "loudnorm",
                    {},
                    raw_options=(
                        f"I={args.target_lufs}:TP={args.true_peak}:"
                        f"LRA={args.loudness_range}:print_format=json"
                    ),
                )
            ]
        )
        measure = await run_ffmpeg(
            ["-i", str(source), "-af", analysis.render(), "-f", "null", "-"],
            total_duration=info.duration,
            on_progress=ctx.make_progress_hook(2.0, 48.0),
            cancel_check=ctx.cancelled,
            loglevel="info",
            settings=ctx.settings,
        )
        measurements = parse_loudnorm_json(measure.stderr)
        if measurements and "input_i" in measurements:
            normalise = measured_loudnorm_filter(
                measurements, args.target_lufs, args.true_peak, args.loudness_range
            )
            notes.append(
                f"Measured {measurements['input_i']} LUFS, corrected to {args.target_lufs}."
            )
        else:
            notes.append("Loudness measurement could not be parsed; fell back to a single pass.")

    await ctx.report(50.0, "Applying normalisation")
    chain = FilterChain(filters=[normalise, Filter("aresample", {}, raw_options=str(SAMPLE_RATE))])
    argv = ["-i", str(source), "-af", chain.render()]
    if info.has_video:
        argv += ["-c:v", "copy"]
    argv += output_args(args.encode, destination, has_video=False, has_audio=True)
    argv += ["-y", str(destination)]
    result = await run_ffmpeg(
        argv,
        total_duration=info.duration,
        on_progress=ctx.make_progress_hook(50.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    notes.append(f"Normalised to {args.target_lufs} LUFS.")
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


# --------------------------------------------------------------------------- #
# fade_audio
# --------------------------------------------------------------------------- #


class FadeAudioArgs(MediaJobArgs):
    """Arguments for audio fades."""

    fade_in: float = Field(default=0.0, ge=0, le=600, description="Fade-in length in seconds.")
    fade_out: float = Field(default=0.0, ge=0, le=600, description="Fade-out length in seconds.")
    encode: EncodeOptions = Field(default_factory=EncodeOptions)

    @model_validator(mode="after")
    def _check_any(self) -> FadeAudioArgs:
        if self.fade_in <= 0 and self.fade_out <= 0:
            raise ValueError("fade_audio needs a fade_in or a fade_out")
        return self


@tool("fade_audio", title="Fade audio", phase=5)
async def fade_audio(args: FadeAudioArgs) -> JobSubmission:
    """Fade a file's audio in at the start, out at the end, or both.

    The fade-out is positioned from the file's measured duration, so you give
    its length rather than working out its start time yourself. Video is stream-
    copied, so only the audio is re-encoded.
    """
    resolve_io(args, operation="faded")
    return queue("fade_audio", args)


@handler("fade_audio")
async def _run_fade_audio(ctx: JobContext) -> JobOutcome:
    args = FadeAudioArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="faded", job_id=ctx.job_id)
    info = await probe(source, ctx.settings)
    if not info.has_audio:
        raise InvalidParameterError("File has no audio to fade.", path=str(source))
    duration = info.duration
    if args.fade_out > 0 and not duration:
        raise InvalidParameterError(
            "Cannot place a fade-out without a known duration.", path=str(source)
        )

    filters: list[Filter] = []
    if args.fade_in > 0:
        filters.append(afade_filter("in", 0.0, args.fade_in))
    if args.fade_out > 0:
        assert duration is not None
        filters.append(afade_filter("out", max(0.0, duration - args.fade_out), args.fade_out))

    argv = ["-i", str(source), "-af", FilterChain(filters=filters).render()]
    if info.has_video:
        argv += ["-c:v", "copy"]
    argv += output_args(args.encode, destination, has_video=False, has_audio=True)
    argv += ["-y", str(destination)]
    result = await run_ffmpeg(
        argv,
        total_duration=duration,
        on_progress=ctx.make_progress_hook(2.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    notes = []
    if args.fade_in:
        notes.append(f"Faded in over {args.fade_in}s.")
    if args.fade_out:
        notes.append(f"Faded out over {args.fade_out}s.")
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


# --------------------------------------------------------------------------- #
# render_timeline
# --------------------------------------------------------------------------- #


class RenderTimelineArgs(StrictModel):
    """Arguments for rendering a full timeline."""

    timeline: Timeline
    output_path: str | None = None
    encode: EncodeOptions = Field(default_factory=EncodeOptions)


@tool("render_timeline", title="Render a timeline", phase=5)
async def render_timeline(args: RenderTimelineArgs) -> JobSubmission:
    """Render a complete edit — clips, transitions, overlays, captions, audio — in one pass.

    This is the entry point for driving the server from a script rather than
    calling tools one at a time, and it is exactly the structure the local UI's
    timeline editor produces.

    The timeline declares an output width, height and frame rate; every clip is
    scaled and padded to fit, so sources may differ. Each clip has in and out
    points into its source, an optional speed, and an optional transition into
    the next one. Text overlays, media overlays (picture-in-picture or
    watermarks), a burned-in subtitle file, and extra audio tracks with optional
    ducking all layer on top.
    """
    for clip in args.timeline.clips:
        validate_input_file(clip.source)
    for overlay in args.timeline.media_overlays:
        validate_input_file(overlay.source)
    for track in args.timeline.audio_tracks:
        validate_input_file(track.source)
    if args.timeline.captions:
        validate_input_file(args.timeline.captions.subtitle_path)
    return queue("render_timeline", args)


@handler("render_timeline")
async def _run_render_timeline(ctx: JobContext) -> JobOutcome:
    args = RenderTimelineArgs.model_validate(ctx.params)
    timeline = args.timeline

    clip_paths = [validate_input_file(c.source, ctx.settings) for c in timeline.clips]
    clip_infos = [await probe(p, ctx.settings) for p in clip_paths]
    overlay_paths = [validate_input_file(o.source, ctx.settings) for o in timeline.media_overlays]
    overlay_infos: list[MediaInfo | None] = []
    for path in overlay_paths:
        try:
            overlay_infos.append(await probe(path, ctx.settings))
        except Exception:
            overlay_infos.append(None)
    audio_paths = [validate_input_file(t.source, ctx.settings) for t in timeline.audio_tracks]
    audio_infos = [await probe(p, ctx.settings) for p in audio_paths]
    if timeline.captions:
        validate_input_file(timeline.captions.subtitle_path, ctx.settings)
    await ctx.report(4.0, f"Probed {len(clip_paths)} clip(s)")

    # Overlay text goes to sidecar files so it never enters the filter graph.
    text_files: list[str] = []
    for index, text in enumerate(timeline.text_overlays):
        path = ctx.workdir / f"timeline_text_{index}.txt"
        path.write_text(text.text, encoding="utf-8")
        text_files.append(str(path))

    resolved = timeline.model_copy(
        update={
            "clips": [
                c.model_copy(update={"source": str(p)})
                for c, p in zip(timeline.clips, clip_paths, strict=True)
            ],
            "media_overlays": [
                o.model_copy(update={"source": str(p)})
                for o, p in zip(timeline.media_overlays, overlay_paths, strict=True)
            ],
            "audio_tracks": [
                t.model_copy(update={"source": str(p)})
                for t, p in zip(timeline.audio_tracks, audio_paths, strict=True)
            ],
        }
    )

    compiled = compile_timeline(
        resolved,
        clip_infos,
        overlay_infos,
        audio_infos,
        text_files=text_files,
        sample_rate=SAMPLE_RATE,
    )
    destination = validate_output_path(
        args.output_path,
        suggested_name="timeline.mp4",
        job_id=ctx.job_id,
        settings=ctx.settings,
    )

    argv: list[str] = []
    for pair in compiled.inputs:
        argv += pair
    argv += [
        "-filter_complex",
        compiled.graph,
        "-map",
        f"[{compiled.video_label}]",
    ]
    if compiled.audio_label:
        argv += ["-map", f"[{compiled.audio_label}]"]
    argv += output_args(
        args.encode, destination, has_video=True, has_audio=bool(compiled.audio_label)
    )
    argv += ["-y", str(destination)]

    result = await run_ffmpeg(
        argv,
        total_duration=compiled.duration,
        on_progress=ctx.make_progress_hook(5.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    notes = [
        f"Rendered {len(timeline.clips)} clip(s) to "
        f"{timeline.width}x{timeline.height} @ {timeline.fps}fps "
        f"({compiled.duration:.2f}s).",
        *compiled.notes,
    ]
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


__all__ = [
    "AddTransitionArgs",
    "AudioSource",
    "FadeAudioArgs",
    "MixAudioArgs",
    "NormalizeAudioArgs",
    "OverlayMediaArgs",
    "RenderTimelineArgs",
    "measured_loudnorm_filter",
    "parse_loudnorm_json",
]
