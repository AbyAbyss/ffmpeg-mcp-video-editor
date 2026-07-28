"""Phase 3 tools: Whisper transcription, translation, and one-shot auto-captioning.

Transcription is CPU-bound and blocking, so it runs in a worker thread while the
event loop keeps serving progress polls and cancellations. Audio is first
extracted to 16 kHz mono WAV through the shared ffmpeg runner, which is both what
Whisper wants and much faster to decode than seeking around a video container.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..errors import InvalidParameterError
from ..ffmpeg.probe import probe
from ..ffmpeg.runner import run_ffmpeg
from ..jobs.worker import JobContext, JobOutcome, handler
from ..models import EncodeOptions, JobSubmission, Segment, StrictModel, WordTiming
from ..paths import validate_input_file, validate_output_path
from ..subtitles import build_srt
from ..transcribe import KNOWN_LANGUAGES, MODEL_SIZES, TranscriptionOutcome, transcribe_file
from .captions import BurnCaptionsArgs, CaptionStyle, burn_captions_into
from .common import queue
from .registry import tool

WhisperModelName = Literal[
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
]


class TranscriptionOptions(StrictModel):
    """Whisper settings shared by transcription, translation, and auto-caption."""

    model: WhisperModelName | None = Field(
        default=None,
        description=(
            "Whisper size. Larger is more accurate and slower; 'base' is a good "
            "default on CPU, 'large-v3' for accuracy. Defaults to the server setting."
        ),
    )
    language: str | None = Field(
        default=None,
        description="ISO code of the spoken language, e.g. 'en'. Omit to auto-detect.",
    )
    word_timestamps: bool = Field(
        default=False,
        description="Also return per-word timings. Slower, but needed for karaoke captions.",
    )
    vad_filter: bool = Field(
        default=True,
        description="Skip silence with voice activity detection. Usually improves timing.",
    )
    beam_size: int = Field(default=5, ge=1, le=10)
    initial_prompt: str | None = Field(
        default=None,
        description="Context to bias decoding, e.g. proper nouns or domain jargon.",
    )

    @model_validator(mode="after")
    def _check_language(self) -> TranscriptionOptions:
        if self.language and self.language.lower() not in KNOWN_LANGUAGES:
            raise ValueError(
                f"unknown language code {self.language!r}; expected one of the Whisper codes"
            )
        return self


class TranscriptResult(StrictModel):
    """A completed transcription."""

    text: str = Field(description="The full transcript as one string.")
    segments: list[Segment] = Field(default_factory=list)
    words: list[WordTiming] = Field(default_factory=list)
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None
    segment_count: int = 0
    notes: list[str] = Field(default_factory=list)


async def extract_audio_for_whisper(
    ctx: JobContext, source: Path, *, floor: float = 0.0, ceiling: float = 10.0
) -> Path:
    """Decode a file's audio to 16 kHz mono WAV in the job directory."""
    info = await probe(source, ctx.settings)
    if not info.has_audio:
        raise InvalidParameterError("File has no audio track to transcribe.", path=str(source))
    destination = ctx.workdir / "audio_16k.wav"
    argv = [
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(destination),
    ]
    await run_ffmpeg(
        argv,
        total_duration=info.duration,
        on_progress=ctx.make_progress_hook(floor, ceiling),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    return destination


async def run_transcription(
    ctx: JobContext,
    source: Path,
    options: TranscriptionOptions,
    *,
    task: str = "transcribe",
    floor: float = 10.0,
    ceiling: float = 95.0,
) -> TranscriptionOutcome:
    """Extract audio and transcribe it, reporting progress across a sub-range."""
    audio = await extract_audio_for_whisper(ctx, source, ceiling=floor)
    model_name = options.model or ctx.settings.whisper_model
    if model_name not in MODEL_SIZES:
        raise InvalidParameterError("Unknown Whisper model.", model=model_name)
    await ctx.report(floor, f"Transcribing with Whisper '{model_name}'")

    loop = asyncio.get_running_loop()
    span = ceiling - floor

    def report(percent: float) -> None:
        # Called from the worker thread, so hop back onto the loop to publish.
        asyncio.run_coroutine_threadsafe(ctx.report(floor + percent / 100.0 * span), loop)

    return await asyncio.to_thread(
        transcribe_file,
        audio,
        model_name=model_name,
        compute_type=ctx.settings.whisper_compute_type,
        language=options.language,
        task=task,
        word_timestamps=options.word_timestamps,
        vad_filter=options.vad_filter,
        beam_size=options.beam_size,
        initial_prompt=options.initial_prompt,
        on_progress=report,
        should_cancel=ctx.cancelled,
    )


def _to_result(outcome: TranscriptionOutcome, notes: list[str] | None = None) -> TranscriptResult:
    return TranscriptResult(
        text=outcome.text,
        segments=outcome.segments,
        words=outcome.words,
        language=outcome.language,
        language_probability=outcome.language_probability,
        duration=outcome.duration,
        segment_count=len(outcome.segments),
        notes=notes or [],
    )


# --------------------------------------------------------------------------- #
# transcribe_audio
# --------------------------------------------------------------------------- #


class TranscribeArgs(StrictModel):
    """Arguments for transcribing a file."""

    input_path: str = Field(description="Audio or video file to transcribe.")
    options: TranscriptionOptions = Field(default_factory=TranscriptionOptions)
    srt_path: str | None = Field(
        default=None,
        description="Also write the transcript as an SRT file at this path.",
    )


@tool("transcribe_audio", title="Transcribe audio", phase=3)
async def transcribe_audio(args: TranscribeArgs) -> JobSubmission:
    """Transcribe speech in an audio or video file using Whisper.

    Returns timed segments and, if word_timestamps is set, per-word timings. The
    spoken language is auto-detected unless you name one. Pass srt_path to have
    the transcript written straight out as a subtitle file.

    This can take a while — roughly real-time on CPU with the 'base' model, and
    several times that with 'large-v3' — so poll job_status. The first run with
    a given model also downloads its weights.
    """
    validate_input_file(args.input_path)
    return queue("transcribe_audio", args)


@handler("transcribe_audio")
async def _run_transcribe(ctx: JobContext) -> JobOutcome:
    args = TranscribeArgs.model_validate(ctx.params)
    source = validate_input_file(args.input_path, ctx.settings)
    outcome = await run_transcription(ctx, source, args.options)

    outputs: list[str] = []
    notes: list[str] = []
    if not outcome.segments:
        notes.append("No speech was detected in this file.")
    if args.srt_path:
        srt = validate_output_path(
            args.srt_path,
            suggested_name=f"{source.stem}.srt",
            job_id=ctx.job_id,
            settings=ctx.settings,
        )
        srt.write_text(build_srt(outcome.segments, max_chars_per_line=42), encoding="utf-8")
        outputs.append(str(srt))
    await ctx.report(99.0, "Transcription complete")
    return JobOutcome(outputs=outputs, result=_to_result(outcome, notes).model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# translate_transcript
# --------------------------------------------------------------------------- #


class TranslateArgs(StrictModel):
    """Arguments for translating speech into English."""

    input_path: str = Field(description="Audio or video file containing non-English speech.")
    options: TranscriptionOptions = Field(default_factory=TranscriptionOptions)
    srt_path: str | None = Field(default=None, description="Also write an English SRT file.")


@tool("translate_transcript", title="Translate speech to English", phase=3)
async def translate_transcript(args: TranslateArgs) -> JobSubmission:
    """Transcribe non-English speech and translate it into English.

    This uses Whisper's built-in translate mode, which only ever outputs
    English — Whisper cannot translate into any other target language. To reach
    a different language you would need a separate translation step applied to
    the transcript this returns.
    """
    validate_input_file(args.input_path)
    return queue("translate_transcript", args)


@handler("translate_transcript")
async def _run_translate(ctx: JobContext) -> JobOutcome:
    args = TranslateArgs.model_validate(ctx.params)
    source = validate_input_file(args.input_path, ctx.settings)
    outcome = await run_transcription(ctx, source, args.options, task="translate")

    notes = ["Whisper's translate mode outputs English only."]
    if not outcome.segments:
        notes.append("No speech was detected in this file.")
    outputs: list[str] = []
    if args.srt_path:
        srt = validate_output_path(
            args.srt_path,
            suggested_name=f"{source.stem}_en.srt",
            job_id=ctx.job_id,
            settings=ctx.settings,
        )
        srt.write_text(build_srt(outcome.segments, max_chars_per_line=42), encoding="utf-8")
        outputs.append(str(srt))
    await ctx.report(99.0, "Translation complete")
    return JobOutcome(outputs=outputs, result=_to_result(outcome, notes).model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# auto_caption
# --------------------------------------------------------------------------- #


class AutoCaptionArgs(StrictModel):
    """Arguments for the transcribe-then-burn convenience tool."""

    input_path: str = Field(description="Video file to caption.")
    output_path: str | None = Field(default=None, description="Captioned video destination.")
    srt_path: str | None = Field(
        default=None, description="Where to keep the generated SRT; defaults to the job directory."
    )
    options: TranscriptionOptions = Field(default_factory=TranscriptionOptions)
    style: CaptionStyle = Field(default_factory=CaptionStyle)
    max_chars_per_line: int = Field(
        default=42, ge=10, le=200, description="Wrap captions to this width."
    )
    max_lines: int = Field(default=2, ge=1, le=5)
    translate_to_english: bool = Field(
        default=False, description="Translate the speech to English before captioning."
    )
    encode: EncodeOptions = Field(default_factory=EncodeOptions)


class AutoCaptionResult(StrictModel):
    """The captioned video plus the transcript it was built from."""

    output_path: str
    srt_path: str
    transcript: TranscriptResult
    notes: list[str] = Field(default_factory=list)


@tool("auto_caption", title="Auto-caption a video", phase=3)
async def auto_caption(args: AutoCaptionArgs) -> JobSubmission:
    """Transcribe a video and burn the captions in, as one job.

    Chains transcription, SRT generation and caption burning so you do not have
    to orchestrate three jobs. The generated SRT is kept alongside the video, so
    you can correct the text and re-burn it with burn_captions if Whisper
    mishears something.

    Set translate_to_english to caption foreign-language speech in English.
    """
    validate_input_file(args.input_path)
    if args.output_path:
        validate_output_path(args.output_path, suggested_name="captioned.mp4")
    return queue("auto_caption", args)


@handler("auto_caption")
async def _run_auto_caption(ctx: JobContext) -> JobOutcome:
    args = AutoCaptionArgs.model_validate(ctx.params)
    source = validate_input_file(args.input_path, ctx.settings)
    info = await probe(source, ctx.settings)
    if not info.has_video:
        raise InvalidParameterError("auto_caption needs a video stream.", path=str(source))

    task = "translate" if args.translate_to_english else "transcribe"
    # Transcription is the slow half, so it owns most of the progress range.
    outcome = await run_transcription(ctx, source, args.options, task=task, floor=8.0, ceiling=70.0)

    srt = validate_output_path(
        args.srt_path,
        suggested_name=f"{source.stem}.srt",
        job_id=ctx.job_id,
        settings=ctx.settings,
    )
    srt.write_text(
        build_srt(
            outcome.segments,
            max_chars_per_line=args.max_chars_per_line,
            max_lines=args.max_lines,
        ),
        encoding="utf-8",
    )
    notes: list[str] = []
    if not outcome.segments:
        notes.append("No speech detected; the video was copied through with no captions.")
    if args.translate_to_english:
        notes.append("Captions were translated to English.")

    await ctx.report(72.0, "Burning captions")
    burn_outcome = await burn_captions_into(
        ctx,
        BurnCaptionsArgs(
            input_path=str(source),
            output_path=args.output_path,
            subtitle_path=str(srt),
            style=args.style,
            encode=args.encode,
        ),
        floor=72.0,
        ceiling=99.0,
    )

    output = burn_outcome.outputs[0]
    result = AutoCaptionResult(
        output_path=output,
        srt_path=str(srt),
        transcript=_to_result(outcome),
        notes=notes + burn_outcome.result.get("notes", []),
    )
    return JobOutcome(
        outputs=[output, str(srt)],
        result=result.model_dump(mode="json"),
        command=burn_outcome.command,
    )


__all__ = [
    "AutoCaptionArgs",
    "AutoCaptionResult",
    "TranscribeArgs",
    "TranscriptResult",
    "TranscriptionOptions",
    "TranslateArgs",
]
