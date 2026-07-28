"""Phase 7 tools: looking at and measuring media, rather than only rendering it.

Editing is a loop — look, cut, look again, grade, measure, adjust — and until
now the server could only do every other step. A caller could render a grade but
not tell whether it had crushed the blacks, could cut a sequence but not see
what was in the footage, and had to decide whether audio was worth keeping
without ever measuring it.

``extract_frame``, ``extract_filmstrip`` and ``analyze_video`` answer
synchronously: each seeks to specific timestamps rather than decoding the whole
file, so the cost is bounded by the number of samples asked for, not the length
of the media. ``measure_audio`` is a job because loudness genuinely requires a
full pass.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..config import Settings
from ..errors import InvalidParameterError
from ..ffmpeg.filters import Filter, FilterChain, FilterGraph, escape_path_for_filter
from ..ffmpeg.probe import probe
from ..ffmpeg.runner import run_ffmpeg
from ..jobs.worker import JobContext, JobOutcome, handler
from ..models import JobSubmission, StrictModel
from ..paths import validate_input_file, validate_output_path
from .common import queue
from .registry import tool

MAX_SAMPLES = 60
MAX_TILES = 36


def sample_times(duration: float, count: int, start: float, end: float | None) -> list[float]:
    """Evenly spaced timestamps across a range, biased off the very edges.

    The first and last frames of a clip are the least representative — a fade,
    a shutter change, a hand still on the camera — so samples sit inside the
    range rather than on its boundaries.
    """
    if count < 1:
        raise InvalidParameterError("count must be at least 1.", count=count)
    stop = end if end is not None else duration
    if stop <= start:
        raise InvalidParameterError("end must be greater than start.", start=start, end=stop)
    span = stop - start
    if count == 1:
        return [round(start + span / 2, 3)]
    step = span / (count + 1)
    return [round(start + step * (i + 1), 3) for i in range(count)]


# --------------------------------------------------------------------------- #
# extract_frame
# --------------------------------------------------------------------------- #


class ExtractFrameArgs(StrictModel):
    """Arguments for grabbing a single still."""

    input_path: str = Field(description="Media file to take the frame from.")
    time: float = Field(default=0.0, ge=0, description="Timestamp in seconds.")
    output_path: str | None = Field(
        default=None, description="Where to write the image; defaults to the workspace."
    )
    width: int | None = Field(
        default=None, gt=0, le=8192, description="Scale the still to this width."
    )
    format: Literal["png", "jpg"] = Field(
        default="png", description="png is lossless; jpg is smaller."
    )


class ExtractedFrame(StrictModel):
    """A still written to disk."""

    output_path: str
    time: float
    width: int
    height: int


async def _write_frame(
    source: Path,
    time: float,
    destination: Path,
    width: int | None,
    settings: Settings | None,
) -> None:
    argv = ["-ss", f"{time:.3f}", "-i", str(source), "-frames:v", "1"]
    if width:
        argv += ["-vf", f"scale={width}:-2"]
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        argv += ["-q:v", "2"]
    argv += ["-y", str(destination)]
    # -ss before -i is a fast seek, so the cost does not grow with file length.
    await run_ffmpeg(argv, settings=settings)


@tool("extract_frame", title="Extract a frame", phase=7, read_only=True)
async def extract_frame(args: ExtractFrameArgs) -> ExtractedFrame:
    """Save a single frame as an image so it can actually be looked at.

    Answers immediately. Seeking is done before decoding, so grabbing a frame
    from an hour-long file costs the same as from a short one.

    Use it to check what is in footage before cutting, and to confirm a render
    looks the way it was meant to.
    """
    source = validate_input_file(args.input_path)
    info = await probe(source)
    if not info.has_video:
        raise InvalidParameterError("File has no video stream.", path=str(source))
    if info.duration and args.time > info.duration:
        raise InvalidParameterError(
            "Timestamp is past the end of the file.",
            time=args.time,
            duration=info.duration,
        )
    destination = validate_output_path(
        args.output_path,
        suggested_name=f"{source.stem}_t{args.time:.2f}.{args.format}",
    )
    await _write_frame(source, args.time, destination, args.width, None)
    still = await probe(destination)
    frame = still.video_streams[0] if still.video_streams else None
    return ExtractedFrame(
        output_path=str(destination),
        time=args.time,
        width=frame.width or 0 if frame else 0,
        height=frame.height or 0 if frame else 0,
    )


# --------------------------------------------------------------------------- #
# extract_filmstrip
# --------------------------------------------------------------------------- #


class FilmstripArgs(StrictModel):
    """Arguments for a contact sheet of frames."""

    input_path: str = Field(description="Video to sample.")
    count: int = Field(default=6, ge=2, le=MAX_TILES, description="How many frames to sample.")
    columns: int = Field(default=6, ge=1, le=12, description="Tiles per row.")
    start: float = Field(default=0.0, ge=0)
    end: float | None = Field(default=None, description="Null samples to the end of the file.")
    tile_width: int = Field(default=200, ge=40, le=1000, description="Width of each tile.")
    output_path: str | None = None


class Filmstrip(StrictModel):
    """A contact sheet and the timestamps it covers."""

    output_path: str
    times: list[float]
    columns: int
    rows: int


@tool("extract_filmstrip", title="Extract a filmstrip", phase=7, read_only=True)
async def extract_filmstrip(args: FilmstripArgs) -> Filmstrip:
    """Tile several frames into one contact sheet, to survey footage at a glance.

    The fastest way to find out what is actually in a clip — where the good
    moments are, where the camera settles, which shots are worth cutting to.
    Answers immediately: each frame is a separate fast seek, so the cost tracks
    the number of frames, not the length of the video.
    """
    source = validate_input_file(args.input_path)
    info = await probe(source)
    if not info.has_video:
        raise InvalidParameterError("File has no video stream.", path=str(source))
    duration = info.duration or 0.0
    times = sample_times(duration, args.count, args.start, args.end)
    destination = validate_output_path(
        args.output_path, suggested_name=f"{source.stem}_filmstrip.png"
    )

    columns = min(args.columns, len(times))
    rows = (len(times) + columns - 1) // columns

    # One input per sample, each with its own fast seek, then tiled in one pass.
    argv: list[str] = []
    for time in times:
        argv += ["-ss", f"{time:.3f}", "-i", str(source)]
    graph = FilterGraph()
    labels = []
    for index in range(len(times)):
        label = f"t{index}"
        graph.chain([f"{index}:v:0"], [label]).add(
            Filter("scale", {}, raw_options=f"{args.tile_width}:-2"),
            Filter("setsar", {}, raw_options="1"),
        )
        labels.append(label)
    graph.chain(labels, ["sheet"]).add(
        Filter("xstack", {}, raw_options=_xstack_layout(len(times), columns))
        if len(times) > 1
        else Filter("null")
    )
    argv += [
        "-filter_complex",
        graph.render(),
        "-map",
        "[sheet]",
        "-frames:v",
        "1",
        "-y",
        str(destination),
    ]
    await run_ffmpeg(argv)
    return Filmstrip(output_path=str(destination), times=times, columns=columns, rows=rows)


def _xstack_layout(count: int, columns: int) -> str:
    """Build an ``xstack`` grid layout string for evenly sized tiles."""
    cells = []
    for index in range(count):
        col, row = index % columns, index // columns
        x = "0" if col == 0 else "+".join(f"w{c}" for c in range(col))
        y = "0" if row == 0 else "+".join(f"h{r * columns}" for r in range(row))
        cells.append(f"{x}_{y}")
    return f"inputs={count}:layout={'|'.join(cells)}:fill=black"


# --------------------------------------------------------------------------- #
# analyze_video
# --------------------------------------------------------------------------- #

_STAT = re.compile(r"lavfi\.signalstats\.([A-Z]+)=([-0-9.]+)")


def parse_signalstats(text: str) -> dict[str, float]:
    """Pull one frame's signalstats values out of the metadata dump."""
    return {key: float(value) for key, value in _STAT.findall(text)}


class FrameStats(StrictModel):
    """Measurements for one sampled frame."""

    time: float
    luma_avg: float = Field(description="Mean brightness, 0-255.")
    luma_min: float
    luma_max: float
    saturation_avg: float = Field(description="Mean colourfulness; 0 is greyscale.")
    saturation_max: float


class VideoAnalysis(StrictModel):
    """What the sampled frames say about the picture."""

    frames: list[FrameStats] = Field(default_factory=list)
    luma_avg: float = Field(description="Mean brightness across the samples.")
    luma_min: float = Field(description="Darkest sampled frame's mean brightness.")
    luma_max: float = Field(description="Brightest sampled frame's mean brightness.")
    saturation_avg: float
    is_greyscale: bool = Field(description="True when no sample carries meaningful colour.")
    crushed_blacks: bool = Field(
        description="True when a sample's mean brightness is very low, so shadow detail is lost."
    )
    blown_highlights: bool = Field(description="True when a sample is close to clipping white.")
    notes: list[str] = Field(default_factory=list)


class AnalyzeArgs(StrictModel):
    """Arguments for measuring the picture."""

    input_path: str
    count: int = Field(default=8, ge=1, le=MAX_SAMPLES, description="Frames to sample.")
    start: float = Field(default=0.0, ge=0)
    end: float | None = None


@tool("analyze_video", title="Analyse brightness and colour", phase=7, read_only=True)
async def analyze_video(args: AnalyzeArgs) -> VideoAnalysis:
    """Measure brightness and colourfulness across sampled frames.

    This is how a grade gets checked rather than guessed at: whether contrast
    crushed the shadows, whether a highlight is clipping, whether a clip is
    genuinely greyscale, and how evenly exposed a cut is across its shots.

    Answers immediately; each sample is a separate fast seek.
    """
    source = validate_input_file(args.input_path)
    info = await probe(source)
    if not info.has_video:
        raise InvalidParameterError("File has no video stream.", path=str(source))
    times = sample_times(info.duration or 0.0, args.count, args.start, args.end)

    from ..paths import job_dir

    scratch = job_dir("analysis")
    samples: list[FrameStats] = []
    for index, time in enumerate(times):
        stats_file = scratch / f"stats_{index}.txt"
        chain = FilterChain(
            filters=[
                Filter("signalstats"),
                Filter(
                    "metadata",
                    {},
                    raw_options=f"print:file={escape_path_for_filter(stats_file)}",
                ),
            ]
        )
        await run_ffmpeg(
            [
                "-ss",
                f"{time:.3f}",
                "-i",
                str(source),
                "-vf",
                chain.render(),
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ]
        )
        values = parse_signalstats(stats_file.read_text(errors="replace"))
        stats_file.unlink(missing_ok=True)
        samples.append(
            FrameStats(
                time=time,
                luma_avg=round(values.get("YAVG", 0.0), 2),
                luma_min=values.get("YMIN", 0.0),
                luma_max=values.get("YMAX", 0.0),
                saturation_avg=round(values.get("SATAVG", 0.0), 2),
                saturation_max=values.get("SATMAX", 0.0),
            )
        )

    return summarise_analysis(samples)


def summarise_analysis(samples: list[FrameStats]) -> VideoAnalysis:
    """Turn per-frame measurements into a verdict. Pure, so it is unit tested."""
    if not samples:
        raise InvalidParameterError("No frames were sampled.")
    lumas = [s.luma_avg for s in samples]
    sats = [s.saturation_avg for s in samples]
    mean_luma = sum(lumas) / len(lumas)
    mean_sat = sum(sats) / len(sats)

    greyscale = max(sats) < 3.0
    crushed = min(lumas) < 25.0
    blown = max(s.luma_max for s in samples) >= 254 and mean_luma > 200

    notes: list[str] = []
    if greyscale:
        notes.append("No meaningful colour in any sample; this is greyscale.")
    if crushed:
        notes.append(
            f"The darkest sample averages {min(lumas):.1f}/255, so shadow detail is "
            "likely lost. Ease the contrast or lift the black point."
        )
    if blown:
        notes.append("Highlights are close to clipping.")
    if max(lumas) - min(lumas) > 90:
        notes.append(
            f"Exposure varies a lot across the samples ({min(lumas):.0f} to "
            f"{max(lumas):.0f}), so the shots may not cut together evenly."
        )
    return VideoAnalysis(
        frames=samples,
        luma_avg=round(mean_luma, 2),
        luma_min=round(min(lumas), 2),
        luma_max=round(max(lumas), 2),
        saturation_avg=round(mean_sat, 2),
        is_greyscale=greyscale,
        crushed_blacks=crushed,
        blown_highlights=blown,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# measure_audio
# --------------------------------------------------------------------------- #

_VOLUME = re.compile(r"(mean_volume|max_volume):\s*(-?[0-9.]+) dB")
_EBUR = re.compile(r"^\s*(I|LRA|Threshold):\s*(-?[0-9.]+)\s*(LUFS|LU)", re.M)


def parse_audio_measurements(stderr: str) -> dict[str, float]:
    """Pull volumedetect and ebur128 figures out of ffmpeg's log."""
    found: dict[str, float] = {key: float(value) for key, value in _VOLUME.findall(stderr)}
    matches = _EBUR.findall(stderr)
    # ebur128 prints Threshold twice; the integrated block comes first.
    for key, value, _unit in matches:
        name = {"I": "integrated_lufs", "LRA": "loudness_range"}.get(key)
        if name and name not in found:
            found[name] = float(value)
    return found


class MeasureAudioArgs(StrictModel):
    """Arguments for measuring loudness."""

    input_path: str


class AudioMeasurement(StrictModel):
    """How loud a file actually is."""

    mean_volume_db: float | None = None
    max_volume_db: float | None = None
    integrated_lufs: float | None = None
    loudness_range: float | None = None
    is_effectively_silent: bool = False
    notes: list[str] = Field(default_factory=list)


@tool("measure_audio", title="Measure loudness", phase=7)
async def measure_audio(args: MeasureAudioArgs) -> JobSubmission:
    """Measure a file's loudness: mean and peak level, and EBU R128 LUFS.

    Use it to decide whether audio is worth keeping — room tone and handling
    noise measure very differently from speech — and to check what a mix or a
    normalisation actually did.

    This one is a job rather than an instant answer: integrated loudness is
    defined over the whole file, so it cannot be sampled.
    """
    validate_input_file(args.input_path)
    return queue("measure_audio", args)


@handler("measure_audio")
async def _run_measure_audio(ctx: JobContext) -> JobOutcome:
    args = MeasureAudioArgs.model_validate(ctx.params)
    source = validate_input_file(args.input_path, ctx.settings)
    info = await probe(source, ctx.settings)
    if not info.has_audio:
        raise InvalidParameterError("File has no audio stream.", path=str(source))

    result = await run_ffmpeg(
        ["-i", str(source), "-af", "volumedetect,ebur128=framelog=quiet", "-f", "null", "-"],
        total_duration=info.duration,
        on_progress=ctx.make_progress_hook(1.0, 99.0),
        cancel_check=ctx.cancelled,
        loglevel="info",
        settings=ctx.settings,
    )
    values = parse_audio_measurements(result.stderr)
    payload = build_audio_measurement(values)
    return JobOutcome(result=payload.model_dump(mode="json"), command=result.command)


def build_audio_measurement(values: dict[str, float]) -> AudioMeasurement:
    """Turn raw ffmpeg figures into a verdict. Pure, so it is unit tested."""
    mean = values.get("mean_volume")
    peak = values.get("max_volume")
    silent = mean is not None and mean < -50.0

    notes: list[str] = []
    if silent:
        notes.append(
            f"Effectively silent at {mean:.1f} dB mean; there is nothing here worth keeping."
        )
    elif mean is not None and mean < -40.0:
        notes.append(
            f"Very quiet at {mean:.1f} dB mean — likely room tone or handling noise "
            "rather than content."
        )
    if peak is not None and peak > -0.5:
        notes.append("Peaks are at or near full scale, so the audio may already be clipping.")
    integrated = values.get("integrated_lufs")
    if integrated is not None and integrated > -9.0:
        notes.append(
            f"Integrated loudness is {integrated:.1f} LUFS, louder than the -14 to -16 "
            "most platforms normalise to."
        )
    return AudioMeasurement(
        mean_volume_db=mean,
        max_volume_db=peak,
        integrated_lufs=integrated,
        loudness_range=values.get("loudness_range"),
        is_effectively_silent=silent,
        notes=notes,
    )


__all__ = [
    "AnalyzeArgs",
    "AudioMeasurement",
    "ExtractFrameArgs",
    "FilmstripArgs",
    "VideoAnalysis",
    "build_audio_measurement",
    "parse_audio_measurements",
    "parse_signalstats",
    "sample_times",
    "summarise_analysis",
]
