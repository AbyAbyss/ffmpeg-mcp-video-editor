"""Phase 2 colour tools: grading, LUTs, and curves."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..errors import InvalidParameterError
from ..ffmpeg.encoding import output_args
from ..ffmpeg.filters import (
    Filter,
    FilterChain,
    FilterGraph,
    curves_filter,
    eq_filter,
    format_filter,
    lut3d_filter,
    temperature_filter,
)
from ..ffmpeg.probe import probe
from ..ffmpeg.runner import run_ffmpeg
from ..jobs.worker import JobContext, JobOutcome, handler
from ..models import EncodeOptions, JobSubmission, StrictModel
from ..paths import validate_input_file
from ..subtitles import validate_cube_file
from .common import MediaJobArgs, finish_media_job, queue, resolve_io
from .registry import tool

CURVE_PRESETS = (
    "none",
    "color_negative",
    "cross_process",
    "darker",
    "increase_contrast",
    "lighter",
    "linear_contrast",
    "medium_contrast",
    "negative",
    "strong_contrast",
    "vintage",
)


# --------------------------------------------------------------------------- #
# color_grade
# --------------------------------------------------------------------------- #


class ColorGradeArgs(MediaJobArgs):
    """Arguments for a primary colour grade."""

    brightness: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Additive lift; 0 leaves it alone."
    )
    contrast: float = Field(
        default=1.0, ge=0.0, le=4.0, description="Multiplier; 1.0 leaves it alone."
    )
    saturation: float = Field(
        default=1.0, ge=0.0, le=3.0, description="Multiplier; 0 is greyscale, 1.0 unchanged."
    )
    gamma: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Midtone curve; 1.0 leaves it alone."
    )
    temperature: float = Field(
        default=0.0,
        ge=-100.0,
        le=100.0,
        description="Relative warmth: negative is cooler and bluer, positive warmer and oranger.",
    )
    encode: EncodeOptions = Field(default_factory=EncodeOptions)

    @model_validator(mode="after")
    def _check_any(self) -> ColorGradeArgs:
        if (
            self.brightness == 0.0
            and self.contrast == 1.0
            and self.saturation == 1.0
            and self.gamma == 1.0
            and self.temperature == 0.0
        ):
            raise ValueError("color_grade needs at least one non-neutral adjustment")
        return self


def build_grade_filters(args: ColorGradeArgs) -> list[Filter]:
    """Build the colour chain, omitting any stage left at its neutral value."""
    filters: list[Filter] = []
    equaliser = eq_filter(
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        gamma=args.gamma,
    )
    if equaliser:
        filters.append(equaliser)
    warmth = temperature_filter(args.temperature)
    if warmth:
        filters.append(warmth)
    return filters


@tool("color_grade", title="Colour grade", phase=2)
async def color_grade(args: ColorGradeArgs) -> JobSubmission:
    """Adjust brightness, contrast, saturation, gamma and colour temperature.

    All five are applied in one filter chain, so the whole grade costs a single
    re-encode. Stages left at their neutral value are omitted entirely.
    Temperature is a relative artistic warm/cool control, not an absolute white
    balance in Kelvin.
    """
    resolve_io(args, operation="graded")
    build_grade_filters(args)
    return queue("color_grade", args)


@handler("color_grade")
async def _run_color_grade(ctx: JobContext) -> JobOutcome:
    args = ColorGradeArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="graded", job_id=ctx.job_id)
    info = await probe(source, ctx.settings)
    if not info.has_video:
        raise InvalidParameterError("color_grade needs a video stream.", path=str(source))

    chain = FilterChain(filters=build_grade_filters(args))
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
# apply_lut
# --------------------------------------------------------------------------- #


class ApplyLutArgs(MediaJobArgs):
    """Arguments for applying a 3D LUT."""

    lut_path: str = Field(description="Path to a .cube LUT file.")
    interpolation: Literal["nearest", "trilinear", "tetrahedral", "pyramid", "prism"] = Field(
        default="tetrahedral",
        description="Sampling mode. Tetrahedral is the usual choice for film LUTs.",
    )
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Blend against the ungraded image; 1.0 applies the LUT fully.",
    )
    encode: EncodeOptions = Field(default_factory=EncodeOptions)


@tool("apply_lut", title="Apply a LUT", phase=2)
async def apply_lut(args: ApplyLutArgs) -> JobSubmission:
    """Apply a .cube colour lookup table.

    The LUT is parsed and validated before the job is queued — declared size,
    row count and numeric content — because ffmpeg reports a malformed cube with
    an error that gives no hint what is wrong. Use 'strength' below 1.0 to blend
    the graded result back against the original.
    """
    resolve_io(args, operation="lut")
    lut = validate_input_file(args.lut_path)
    validate_cube_file(lut)
    return queue("apply_lut", args)


def build_lut_graph(lut_path: str, interpolation: str, strength: float) -> str:
    """Build the LUT graph, blending against the source when strength < 1.

    A partial-strength LUT needs the original image kept alongside the graded
    one, so the graph splits the input and recombines it with ``blend``.
    """
    lut = lut3d_filter(lut_path, interpolation)
    if strength >= 1.0:
        return FilterChain(inputs=["0:v:0"], outputs=["vout"]).add(lut).render()

    graph = FilterGraph()
    graph.chain(["0:v:0"], ["orig", "tograde"]).add(Filter("split", {}, raw_options="2"))
    graph.chain(["tograde"], ["graded"]).add(lut)
    graph.chain(["orig", "graded"], ["vout"]).add(
        Filter("blend", {}, raw_options=f"all_mode=normal:all_opacity={strength:.4f}")
    )
    return graph.render()


@handler("apply_lut")
async def _run_apply_lut(ctx: JobContext) -> JobOutcome:
    args = ApplyLutArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="lut", job_id=ctx.job_id)
    lut = validate_input_file(args.lut_path, ctx.settings)
    info = validate_cube_file(lut)
    media = await probe(source, ctx.settings)
    if not media.has_video:
        raise InvalidParameterError("apply_lut needs a video stream.", path=str(source))
    if info.dimensions != 3:
        raise InvalidParameterError(
            "Only 3D LUTs are supported by this tool.", dimensions=info.dimensions
        )

    graph = build_lut_graph(str(lut), args.interpolation, args.strength)
    argv = ["-i", str(source), "-filter_complex", graph, "-map", "[vout]"]
    if media.has_audio:
        argv += ["-map", "0:a:0"]
    argv += output_args(args.encode, destination, has_video=True, has_audio=media.has_audio)
    argv += ["-y", str(destination)]
    result = await run_ffmpeg(
        argv,
        total_duration=media.duration,
        on_progress=ctx.make_progress_hook(1.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    notes = [f"Applied {info.size}x{info.size}x{info.size} LUT ({info.title or lut.name})."]
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


# --------------------------------------------------------------------------- #
# apply_curves
# --------------------------------------------------------------------------- #


class CurvePoint(StrictModel):
    """One control point on a tone curve, in 0..1 input/output coordinates."""

    x: float = Field(ge=0.0, le=1.0, description="Input level.")
    y: float = Field(ge=0.0, le=1.0, description="Output level.")


class CurvesArgs(MediaJobArgs):
    """Arguments for a curves adjustment."""

    preset: str | None = Field(default=None, description=f"One of: {', '.join(CURVE_PRESETS)}.")
    master: list[CurvePoint] = Field(
        default_factory=list, description="Control points applied to all channels."
    )
    red: list[CurvePoint] = Field(default_factory=list)
    green: list[CurvePoint] = Field(default_factory=list)
    blue: list[CurvePoint] = Field(default_factory=list)
    encode: EncodeOptions = Field(default_factory=EncodeOptions)

    @model_validator(mode="after")
    def _check_input(self) -> CurvesArgs:
        if self.preset and self.preset not in CURVE_PRESETS:
            raise ValueError(f"unknown curve preset {self.preset!r}")
        if not self.preset and not (self.master or self.red or self.green or self.blue):
            raise ValueError("apply_curves needs a preset or at least one set of control points")
        if self.preset and (self.master or self.red or self.green or self.blue):
            raise ValueError("give either a preset or control points, not both")
        return self


def _points(values: list[CurvePoint]) -> list[tuple[float, float]] | None:
    return [(p.x, p.y) for p in values] or None


def build_curves_filter(args: CurvesArgs) -> Filter:
    """Build the curves filter from either a preset or explicit control points."""
    return curves_filter(
        preset=args.preset,
        master=_points(args.master),
        red=_points(args.red),
        green=_points(args.green),
        blue=_points(args.blue),
    )


@tool("apply_curves", title="Adjust curves", phase=2)
async def apply_curves(args: CurvesArgs) -> JobSubmission:
    """Apply a tone curve, either a named preset or your own control points.

    Points are given in 0..1 input/output coordinates and may be listed in any
    order; they are sorted and validated before the job runs, because ffmpeg
    silently ignores a malformed curve rather than reporting an error. Give a
    preset or points, not both.
    """
    resolve_io(args, operation="curves")
    build_curves_filter(args)
    return queue("apply_curves", args)


@handler("apply_curves")
async def _run_curves(ctx: JobContext) -> JobOutcome:
    args = CurvesArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="curves", job_id=ctx.job_id)
    info = await probe(source, ctx.settings)
    if not info.has_video:
        raise InvalidParameterError("apply_curves needs a video stream.", path=str(source))

    chain = FilterChain(filters=[build_curves_filter(args), format_filter("yuv420p")])
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


__all__ = [
    "ApplyLutArgs",
    "ColorGradeArgs",
    "CurvePoint",
    "CurvesArgs",
    "build_curves_filter",
    "build_grade_filters",
    "build_lut_graph",
]
