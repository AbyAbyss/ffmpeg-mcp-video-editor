"""Phase 2 text tools: burning subtitle files, text overlays, and SRT generation.

Caption text is the highest-risk input in this project: it is arbitrary
user-supplied text going into a filter graph whose syntax uses colons, commas,
brackets and quotes structurally. Two defences are used here. Subtitle files are
passed to ffmpeg by path, so the text itself never enters the graph. Overlay text
is written to a sidecar file and referenced with ``textfile=``, with
``expansion=none`` so ffmpeg does not evaluate ``%{...}`` sequences inside it.
Everything that does reach the graph goes through
:func:`~ffmpeg_mcp.ffmpeg.filters.escape_filter_value`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..errors import InvalidParameterError
from ..ffmpeg.encoding import output_args
from ..ffmpeg.filters import (
    Filter,
    FilterChain,
    between_expr,
    drawtext_filter,
    fade_alpha_expr,
    subtitles_filter,
)
from ..ffmpeg.probe import probe
from ..ffmpeg.runner import run_ffmpeg
from ..jobs.worker import JobContext, JobOutcome, handler
from ..models import EncodeOptions, JobSubmission, Segment, StrictModel
from ..paths import (
    validate_font_dir,
    validate_font_file,
    validate_input_file,
    validate_output_path,
)
from ..subtitles import build_srt, parse_srt
from .common import MediaJobArgs, finish_media_job, queue, resolve_io
from .registry import tool

SUBTITLE_SUFFIXES = {".srt", ".ass", ".ssa", ".vtt"}

Position = Literal[
    "top-left",
    "top-center",
    "top-right",
    "middle-left",
    "center",
    "middle-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
    "lower-third",
]

# drawtext expressions per named position. 'text_w'/'text_h' are the rendered
# text's dimensions, 'w'/'h' the frame's.
_POSITION_EXPRESSIONS: dict[str, tuple[str, str]] = {
    "top-left": ("{m}", "{m}"),
    "top-center": ("(w-text_w)/2", "{m}"),
    "top-right": ("w-text_w-{m}", "{m}"),
    "middle-left": ("{m}", "(h-text_h)/2"),
    "center": ("(w-text_w)/2", "(h-text_h)/2"),
    "middle-right": ("w-text_w-{m}", "(h-text_h)/2"),
    "bottom-left": ("{m}", "h-text_h-{m}"),
    "bottom-center": ("(w-text_w)/2", "h-text_h-{m}"),
    "bottom-right": ("w-text_w-{m}", "h-text_h-{m}"),
    "lower-third": ("{m}", "h*0.72"),
}

# ASS alignment codes follow the numeric keypad: 1-3 bottom, 4-6 middle, 7-9 top.
_ASS_ALIGNMENT: dict[str, int] = {
    "bottom-left": 1,
    "bottom-center": 2,
    "bottom-right": 3,
    "middle-left": 4,
    "center": 5,
    "middle-right": 6,
    "top-left": 7,
    "top-center": 8,
    "top-right": 9,
    "lower-third": 1,
}

_HEX_COLOUR = re.compile(r"^#?([0-9a-fA-F]{6})([0-9a-fA-F]{2})?$")


def position_expressions(position: str, margin: int) -> tuple[str, str]:
    """Return the (x, y) drawtext expressions for a named position."""
    template = _POSITION_EXPRESSIONS.get(position)
    if template is None:
        raise InvalidParameterError(
            "Unknown position.", position=position, allowed=sorted(_POSITION_EXPRESSIONS)
        )
    return template[0].format(m=margin), template[1].format(m=margin)


def to_ass_colour(colour: str) -> str:
    """Convert ``#RRGGBB`` or ``#RRGGBBAA`` to an ASS ``&HAABBGGRR`` value.

    ASS stores colour as BGR with an *inverted* alpha, where 00 is opaque. Doing
    this by hand is the usual reason burned-in captions come out the wrong
    colour, so it is centralised and tested here.
    """
    match = _HEX_COLOUR.match(colour.strip())
    if not match:
        raise InvalidParameterError(
            "Colour must be a hex value like '#FFCC00' or '#FFCC00FF'.", colour=colour
        )
    rgb = match.group(1)
    red, green, blue = rgb[0:2], rgb[2:4], rgb[4:6]
    alpha_hex = match.group(2)
    # ASS alpha is transparency: 00 fully opaque, FF fully transparent.
    alpha = f"{255 - int(alpha_hex, 16):02X}" if alpha_hex else "00"
    return f"&H{alpha}{blue}{green}{red}".upper()


class CaptionStyle(StrictModel):
    """Styling for burned-in captions."""

    font_name: str = Field(default="Arial", description="Font family name as installed.")
    font_size: int = Field(default=24, ge=6, le=400)
    font_color: str = Field(default="#FFFFFF", description="Hex fill colour.")
    outline_color: str = Field(default="#000000", description="Hex outline colour.")
    outline_width: float = Field(default=2.0, ge=0.0, le=10.0)
    shadow: float = Field(default=0.0, ge=0.0, le=10.0)
    bold: bool = False
    italic: bool = False
    position: Position = "bottom-center"
    margin_vertical: int = Field(default=30, ge=0, le=2000)
    background_box: bool = Field(
        default=False, description="Draw an opaque box behind the text instead of an outline."
    )


def build_force_style(style: CaptionStyle) -> dict[str, str | int | float]:
    """Translate a CaptionStyle into ASS ``force_style`` overrides."""
    overrides: dict[str, str | int | float] = {
        "FontName": style.font_name,
        "FontSize": style.font_size,
        "PrimaryColour": to_ass_colour(style.font_color),
        "OutlineColour": to_ass_colour(style.outline_color),
        "Outline": style.outline_width,
        "Shadow": style.shadow,
        "Alignment": _ASS_ALIGNMENT[style.position],
        "MarginV": style.margin_vertical,
        "Bold": -1 if style.bold else 0,
        "Italic": -1 if style.italic else 0,
    }
    if style.background_box:
        # BorderStyle 3 draws a filled box using OutlineColour as the fill.
        overrides["BorderStyle"] = 3
    return overrides


# --------------------------------------------------------------------------- #
# build_srt
# --------------------------------------------------------------------------- #


class BuildSrtArgs(StrictModel):
    """Arguments for generating an SRT file from timed segments."""

    segments: list[Segment] = Field(min_length=1, description="Timed text, in any order.")
    output_path: str | None = Field(
        default=None, description="Where to write the .srt file; defaults to the workspace."
    )
    max_chars_per_line: int | None = Field(
        default=42,
        ge=10,
        le=200,
        description="Wrap cues to this width. Null leaves the text unwrapped.",
    )
    max_lines: int = Field(default=2, ge=1, le=5)


class BuildSrtResult(StrictModel):
    """The generated subtitle file."""

    output_path: str
    cue_count: int
    duration: float = Field(description="End time of the last cue, in seconds.")
    content_preview: str = Field(description="First few cues, for a quick sanity check.")


@tool("build_srt", title="Build an SRT file", phase=2)
async def build_srt_tool(args: BuildSrtArgs) -> BuildSrtResult:
    """Turn a list of timed text segments into a valid SRT subtitle file.

    Pure text handling — no ffmpeg call, so this answers immediately. Segments
    are sorted, empty ones dropped, and overlapping cues truncated so they do
    not fight each other on screen. Long lines are word-wrapped. Feed the result
    to burn_captions, or edit it first.
    """
    destination = validate_output_path(args.output_path, suggested_name="captions.srt")
    if destination.suffix.lower() != ".srt":
        destination = destination.with_suffix(".srt")
    content = build_srt(
        args.segments,
        max_chars_per_line=args.max_chars_per_line,
        max_lines=args.max_lines,
    )
    destination.write_text(content, encoding="utf-8")
    cues = parse_srt(content)
    return BuildSrtResult(
        output_path=str(destination),
        cue_count=len(cues),
        duration=max((c.end for c in cues), default=0.0),
        content_preview="\n".join(content.splitlines()[:12]),
    )


# --------------------------------------------------------------------------- #
# burn_captions
# --------------------------------------------------------------------------- #


class BurnCaptionsArgs(MediaJobArgs):
    """Arguments for burning a subtitle file into the picture."""

    subtitle_path: str = Field(description="Path to an .srt, .ass, .ssa or .vtt file.")
    style: CaptionStyle = Field(default_factory=CaptionStyle)
    fonts_dir: str | None = Field(
        default=None, description="Directory to search for the font, if it is not installed."
    )
    encode: EncodeOptions = Field(default_factory=EncodeOptions)


@tool("burn_captions", title="Burn in captions", phase=2)
async def burn_captions(args: BurnCaptionsArgs) -> JobSubmission:
    """Burn an SRT or ASS subtitle file permanently into the video.

    Font, size, fill and outline colour, position and margins are all
    controllable. Colours are given as hex (#RRGGBB) and converted to ASS's
    inverted BGR form internally. The subtitle file is handed to ffmpeg by path,
    so caption text containing colons, commas or brackets cannot corrupt the
    filter graph.

    For styling of an .ass file's own embedded styles, note that these overrides
    replace them.
    """
    resolve_io(args, operation="captioned")
    subtitle = validate_input_file(args.subtitle_path)
    if subtitle.suffix.lower() not in SUBTITLE_SUFFIXES:
        raise InvalidParameterError(
            "Unsupported subtitle format.",
            path=str(subtitle),
            supported=sorted(SUBTITLE_SUFFIXES),
        )
    build_force_style(args.style)  # validate colours before queueing
    return queue("burn_captions", args)


async def burn_captions_into(
    ctx: JobContext,
    args: BurnCaptionsArgs,
    *,
    floor: float = 1.0,
    ceiling: float = 99.0,
) -> JobOutcome:
    """Burn a subtitle file into a video, reporting progress across a sub-range.

    Split out from the tool handler so phase 3's auto_caption can reuse it as the
    second half of a longer job without duplicating the burn logic or resetting
    the job's progress back to zero.
    """
    source, destination = resolve_io(args, operation="captioned", job_id=ctx.job_id)
    subtitle = validate_input_file(args.subtitle_path, ctx.settings)
    info = await probe(source, ctx.settings)
    if not info.has_video:
        raise InvalidParameterError("burn_captions needs a video stream.", path=str(source))

    fonts_dir = None
    if args.fonts_dir:
        fonts_dir = validate_font_dir(args.fonts_dir, ctx.settings)

    chain = FilterChain(
        filters=[
            subtitles_filter(
                subtitle, force_style=build_force_style(args.style), fonts_dir=fonts_dir
            )
        ]
    )
    argv = ["-i", str(source), "-vf", chain.render()]
    argv += output_args(args.encode, destination, has_video=True, has_audio=info.has_audio)
    argv += ["-y", str(destination)]
    result = await run_ffmpeg(
        argv,
        total_duration=info.duration,
        on_progress=ctx.make_progress_hook(floor, ceiling),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    cue_count = (
        len(parse_srt(subtitle.read_text(encoding="utf-8", errors="replace")))
        if subtitle.suffix.lower() == ".srt"
        else 0
    )
    notes = [f"Burned {cue_count} cues." if cue_count else "Burned subtitle track."]
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


@handler("burn_captions")
async def _run_burn_captions(ctx: JobContext) -> JobOutcome:
    return await burn_captions_into(ctx, BurnCaptionsArgs.model_validate(ctx.params))


# --------------------------------------------------------------------------- #
# text_overlay
# --------------------------------------------------------------------------- #

Animation = Literal["none", "fade", "slide-left", "slide-right", "slide-up", "slide-down"]


class TextOverlayItem(StrictModel):
    """One title or lower-third with its timing, placement, and animation."""

    text: str = Field(min_length=1, description="The text to draw. Newlines are honoured.")
    start: float = Field(default=0.0, ge=0.0, description="When it appears, in seconds.")
    end: float | None = Field(
        default=None, description="When it disappears, in seconds. Null means to the end."
    )
    position: Position = "bottom-center"
    x: str | None = Field(default=None, description="Explicit x in pixels, overriding 'position'.")
    y: str | None = Field(default=None, description="Explicit y in pixels, overriding 'position'.")
    margin: int = Field(default=40, ge=0, le=2000)
    font_size: int = Field(default=48, ge=6, le=400)
    font_color: str = Field(default="#FFFFFF")
    font_file: str | None = Field(
        default=None, description="Path to a .ttf/.otf file, if the default font is not wanted."
    )
    outline_width: int = Field(default=0, ge=0, le=20)
    outline_color: str = Field(default="#000000")
    background_box: bool = False
    background_color: str = Field(
        default="#000000", description="Box fill colour when background_box is set."
    )
    background_opacity: float = Field(default=0.5, ge=0.0, le=1.0)
    animation: Animation = "none"
    animation_duration: float = Field(
        default=0.5, ge=0.0, le=10.0, description="Length of the in and out animation."
    )

    @model_validator(mode="after")
    def _check_window(self) -> TextOverlayItem:
        if self.end is not None and self.end <= self.start:
            raise ValueError("end must be greater than start")
        if (self.x is None) != (self.y is None):
            raise ValueError("give both x and y, or neither")
        return self


class TextOverlayArgs(MediaJobArgs):
    """Arguments for drawing one or more text overlays."""

    items: list[TextOverlayItem] = Field(min_length=1, max_length=50)
    encode: EncodeOptions = Field(default_factory=EncodeOptions)


def _hex_to_drawtext_colour(colour: str, opacity: float | None = None) -> str:
    """Convert ``#RRGGBB`` to ffmpeg's ``0xRRGGBB`` or ``0xRRGGBB@opacity`` form."""
    match = _HEX_COLOUR.match(colour.strip())
    if not match:
        raise InvalidParameterError("Colour must be a hex value like '#FFCC00'.", colour=colour)
    value = f"0x{match.group(1).upper()}"
    if opacity is not None:
        return f"{value}@{opacity:.3f}"
    return value


def build_overlay_filter(
    item: TextOverlayItem, textfile: Path, duration: float | None, font_file: Path | None
) -> Filter:
    """Build the drawtext filter for one overlay, including its animation.

    Timing, placement and animation are all derived from the item's fields as
    ffmpeg expressions rather than hand-written per call, so the ``enable``
    window and the fade/slide curves always agree with each other.
    """
    end = item.end if item.end is not None else (duration if duration else item.start + 5.0)
    if end <= item.start:
        raise InvalidParameterError("Overlay ends before it starts.", start=item.start, end=end)

    if item.x is not None and item.y is not None:
        x_expr, y_expr = item.x, item.y
    else:
        x_expr, y_expr = position_expressions(item.position, item.margin)

    alpha_expr: str | None = None
    slide = item.animation.startswith("slide-")
    if item.animation == "fade" and item.animation_duration > 0:
        alpha_expr = fade_alpha_expr(
            item.start, end, item.animation_duration, item.animation_duration
        )
    elif slide and item.animation_duration > 0:
        x_expr, y_expr = _slide_expressions(item, x_expr, y_expr, end)

    return drawtext_filter(
        textfile=textfile,
        fontfile=font_file,
        fontsize=item.font_size,
        fontcolor=_hex_to_drawtext_colour(item.font_color),
        x=x_expr,
        y=y_expr,
        box=item.background_box,
        boxcolor=_hex_to_drawtext_colour(item.background_color, item.background_opacity),
        borderw=item.outline_width,
        bordercolor=_hex_to_drawtext_colour(item.outline_color),
        alpha_expr=alpha_expr,
        enable_expr=between_expr(item.start, end),
    )


def _slide_expressions(
    item: TextOverlayItem, x_expr: str, y_expr: str, end: float
) -> tuple[str, str]:
    """Offset the resting position during the slide-in window."""
    seconds = item.animation_duration
    start, stop = item.start, item.start + seconds
    progress = f"(1-(t-{start:.6f})/{seconds:.6f})"
    travel = 200
    if item.animation == "slide-left":
        return f"if(lt(t,{stop:.6f}),({x_expr})+{travel}*{progress},{x_expr})", y_expr
    if item.animation == "slide-right":
        return f"if(lt(t,{stop:.6f}),({x_expr})-{travel}*{progress},{x_expr})", y_expr
    if item.animation == "slide-up":
        return x_expr, f"if(lt(t,{stop:.6f}),({y_expr})+{travel}*{progress},{y_expr})"
    return x_expr, f"if(lt(t,{stop:.6f}),({y_expr})-{travel}*{progress},{y_expr})"


@tool("text_overlay", title="Add text overlays", phase=2)
async def text_overlay(args: TextOverlayArgs) -> JobSubmission:
    """Draw titles and lower-thirds over the video, with timing and animation.

    Each item has its own in and out point, named position (or explicit x/y),
    font size and colour, optional outline or background box, and an animation:
    fade, or a slide from any direction. Everything renders in one pass.

    The text itself is written to a sidecar file and referenced by path rather
    than embedded in the filter graph, so any characters are safe — including
    colons, commas, brackets, quotes and %{...} sequences, which are drawn
    literally.
    """
    resolve_io(args, operation="titled")
    for item in args.items:
        _hex_to_drawtext_colour(item.font_color)
        if item.x is None:
            position_expressions(item.position, item.margin)
        if item.font_file:
            validate_font_file(item.font_file)
    return queue("text_overlay", args)


@handler("text_overlay")
async def _run_text_overlay(ctx: JobContext) -> JobOutcome:
    args = TextOverlayArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="titled", job_id=ctx.job_id)
    info = await probe(source, ctx.settings)
    if not info.has_video:
        raise InvalidParameterError("text_overlay needs a video stream.", path=str(source))

    filters: list[Filter] = []
    for index, item in enumerate(args.items):
        textfile = ctx.workdir / f"overlay_{index}.txt"
        textfile.write_text(item.text, encoding="utf-8")
        font_file = validate_font_file(item.font_file, ctx.settings) if item.font_file else None
        filters.append(build_overlay_filter(item, textfile, info.duration, font_file))

    chain = FilterChain(filters=filters)
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
    notes = [f"Drew {len(args.items)} overlay(s)."]
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


__all__ = [
    "BuildSrtArgs",
    "BurnCaptionsArgs",
    "CaptionStyle",
    "TextOverlayArgs",
    "TextOverlayItem",
    "build_force_style",
    "build_overlay_filter",
    "position_expressions",
    "to_ass_colour",
]
