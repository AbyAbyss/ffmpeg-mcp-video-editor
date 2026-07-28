"""Resolution and aspect-ratio conversion.

The common request — "make this YouTube video into a Reel" — is a change of both
resolution and shape, and the interesting part is what happens to the picture
that no longer fits. Four fit modes cover it:

* ``cover``   crop to fill the target, losing the edges (choose what survives
  with ``focus``),
* ``contain`` letterbox with solid bars, losing nothing,
* ``blur``    letterbox with a blurred, zoomed copy of the video behind — what
  most social reframing tools do, since it fills the frame without cropping,
* ``stretch`` distort to fit, which is almost never what anyone wants but is
  occasionally asked for.

``transform`` in phase 1 still handles explicit pixel crops and one-off scales;
this tool is the shape-aware layer above it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..errors import InvalidParameterError
from ..ffmpeg.encoding import output_args
from ..ffmpeg.filters import Filter, FilterGraph, gblur_filter
from ..ffmpeg.probe import probe
from ..ffmpeg.runner import run_ffmpeg
from ..jobs.worker import JobContext, JobOutcome, handler
from ..models import EncodeOptions, JobSubmission, StrictModel
from ..vision.tracking import parse_aspect_ratio
from .common import MediaJobArgs, finish_media_job, queue, resolve_io
from .registry import tool

# Named targets for the platforms people actually ask for by name.
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    # Landscape
    "youtube_1080p": (1920, 1080),
    "youtube_1440p": (2560, 1440),
    "youtube_4k": (3840, 2160),
    "youtube_720p": (1280, 720),
    "twitter_landscape": (1280, 720),
    "linkedin_landscape": (1920, 1080),
    "cinema_2k_flat": (1998, 1080),
    "cinema_scope": (2048, 858),
    # Vertical
    "reel": (1080, 1920),
    "instagram_reel": (1080, 1920),
    "tiktok": (1080, 1920),
    "youtube_short": (1080, 1920),
    "story": (1080, 1920),
    "snapchat": (1080, 1920),
    # Square and portrait
    "instagram_square": (1080, 1080),
    "instagram_portrait": (1080, 1350),
    "pinterest": (1000, 1500),
    # Small
    "preview_480p": (854, 480),
    "thumbnail_360p": (640, 360),
}

FitMode = Literal["cover", "contain", "blur", "stretch"]
Focus = Literal["center", "top", "bottom", "left", "right"]

# Where the crop window sits when 'cover' has to discard part of the picture.
# Values are the fraction of the overflow removed from the leading edge.
_FOCUS_BIAS: dict[str, tuple[float, float]] = {
    "center": (0.5, 0.5),
    "top": (0.5, 0.0),
    "bottom": (0.5, 1.0),
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
}


def resolve_target_size(
    *,
    source_width: int,
    source_height: int,
    preset: str | None,
    width: int | None,
    height: int | None,
    aspect_ratio: str | None,
) -> tuple[int, int]:
    """Work out the output size from a preset, explicit dimensions, or a ratio.

    Precedence is preset, then explicit width/height, then aspect_ratio. Giving
    an aspect ratio with only one dimension derives the other; giving it with
    neither keeps roughly the source's pixel count so a reframe does not silently
    upscale or shrink.

    Returns:
        An even-numbered (width, height), since yuv420p requires it.
    """
    if preset:
        key = preset.strip().lower()
        if key not in RESOLUTION_PRESETS:
            raise InvalidParameterError(
                "Unknown resolution preset.", preset=preset, available=sorted(RESOLUTION_PRESETS)
            )
        return RESOLUTION_PRESETS[key]

    if aspect_ratio:
        ratio = parse_aspect_ratio(aspect_ratio)
        if width and height:
            target_width, target_height = width, height
        elif width:
            target_width, target_height = width, round(width / ratio)
        elif height:
            target_width, target_height = round(height * ratio), height
        else:
            # Preserve the source's area so the result is neither upscaled nor
            # shrunk just because its shape changed.
            area = source_width * source_height
            target_height = round((area / ratio) ** 0.5)
            target_width = round(target_height * ratio)
        return _make_even(target_width, target_height)

    if width and height:
        return _make_even(width, height)
    if width:
        return _make_even(width, round(width * source_height / source_width))
    if height:
        return _make_even(round(height * source_width / source_height), height)

    raise InvalidParameterError("Give a preset, a width and/or height, or an aspect_ratio.")


def _make_even(width: int, height: int) -> tuple[int, int]:
    width = max(2, min(16384, width))
    height = max(2, min(16384, height))
    return width - width % 2, height - height % 2


def crop_offsets(
    scaled_width: int, scaled_height: int, target_width: int, target_height: int, focus: str
) -> tuple[str, str]:
    """Crop origin expressions for 'cover', biased by the focus point."""
    bias = _FOCUS_BIAS.get(focus)
    if bias is None:
        raise InvalidParameterError(
            "Unknown focus point.", focus=focus, allowed=sorted(_FOCUS_BIAS)
        )
    bias_x, bias_y = bias
    # Expressed against the scaled input so ffmpeg resolves the exact overflow.
    return f"(iw-{target_width})*{bias_x}", f"(ih-{target_height})*{bias_y}"


def build_resize_graph(
    *,
    target_width: int,
    target_height: int,
    fit: str,
    focus: str = "center",
    background_color: str = "black",
    blur_strength: float = 25.0,
    input_label: str = "0:v:0",
    output_label: str = "vout",
) -> str:
    """Build the filter graph for one resize, per fit mode.

    Pure, so each mode's graph shape is unit tested without running ffmpeg.
    """
    graph = FilterGraph()

    if fit == "stretch":
        graph.chain([input_label], [output_label]).add(
            Filter("scale", {}, raw_options=f"w={target_width}:h={target_height}"),
            Filter("setsar", {}, raw_options="1"),
            Filter("format", {}, raw_options="yuv420p"),
        )
        return graph.render()

    if fit == "cover":
        x_expr, y_expr = crop_offsets(
            target_width, target_height, target_width, target_height, focus
        )
        graph.chain([input_label], [output_label]).add(
            # 'increase' scales until the frame is covered, then the excess is cropped.
            Filter(
                "scale",
                {},
                raw_options=(
                    f"w={target_width}:h={target_height}:force_original_aspect_ratio=increase"
                ),
            ),
            Filter(
                "crop",
                {},
                raw_options=f"w={target_width}:h={target_height}:x='{x_expr}':y='{y_expr}'",
            ),
            Filter("setsar", {}, raw_options="1"),
            Filter("format", {}, raw_options="yuv420p"),
        )
        return graph.render()

    if fit == "contain":
        from ..ffmpeg.filters import escape_filter_value

        graph.chain([input_label], [output_label]).add(
            Filter(
                "scale",
                {},
                raw_options=(
                    f"w={target_width}:h={target_height}:force_original_aspect_ratio=decrease"
                ),
            ),
            Filter(
                "pad",
                {},
                raw_options=(
                    f"w={target_width}:h={target_height}:x=(ow-iw)/2:y=(oh-ih)/2:"
                    f"color={escape_filter_value(background_color)}"
                ),
            ),
            Filter("setsar", {}, raw_options="1"),
            Filter("format", {}, raw_options="yuv420p"),
        )
        return graph.render()

    if fit == "blur":
        # Background: a zoomed, cropped, blurred copy filling the frame.
        # Foreground: the whole picture fitted inside, centred on top.
        graph.chain([input_label], ["bgsrc", "fgsrc"]).add(Filter("split", {}, raw_options="2"))
        graph.chain(["bgsrc"], ["bg"]).add(
            Filter(
                "scale",
                {},
                raw_options=(
                    f"w={target_width}:h={target_height}:force_original_aspect_ratio=increase"
                ),
            ),
            Filter(
                "crop",
                {},
                raw_options=f"w={target_width}:h={target_height}",
            ),
            gblur_filter(blur_strength),
            Filter("setsar", {}, raw_options="1"),
        )
        graph.chain(["fgsrc"], ["fg"]).add(
            Filter(
                "scale",
                {},
                raw_options=(
                    f"w={target_width}:h={target_height}:force_original_aspect_ratio=decrease"
                ),
            ),
            Filter("setsar", {}, raw_options="1"),
        )
        graph.chain(["bg", "fg"], [output_label]).add(
            Filter("overlay", {}, raw_options="x=(W-w)/2:y=(H-h)/2"),
            Filter("format", {}, raw_options="yuv420p"),
        )
        return graph.render()

    raise InvalidParameterError(
        "Unknown fit mode.", fit=fit, allowed=["cover", "contain", "blur", "stretch"]
    )


class ResizeArgs(MediaJobArgs):
    """Arguments for changing a video's resolution or aspect ratio."""

    preset: str | None = Field(
        default=None,
        description=(
            "Named target, e.g. 'reel', 'tiktok', 'youtube_short', 'youtube_1080p', "
            "'youtube_4k', 'instagram_square', 'instagram_portrait'. Takes precedence "
            "over width/height and aspect_ratio."
        ),
    )
    width: int | None = Field(default=None, gt=0, le=16384)
    height: int | None = Field(default=None, gt=0, le=16384)
    aspect_ratio: str | None = Field(
        default=None,
        description=(
            "Target shape such as '9:16', '16:9', '1:1', '4:5'. Combined with a "
            "width or height to fix the size; on its own the source's pixel count "
            "is preserved."
        ),
    )
    fit: FitMode = Field(
        default="cover",
        description=(
            "How the picture fills a different shape. 'cover' crops to fill; "
            "'contain' letterboxes with solid bars; 'blur' letterboxes over a "
            "blurred zoomed copy of the video; 'stretch' distorts to fit."
        ),
    )
    focus: Focus = Field(
        default="center",
        description="Which part of the picture 'cover' keeps when it has to crop.",
    )
    background_color: str = Field(default="black", description="Bar colour for 'contain'.")
    blur_strength: float = Field(
        default=25.0, gt=0, le=200, description="Background blur amount for 'blur'."
    )
    encode: EncodeOptions = Field(default_factory=EncodeOptions)

    @model_validator(mode="after")
    def _check_target(self) -> ResizeArgs:
        if not (self.preset or self.width or self.height or self.aspect_ratio):
            raise ValueError("give a preset, a width and/or height, or an aspect_ratio")
        return self


class ResizeResult(StrictModel):
    """What a resize produced, and what it did to the picture."""

    output_path: str
    width: int
    height: int
    source_width: int
    source_height: int
    fit: str
    notes: list[str] = Field(default_factory=list)


@tool("resize_video", title="Resize / change aspect ratio", phase=5)
async def resize_video(args: ResizeArgs) -> JobSubmission:
    """Convert a video to a different resolution or aspect ratio.

    Use a named preset for the common targets — 'reel', 'tiktok',
    'youtube_short' and 'story' are all 1080x1920; 'youtube_1080p',
    'youtube_4k', 'instagram_square' and 'instagram_portrait' do what they say —
    or give an explicit width and height, or an aspect_ratio such as '9:16'.

    'fit' decides what happens to the picture that no longer fits when the shape
    changes:
      - cover   (default) zooms and crops to fill; nothing is letterboxed but
                the edges are lost. Use 'focus' to choose which part survives.
      - contain fits the whole picture inside and pads with solid bars.
      - blur    fits the whole picture inside over a blurred, zoomed copy of
                itself — the usual look for turning landscape footage vertical.
      - stretch distorts the picture to fit exactly.

    For a talking-head video where the subject must stay in frame, prefer
    track_and_crop, which follows the face instead of cropping to a fixed point.
    """
    resolve_io(args, operation="resized")
    if args.preset:
        resolve_target_size(
            source_width=1920,
            source_height=1080,
            preset=args.preset,
            width=None,
            height=None,
            aspect_ratio=None,
        )
    if args.aspect_ratio:
        parse_aspect_ratio(args.aspect_ratio)
    return queue("resize_video", args)


@handler("resize_video")
async def _run_resize(ctx: JobContext) -> JobOutcome:
    args = ResizeArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="resized", job_id=ctx.job_id)
    info = await probe(source, ctx.settings)
    video = info.primary_video
    if video is None or not video.width or not video.height:
        raise InvalidParameterError("resize_video needs a video stream.", path=str(source))

    target_width, target_height = resolve_target_size(
        source_width=video.width,
        source_height=video.height,
        preset=args.preset,
        width=args.width,
        height=args.height,
        aspect_ratio=args.aspect_ratio,
    )
    graph = build_resize_graph(
        target_width=target_width,
        target_height=target_height,
        fit=args.fit,
        focus=args.focus,
        background_color=args.background_color,
        blur_strength=args.blur_strength,
    )

    argv = ["-i", str(source), "-filter_complex", graph, "-map", "[vout]"]
    if info.has_audio:
        argv += ["-map", "0:a:0"]
    argv += output_args(args.encode, destination, has_video=True, has_audio=info.has_audio)
    argv += ["-y", str(destination)]

    result = await run_ffmpeg(
        argv,
        total_duration=info.duration,
        on_progress=ctx.make_progress_hook(2.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )

    source_ratio = video.width / video.height
    target_ratio = target_width / target_height
    notes = [
        f"Resized {video.width}x{video.height} to {target_width}x{target_height} "
        f"using fit='{args.fit}'."
    ]
    if abs(source_ratio - target_ratio) > 0.01:
        if args.fit == "cover":
            notes.append(
                "The aspect ratio changed, so the edges of the picture were cropped "
                f"(focus='{args.focus}'). Use fit='contain' or 'blur' to keep the whole frame."
            )
        elif args.fit == "stretch":
            notes.append("The aspect ratio changed and the picture was distorted to fit.")
        else:
            notes.append("The aspect ratio changed; the whole frame was kept and padded.")

    payload = ResizeResult(
        output_path=str(destination),
        width=target_width,
        height=target_height,
        source_width=video.width,
        source_height=video.height,
        fit=args.fit,
        notes=notes,
    )
    outcome = await finish_media_job(ctx, destination, notes=notes, command=result.command)
    outcome.result.update(payload.model_dump(mode="json"))
    return outcome


class ResolutionPresetsArgs(StrictModel):
    """No arguments; lists the built-in presets."""


class ResolutionPresets(StrictModel):
    """The named resolution targets this server understands."""

    presets: dict[str, list[int]] = Field(description="Preset name to [width, height].")
    fit_modes: list[str] = Field(default_factory=list)
    focus_points: list[str] = Field(default_factory=list)


@tool("list_resolution_presets", title="List resolution presets", phase=5, read_only=True)
async def list_resolution_presets(args: ResolutionPresetsArgs) -> ResolutionPresets:
    """List the named resolution presets, fit modes and focus points resize_video accepts.

    Answers immediately. Use it when you want to name a target platform rather
    than work out its pixel dimensions.
    """
    return ResolutionPresets(
        presets={name: [w, h] for name, (w, h) in sorted(RESOLUTION_PRESETS.items())},
        fit_modes=["cover", "contain", "blur", "stretch"],
        focus_points=sorted(_FOCUS_BIAS),
    )


__all__ = [
    "RESOLUTION_PRESETS",
    "ResizeArgs",
    "ResizeResult",
    "build_resize_graph",
    "crop_offsets",
    "resolve_target_size",
]
