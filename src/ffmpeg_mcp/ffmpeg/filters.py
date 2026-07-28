"""Filter graph construction and escaping.

Every filter string in this project is built here. Nothing concatenates
user-supplied text into a graph directly, because ffmpeg's filter syntax gives
structural meaning to ``: , ; [ ] ' \\`` and a caption containing a colon would
otherwise silently corrupt the whole graph.

ffmpeg unescapes filter descriptions in two passes (see "Notes on filtergraph
escaping" in the ffmpeg-filters manual):

* level 1 -- inside a single option value, where ``:`` separates options,
* level 2 -- across the whole filter description, where ``[ ] , ;`` are
  structural.

:func:`escape_filter_value` applies both, in that order. There is no level 3
here because arguments are passed to the subprocess as a list, never a shell
string.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import InvalidParameterError

_LEVEL2_SPECIALS = "[],;"


def escape_filter_value(value: str) -> str:
    """Escape an arbitrary string for use as a filter option value.

    Args:
        value: Raw text, potentially containing any character.

    Returns:
        The value with both levels of ffmpeg filter escaping applied.
    """
    # Level 1: within one option value.
    out = value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    # Level 2: across the filter description. Re-escapes the backslashes added
    # above, which is exactly what ffmpeg's two unescape passes expect.
    out = out.replace("\\", "\\\\").replace("'", "\\'")
    for char in _LEVEL2_SPECIALS:
        out = out.replace(char, "\\" + char)
    return out


def escape_path_for_filter(path: str | Path) -> str:
    """Escape a filesystem path for a filter option such as ``subtitles=f=...``.

    Windows paths need their drive colon and backslashes escaped, which the
    generic value escaper already handles.
    """
    return escape_filter_value(str(path))


def _format_number(value: float) -> str:
    if isinstance(value, bool):  # pragma: no cover - guard against silent bool coercion
        raise InvalidParameterError("Boolean passed where a number was expected.")
    if not math.isfinite(value):
        raise InvalidParameterError("Filter parameters must be finite numbers.", value=str(value))
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_option_value(value: str | int | float | bool) -> str:
    """Render one option value, escaping strings and normalising numbers."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return _format_number(float(value))
    return escape_filter_value(value)


@dataclass(frozen=True)
class Filter:
    """A single filter with its options, e.g. ``scale=w=1280:h=720``."""

    name: str
    options: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    raw_options: str | None = field(
        default=None,
        compare=False,
        metadata={"doc": "Pre-rendered option string, for expressions built elsewhere."},
    )

    def render(self) -> str:
        """Render the filter as ``name=k=v:k=v``."""
        if self.raw_options is not None:
            return f"{self.name}={self.raw_options}" if self.raw_options else self.name
        if not self.options:
            return self.name
        rendered = ":".join(
            f"{key}={format_option_value(value)}" for key, value in self.options.items()
        )
        return f"{self.name}={rendered}"


@dataclass
class FilterChain:
    """An ordered list of filters applied to one labelled stream."""

    filters: list[Filter] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def add(self, *filters: Filter) -> FilterChain:
        """Append filters to the chain and return self, for fluent building."""
        self.filters.extend(filters)
        return self

    def extend(self, filters: Iterable[Filter]) -> FilterChain:
        """Append an iterable of filters."""
        self.filters.extend(filters)
        return self

    def render(self) -> str:
        """Render as ``[in]filter,filter[out]``."""
        body = ",".join(f.render() for f in self.filters) or "null"
        prefix = "".join(f"[{label}]" for label in self.inputs)
        suffix = "".join(f"[{label}]" for label in self.outputs)
        return f"{prefix}{body}{suffix}"

    def __bool__(self) -> bool:
        return bool(self.filters)


@dataclass
class FilterGraph:
    """A complete ``-filter_complex`` graph: several chains joined by ``;``."""

    chains: list[FilterChain] = field(default_factory=list)

    def chain(self, inputs: Sequence[str] = (), outputs: Sequence[str] = ()) -> FilterChain:
        """Create, register, and return a new chain."""
        new = FilterChain(inputs=list(inputs), outputs=list(outputs))
        self.chains.append(new)
        return new

    def render(self) -> str:
        """Render the whole graph."""
        return ";".join(c.render() for c in self.chains)

    def __bool__(self) -> bool:
        return any(self.chains)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def scale_filter(
    width: int | None,
    height: int | None,
    *,
    keep_aspect: bool = True,
    force_even: bool = True,
) -> Filter:
    """Build a ``scale`` filter.

    A ``None`` dimension is derived from the other one. ``force_even`` rounds the
    derived side to a multiple of two, which yuv420p encoders require.
    """
    if width is None and height is None:
        raise InvalidParameterError("scale requires at least one of width or height.")
    divisor = "-2" if force_even else "-1"
    w = str(width) if width is not None else divisor
    h = str(height) if height is not None else divisor
    if width is not None and height is not None and keep_aspect:
        # Fit inside the box, then pad is applied separately if requested.
        return Filter("scale", {}, raw_options=f"w={w}:h={h}:force_original_aspect_ratio=decrease")
    return Filter("scale", {}, raw_options=f"w={w}:h={h}")


def pad_to_filter(width: int, height: int, color: str = "black") -> Filter:
    """Pad a scaled image out to an exact box, centring it."""
    return Filter(
        "pad",
        {},
        raw_options=f"w={width}:h={height}:x=(ow-iw)/2:y=(oh-ih)/2:color={escape_filter_value(color)}",
    )


def crop_filter(x: int, y: int, width: int, height: int) -> Filter:
    """Build a static ``crop`` filter."""
    return Filter("crop", {"w": width, "h": height, "x": x, "y": y})


def crop_expr_filter(width: int, height: int, x_expr: str, y_expr: str) -> Filter:
    """Build a ``crop`` whose origin is an ffmpeg expression (used by face tracking)."""
    return Filter("crop", {}, raw_options=f"w={width}:h={height}:x='{x_expr}':y='{y_expr}'")


def rotate_filter(degrees: int) -> list[Filter]:
    """Rotate by a multiple of 90 degrees using lossless transpose steps."""
    normalised = degrees % 360
    if normalised == 0:
        return []
    if normalised == 90:
        return [Filter("transpose", {"dir": 1})]
    if normalised == 180:
        return [Filter("transpose", {"dir": 1}), Filter("transpose", {"dir": 1})]
    if normalised == 270:
        return [Filter("transpose", {"dir": 2})]
    raise InvalidParameterError("Rotation must be a multiple of 90 degrees.", degrees=degrees)


def flip_filters(horizontal: bool, vertical: bool) -> list[Filter]:
    """Mirror filters for horizontal and/or vertical flips."""
    out: list[Filter] = []
    if horizontal:
        out.append(Filter("hflip"))
    if vertical:
        out.append(Filter("vflip"))
    return out


def fps_filter(fps: float) -> Filter:
    """Resample to a constant frame rate."""
    if fps <= 0:
        raise InvalidParameterError("fps must be positive.", fps=fps)
    return Filter("fps", {"fps": fps})


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #


def atempo_chain(factor: float) -> list[Filter]:
    """Decompose a tempo change into ``atempo`` steps each within 0.5x-2.0x.

    Older ffmpeg builds reject a single ``atempo`` outside that range, so a 4x
    speed-up becomes ``atempo=2.0,atempo=2.0``.
    """
    if factor <= 0:
        raise InvalidParameterError("Speed factor must be positive.", factor=factor)
    if math.isclose(factor, 1.0, rel_tol=1e-9):
        return []
    steps: list[float] = []
    remaining = factor
    while remaining > 2.0:
        steps.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        steps.append(0.5)
        remaining /= 0.5
    if not math.isclose(remaining, 1.0, rel_tol=1e-9):
        steps.append(remaining)
    return [Filter("atempo", {"tempo": round(step, 6)}) for step in steps]


def setpts_filter(speed: float) -> Filter:
    """Scale video presentation timestamps for a speed change."""
    if speed <= 0:
        raise InvalidParameterError("Speed factor must be positive.", speed=speed)
    return Filter("setpts", {}, raw_options=f"{_format_number(1.0 / speed)}*PTS")


def asetpts_reset() -> Filter:
    """Reset audio timestamps after concatenation or tempo changes."""
    return Filter("asetpts", {}, raw_options="N/SR/TB")


# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #


def eq_filter(
    *,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
) -> Filter | None:
    """Build an ``eq`` filter, or None if every parameter is at its neutral value."""
    if (
        math.isclose(brightness, 0.0)
        and math.isclose(contrast, 1.0)
        and math.isclose(saturation, 1.0)
        and math.isclose(gamma, 1.0)
    ):
        return None
    return Filter(
        "eq",
        {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "gamma": gamma,
        },
    )


def temperature_filter(kelvin_shift: float) -> Filter | None:
    """Approximate a colour-temperature shift with per-channel gain.

    ``kelvin_shift`` runs -100 (cooler/bluer) to +100 (warmer/oranger); it is a
    relative artistic control, not an absolute white point.
    """
    if math.isclose(kelvin_shift, 0.0):
        return None
    if not -100.0 <= kelvin_shift <= 100.0:
        raise InvalidParameterError(
            "temperature must be between -100 and 100.", temperature=kelvin_shift
        )
    amount = kelvin_shift / 100.0 * 0.3
    red = 1.0 + amount
    blue = 1.0 - amount
    return Filter(
        "colorchannelmixer",
        {"rr": round(red, 4), "gg": 1.0, "bb": round(blue, 4)},
    )


def lut3d_filter(path: str | Path, interp: str = "tetrahedral") -> Filter:
    """Apply a 3D LUT from a ``.cube`` file."""
    allowed = {"nearest", "trilinear", "tetrahedral", "pyramid", "prism"}
    if interp not in allowed:
        raise InvalidParameterError(
            "Unsupported LUT interpolation mode.", interp=interp, allowed=sorted(allowed)
        )
    return Filter("lut3d", {}, raw_options=f"file={escape_path_for_filter(path)}:interp={interp}")


def curves_filter(
    *,
    preset: str | None = None,
    master: Sequence[tuple[float, float]] | None = None,
    red: Sequence[tuple[float, float]] | None = None,
    green: Sequence[tuple[float, float]] | None = None,
    blue: Sequence[tuple[float, float]] | None = None,
) -> Filter:
    """Build a ``curves`` filter from a preset name or explicit control points."""
    if preset:
        return Filter("curves", {"preset": preset})
    parts: list[str] = []
    for key, points in (("m", master), ("r", red), ("g", green), ("b", blue)):
        if not points:
            continue
        parts.append(f"{key}={format_curve_points(points)}")
    if not parts:
        raise InvalidParameterError("curves requires a preset or at least one set of points.")
    return Filter("curves", {}, raw_options=":".join(parts))


def format_curve_points(points: Sequence[tuple[float, float]]) -> str:
    """Render control points as ffmpeg's ``x0/y0 x1/y1`` curve syntax.

    Points are sorted by x and validated to sit inside the 0..1 unit square,
    because ffmpeg silently ignores a malformed curve rather than erroring.
    """
    if len(points) < 2:
        raise InvalidParameterError("A curve needs at least two control points.")
    for x, y in points:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise InvalidParameterError("Curve control points must lie within 0..1.", point=[x, y])
    ordered = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in ordered]
    if len(set(xs)) != len(xs):
        raise InvalidParameterError("Curve control points must have distinct x values.")
    return " ".join(f"{_format_number(x)}/{_format_number(y)}" for x, y in ordered)


# --------------------------------------------------------------------------- #
# Text and subtitles
# --------------------------------------------------------------------------- #


def subtitles_filter(
    path: str | Path,
    *,
    force_style: Mapping[str, str | int | float] | None = None,
    fonts_dir: str | Path | None = None,
) -> Filter:
    """Burn a subtitle file into the video.

    ``force_style`` is an ASS style override; its ``,`` separators are escaped
    along with everything else, so a font name containing a comma cannot break
    out of the option.
    """
    options = f"f={escape_path_for_filter(path)}"
    if fonts_dir is not None:
        options += f":fontsdir={escape_path_for_filter(fonts_dir)}"
    if force_style:
        style = ",".join(f"{k}={v}" for k, v in force_style.items())
        options += f":force_style={escape_filter_value(style)}"
    return Filter("subtitles", {}, raw_options=options)


def drawtext_filter(
    *,
    textfile: str | Path | None = None,
    text: str | None = None,
    fontfile: str | Path | None = None,
    fontsize: int = 48,
    fontcolor: str = "white",
    x: str = "(w-text_w)/2",
    y: str = "h-text_h-40",
    box: bool = False,
    boxcolor: str = "black@0.5",
    boxborderw: int = 12,
    borderw: int = 0,
    bordercolor: str = "black",
    shadowx: int = 0,
    shadowy: int = 0,
    shadowcolor: str = "black@0.6",
    alpha_expr: str | None = None,
    enable_expr: str | None = None,
) -> Filter:
    """Build a ``drawtext`` filter.

    Prefer ``textfile`` over ``text``: it avoids embedding arbitrary user text in
    the graph at all. Either way ``expansion=none`` is set, so ``%{...}`` and
    ``{}`` sequences in the caption are drawn literally instead of being
    evaluated as ffmpeg expressions.

    ``x``, ``y``, ``alpha_expr`` and ``enable_expr`` are ffmpeg expressions and
    are deliberately *not* escaped; they are built by this module, never taken
    from a client verbatim.
    """
    if (textfile is None) == (text is None):
        raise InvalidParameterError("drawtext requires exactly one of textfile or text.")
    parts: list[str] = []
    if textfile is not None:
        parts.append(f"textfile={escape_path_for_filter(textfile)}")
    else:
        assert text is not None
        parts.append(f"text={escape_filter_value(text)}")
    parts.append("expansion=none")
    if fontfile is not None:
        parts.append(f"fontfile={escape_path_for_filter(fontfile)}")
    parts.append(f"fontsize={int(fontsize)}")
    parts.append(f"fontcolor={escape_filter_value(fontcolor)}")
    # x/y must be quoted: an animated coordinate is an if(...) expression whose
    # commas would otherwise be read as filter separators and break the graph.
    parts.append(f"x='{x}'")
    parts.append(f"y='{y}'")
    if box:
        parts.append("box=1")
        parts.append(f"boxcolor={escape_filter_value(boxcolor)}")
        parts.append(f"boxborderw={int(boxborderw)}")
    if borderw:
        parts.append(f"borderw={int(borderw)}")
        parts.append(f"bordercolor={escape_filter_value(bordercolor)}")
    if shadowx or shadowy:
        parts.append(f"shadowx={int(shadowx)}")
        parts.append(f"shadowy={int(shadowy)}")
        parts.append(f"shadowcolor={escape_filter_value(shadowcolor)}")
    if alpha_expr is not None:
        parts.append(f"alpha='{alpha_expr}'")
    if enable_expr is not None:
        parts.append(f"enable='{enable_expr}'")
    return Filter("drawtext", {}, raw_options=":".join(parts))


def between_expr(start: float, end: float) -> str:
    """An ffmpeg ``enable`` expression restricting a filter to a time window."""
    if end <= start:
        raise InvalidParameterError("end must be greater than start.", start=start, end=end)
    return f"between(t,{_format_number(start)},{_format_number(end)})"


def fade_alpha_expr(start: float, end: float, fade_in: float, fade_out: float) -> str:
    """An alpha expression that fades a text overlay in and out within its window.

    Built as nested ``if`` expressions rather than hand-written per call, so the
    timings always stay consistent with the ``enable`` window.
    """
    if end <= start:
        raise InvalidParameterError("end must be greater than start.", start=start, end=end)
    duration = end - start
    fade_in = max(0.0, min(fade_in, duration / 2))
    fade_out = max(0.0, min(fade_out, duration / 2))
    s, e = _format_number(start), _format_number(end)
    terms = []
    if fade_in > 0:
        terms.append(
            f"if(lt(t,{_format_number(start + fade_in)}),(t-{s})/{_format_number(fade_in)}"
        )
    if fade_out > 0:
        terms.append(
            f"if(gt(t,{_format_number(end - fade_out)}),({e}-t)/{_format_number(fade_out)}"
        )
    if not terms:
        return "1"
    expr = ",".join(terms) + ",1" + ")" * len(terms)
    return expr


def slide_x_expr(start: float, target_x: str, distance: int, slide_seconds: float) -> str:
    """An x expression that slides an overlay in from the left over ``slide_seconds``."""
    if slide_seconds <= 0:
        return target_x
    s = _format_number(start)
    d = _format_number(slide_seconds)
    return (
        f"if(lt(t,{_format_number(start + slide_seconds)}),"
        f"({target_x})-{int(distance)}*(1-(t-{s})/{d}),{target_x})"
    )


# --------------------------------------------------------------------------- #
# Compositing, transitions, audio
# --------------------------------------------------------------------------- #


def overlay_filter(
    x: str | int, y: str | int, *, enable_expr: str | None = None, eof_action: str = "pass"
) -> Filter:
    """Composite one video stream on top of another."""
    parts = [f"x={x}", f"y={y}", f"eof_action={eof_action}"]
    if enable_expr is not None:
        parts.append(f"enable='{enable_expr}'")
    return Filter("overlay", {}, raw_options=":".join(parts))


def chromakey_filter(color: str, similarity: float, blend: float) -> Filter:
    """Key out a colour so the layer below shows through."""
    if not 0.0 < similarity <= 1.0:
        raise InvalidParameterError("similarity must be in (0, 1].", similarity=similarity)
    if not 0.0 <= blend <= 1.0:
        raise InvalidParameterError("blend must be in [0, 1].", blend=blend)
    return Filter(
        "chromakey",
        {"color": color, "similarity": similarity, "blend": blend},
    )


def xfade_filter(transition: str, duration: float, offset: float) -> Filter:
    """Cross-fade or wipe between two video streams."""
    if duration <= 0:
        raise InvalidParameterError("Transition duration must be positive.", duration=duration)
    if offset < 0:
        raise InvalidParameterError("Transition offset cannot be negative.", offset=offset)
    return Filter(
        "xfade",
        {"transition": transition, "duration": round(duration, 6), "offset": round(offset, 6)},
    )


def acrossfade_filter(duration: float, curve: str = "tri") -> Filter:
    """Cross-fade two audio streams."""
    return Filter("acrossfade", {"d": round(duration, 6), "c1": curve, "c2": curve})


def afade_filter(direction: str, start: float, duration: float) -> Filter:
    """Fade audio in or out."""
    if direction not in {"in", "out"}:
        raise InvalidParameterError("Fade direction must be 'in' or 'out'.", direction=direction)
    if duration <= 0:
        raise InvalidParameterError("Fade duration must be positive.", duration=duration)
    return Filter(
        "afade",
        {"t": direction, "st": round(start, 6), "d": round(duration, 6)},
    )


def fade_filter(direction: str, start: float, duration: float) -> Filter:
    """Fade video to or from black."""
    if direction not in {"in", "out"}:
        raise InvalidParameterError("Fade direction must be 'in' or 'out'.", direction=direction)
    return Filter(
        "fade",
        {"t": direction, "st": round(start, 6), "d": round(duration, 6)},
    )


def loudnorm_filter(target_lufs: float, true_peak: float, lra: float) -> Filter:
    """EBU R128 loudness normalisation."""
    if not -70.0 <= target_lufs <= -5.0:
        raise InvalidParameterError(
            "Target loudness must be between -70 and -5 LUFS.", target_lufs=target_lufs
        )
    return Filter("loudnorm", {"I": target_lufs, "TP": true_peak, "LRA": lra})


def volume_filter(gain_db: float) -> Filter:
    """Apply a fixed gain in decibels."""
    return Filter("volume", {}, raw_options=f"{_format_number(gain_db)}dB")


def sidechain_duck_filter(threshold: float, ratio: float, attack: float, release: float) -> Filter:
    """Duck one stream under another using ``sidechaincompress``."""
    return Filter(
        "sidechaincompress",
        {
            "threshold": threshold,
            "ratio": ratio,
            "attack": attack,
            "release": release,
            "makeup": 1,
        },
    )


def boxblur_filter(strength: int) -> Filter:
    """Blur, with strength expressed as the box radius in pixels."""
    if strength < 1:
        raise InvalidParameterError("Blur strength must be at least 1.", strength=strength)
    return Filter(
        "boxblur", {}, raw_options=f"luma_radius={strength}:chroma_radius={strength}:luma_power=2"
    )


def gblur_filter(sigma: float) -> Filter:
    """Gaussian blur with the given sigma."""
    if sigma <= 0:
        raise InvalidParameterError("Blur sigma must be positive.", sigma=sigma)
    return Filter("gblur", {"sigma": round(sigma, 4)})


def format_filter(pix_fmt: str) -> Filter:
    """Force a pixel format mid-graph."""
    return Filter("format", {}, raw_options=escape_filter_value(pix_fmt))
