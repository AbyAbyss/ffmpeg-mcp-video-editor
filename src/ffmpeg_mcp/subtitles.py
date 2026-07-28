"""Subtitle file generation and LUT parsing. Pure logic, no ffmpeg involved.

Kept separate from the tool layer so phase 3's transcription pipeline can reuse
:func:`build_srt` without going through a job, and so all of it is unit testable
with no binary present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .errors import InvalidParameterError
from .models import Segment

_MAX_CUBE_SIZE = 256


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp (``HH:MM:SS,mmm``).

    Negative inputs clamp to zero; SRT has no way to express them and ffmpeg
    silently drops a cue with a malformed start time.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_ass_timestamp(seconds: float) -> str:
    """Format seconds as an ASS timestamp (``H:MM:SS.cc``)."""
    if seconds < 0:
        seconds = 0.0
    total_cs = round(seconds * 100)
    hours, remainder = divmod(total_cs, 360_000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def normalise_segments(segments: list[Segment], *, min_duration: float = 0.05) -> list[Segment]:
    """Sort segments, drop empty text, and stop cues overlapping the next one.

    Overlapping cues make a burned-in caption track flicker between two lines,
    so a cue that runs past the next one's start is truncated rather than left
    to fight with it.
    """
    kept = [s for s in segments if s.text.strip()]
    kept.sort(key=lambda s: (s.start, s.end))
    result: list[Segment] = []
    for index, segment in enumerate(kept):
        end = segment.end
        if index + 1 < len(kept):
            end = min(end, kept[index + 1].start)
        if end - segment.start < min_duration:
            end = segment.start + min_duration
        result.append(Segment(start=segment.start, end=end, text=segment.text.strip()))
    return result


def wrap_caption_text(text: str, max_chars_per_line: int, max_lines: int = 2) -> str:
    """Wrap a caption onto at most ``max_lines`` lines at word boundaries.

    Long single-line captions run off the side of the frame, and ffmpeg does no
    wrapping of its own for SRT.
    """
    if max_chars_per_line <= 0:
        raise InvalidParameterError("max_chars_per_line must be positive.")
    words = text.split()
    if not words:
        return ""
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars_per_line or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        # Fold the overflow back into the last allowed line rather than dropping it.
        head = lines[: max_lines - 1]
        tail = " ".join(lines[max_lines - 1 :])
        lines = [*head, tail]
    return "\n".join(lines)


def build_srt(
    segments: list[Segment],
    *,
    max_chars_per_line: int | None = None,
    max_lines: int = 2,
) -> str:
    """Render timed segments as a valid SRT document.

    Args:
        segments: Timed text, in any order.
        max_chars_per_line: Wrap each cue to this width; None leaves text alone.
        max_lines: Maximum lines per cue when wrapping.

    Returns:
        The SRT document, ending in a trailing newline.
    """
    prepared = normalise_segments(segments)
    blocks: list[str] = []
    for index, segment in enumerate(prepared, start=1):
        text = segment.text
        if max_chars_per_line:
            text = wrap_caption_text(text, max_chars_per_line, max_lines)
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}\n"
            f"{text}\n"
        )
    return "\n".join(blocks)


_SRT_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def parse_srt(text: str) -> list[Segment]:
    """Parse an SRT document back into segments. Used to validate and to re-time."""
    segments: list[Segment] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timing_index = next((i for i, line in enumerate(lines) if _SRT_TIME.search(line)), None)
        if timing_index is None:
            continue
        match = _SRT_TIME.search(lines[timing_index])
        assert match is not None
        values = [int(g) for g in match.groups()]
        start = values[0] * 3600 + values[1] * 60 + values[2] + values[3] / 1000
        end = values[4] * 3600 + values[5] * 60 + values[6] + values[7] / 1000
        body = "\n".join(lines[timing_index + 1 :]).strip()
        if body:
            segments.append(Segment(start=start, end=max(end, start), text=body))
    return segments


# --------------------------------------------------------------------------- #
# LUT parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CubeInfo:
    """What a validated ``.cube`` file contains."""

    title: str | None
    dimensions: int  # 1 or 3
    size: int
    entries: int
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]


def parse_cube(text: str) -> CubeInfo:
    """Validate an Adobe ``.cube`` LUT and report its shape.

    ffmpeg's ``lut3d`` filter fails with an unhelpful message on a malformed
    cube, so the file is checked here first: declared size present and sane, and
    the actual number of data rows matching what the size implies.
    """
    title: str | None = None
    size: int | None = None
    dimensions = 0
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    rows = 0

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("TITLE"):
            title = line[5:].strip().strip('"')
        elif upper.startswith("LUT_3D_SIZE"):
            size, dimensions = _parse_size(line, 3)
        elif upper.startswith("LUT_1D_SIZE"):
            size, dimensions = _parse_size(line, 1)
        elif upper.startswith("DOMAIN_MIN"):
            domain_min = _parse_triple(line)
        elif upper.startswith("DOMAIN_MAX"):
            domain_max = _parse_triple(line)
        else:
            parts = line.split()
            if len(parts) != 3:
                raise InvalidParameterError(
                    "LUT data rows must contain exactly three float values.", line=line
                )
            try:
                [float(p) for p in parts]
            except ValueError as exc:
                raise InvalidParameterError("LUT data row is not numeric.", line=line) from exc
            rows += 1

    if size is None or dimensions == 0:
        raise InvalidParameterError("LUT file declares no LUT_3D_SIZE or LUT_1D_SIZE.")
    if not 2 <= size <= _MAX_CUBE_SIZE:
        raise InvalidParameterError("LUT size out of range.", size=size)
    expected = size**3 if dimensions == 3 else size
    if rows != expected:
        raise InvalidParameterError(
            "LUT data row count does not match the declared size.",
            declared_size=size,
            expected_rows=expected,
            actual_rows=rows,
        )
    return CubeInfo(
        title=title,
        dimensions=dimensions,
        size=size,
        entries=rows,
        domain_min=domain_min,
        domain_max=domain_max,
    )


def _parse_size(line: str, dimensions: int) -> tuple[int, int]:
    parts = line.split()
    if len(parts) != 2:
        raise InvalidParameterError("Malformed LUT size declaration.", line=line)
    try:
        return int(parts[1]), dimensions
    except ValueError as exc:
        raise InvalidParameterError("LUT size is not an integer.", line=line) from exc


def _parse_triple(line: str) -> tuple[float, float, float]:
    parts = line.split()
    if len(parts) != 4:
        raise InvalidParameterError("Malformed LUT domain declaration.", line=line)
    try:
        return (float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError as exc:
        raise InvalidParameterError("LUT domain values are not numeric.", line=line) from exc


def validate_cube_file(path: Path, max_bytes: int = 128 * 1024 * 1024) -> CubeInfo:
    """Read and validate a ``.cube`` file from disk."""
    if path.suffix.lower() != ".cube":
        raise InvalidParameterError("LUT file must have a .cube extension.", path=str(path))
    if path.stat().st_size > max_bytes:
        raise InvalidParameterError("LUT file is implausibly large.", path=str(path))
    return parse_cube(path.read_text(encoding="utf-8", errors="replace"))
