"""Serving media files to the browser with byte-range support.

A ``<video>`` element cannot seek unless the server honours Range requests, so
this is implemented properly rather than falling back to whole-file responses.
Every path still goes through the same allowlist validation as the tools.
"""

from __future__ import annotations

import mimetypes
import re
from collections.abc import Iterator
from pathlib import Path

from starlette.responses import FileResponse, Response, StreamingResponse

from ..config import Settings
from ..paths import validate_input_file

_RANGE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK = 512 * 1024

_EXTRA_TYPES = {
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".m4a": "audio/mp4",
    ".opus": "audio/opus",
    ".srt": "text/plain; charset=utf-8",
    ".ass": "text/plain; charset=utf-8",
    ".vtt": "text/vtt",
}


def guess_media_type(path: Path) -> str:
    """Best-effort content type for a media file."""
    suffix = path.suffix.lower()
    if suffix in _EXTRA_TYPES:
        return _EXTRA_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Parse a ``Range: bytes=...`` header into an inclusive (start, end).

    Returns None when the header is absent or unsatisfiable, in which case the
    caller should send the whole file.
    """
    match = _RANGE.fullmatch(header.strip())
    if not match:
        return None
    raw_start, raw_end = match.group(1), match.group(2)
    if raw_start == "" and raw_end == "":
        return None
    if raw_start == "":
        # A suffix range: the last N bytes.
        length = int(raw_end)
        if length <= 0:
            return None
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    if start >= size or start > end:
        return None
    return start, min(end, size - 1)


def _iter_range(path: Path, start: int, end: int) -> Iterator[bytes]:
    remaining = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def serve_media(raw_path: str, range_header: str | None, settings: Settings) -> Response:
    """Serve a validated media file, honouring a Range request when present."""
    path = validate_input_file(raw_path, settings)
    size = path.stat().st_size
    media_type = guess_media_type(path)

    requested = parse_range(range_header, size) if range_header else None
    if requested is None:
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache"},
        )

    start, end = requested
    return StreamingResponse(
        _iter_range(path, start, end),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )
