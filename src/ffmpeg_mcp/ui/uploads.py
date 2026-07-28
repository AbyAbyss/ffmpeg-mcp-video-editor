"""Accepting file uploads from the browser.

Uploads are the one place where a name chosen entirely by the client becomes a
filesystem path, so the name is rebuilt from scratch rather than filtered: the
directory component is discarded, the stem is reduced to a known-safe character
set, and the extension must be one this server actually handles. The size cap is
enforced while streaming, so an oversized upload is abandoned rather than being
written out in full and rejected afterwards.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO, Protocol

from ..config import Settings
from ..errors import FileTooLargeError, InvalidParameterError

CHUNK = 1024 * 1024
MAX_STEM = 80

# Deliberately narrower than the browse allowlist: these are the formats a tool
# can actually read, and nothing here is executable or interpreted.
UPLOAD_SUFFIXES = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".opus",
    ".aac",
    ".ogg",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".srt",
    ".ass",
    ".vtt",
    ".cube",
    ".ttf",
    ".otf",
}

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class UploadSource(Protocol):
    """The part of Starlette's UploadFile this module needs."""

    filename: str | None
    file: BinaryIO


def safe_upload_name(raw: str | None) -> str:
    """Rebuild a client-supplied filename into one that is safe to join.

    Any directory component is dropped, so ``../../etc/passwd`` becomes
    ``passwd``; everything outside ``[A-Za-z0-9._-]`` is replaced; and the
    extension must be one of :data:`UPLOAD_SUFFIXES`.
    """
    candidate = (raw or "").replace("\x00", "").strip()
    # PurePosixPath and Path disagree about separators across platforms, so
    # take the last component under either convention.
    candidate = candidate.replace("\\", "/").split("/")[-1]
    if not candidate:
        raise InvalidParameterError("Upload has no filename.")

    suffix = Path(candidate).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise InvalidParameterError(
            "Unsupported upload type.",
            filename=candidate,
            supported=sorted(UPLOAD_SUFFIXES),
        )

    stem = _UNSAFE.sub("_", Path(candidate).stem).strip("._")[:MAX_STEM]
    return f"{stem or 'upload'}{suffix}"


def unique_path(directory: Path, name: str) -> Path:
    """Return a path in ``directory`` that does not collide with an existing file."""
    target = directory / name
    if not target.exists():
        return target
    stem, suffix = Path(name).stem, Path(name).suffix
    for index in range(1, 10_000):
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise InvalidParameterError("Too many files with that name.", filename=name)


def store_upload(upload: UploadSource, settings: Settings) -> Path:
    """Stream one upload into the workspace's uploads directory.

    Returns:
        The path the file was written to.

    Raises:
        InvalidParameterError: The filename or extension is unacceptable.
        FileTooLargeError: The upload exceeded the configured maximum size.
    """
    settings.ensure_dirs()
    destination = unique_path(settings.uploads_dir, safe_upload_name(upload.filename))

    written = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = upload.file.read(CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > settings.max_input_bytes:
                    raise FileTooLargeError(
                        "Upload exceeds the configured maximum size.",
                        filename=destination.name,
                        max_bytes=settings.max_input_bytes,
                    )
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    if written == 0:
        destination.unlink(missing_ok=True)
        raise InvalidParameterError("Upload is empty.", filename=destination.name)
    return destination
