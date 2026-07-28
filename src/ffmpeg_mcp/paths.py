"""Path validation.

Every path that reaches a subprocess passes through here first. Resolution is
done with ``Path.resolve()`` so symlinks are followed *before* the allowlist
check, which is the part that actually stops a symlink in the workspace from
pointing at ``/etc``.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from .config import Settings, get_settings
from .errors import FileTooLargeError, InvalidPathError


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_within_roots(raw: str | Path, settings: Settings | None = None) -> Path:
    """Resolve ``raw`` and assert it lives under one of the allowed roots.

    Args:
        raw: A path as supplied by a client. May be relative to the workspace.
        settings: Optional settings override; defaults to the process settings.

    Returns:
        The fully resolved, symlink-free absolute path.

    Raises:
        InvalidPathError: If the path escapes every allowed root.
    """
    settings = settings or get_settings()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = settings.workspace / path
    # strict=False so we can validate output paths that do not exist yet; the
    # parent chain that *does* exist is still resolved, so symlink escapes are
    # caught.
    resolved = path.resolve(strict=False)

    roots = [r.resolve(strict=False) for r in settings.allowed_roots]
    if not any(_is_within(resolved, root) for root in roots):
        raise InvalidPathError(
            "Path resolves outside the configured allowed roots.",
            path=str(resolved),
            allowed_roots=[str(r) for r in roots],
        )
    return resolved


def validate_input_file(raw: str | Path, settings: Settings | None = None) -> Path:
    """Resolve an input path, and check it exists, is a file, and is not oversized."""
    settings = settings or get_settings()
    path = resolve_within_roots(raw, settings)
    if not path.exists():
        raise InvalidPathError("Input file does not exist.", path=str(path))
    if not path.is_file():
        raise InvalidPathError("Input path is not a regular file.", path=str(path))
    size = path.stat().st_size
    if size > settings.max_input_bytes:
        raise FileTooLargeError(
            "Input file exceeds the configured maximum size.",
            path=str(path),
            size_bytes=size,
            max_bytes=settings.max_input_bytes,
        )
    return path


def validate_output_path(
    raw: str | Path | None,
    *,
    suggested_name: str,
    job_id: str | None = None,
    settings: Settings | None = None,
) -> Path:
    """Resolve an output path, creating its parent directory.

    When ``raw`` is ``None`` the file is placed inside the job's own workspace
    directory, which is what the retention policy later cleans up.
    """
    settings = settings or get_settings()
    if raw is None:
        directory = settings.jobs_dir / (job_id or uuid.uuid4().hex)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / suggested_name
    path = resolve_within_roots(raw, settings)
    if path.exists() and path.is_dir():
        path = path / suggested_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def job_dir(job_id: str, settings: Settings | None = None) -> Path:
    """Return (and create) the workspace directory owned by a job."""
    settings = settings or get_settings()
    directory = settings.jobs_dir / job_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory
