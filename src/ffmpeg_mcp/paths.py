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

# Paths that belong to an assistant's own sandbox rather than to this machine.
# A model that attached a file to the conversation will reach for one of these,
# and the plain "outside the allowed roots" message sends it looking for a
# misconfiguration that does not exist.
_ASSISTANT_SANDBOX_PREFIXES = (
    "/mnt/user-data",
    "/mnt/outputs",
    "/mnt/skills",
    "/mnt/knowledge",
    "/home/claude",
    "/tmp/outputs",
)


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def looks_like_assistant_sandbox(path: Path) -> bool:
    """Whether a path looks like it belongs to an assistant's sandbox, not this host."""
    text = str(path)
    return any(
        text == prefix or text.startswith(prefix + "/") for prefix in _ASSISTANT_SANDBOX_PREFIXES
    )


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
        # Check the path as supplied as well as resolved: on macOS /tmp is a
        # symlink to /private/tmp and /home is an autofs mount, so resolution
        # rewrites exactly the prefixes being looked for.
        if looks_like_assistant_sandbox(resolved) or looks_like_assistant_sandbox(path):
            # The likeliest cause by far is a file attached to the conversation,
            # which lives in the assistant's sandbox and not on this machine.
            raise InvalidPathError(
                "That path is not on this machine. It looks like a file inside the "
                "assistant's own sandbox, such as an attachment uploaded to the "
                "conversation. This server runs on the user's computer and can only "
                "read files that exist there. Ask the user for the file's real path "
                "on their machine, for example '~/Downloads/clip.mov'.",
                path=str(resolved),
                allowed_roots=[str(r) for r in roots],
                reason="assistant_sandbox_path",
            )
        raise InvalidPathError(
            "Path resolves outside the configured allowed roots."
            + ("" if resolved.exists() else " It also does not exist on this machine."),
            path=str(resolved),
            allowed_roots=[str(r) for r in roots],
            reason="outside_allowed_roots",
            hint=(
                "Give a path under one of the allowed roots, or set "
                "FFMPEG_MCP_ALLOWED_ROOTS to include this location."
            ),
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
