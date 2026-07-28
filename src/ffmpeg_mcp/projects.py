"""Projects: a named scope for jobs and their outputs.

Two people — or two Claude sessions — editing different videos against the same
server previously shared one undifferentiated job list and one output directory.
Nothing broke, but everything was mixed together, and the only way to separate
them was to point each server at a different workspace, which meant restarting.

A project is a label carried by each job plus a directory its outputs land in.
The active project is per server process, which for stdio transport is exactly
one client, so it behaves like a session. Workers still share the queue, so
projects divide the bookkeeping without dividing the compute.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import Settings, get_settings
from .errors import InvalidParameterError

DEFAULT_PROJECT = "default"
MAX_NAME = 64

# A project name becomes a directory name, so it is rebuilt rather than trusted.
_VALID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_active: str = DEFAULT_PROJECT


def validate_project_name(name: str) -> str:
    """Check a project name is safe to use as a directory component."""
    cleaned = (name or "").strip()
    if not cleaned:
        raise InvalidParameterError("Project name cannot be empty.")
    if len(cleaned) > MAX_NAME:
        raise InvalidParameterError("Project name is too long.", name=cleaned, max_length=MAX_NAME)
    if not _VALID.match(cleaned):
        raise InvalidParameterError(
            "Project names may contain letters, digits, dot, dash and underscore, "
            "and must start with a letter or digit.",
            name=cleaned,
        )
    return cleaned


def active_project() -> str:
    """The project new jobs are filed under in this process."""
    return _active


def set_active_project(name: str) -> str:
    """Set the active project for this process. Returns the validated name."""
    global _active
    _active = validate_project_name(name)
    return _active


def reset_active_project() -> None:
    """Return to the default project. Used by tests."""
    global _active
    _active = DEFAULT_PROJECT


def project_dir(name: str | None = None, settings: Settings | None = None) -> Path:
    """Return (creating if needed) the directory a project's outputs live in."""
    settings = settings or get_settings()
    project = validate_project_name(name or active_project())
    path = settings.workspace / "projects" / project
    path.mkdir(parents=True, exist_ok=True)
    return path
