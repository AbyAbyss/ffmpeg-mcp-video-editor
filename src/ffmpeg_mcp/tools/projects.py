"""Project tools: keeping separate pieces of work separate.

The active project is per server process. Under stdio transport that is one
client, so switching projects scopes everything that session goes on to do —
where its outputs land, and which jobs it sees — without disturbing anyone else
using the same server.
"""

from __future__ import annotations

from pydantic import Field

from ..jobs.store import get_store
from ..models import StrictModel
from ..projects import (
    DEFAULT_PROJECT,
    active_project,
    project_dir,
    set_active_project,
    validate_project_name,
)
from .registry import tool


class SetProjectArgs(StrictModel):
    """Arguments for switching project."""

    name: str = Field(
        description="Project name: letters, digits, dot, dash and underscore.",
        min_length=1,
        max_length=64,
    )


class ProjectInfo(StrictModel):
    """The project this session is now working in."""

    name: str
    output_directory: str
    previous: str
    message: str


@tool("set_project", title="Switch project", phase=1)
async def set_project(args: SetProjectArgs) -> ProjectInfo:
    """Work under a named project, so parallel sessions do not mix together.

    Everything queued afterwards is filed under this name, and outputs without
    an explicit path land in the project's own directory. list_jobs then shows
    only this project by default, which keeps a busy shared queue readable.

    Two sessions editing different videos should each set their own project.
    The name is created on first use — there is nothing to set up beforehand.
    """
    previous = active_project()
    name = set_active_project(args.name)
    directory = project_dir(name)
    return ProjectInfo(
        name=name,
        output_directory=str(directory),
        previous=previous,
        message=(
            f"Now working in {name!r}. Jobs queued from here are filed under it, and "
            f"outputs without an explicit path go to {directory}."
        ),
    )


class NoArgs(StrictModel):
    """No arguments."""


class ProjectSummary(StrictModel):
    """One project and how much work it holds."""

    name: str
    jobs: int
    queued: int
    running: int
    done: int
    failed: int
    last_activity: float | None = None
    is_active: bool = False


class ProjectList(StrictModel):
    """Every project the job store has seen."""

    active: str
    projects: list[ProjectSummary] = Field(default_factory=list)


@tool("list_projects", title="List projects", phase=1, read_only=True)
async def list_projects(args: NoArgs) -> ProjectList:
    """List every project in the job store, with how much work each holds.

    Shows which one this session is currently working in, and how many jobs each
    project has queued, running, done and failed — a quick way to find the name
    of work started earlier, or in another session.
    """
    current = active_project()
    rows = get_store().list_projects()
    known = {row["name"] for row in rows}
    summaries = [
        ProjectSummary(
            name=row["name"],
            jobs=row["jobs"],
            queued=row["queued"] or 0,
            running=row["running"] or 0,
            done=row["done"] or 0,
            failed=row["failed"] or 0,
            last_activity=row["last_activity"],
            is_active=row["name"] == current,
        )
        for row in rows
    ]
    # A project switched into but not yet used has no rows, and would otherwise
    # vanish from its own listing.
    if current not in known:
        summaries.insert(
            0,
            ProjectSummary(
                name=current, jobs=0, queued=0, running=0, done=0, failed=0, is_active=True
            ),
        )
    return ProjectList(active=current, projects=summaries)


__all__ = ["DEFAULT_PROJECT", "ProjectList", "SetProjectArgs", "validate_project_name"]
