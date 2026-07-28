"""The local web UI's FastAPI application.

The UI is a second front door onto the same backend, not a second
implementation: it reads and writes the same SQLite job store, runs the same
job handlers through its own worker pool, and calls the same tool functions the
MCP server exposes. Anything it can do, an MCP client can do too.

Trust boundary: the tools can read and write anywhere inside the configured
allowed roots, so binding anywhere other than loopback requires a token.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from ..config import Settings, get_settings
from ..errors import FFmpegMCPError
from ..jobs.store import JobStore, get_store
from ..jobs.worker import WorkerPool
from ..models import JobStatus
from ..paths import resolve_within_roots
from ..tools.registry import ToolSpec, get_tool, load_all_tools
from .media import serve_media
from .uploads import UPLOAD_SUFFIXES, store_upload

log = logging.getLogger("ffmpeg_mcp.ui")

STATIC_DIR = Path(__file__).parent / "static"
MEDIA_SUFFIXES = {
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
}


class UiState:
    """Process-wide state for the UI server."""

    def __init__(self, settings: Settings, token: str | None) -> None:
        self.settings = settings
        self.token = token
        self.store: JobStore = get_store(settings)
        self.pool = WorkerPool(settings, self.store)
        self.tools: list[ToolSpec] = load_all_tools()


_state: UiState | None = None


def state() -> UiState:
    """Return the UI state, which is set up during application startup."""
    if _state is None:  # pragma: no cover - only reachable before startup
        raise RuntimeError("UI state is not initialised")
    return _state


def require_token(request: Request) -> None:
    """Reject unauthenticated requests when a token is configured."""
    current = state()
    if current.token is None:
        return
    supplied = request.headers.get("x-auth-token") or request.query_params.get("token")
    if not supplied or not secrets.compare_digest(supplied, current.token):
        raise HTTPException(status_code=401, detail="Invalid or missing token.")


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class JobView(BaseModel):
    """A job as the UI displays it."""

    job_id: str
    tool: str
    status: JobStatus
    progress: float
    message: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    outputs: list[str] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    command: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    project: str | None = None


class JobsResponse(BaseModel):
    """A page of jobs plus counts for the scope being listed."""

    jobs: list[JobView]
    counts: dict[str, int]
    total: int = 0
    offset: int = 0
    limit: int = 100
    has_more: bool = False


class ProjectRow(BaseModel):
    """One project and how much work it holds."""

    name: str
    jobs: int
    queued: int = 0
    running: int = 0
    done: int = 0
    failed: int = 0
    last_activity: float | None = None


class ToolView(BaseModel):
    """A tool's schema, for the UI's generated forms."""

    name: str
    title: str
    description: str
    phase: int
    read_only: bool
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class FileView(BaseModel):
    """One file the UI can preview or feed into a tool."""

    path: str
    name: str
    size_bytes: int
    modified_at: float
    kind: str


class WorkspaceInfo(BaseModel):
    """Where the UI is pointed and what it can reach."""

    workspace: str
    allowed_roots: list[str]
    jobs_dir: str
    uploads_dir: str


def _to_view(record: Any) -> JobView:
    return JobView(
        job_id=record.job_id,
        tool=record.tool,
        status=record.status,
        progress=record.progress,
        message=record.message,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        outputs=record.outputs,
        result=record.result,
        error=record.error.model_dump() if record.error else None,
        command=record.command,
        params=record.params,
        project=record.project or "default",
    )


def _classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        return "video"
    if suffix in {".mp3", ".wav", ".m4a", ".flac", ".opus", ".aac", ".ogg"}:
        return "audio"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return "image"
    if suffix in {".srt", ".ass", ".vtt"}:
        return "subtitle"
    return "other"


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


def create_app(settings: Settings | None = None, token: str | None = None) -> FastAPI:
    """Build the FastAPI application."""
    global _state
    resolved = settings or get_settings()
    resolved.ensure_dirs()
    _state = UiState(resolved, token)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        current = state()
        current.store.reclaim_stale()
        await current.pool.start()
        log.info("UI worker pool started; workspace %s", current.settings.workspace)
        try:
            yield
        finally:
            await current.pool.stop()

    app = FastAPI(title="ffmpeg-mcp", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(FFmpegMCPError)
    async def _domain_error(_request: Request, exc: FFmpegMCPError) -> JSONResponse:
        return JSONResponse(status_code=400, content=exc.to_dict())

    # -- workspace ------------------------------------------------------- #

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """Liveness plus enough context for the UI header."""
        current = state()
        return {
            "status": "ok",
            "workspace": str(current.settings.workspace),
            "auth_required": current.token is not None,
            "tool_count": len(current.tools),
        }

    @app.get("/api/workspace", dependencies=[Depends(require_token)])
    async def workspace() -> WorkspaceInfo:
        """Where outputs land and which roots are readable."""
        current = state()
        return WorkspaceInfo(
            workspace=str(current.settings.workspace),
            allowed_roots=[str(r) for r in current.settings.allowed_roots],
            jobs_dir=str(current.settings.jobs_dir),
            uploads_dir=str(current.settings.uploads_dir),
        )

    @app.get("/api/files", dependencies=[Depends(require_token)])
    async def files(
        directory: str | None = Query(default=None, description="Directory to list."),
        limit: int = Query(default=300, ge=1, le=2000),
    ) -> list[FileView]:
        """List media files in a directory inside the allowed roots."""
        current = state()
        target = (
            resolve_within_roots(directory, current.settings)
            if directory
            else current.settings.workspace
        )
        if not target.is_dir():
            raise HTTPException(status_code=400, detail="Not a directory.")
        found: list[FileView] = []
        for path in sorted(target.rglob("*")):
            if len(found) >= limit:
                break
            if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            stat = path.stat()
            found.append(
                FileView(
                    path=str(path),
                    name=path.name,
                    size_bytes=stat.st_size,
                    modified_at=stat.st_mtime,
                    kind=_classify(path),
                )
            )
        found.sort(key=lambda f: f.modified_at, reverse=True)
        return found

    @app.post("/api/upload", dependencies=[Depends(require_token)])
    async def upload(files: list[UploadFile] = File(...)) -> list[FileView]:
        """Take files from the browser into the workspace's uploads directory.

        Lets the UI work on media from anywhere on the machine without the user
        first copying it under an allowed root by hand. Filenames are rebuilt
        rather than trusted, and the size cap is applied while streaming.
        """
        current = state()
        stored: list[FileView] = []
        for item in files:
            path = await asyncio.to_thread(store_upload, item, current.settings)
            stat = path.stat()
            stored.append(
                FileView(
                    path=str(path),
                    name=path.name,
                    size_bytes=stat.st_size,
                    modified_at=stat.st_mtime,
                    kind=_classify(path),
                )
            )
        return stored

    @app.get("/api/upload/accepts", dependencies=[Depends(require_token)])
    async def upload_accepts() -> dict[str, Any]:
        """What the upload endpoint will take, for the browser's file picker."""
        return {
            "suffixes": sorted(UPLOAD_SUFFIXES),
            "max_bytes": state().settings.max_input_bytes,
        }

    @app.get("/api/media", dependencies=[Depends(require_token)])
    async def media(request: Request, path: str = Query(...)) -> Response:
        """Stream a media file to the browser, honouring Range requests."""
        return serve_media(path, request.headers.get("range"), state().settings)

    # -- jobs ------------------------------------------------------------ #

    @app.get("/api/projects", dependencies=[Depends(require_token)])
    async def projects() -> list[ProjectRow]:
        """Every project in the store, so the UI can offer them as filters."""
        return [ProjectRow(**row) for row in state().store.list_projects()]

    @app.get("/api/jobs", dependencies=[Depends(require_token)])
    async def list_jobs(
        status: JobStatus | None = None,
        project: str | None = Query(default=None, description="Omit or 'all' for every project."),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> JobsResponse:
        """A page of jobs, newest first, optionally scoped to one project."""
        current = state()
        selector = None if project in (None, "all") else project
        records = current.store.list_jobs(
            status=status, project=selector, limit=limit, offset=offset
        )
        total = current.store.count_jobs(status, selector)
        return JobsResponse(
            jobs=[_to_view(r) for r in records],
            counts=current.store.counts_by_status(selector),
            total=total,
            offset=offset,
            limit=limit,
            has_more=offset + len(records) < total,
        )

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(require_token)])
    async def get_job(job_id: str) -> JobView:
        """One job in full, including its resolved ffmpeg command."""
        return _to_view(state().store.get(job_id))

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_token)])
    async def cancel_job(job_id: str) -> JobView:
        """Cancel a queued or running job.

        Uses the same cooperative flag the MCP cancel tool sets, so this works
        even when the job is being run by the MCP server's worker pool.
        """
        current = state()
        current.store.request_cancel(job_id)
        return _to_view(current.store.get(job_id))

    # -- tools ----------------------------------------------------------- #

    @app.get("/api/tools", dependencies=[Depends(require_token)])
    async def tools() -> list[ToolView]:
        """Every registered tool with its JSON schema, for the generated forms."""
        return [
            ToolView(
                name=spec.name,
                title=spec.title,
                description=spec.description,
                phase=spec.phase,
                read_only=spec.read_only,
                input_schema=spec.input_schema(),
                output_schema=spec.output_schema(),
            )
            for spec in state().tools
        ]

    @app.post("/api/tools/{name}", dependencies=[Depends(require_token)])
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool exactly as an MCP client would."""
        spec = get_tool(name)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {name}")
        try:
            return await spec.call(arguments)
        except ValidationError as exc:
            # include_context=False drops the original exception objects, which
            # are not JSON serialisable.
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_url=False, include_context=False),
            ) from exc

    # -- static frontend -------------------------------------------------- #

    if STATIC_DIR.is_dir() and (STATIC_DIR / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{full_path:path}")
        async def spa(full_path: str) -> Response:
            """Serve the built single-page app, falling back to its index."""
            candidate = (STATIC_DIR / full_path).resolve()
            if full_path and candidate.is_file() and candidate.is_relative_to(STATIC_DIR.resolve()):
                return FileResponse(candidate)
            return FileResponse(STATIC_DIR / "index.html")

    # -- websocket -------------------------------------------------------- #

    @app.websocket("/ws/jobs")
    async def jobs_socket(websocket: WebSocket) -> None:
        """Push job updates so the queue view stays live without polling.

        The store is polled here rather than in the browser, and only changed
        jobs are sent, which keeps the view within a few hundred milliseconds of
        the truth however the job was started.
        """
        current = state()
        if current.token is not None:
            supplied = websocket.query_params.get("token")
            if not supplied or not secrets.compare_digest(supplied, current.token):
                await websocket.close(code=4401)
                return
        await websocket.accept()
        previous: dict[str, tuple[str, float, str | None]] = {}
        try:
            while True:
                records = await asyncio.to_thread(current.store.list_jobs, limit=200)
                changed = []
                seen = set()
                for record in records:
                    key = (record.status.value, record.progress, record.message)
                    seen.add(record.job_id)
                    if previous.get(record.job_id) != key:
                        previous[record.job_id] = key
                        changed.append(_to_view(record).model_dump(mode="json"))
                removed = [job_id for job_id in previous if job_id not in seen]
                for job_id in removed:
                    previous.pop(job_id, None)
                if changed or removed:
                    counts = await asyncio.to_thread(current.store.counts_by_status)
                    await websocket.send_json(
                        {"type": "jobs", "jobs": changed, "removed": removed, "counts": counts}
                    )
                await asyncio.sleep(0.4)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def _is_loopback(host: str) -> bool:
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    """Console-script entry point: run the local UI server."""
    parser = argparse.ArgumentParser(description="Run the ffmpeg-mcp local UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8756, help="Port to listen on.")
    parser.add_argument(
        "--token",
        default=None,
        help="Require this token. Generated automatically for non-loopback binds.",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, (args.log_level or settings.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    token = args.token
    if token is None and not _is_loopback(args.host):
        # The UI can trigger arbitrary local reads and writes through the same
        # tools the MCP server exposes, so it is never opened up unauthenticated.
        token = secrets.token_urlsafe(24)
        log.warning(
            "Binding to %s, which is reachable from the network. Access token: %s",
            args.host,
            token,
        )

    import uvicorn

    app = create_app(settings, token)
    url = f"http://{args.host}:{args.port}"
    log.info("ffmpeg-mcp UI on %s%s", url, f"?token={token}" if token else "")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
