"""MCP server entry point.

Uses the low-level SDK server rather than the decorator helper so each tool's
JSON schema comes straight from its pydantic model, flat and without an extra
wrapper object around the arguments.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from .binaries import get_binaries
from .config import get_settings
from .errors import FFmpegMCPError
from .jobs.store import get_store
from .jobs.worker import WorkerPool
from .tools.registry import ToolSpec, get_tool, load_all_tools

log = logging.getLogger("ffmpeg_mcp")

SERVER_NAME = "ffmpeg-mcp"
SERVER_VERSION = "0.1.0"

INSTRUCTIONS = """\
Video and image editing through ffmpeg, with Whisper transcription and
MediaPipe/OpenCV vision tools.

How to drive this server:

1. Call probe_media first when an edit depends on the source duration or size.
   It answers immediately, as does list_capabilities.
2. Every tool that encodes frames returns a job id straight away. Poll
   job_status until it reports 'done', then call job_result for the output path.
   cancel_job stops a job that is queued or running.
3. Outputs go to the path you name. Omit output_path and the file is written
   into the job's own workspace directory, which is cleaned up after the
   retention window.
4. Input and output paths must sit under the configured allowed roots.
"""


def _configure_logging() -> None:
    """Log to stderr; stdout belongs to the MCP stdio transport."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


class ToolFailure(Exception):
    """Carries a structured error payload back to the client as tool output."""


def _to_mcp_tool(spec: ToolSpec) -> types.Tool:
    return types.Tool(
        name=spec.name,
        title=spec.title,
        description=spec.description,
        inputSchema=spec.input_schema(),
        outputSchema=spec.output_schema(),
        annotations=types.ToolAnnotations(
            title=spec.title,
            readOnlyHint=spec.read_only,
            destructiveHint=False,
            idempotentHint=spec.read_only,
            openWorldHint=False,
        ),
    )


def build_server() -> Server[Any, Any]:
    """Construct the MCP server with every registered tool wired up."""
    specs = load_all_tools()
    server: Server[Any, Any] = Server(
        SERVER_NAME, version=SERVER_VERSION, instructions=INSTRUCTIONS
    )

    # The SDK's decorators are untyped, so these two ignores are unavoidable.
    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def _list_tools() -> list[types.Tool]:
        return [_to_mcp_tool(spec) for spec in specs]

    # validate_input is off because each ToolSpec validates with its own pydantic
    # model, which produces better messages than raw JSON-schema errors.
    @server.call_tool(validate_input=False)  # type: ignore[untyped-decorator]
    async def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        spec = get_tool(name)
        if spec is None:
            raise ValueError(f"Unknown tool: {name}")
        try:
            return await spec.call(arguments)
        except FFmpegMCPError as exc:
            log.warning("Tool %s failed: %s", name, exc.message)
            raise ToolFailure(json.dumps(exc.to_dict())) from exc

    return server


async def _serve() -> None:
    _configure_logging()
    settings = get_settings()
    settings.ensure_dirs()
    log.info("Workspace: %s", settings.workspace)
    log.info("Allowed roots: %s", ", ".join(str(r) for r in settings.allowed_roots))

    try:
        binaries = get_binaries(settings)
        log.info("Using %s (%s, %s)", binaries.ffmpeg, binaries.version, binaries.source)
    except FFmpegMCPError as exc:
        # Do not abort startup: list_capabilities and the job tools still work,
        # and the error is far more useful surfaced through a tool call.
        log.error("ffmpeg unavailable: %s", exc.message)

    store = get_store(settings)
    store.reclaim_stale()
    pool = WorkerPool(settings, store)
    await pool.start()

    server = build_server()
    options = InitializationOptions(
        server_name=SERVER_NAME,
        server_version=SERVER_VERSION,
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(), experimental_capabilities={}
        ),
        instructions=INSTRUCTIONS,
    )
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, options)
    finally:
        await pool.stop()


def main() -> None:
    """Console-script entry point: run the MCP server over stdio."""
    with contextlib.suppress(KeyboardInterrupt):  # pragma: no cover - interactive
        asyncio.run(_serve())


if __name__ == "__main__":  # pragma: no cover
    main()
