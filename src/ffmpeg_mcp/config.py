"""Runtime configuration, sourced from environment variables with safe defaults.

Both processes that make up this project (the MCP server and the optional UI)
load the same settings, which is what lets them share one workspace and one job
store.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ENV_PREFIX = "FFMPEG_MCP_"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _default_workspace() -> Path:
    raw = _env("WORKSPACE")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".ffmpeg-mcp" / "workspace").resolve()


def _default_allowed_roots(workspace: Path) -> list[Path]:
    """Roots that inputs and outputs may live under.

    Defaults to the workspace plus the user's home directory, which is the
    common case for a locally-run editing tool. Set ``FFMPEG_MCP_ALLOWED_ROOTS``
    (os.pathsep separated) to tighten or widen it.
    """
    raw = _env("ALLOWED_ROOTS")
    if raw:
        roots = [Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p.strip()]
    else:
        roots = [Path.home().resolve()]
    if workspace not in roots and not any(workspace.is_relative_to(r) for r in roots):
        roots.append(workspace)
    return roots


@dataclass(frozen=True)
class Settings:
    """Resolved server configuration."""

    workspace: Path
    allowed_roots: list[Path]
    max_input_bytes: int
    max_job_seconds: int
    retention_hours: int
    worker_concurrency: int
    ffmpeg_path: str | None
    ffprobe_path: str | None
    auto_download_ffmpeg: bool
    whisper_model: str
    whisper_compute_type: str
    log_level: str

    @property
    def jobs_dir(self) -> Path:
        return self.workspace / "jobs"

    @property
    def bin_dir(self) -> Path:
        return self.workspace / "bin"

    @property
    def db_path(self) -> Path:
        return self.workspace / "jobs.db"

    @property
    def uploads_dir(self) -> Path:
        return self.workspace / "uploads"

    def ensure_dirs(self) -> None:
        """Create the workspace tree if it does not exist yet."""
        for path in (self.workspace, self.jobs_dir, self.bin_dir, self.uploads_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class _Override:
    settings: Settings | None = field(default=None)


_override = _Override()


def _build_settings() -> Settings:
    workspace = _default_workspace()
    return Settings(
        workspace=workspace,
        allowed_roots=_default_allowed_roots(workspace),
        max_input_bytes=_env_int("MAX_INPUT_BYTES", 16 * 1024**3),
        max_job_seconds=_env_int("MAX_JOB_SECONDS", 3 * 60 * 60),
        retention_hours=_env_int("RETENTION_HOURS", 24),
        worker_concurrency=_env_int("WORKER_CONCURRENCY", 2),
        ffmpeg_path=_env("FFMPEG_PATH"),
        ffprobe_path=_env("FFPROBE_PATH"),
        auto_download_ffmpeg=(_env("AUTO_DOWNLOAD", "1") or "1").lower()
        not in {"0", "false", "no"},
        whisper_model=_env("WHISPER_MODEL", "base") or "base",
        whisper_compute_type=_env("WHISPER_COMPUTE_TYPE", "int8") or "int8",
        log_level=(_env("LOG_LEVEL", "INFO") or "INFO").upper(),
    )


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return _build_settings()


def get_settings() -> Settings:
    """Return the process-wide settings, building them on first use."""
    if _override.settings is not None:
        return _override.settings
    return _cached_settings()


def set_settings(settings: Settings | None) -> None:
    """Override settings for the current process. Tests use this; nothing else should."""
    _override.settings = settings
