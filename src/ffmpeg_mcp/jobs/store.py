"""The shared job store.

Backed by a single SQLite file inside the workspace. SQLite is used rather than
an in-memory dict because the MCP server and the optional UI run as separate
processes and must see the same queue: either process can enqueue, either can
run workers, and either can request cancellation of a job the other one owns.

Cancellation is cooperative and therefore cross-process safe: the canceller sets
``cancel_requested``, and whichever worker owns the subprocess notices on its
next poll and terminates it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..errors import JobNotFoundError
from ..models import JobError, JobRecord, JobStatus

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,
    tool             TEXT NOT NULL,
    status           TEXT NOT NULL,
    progress         REAL NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL,
    started_at       REAL,
    finished_at      REAL,
    params           TEXT NOT NULL DEFAULT '{}',
    outputs          TEXT NOT NULL DEFAULT '[]',
    result           TEXT NOT NULL DEFAULT '{}',
    error            TEXT,
    command          TEXT,
    message          TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    worker_pid       INTEGER,
    heartbeat_at     REAL,
    schema_fingerprint TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);
"""

_STALE_HEARTBEAT_SECONDS = 90.0


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str)


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class JobStore:
    """CRUD and queue operations over the jobs table."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.db_path = self.settings.db_path
        self._initialise()

    # -- connection handling ------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @contextmanager
    def _cursor(self, *, immediate: bool = False) -> Iterator[sqlite3.Cursor]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            cursor = conn.cursor()
            yield cursor
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _initialise(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            # Older stores predate the fingerprint; add it rather than making
            # anyone delete their job history to upgrade.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "schema_fingerprint" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN schema_fingerprint TEXT")
        finally:
            conn.close()

    # -- serialisation ------------------------------------------------------ #

    @staticmethod
    def _to_record(row: sqlite3.Row) -> JobRecord:
        error_raw = row["error"]
        return JobRecord(
            job_id=row["job_id"],
            tool=row["tool"],
            status=JobStatus(row["status"]),
            progress=row["progress"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            params=json.loads(row["params"]),
            outputs=json.loads(row["outputs"]),
            result=json.loads(row["result"]),
            error=JobError.model_validate(json.loads(error_raw)) if error_raw else None,
            command=row["command"],
            message=row["message"],
            cancel_requested=bool(row["cancel_requested"]),
            worker_pid=row["worker_pid"],
        )

    # -- writes ------------------------------------------------------------- #

    def create(
        self, tool: str, params: dict[str, Any], schema_fingerprint: str | None = None
    ) -> JobRecord:
        """Enqueue a new job and return its record."""
        job_id = uuid.uuid4().hex[:16]
        now = time.time()
        with self._cursor() as cursor:
            cursor.execute(
                "INSERT INTO jobs (job_id, tool, status, created_at, params, schema_fingerprint) "
                "VALUES (?,?,?,?,?,?)",
                (job_id, tool, JobStatus.QUEUED.value, now, _dumps(params), schema_fingerprint),
            )
        log.info("Job %s queued for tool %s", job_id, tool)
        return JobRecord(
            job_id=job_id, tool=tool, status=JobStatus.QUEUED, created_at=now, params=params
        )

    def claim_next(self, worker_pid: int, known: dict[str, str] | None = None) -> JobRecord | None:
        """Atomically take the oldest queued job. Returns None if the queue is empty.

        ``BEGIN IMMEDIATE`` takes the write lock before the select, so two
        workers -- in this process or another one -- cannot claim the same job.
        """
        now = time.time()
        with self._cursor(immediate=True) as cursor:
            if known is None:
                row = cursor.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at LIMIT 1",
                    (JobStatus.QUEUED.value,),
                ).fetchone()
            else:
                # Leave work stamped by a build whose schema differs to whichever
                # worker can actually run it. A NULL stamp is pre-upgrade, so it
                # is claimable by anyone.
                pairs = [f"{tool}:{fp}" for tool, fp in known.items()]
                placeholders = ",".join("?" * len(pairs)) or "NULL"
                row = cursor.execute(
                    "SELECT * FROM jobs WHERE status = ? AND ("
                    "  schema_fingerprint IS NULL"
                    f"  OR (tool || ':' || schema_fingerprint) IN ({placeholders})"
                    ") ORDER BY created_at LIMIT 1",
                    (JobStatus.QUEUED.value, *pairs),
                ).fetchone()
            if row is None:
                return None
            cursor.execute(
                "UPDATE jobs SET status=?, started_at=?, worker_pid=?, heartbeat_at=? "
                "WHERE job_id=?",
                (JobStatus.RUNNING.value, now, worker_pid, now, row["job_id"]),
            )
            record = self._to_record(row)
        record.status = JobStatus.RUNNING
        record.started_at = now
        record.worker_pid = worker_pid
        return record

    def update_progress(
        self,
        job_id: str,
        progress: float | None = None,
        message: str | None = None,
        command: str | None = None,
    ) -> None:
        """Record progress, the current step, and/or the resolved command line."""
        assignments = ["heartbeat_at = ?"]
        values: list[Any] = [time.time()]
        if progress is not None:
            assignments.append("progress = ?")
            values.append(max(0.0, min(100.0, progress)))
        if message is not None:
            assignments.append("message = ?")
            values.append(message)
        if command is not None:
            assignments.append("command = ?")
            values.append(command)
        values.append(job_id)
        with self._cursor() as cursor:
            cursor.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE job_id = ?", values)

    def finish(
        self, job_id: str, outputs: list[str], result: dict[str, Any], command: str | None = None
    ) -> None:
        """Mark a job done with its outputs and structured result."""
        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE jobs SET status=?, progress=100, finished_at=?, outputs=?, result=?, "
                "command=COALESCE(?, command), message=? WHERE job_id=?",
                (
                    JobStatus.DONE.value,
                    time.time(),
                    _dumps(outputs),
                    _dumps(result),
                    command,
                    "Completed.",
                    job_id,
                ),
            )
        log.info("Job %s finished", job_id)

    def fail(self, job_id: str, error: JobError, command: str | None = None) -> None:
        """Mark a job failed with a structured error."""
        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE jobs SET status=?, finished_at=?, error=?, command=COALESCE(?, command), "
                "message=? WHERE job_id=?",
                (
                    JobStatus.FAILED.value,
                    time.time(),
                    _dumps(error.model_dump()),
                    command,
                    error.message,
                    job_id,
                ),
            )
        log.warning("Job %s failed: %s", job_id, error.message)

    def mark_cancelled(self, job_id: str) -> None:
        """Move a job to the cancelled terminal state."""
        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE jobs SET status=?, finished_at=?, message=? WHERE job_id=?",
                (JobStatus.CANCELLED.value, time.time(), "Cancelled.", job_id),
            )

    def request_cancel(self, job_id: str) -> JobRecord:
        """Ask for a job to stop.

        A queued job is cancelled immediately; a running one gets a flag that its
        owning worker picks up, which is what makes cancel work across processes.
        """
        with self._cursor(immediate=True) as cursor:
            row = cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError("No such job.", job_id=job_id)
            record = self._to_record(row)
            if record.status.is_terminal:
                return record
            cursor.execute("UPDATE jobs SET cancel_requested=1 WHERE job_id=?", (job_id,))
            if record.status is JobStatus.QUEUED:
                cursor.execute(
                    "UPDATE jobs SET status=?, finished_at=?, message=? WHERE job_id=?",
                    (JobStatus.CANCELLED.value, time.time(), "Cancelled.", job_id),
                )
                record.status = JobStatus.CANCELLED
            record.cancel_requested = True
        return record

    # -- reads -------------------------------------------------------------- #

    def get(self, job_id: str) -> JobRecord:
        """Fetch one job, raising :class:`JobNotFoundError` if it is unknown."""
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise JobNotFoundError("No such job.", job_id=job_id)
        return self._to_record(row)

    def is_cancel_requested(self, job_id: str) -> bool:
        """Cheap poll used by the ffmpeg runner between progress ticks."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT cancel_requested FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        finally:
            conn.close()
        return bool(row and row["cancel_requested"])

    def list_jobs(
        self, *, status: JobStatus | None = None, limit: int = 100, offset: int = 0
    ) -> list[JobRecord]:
        """List jobs newest first, optionally filtered by status."""
        conn = self._connect()
        try:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (status.value, limit, offset),
                ).fetchall()
        finally:
            conn.close()
        return [self._to_record(row) for row in rows]

    def counts_by_status(self) -> dict[str, int]:
        """Queue summary for the UI header."""
        conn = self._connect()
        try:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        finally:
            conn.close()
        counts = {status.value: 0 for status in JobStatus}
        for row in rows:
            counts[row["status"]] = row["n"]
        return counts

    # -- maintenance -------------------------------------------------------- #

    def reclaim_stale(self) -> int:
        """Fail jobs whose owning worker process has died.

        Without this, a crashed worker leaves jobs stuck in ``running`` forever.
        """
        cutoff = time.time() - _STALE_HEARTBEAT_SECONDS
        stale: list[str] = []
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT job_id, worker_pid, heartbeat_at FROM jobs WHERE status = ?",
                (JobStatus.RUNNING.value,),
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            heartbeat = row["heartbeat_at"] or 0.0
            if heartbeat < cutoff and not _pid_alive(row["worker_pid"]):
                stale.append(row["job_id"])
        for job_id in stale:
            self.fail(
                job_id,
                JobError(
                    code="worker_lost",
                    message="The worker running this job exited before it completed.",
                ),
            )
        return len(stale)

    def cleanup(self, retention_hours: int | None = None) -> int:
        """Delete finished jobs older than the retention window, plus their files."""
        hours = self.settings.retention_hours if retention_hours is None else retention_hours
        cutoff = time.time() - hours * 3600
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
        removed = 0
        for row in rows:
            job_id = row["job_id"]
            directory = self.settings.jobs_dir / job_id
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)
            with self._cursor() as cursor:
                cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            removed += 1
        if removed:
            log.info("Retention: removed %d expired jobs", removed)
        return removed

    def job_dir(self, job_id: str) -> Path:
        """Return (creating if needed) the workspace directory owned by a job."""
        directory = self.settings.jobs_dir / job_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory


_store: JobStore | None = None


def get_store(settings: Settings | None = None) -> JobStore:
    """Return the process-wide job store, creating it on first use."""
    global _store
    if _store is None or (settings is not None and _store.settings is not settings):
        _store = JobStore(settings)
    return _store


def reset_store() -> None:
    """Drop the cached store. Used by tests."""
    global _store
    _store = None
