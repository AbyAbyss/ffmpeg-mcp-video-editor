"""Unit tests for the job store and worker dispatch. No ffmpeg involved."""

from __future__ import annotations

import asyncio
import time

import pytest

from ffmpeg_mcp.config import Settings
from ffmpeg_mcp.errors import JobNotFoundError
from ffmpeg_mcp.jobs.store import JobStore
from ffmpeg_mcp.jobs.worker import (
    JobContext,
    JobOutcome,
    execute_job,
    get_handler,
    register_handler,
)
from ffmpeg_mcp.models import JobError, JobStatus


class TestJobStore:
    def test_create_returns_a_queued_job(self, store: JobStore) -> None:
        record = store.create("trim", {"input_path": "a.mp4"})
        assert record.status is JobStatus.QUEUED
        assert store.get(record.job_id).params == {"input_path": "a.mp4"}

    def test_unknown_job_raises(self, store: JobStore) -> None:
        with pytest.raises(JobNotFoundError):
            store.get("does-not-exist")

    def test_claim_takes_the_oldest_job_first(self, store: JobStore) -> None:
        first = store.create("trim", {})
        time.sleep(0.01)
        second = store.create("trim", {})
        assert store.claim_next(1).job_id == first.job_id  # type: ignore[union-attr]
        assert store.claim_next(1).job_id == second.job_id  # type: ignore[union-attr]
        assert store.claim_next(1) is None

    def test_claim_is_exclusive(self, store: JobStore) -> None:
        # Two workers must never receive the same job.
        for _ in range(5):
            store.create("trim", {})
        claimed = [store.claim_next(1) for _ in range(10)]
        ids = [c.job_id for c in claimed if c is not None]
        assert len(ids) == 5
        assert len(set(ids)) == 5

    def test_claim_is_exclusive_across_separate_store_instances(self, settings: Settings) -> None:
        # Stands in for the MCP server and the UI running as two processes.
        writer = JobStore(settings)
        job = writer.create("trim", {})
        reader = JobStore(settings)
        first = writer.claim_next(1)
        second = reader.claim_next(2)
        assert first is not None and first.job_id == job.job_id
        assert second is None

    def test_progress_updates_are_clamped(self, store: JobStore) -> None:
        record = store.create("trim", {})
        store.update_progress(record.job_id, 150.0, "over")
        assert store.get(record.job_id).progress == 100.0
        store.update_progress(record.job_id, -5.0)
        assert store.get(record.job_id).progress == 0.0

    def test_finish_records_outputs_and_completes(self, store: JobStore) -> None:
        record = store.create("trim", {})
        store.finish(record.job_id, ["/tmp/out.mp4"], {"output_path": "/tmp/out.mp4"})
        done = store.get(record.job_id)
        assert done.status is JobStatus.DONE
        assert done.progress == 100.0
        assert done.outputs == ["/tmp/out.mp4"]
        assert done.finished_at is not None

    def test_fail_records_a_structured_error(self, store: JobStore) -> None:
        record = store.create("trim", {})
        store.fail(record.job_id, JobError(code="ffmpeg_failed", message="boom", details={"x": 1}))
        failed = store.get(record.job_id)
        assert failed.status is JobStatus.FAILED
        assert failed.error is not None
        assert failed.error.code == "ffmpeg_failed"
        assert failed.error.details == {"x": 1}

    def test_cancelling_a_queued_job_is_immediate(self, store: JobStore) -> None:
        record = store.create("trim", {})
        result = store.request_cancel(record.job_id)
        assert result.status is JobStatus.CANCELLED
        assert store.get(record.job_id).status is JobStatus.CANCELLED

    def test_cancelling_a_running_job_sets_the_flag(self, store: JobStore) -> None:
        record = store.create("trim", {})
        store.claim_next(1)
        result = store.request_cancel(record.job_id)
        assert result.status is JobStatus.RUNNING
        assert store.is_cancel_requested(record.job_id) is True

    def test_cancel_is_visible_to_another_store_instance(self, settings: Settings) -> None:
        # This is what makes cancel work from the UI process.
        writer = JobStore(settings)
        record = writer.create("trim", {})
        writer.claim_next(1)
        JobStore(settings).request_cancel(record.job_id)
        assert writer.is_cancel_requested(record.job_id) is True

    def test_cancelling_a_finished_job_is_a_no_op(self, store: JobStore) -> None:
        record = store.create("trim", {})
        store.finish(record.job_id, [], {})
        assert store.request_cancel(record.job_id).status is JobStatus.DONE

    def test_cancelling_an_unknown_job_raises(self, store: JobStore) -> None:
        with pytest.raises(JobNotFoundError):
            store.request_cancel("nope")

    def test_listing_filters_by_status(self, store: JobStore) -> None:
        done = store.create("trim", {})
        store.finish(done.job_id, [], {})
        store.create("trim", {})
        assert len(store.list_jobs(status=JobStatus.DONE)) == 1
        assert len(store.list_jobs(status=JobStatus.QUEUED)) == 1
        assert len(store.list_jobs()) == 2

    def test_counts_cover_every_status(self, store: JobStore) -> None:
        store.create("trim", {})
        counts = store.counts_by_status()
        assert set(counts) == {s.value for s in JobStatus}
        assert counts["queued"] == 1

    def test_retention_removes_old_jobs_and_their_files(self, store: JobStore) -> None:
        record = store.create("trim", {})
        directory = store.job_dir(record.job_id)
        (directory / "out.mp4").write_bytes(b"x")
        store.finish(record.job_id, [], {})
        # Backdate the completion past the retention window.
        conn = store._connect()
        conn.execute(
            "UPDATE jobs SET finished_at = ? WHERE job_id = ?",
            (time.time() - 48 * 3600, record.job_id),
        )
        conn.close()
        assert store.cleanup(retention_hours=24) == 1
        assert not directory.exists()
        with pytest.raises(JobNotFoundError):
            store.get(record.job_id)

    def test_retention_keeps_recent_jobs(self, store: JobStore) -> None:
        record = store.create("trim", {})
        store.finish(record.job_id, [], {})
        assert store.cleanup(retention_hours=24) == 0
        assert store.get(record.job_id).status is JobStatus.DONE

    def test_stale_running_jobs_from_a_dead_worker_are_reclaimed(self, store: JobStore) -> None:
        record = store.create("trim", {})
        store.claim_next(999_999)  # a pid that does not exist
        conn = store._connect()
        conn.execute(
            "UPDATE jobs SET heartbeat_at = ? WHERE job_id = ?",
            (time.time() - 600, record.job_id),
        )
        conn.close()
        assert store.reclaim_stale() == 1
        reclaimed = store.get(record.job_id)
        assert reclaimed.status is JobStatus.FAILED
        assert reclaimed.error is not None
        assert reclaimed.error.code == "worker_lost"

    def test_live_workers_are_not_reclaimed(self, store: JobStore) -> None:
        import os

        store.create("trim", {})
        store.claim_next(os.getpid())
        assert store.reclaim_stale() == 0


class TestExecuteJob:
    async def test_successful_handler_marks_the_job_done(
        self, store: JobStore, settings: Settings
    ) -> None:
        async def ok(ctx: JobContext) -> JobOutcome:
            await ctx.report(50.0, "halfway")
            return JobOutcome(outputs=["/tmp/a.mp4"], result={"output_path": "/tmp/a.mp4"})

        register_handler("unit_ok", ok)
        record = store.create("unit_ok", {})
        claimed = store.claim_next(1)
        assert claimed is not None
        await execute_job(claimed, store, settings)
        finished = store.get(record.job_id)
        assert finished.status is JobStatus.DONE
        assert finished.outputs == ["/tmp/a.mp4"]

    async def test_a_raising_handler_fails_the_job_without_killing_the_worker(
        self, store: JobStore, settings: Settings
    ) -> None:
        async def boom(ctx: JobContext) -> JobOutcome:
            raise RuntimeError("kaboom")

        register_handler("unit_boom", boom)
        record = store.create("unit_boom", {})
        claimed = store.claim_next(1)
        assert claimed is not None
        await execute_job(claimed, store, settings)
        failed = store.get(record.job_id)
        assert failed.status is JobStatus.FAILED
        assert failed.error is not None
        assert "kaboom" in failed.error.message

    async def test_a_domain_error_keeps_its_code(self, store: JobStore, settings: Settings) -> None:
        from ffmpeg_mcp.errors import UnsupportedCodecError

        async def unsupported(ctx: JobContext) -> JobOutcome:
            raise UnsupportedCodecError("nope", codec="h265")

        register_handler("unit_unsupported", unsupported)
        record = store.create("unit_unsupported", {})
        claimed = store.claim_next(1)
        assert claimed is not None
        await execute_job(claimed, store, settings)
        failed = store.get(record.job_id)
        assert failed.error is not None
        assert failed.error.code == "unsupported_codec"
        assert failed.error.details == {"codec": "h265"}

    async def test_an_unregistered_tool_fails_cleanly(
        self, store: JobStore, settings: Settings
    ) -> None:
        record = store.create("no_such_tool", {})
        claimed = store.claim_next(1)
        assert claimed is not None
        await execute_job(claimed, store, settings)
        failed = store.get(record.job_id)
        assert failed.error is not None
        assert failed.error.code == "unknown_tool"

    async def test_a_job_cancelled_mid_run_ends_as_cancelled(
        self, store: JobStore, settings: Settings
    ) -> None:
        async def slow(ctx: JobContext) -> JobOutcome:
            for _ in range(20):
                if ctx.cancelled():
                    from ffmpeg_mcp.errors import JobCancelledError

                    raise JobCancelledError("stopped")
                await asyncio.sleep(0.01)
            return JobOutcome()

        register_handler("unit_slow", slow)
        record = store.create("unit_slow", {})
        claimed = store.claim_next(1)
        assert claimed is not None
        task = asyncio.create_task(execute_job(claimed, store, settings))
        await asyncio.sleep(0.03)
        store.request_cancel(record.job_id)
        await task
        assert store.get(record.job_id).status is JobStatus.CANCELLED

    def test_duplicate_handler_registration_is_rejected(self) -> None:
        async def noop(ctx: JobContext) -> JobOutcome:
            return JobOutcome()

        register_handler("unit_dupe", noop)
        with pytest.raises(ValueError, match="Duplicate"):
            register_handler("unit_dupe", noop)

    def test_phase_one_tools_all_have_handlers(self) -> None:
        from ffmpeg_mcp.tools.registry import load_all_tools

        load_all_tools()
        for name in ("trim", "concat", "convert_format", "transform", "speed_ramp"):
            assert get_handler(name) is not None, f"{name} has no job handler"
