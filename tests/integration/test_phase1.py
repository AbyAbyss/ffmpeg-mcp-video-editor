"""Phase 1 integration tests: every core tool exercised against a real fixture clip.

Each assertion checks the rendered output with the probe tool, per the phase
definition of done.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from ffmpeg_mcp.config import Settings
from ffmpeg_mcp.jobs.store import get_store
from ffmpeg_mcp.jobs.worker import WorkerPool, execute_job
from ffmpeg_mcp.models import JobStatus

from .helpers import call_tool, output_path, probe_output, run_job, run_job_ok

pytestmark = [pytest.mark.integration]


class TestProbe:
    async def test_probe_reports_the_fixture_accurately(
        self, settings: Settings, clip: Path
    ) -> None:
        info = await call_tool("probe_media", {"input_path": str(clip)})
        assert info["duration"] == pytest.approx(3.0, abs=0.2)
        assert info["video_streams"][0]["width"] == 320
        assert info["video_streams"][0]["height"] == 240
        assert info["video_streams"][0]["codec"] == "h264"
        assert info["video_streams"][0]["fps"] == pytest.approx(30.0, abs=0.1)
        assert info["audio_streams"][0]["codec"] == "aac"

    async def test_probing_a_missing_file_raises(self, settings: Settings) -> None:
        from ffmpeg_mcp.errors import InvalidPathError

        with pytest.raises(InvalidPathError):
            await call_tool("probe_media", {"input_path": str(settings.workspace / "no.mp4")})

    async def test_probing_outside_the_allowlist_raises(self, settings: Settings) -> None:
        from ffmpeg_mcp.errors import InvalidPathError

        with pytest.raises(InvalidPathError):
            await call_tool("probe_media", {"input_path": "/etc/hosts"})


class TestCapabilities:
    async def test_capabilities_report_the_resolved_binary(self, settings: Settings) -> None:
        caps = await call_tool(
            "list_capabilities",
            {
                "check_encoders": ["libx264", "definitely_not_an_encoder"],
                "check_filters": ["scale"],
            },
        )
        assert caps["version"].startswith("ffmpeg version")
        assert caps["source"] in {"system", "cached", "configured", "downloaded"}
        assert caps["encoders_available"]["libx264"] is True
        assert caps["encoders_available"]["definitely_not_an_encoder"] is False
        assert caps["filters_available"]["scale"] is True
        assert set(caps["optional_features"]) == {"whisper", "vision"}


class TestTrim:
    async def test_reencode_trim_is_frame_accurate(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "trim",
            {"input_path": str(clip), "start": 0.5, "end": 2.0, "mode": "reencode"},
            settings,
        )
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(1.5, abs=0.1)

    async def test_stream_copy_trim_produces_playable_output(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "trim", {"input_path": str(clip), "start": 0.0, "end": 2.0, "mode": "copy"}, settings
        )
        info = await probe_output(record, settings)
        assert info.duration is not None and info.duration > 0.5
        assert info.video_streams[0].codec == "h264"
        assert any("keyframe" in note for note in record.result["notes"])

    async def test_trim_to_an_explicit_path(self, settings: Settings, clip: Path) -> None:
        target = settings.workspace / "out" / "cut.mp4"
        record = await run_job_ok(
            "trim",
            {"input_path": str(clip), "start": 0.0, "end": 1.0, "output_path": str(target)},
            settings,
        )
        assert output_path(record) == target

    async def test_trim_without_an_output_path_lands_in_the_job_directory(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "trim", {"input_path": str(clip), "start": 0.0, "end": 1.0}, settings
        )
        assert output_path(record).parent == settings.jobs_dir / record.job_id

    async def test_open_ended_trim_runs_to_the_end_of_the_file(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "trim", {"input_path": str(clip), "start": 1.0, "mode": "reencode"}, settings
        )
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(2.0, abs=0.2)

    async def test_an_inverted_range_is_rejected_before_queueing(
        self, settings: Settings, clip: Path
    ) -> None:
        with pytest.raises(Exception, match="end must be greater than start"):
            await call_tool("trim", {"input_path": str(clip), "start": 2.0, "end": 1.0})

    async def test_a_start_past_the_end_fails_the_job_with_a_clear_code(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job(
            "trim", {"input_path": str(clip), "start": 99.0, "mode": "reencode"}, settings
        )
        assert record.status is JobStatus.FAILED
        assert record.error is not None
        assert record.error.code == "invalid_parameter"

    async def test_the_resolved_command_is_recorded_for_audit(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "trim", {"input_path": str(clip), "start": 0.0, "end": 1.0}, settings
        )
        assert record.command is not None
        assert "ffmpeg" in record.command and str(clip) in record.command


class TestConcat:
    async def test_identical_clips_concatenate_without_reencoding(
        self, settings: Settings, clip: Path
    ) -> None:
        import shutil

        second = settings.workspace / "clip2.mp4"
        shutil.copy(clip, second)
        record = await run_job_ok("concat", {"input_paths": [str(clip), str(second)]}, settings)
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(6.0, abs=0.3)
        assert any("without re-encoding" in note for note in record.result["notes"])

    async def test_mismatched_clips_are_normalised_then_joined(
        self, settings: Settings, clip: Path, clip_alt: Path
    ) -> None:
        record = await run_job_ok("concat", {"input_paths": [str(clip), str(clip_alt)]}, settings)
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(5.0, abs=0.4)
        # Normalised to the first clip's resolution.
        assert (info.video_streams[0].width, info.video_streams[0].height) == (320, 240)
        assert any("normalised" in note for note in record.result["notes"])

    async def test_an_explicit_target_resolution_is_honoured(
        self, settings: Settings, clip: Path, clip_alt: Path
    ) -> None:
        record = await run_job_ok(
            "concat",
            {"input_paths": [str(clip), str(clip_alt)], "target_resolution": "640x480"},
            settings,
        )
        info = await probe_output(record, settings)
        assert (info.video_streams[0].width, info.video_streams[0].height) == (640, 480)

    async def test_fewer_than_two_clips_is_rejected(self, settings: Settings, clip: Path) -> None:
        with pytest.raises(Exception, match="at least 2"):
            await call_tool("concat", {"input_paths": [str(clip)]})


class TestConvert:
    async def test_convert_to_webm(self, settings: Settings, clip: Path) -> None:
        target = settings.workspace / "out.webm"
        record = await run_job_ok(
            "convert_format", {"input_path": str(clip), "output_path": str(target)}, settings
        )
        info = await probe_output(record, settings)
        assert info.video_streams[0].codec == "vp9"
        assert info.audio_streams[0].codec == "opus"

    async def test_audio_only_extraction(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "convert_format",
            {"input_path": str(clip), "container": "mp3", "audio_only": True},
            settings,
        )
        info = await probe_output(record, settings)
        assert info.has_video is False
        assert info.audio_streams[0].codec == "mp3"
        assert info.duration == pytest.approx(3.0, abs=0.3)

    async def test_explicit_encoder_settings_are_applied(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "convert_format",
            {
                "input_path": str(clip),
                "container": "mkv",
                "encode": {"video_codec": "libx265", "crf": 30, "preset": "ultrafast"},
            },
            settings,
        )
        info = await probe_output(record, settings)
        assert info.video_streams[0].codec == "hevc"


class TestTransform:
    async def test_scale_preserves_aspect_ratio_from_one_dimension(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "transform", {"input_path": str(clip), "scale_width": 160}, settings
        )
        info = await probe_output(record, settings)
        assert (info.video_streams[0].width, info.video_streams[0].height) == (160, 120)

    async def test_crop(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "transform", {"input_path": str(clip), "crop": "10,20,100,80"}, settings
        )
        info = await probe_output(record, settings)
        assert (info.video_streams[0].width, info.video_streams[0].height) == (100, 80)

    async def test_rotation_swaps_the_dimensions(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok("transform", {"input_path": str(clip), "rotate": 90}, settings)
        info = await probe_output(record, settings)
        assert (info.video_streams[0].width, info.video_streams[0].height) == (240, 320)

    async def test_crop_scale_and_rotate_compose_in_one_pass(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "transform",
            {
                "input_path": str(clip),
                "crop": "0,0,320,180",
                "scale_width": 160,
                "rotate": 270,
                "flip_horizontal": True,
            },
            settings,
        )
        info = await probe_output(record, settings)
        # 320x180 cropped, scaled to 160x90, then rotated to 90x160.
        assert (info.video_streams[0].width, info.video_streams[0].height) == (90, 160)
        assert record.command is not None and record.command.count("-vf") == 1

    async def test_pad_to_fit_letterboxes_instead_of_stretching(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "transform",
            {
                "input_path": str(clip),
                "scale_width": 1080,
                "scale_height": 1920,
                "pad_to_fit": True,
            },
            settings,
        )
        info = await probe_output(record, settings)
        assert (info.video_streams[0].width, info.video_streams[0].height) == (1080, 1920)

    async def test_a_transform_with_no_operations_is_rejected(
        self, settings: Settings, clip: Path
    ) -> None:
        with pytest.raises(Exception, match="at least one"):
            await call_tool("transform", {"input_path": str(clip)})

    async def test_a_bad_crop_string_is_rejected_before_queueing(
        self, settings: Settings, clip: Path
    ) -> None:
        from ffmpeg_mcp.errors import InvalidParameterError

        with pytest.raises(InvalidParameterError):
            await call_tool("transform", {"input_path": str(clip), "crop": "10,20,100"})


class TestSpeed:
    async def test_double_speed_halves_the_duration(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok("speed_ramp", {"input_path": str(clip), "speed": 2.0}, settings)
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(1.5, abs=0.2)
        assert info.has_audio is True

    async def test_half_speed_doubles_the_duration(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok("speed_ramp", {"input_path": str(clip), "speed": 0.5}, settings)
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(6.0, abs=0.3)

    async def test_a_large_factor_chains_atempo_correctly(
        self, settings: Settings, clip: Path
    ) -> None:
        # 8x needs three atempo stages; a single one would be rejected.
        record = await run_job_ok("speed_ramp", {"input_path": str(clip), "speed": 8.0}, settings)
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(0.375, abs=0.1)
        assert record.command is not None and record.command.count("atempo") == 3

    async def test_dropping_audio(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "speed_ramp", {"input_path": str(clip), "speed": 2.0, "drop_audio": True}, settings
        )
        info = await probe_output(record, settings)
        assert info.has_audio is False

    async def test_pitch_shifting_mode_still_renders(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "speed_ramp", {"input_path": str(clip), "speed": 1.5, "keep_pitch": False}, settings
        )
        info = await probe_output(record, settings)
        assert info.has_audio is True
        assert record.command is not None and "asetrate" in record.command


class TestJobLifecycle:
    async def test_status_and_result_follow_a_job_to_completion(
        self, settings: Settings, clip: Path
    ) -> None:
        submission = await call_tool("trim", {"input_path": str(clip), "start": 0.0, "end": 1.0})
        job_id = submission["job_id"]

        queued = await call_tool("job_status", {"job_id": job_id})
        assert queued["status"] == "queued"

        # Asking for the result too early must error rather than return nothing.
        from ffmpeg_mcp.errors import JobNotFinishedError

        with pytest.raises(JobNotFinishedError):
            await call_tool("job_result", {"job_id": job_id})

        store = get_store(settings)
        claimed = store.claim_next(os.getpid())
        assert claimed is not None
        await execute_job(claimed, store, settings)

        done = await call_tool("job_status", {"job_id": job_id})
        assert done["status"] == "done"
        assert done["progress"] == 100.0
        assert done["elapsed_seconds"] is not None

        result = await call_tool("job_result", {"job_id": job_id})
        assert Path(result["outputs"][0]).exists()
        assert result["result"]["media"]["duration"] == pytest.approx(1.0, abs=0.15)

    async def test_progress_advances_while_a_job_runs(self, settings: Settings, clip: Path) -> None:
        # A slow enough render to observe intermediate progress.
        submission = await call_tool(
            "convert_format",
            {
                "input_path": str(clip),
                "container": "mkv",
                "encode": {"video_codec": "libx265", "crf": 18, "preset": "veryslow"},
            },
        )
        job_id = submission["job_id"]
        store = get_store(settings)
        claimed = store.claim_next(os.getpid())
        assert claimed is not None

        seen: list[float] = []

        async def watch() -> None:
            while True:
                seen.append(store.get(job_id).progress)
                await asyncio.sleep(0.05)

        watcher = asyncio.create_task(watch())
        await execute_job(claimed, store, settings)
        watcher.cancel()

        assert max(seen, default=0) > 0, "progress never advanced past zero"
        assert store.get(job_id).progress == 100.0

    async def test_listing_jobs_reflects_the_queue(self, settings: Settings, clip: Path) -> None:
        await call_tool("trim", {"input_path": str(clip), "start": 0.0, "end": 1.0})
        listing = await call_tool("list_jobs", {})
        assert listing["counts"]["queued"] == 1
        assert listing["jobs"][0]["tool"] == "trim"

    async def test_cancelling_a_queued_job(self, settings: Settings, clip: Path) -> None:
        submission = await call_tool("trim", {"input_path": str(clip), "start": 0.0, "end": 1.0})
        cancelled = await call_tool("cancel_job", {"job_id": submission["job_id"]})
        assert cancelled["cancelled"] is True
        assert cancelled["status"] == "cancelled"

    async def test_cancelling_a_running_job_stops_ffmpeg(
        self, settings: Settings, clip: Path
    ) -> None:
        # A deliberately slow encode, so there is a running process to interrupt.
        submission = await call_tool(
            "convert_format",
            {
                "input_path": str(clip),
                "container": "mkv",
                "encode": {"video_codec": "libx265", "crf": 12, "preset": "veryslow"},
            },
        )
        job_id = submission["job_id"]
        store = get_store(settings)
        claimed = store.claim_next(os.getpid())
        assert claimed is not None

        task = asyncio.create_task(execute_job(claimed, store, settings))
        await asyncio.sleep(0.4)
        # Cancel through a separate store instance, standing in for the UI process.
        from ffmpeg_mcp.jobs.store import JobStore

        JobStore(settings).request_cancel(job_id)
        await asyncio.wait_for(task, timeout=30)
        assert store.get(job_id).status is JobStatus.CANCELLED


class TestWorkerPool:
    async def test_the_pool_drains_queued_jobs(self, settings: Settings, clip: Path) -> None:
        store = get_store(settings)
        pool = WorkerPool(settings, store)
        await pool.start()
        try:
            ids = []
            for start in (0.0, 0.5, 1.0):
                submission = await call_tool(
                    "trim", {"input_path": str(clip), "start": start, "end": start + 0.5}
                )
                ids.append(submission["job_id"])
            deadline = asyncio.get_running_loop().time() + 60
            while asyncio.get_running_loop().time() < deadline:
                if all(store.get(i).status.is_terminal for i in ids):
                    break
                await asyncio.sleep(0.1)
            for job_id in ids:
                record = store.get(job_id)
                assert record.status is JobStatus.DONE, record.error
                assert Path(record.outputs[0]).exists()
        finally:
            await pool.stop()
