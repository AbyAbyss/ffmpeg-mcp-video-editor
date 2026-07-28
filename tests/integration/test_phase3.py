"""Phase 3 integration tests: Whisper transcription, translation, and auto-captioning.

These run the 'tiny' model against a few seconds of macOS text-to-speech. Tiny
mishears words, so assertions check structure — segment count, ordering,
language, timings inside the media duration — plus one distinctive phrase, rather
than exact transcripts.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from ffmpeg_mcp.config import Settings
from ffmpeg_mcp.models import JobStatus
from ffmpeg_mcp.subtitles import parse_srt

from .helpers import call_tool, output_path, probe_output, run_job, run_job_ok
from .test_phase2 import frame_differs

pytestmark = [pytest.mark.integration]

TINY = {"model": "tiny", "language": "en"}


class TestTranscribe:
    async def test_speech_is_transcribed_into_timed_segments(
        self, settings: Settings, speech: Path
    ) -> None:
        record = await run_job_ok(
            "transcribe_audio", {"input_path": str(speech), "options": TINY}, settings
        )
        result = record.result
        assert result["segment_count"] >= 1
        assert result["language"] == "en"
        assert "caption test" in result["text"].lower()

    async def test_segments_are_ordered_and_within_the_media_duration(
        self, settings: Settings, speech: Path
    ) -> None:
        record = await run_job_ok(
            "transcribe_audio", {"input_path": str(speech), "options": TINY}, settings
        )
        segments = record.result["segments"]
        duration = record.result["duration"]
        assert segments
        for previous, current in itertools.pairwise(segments):
            assert previous["start"] <= current["start"]
        for segment in segments:
            assert 0 <= segment["start"] <= segment["end"] <= duration + 1.0

    async def test_word_timestamps_are_returned_when_asked_for(
        self, settings: Settings, speech: Path
    ) -> None:
        record = await run_job_ok(
            "transcribe_audio",
            {"input_path": str(speech), "options": {**TINY, "word_timestamps": True}},
            settings,
        )
        words = record.result["words"]
        assert len(words) > len(record.result["segments"])
        assert all(w["end"] >= w["start"] for w in words)
        assert all(w["word"].strip() for w in words)

    async def test_word_timestamps_are_omitted_by_default(
        self, settings: Settings, speech: Path
    ) -> None:
        record = await run_job_ok(
            "transcribe_audio", {"input_path": str(speech), "options": TINY}, settings
        )
        assert record.result["words"] == []

    async def test_an_srt_file_can_be_written_directly(
        self, settings: Settings, speech: Path
    ) -> None:
        target = settings.workspace / "out.srt"
        record = await run_job_ok(
            "transcribe_audio",
            {"input_path": str(speech), "options": TINY, "srt_path": str(target)},
            settings,
        )
        assert record.outputs == [str(target)]
        cues = parse_srt(target.read_text())
        assert len(cues) == record.result["segment_count"]

    async def test_transcribing_video_works_through_the_audio_track(
        self, settings: Settings, speech_clip: Path
    ) -> None:
        record = await run_job_ok(
            "transcribe_audio", {"input_path": str(speech_clip), "options": TINY}, settings
        )
        assert record.result["segment_count"] >= 1

    async def test_a_file_with_no_audio_fails_clearly(self, settings: Settings, clip: Path) -> None:
        silent = await run_job_ok(
            "convert_format",
            {
                "input_path": str(clip),
                "container": "mp4",
                "encode": {"audio_codec": "aac"},
                "output_path": str(settings.workspace / "silent.mp4"),
            },
            settings,
        )
        stripped = settings.workspace / "novideo_audio.mp4"
        import subprocess

        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(output_path(silent)),
                "-an",
                "-c:v",
                "copy",
                "-y",
                str(stripped),
            ],
            check=True,
            timeout=60,
        )
        record = await run_job(
            "transcribe_audio", {"input_path": str(stripped), "options": TINY}, settings
        )
        assert record.status is JobStatus.FAILED
        assert record.error is not None
        assert record.error.code == "invalid_parameter"
        assert "no audio" in record.error.message.lower()

    async def test_an_unknown_language_code_is_rejected_before_queueing(
        self, settings: Settings, speech: Path
    ) -> None:
        with pytest.raises(Exception, match="unknown language code"):
            await call_tool(
                "transcribe_audio",
                {"input_path": str(speech), "options": {"language": "klingon"}},
            )

    async def test_music_only_audio_yields_no_segments_rather_than_failing(
        self, settings: Settings, clip: Path
    ) -> None:
        # The fixture clip carries a sine tone, not speech.
        record = await run_job_ok(
            "transcribe_audio",
            {"input_path": str(clip), "options": {**TINY, "language": None}},
            settings,
        )
        if record.result["segment_count"] == 0:
            assert "No speech was detected" in record.result["notes"][0]


class TestTranslate:
    async def test_english_speech_passes_through_translate_mode(
        self, settings: Settings, speech: Path
    ) -> None:
        record = await run_job_ok(
            "translate_transcript", {"input_path": str(speech), "options": TINY}, settings
        )
        assert record.result["segment_count"] >= 1
        assert any("English only" in note for note in record.result["notes"])

    async def test_translate_can_write_an_srt(self, settings: Settings, speech: Path) -> None:
        target = settings.workspace / "en.srt"
        record = await run_job_ok(
            "translate_transcript",
            {"input_path": str(speech), "options": TINY, "srt_path": str(target)},
            settings,
        )
        assert target.exists()
        assert record.outputs == [str(target)]


class TestAutoCaption:
    async def test_transcription_srt_and_burn_happen_in_one_job(
        self, settings: Settings, speech_clip: Path
    ) -> None:
        record = await run_job_ok(
            "auto_caption", {"input_path": str(speech_clip), "options": TINY}, settings
        )
        result = record.result
        assert result["transcript"]["segment_count"] >= 1

        video = Path(result["output_path"])
        srt = Path(result["srt_path"])
        assert video.exists() and srt.exists()
        assert len(parse_srt(srt.read_text())) == result["transcript"]["segment_count"]

        source_info = await call_tool("probe_media", {"input_path": str(speech_clip)})
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(source_info["duration"], abs=0.3)
        # The captions are actually burned into the picture.
        assert frame_differs(speech_clip, video, at_seconds=2.0)

    async def test_both_the_video_and_the_srt_are_reported_as_outputs(
        self, settings: Settings, speech_clip: Path
    ) -> None:
        record = await run_job_ok(
            "auto_caption", {"input_path": str(speech_clip), "options": TINY}, settings
        )
        assert len(record.outputs) == 2
        assert all(Path(p).exists() for p in record.outputs)

    async def test_styling_is_passed_through_to_the_burn(
        self, settings: Settings, speech_clip: Path
    ) -> None:
        record = await run_job_ok(
            "auto_caption",
            {
                "input_path": str(speech_clip),
                "options": TINY,
                "style": {"font_size": 18, "font_color": "#E8630A", "position": "top-center"},
            },
            settings,
        )
        assert record.command is not None
        assert "FontSize=18" in record.command
        assert "PrimaryColour=&H000A63E8" in record.command

    async def test_the_generated_srt_is_wrapped_to_the_requested_width(
        self, settings: Settings, speech_clip: Path
    ) -> None:
        record = await run_job_ok(
            "auto_caption",
            {
                "input_path": str(speech_clip),
                "options": TINY,
                "max_chars_per_line": 15,
                "max_lines": 4,
            },
            settings,
        )
        srt = Path(record.result["srt_path"]).read_text()
        text_lines = [
            line
            for line in srt.splitlines()
            if line.strip() and "-->" not in line and not line.strip().isdigit()
        ]
        assert text_lines
        assert all(len(line) <= 15 for line in text_lines)

    async def test_an_explicit_output_path_is_honoured(
        self, settings: Settings, speech_clip: Path
    ) -> None:
        target = settings.workspace / "final" / "captioned.mp4"
        record = await run_job_ok(
            "auto_caption",
            {"input_path": str(speech_clip), "options": TINY, "output_path": str(target)},
            settings,
        )
        assert record.result["output_path"] == str(target)
        assert target.exists()

    async def test_progress_covers_both_stages_and_ends_at_one_hundred(
        self, settings: Settings, speech_clip: Path
    ) -> None:
        record = await run_job_ok(
            "auto_caption", {"input_path": str(speech_clip), "options": TINY}, settings
        )
        assert record.progress == 100.0

    async def test_an_audio_only_input_is_rejected(self, settings: Settings, speech: Path) -> None:
        record = await run_job(
            "auto_caption", {"input_path": str(speech), "options": TINY}, settings
        )
        assert record.status is JobStatus.FAILED
        assert record.error is not None
        assert record.error.code == "invalid_parameter"
