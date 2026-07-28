"""Unit tests for transcription argument validation and result mapping."""

from __future__ import annotations

import pytest

from ffmpeg_mcp.models import Segment, WordTiming
from ffmpeg_mcp.tools.transcription import TranscriptionOptions, _to_result
from ffmpeg_mcp.transcribe import TranscriptionOutcome


class TestTranscriptionOptions:
    def test_defaults_are_conservative(self) -> None:
        options = TranscriptionOptions()
        assert options.model is None  # falls back to the server setting
        assert options.language is None  # auto-detect
        assert options.vad_filter is True
        assert options.word_timestamps is False

    @pytest.mark.parametrize("code", ["en", "fr", "ja", "yue", "EN"])
    def test_known_language_codes_are_accepted(self, code: str) -> None:
        assert TranscriptionOptions(language=code).language == code

    @pytest.mark.parametrize("code", ["klingon", "english", "zz"])
    def test_unknown_language_codes_are_rejected(self, code: str) -> None:
        with pytest.raises(ValueError, match="unknown language code"):
            TranscriptionOptions(language=code)

    def test_an_out_of_range_beam_size_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            TranscriptionOptions(beam_size=99)

    def test_an_unknown_model_name_is_rejected_by_the_schema(self) -> None:
        with pytest.raises(ValueError):
            TranscriptionOptions(model="enormous")  # type: ignore[arg-type]


class TestOutcomeMapping:
    def test_full_text_joins_the_segments(self) -> None:
        outcome = TranscriptionOutcome(
            segments=[
                Segment(start=0, end=1, text="Hello"),
                Segment(start=1, end=2, text="world"),
            ]
        )
        assert outcome.text == "Hello world"

    def test_empty_transcription_has_empty_text(self) -> None:
        assert TranscriptionOutcome().text == ""

    def test_the_result_carries_counts_and_metadata(self) -> None:
        outcome = TranscriptionOutcome(
            segments=[Segment(start=0, end=1, text="hi")],
            words=[WordTiming(start=0, end=0.5, word="hi", probability=0.9)],
            language="en",
            language_probability=0.98,
            duration=1.0,
        )
        result = _to_result(outcome, ["a note"])
        assert result.segment_count == 1
        assert result.language == "en"
        assert result.language_probability == 0.98
        assert result.notes == ["a note"]
        assert result.words[0].word == "hi"
