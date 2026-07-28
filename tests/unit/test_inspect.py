"""Unit tests for the inspection tools' pure logic."""

from __future__ import annotations

import itertools

import pytest

from ffmpeg_mcp.errors import InvalidParameterError
from ffmpeg_mcp.tools.inspect import (
    FrameStats,
    build_audio_measurement,
    parse_audio_measurements,
    parse_signalstats,
    sample_times,
    summarise_analysis,
)


class TestSampleTimes:
    def test_samples_avoid_the_very_edges(self) -> None:
        # The first and last frames are the least representative of a shot.
        times = sample_times(10.0, 4, 0.0, None)
        assert times[0] > 0.0
        assert times[-1] < 10.0

    def test_samples_are_evenly_spaced_and_ordered(self) -> None:
        times = sample_times(10.0, 4, 0.0, None)
        gaps = [round(b - a, 3) for a, b in itertools.pairwise(times)]
        assert len(set(gaps)) == 1
        assert times == sorted(times)

    def test_a_single_sample_lands_in_the_middle(self) -> None:
        assert sample_times(10.0, 1, 0.0, None) == [5.0]

    def test_an_explicit_range_is_respected(self) -> None:
        times = sample_times(60.0, 3, 10.0, 20.0)
        assert all(10.0 < t < 20.0 for t in times)

    def test_the_requested_count_is_returned(self) -> None:
        assert len(sample_times(10.0, 7, 0.0, None)) == 7

    def test_an_inverted_range_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            sample_times(10.0, 3, 8.0, 2.0)

    def test_a_zero_count_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            sample_times(10.0, 0, 0.0, None)


class TestSignalstatsParsing:
    DUMP = """frame:0    pts:0       pts_time:0
lavfi.signalstats.YMIN=16
lavfi.signalstats.YAVG=156.975
lavfi.signalstats.YMAX=255
lavfi.signalstats.SATAVG=2.51
lavfi.signalstats.SATMAX=41
"""

    def test_values_are_extracted(self) -> None:
        values = parse_signalstats(self.DUMP)
        assert values["YAVG"] == pytest.approx(156.975)
        assert values["YMIN"] == 16
        assert values["SATAVG"] == pytest.approx(2.51)

    def test_unrelated_text_is_ignored(self) -> None:
        assert parse_signalstats("nothing to see here") == {}

    def test_negative_values_parse(self) -> None:
        assert parse_signalstats("lavfi.signalstats.HUEAVG=-12.5")["HUEAVG"] == -12.5


def frame(t: float, luma: float, sat: float, ymax: float = 200) -> FrameStats:
    return FrameStats(
        time=t,
        luma_avg=luma,
        luma_min=0,
        luma_max=ymax,
        saturation_avg=sat,
        saturation_max=sat * 4,
    )


class TestAnalysisVerdict:
    def test_a_greyscale_clip_is_recognised(self) -> None:
        result = summarise_analysis([frame(1, 120, 0.4), frame(2, 130, 0.6)])
        assert result.is_greyscale is True
        assert any("greyscale" in n for n in result.notes)

    def test_a_colourful_clip_is_not_called_greyscale(self) -> None:
        assert summarise_analysis([frame(1, 120, 40)]).is_greyscale is False

    def test_crushed_blacks_are_flagged_with_advice(self) -> None:
        result = summarise_analysis([frame(1, 120, 20), frame(2, 8, 20)])
        assert result.crushed_blacks is True
        assert any("shadow detail" in n for n in result.notes)

    def test_a_well_exposed_clip_is_not_flagged(self) -> None:
        result = summarise_analysis([frame(1, 120, 20), frame(2, 130, 22)])
        assert result.crushed_blacks is False
        assert result.blown_highlights is False
        assert result.notes == []

    def test_uneven_exposure_across_shots_is_flagged(self) -> None:
        result = summarise_analysis([frame(1, 40, 20), frame(2, 200, 20)])
        assert any("Exposure varies" in n for n in result.notes)

    def test_aggregates_are_computed(self) -> None:
        result = summarise_analysis([frame(1, 100, 10), frame(2, 200, 30)])
        assert result.luma_avg == 150
        assert result.luma_min == 100
        assert result.luma_max == 200
        assert result.saturation_avg == 20

    def test_no_samples_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            summarise_analysis([])


class TestAudioParsing:
    LOG = """
[Parsed_volumedetect_0 @ 0x1] mean_volume: -35.3 dB
[Parsed_volumedetect_0 @ 0x1] max_volume: -8.8 dB
[Parsed_ebur128_1 @ 0x2] Integrated loudness:
    I:         -33.2 LUFS
    Threshold: -43.5 LUFS
  Loudness range:
    LRA:         4.9 LU
    Threshold: -52.7 LUFS
"""

    def test_volume_figures_are_extracted(self) -> None:
        values = parse_audio_measurements(self.LOG)
        assert values["mean_volume"] == -35.3
        assert values["max_volume"] == -8.8

    def test_loudness_figures_are_extracted(self) -> None:
        values = parse_audio_measurements(self.LOG)
        assert values["integrated_lufs"] == -33.2
        assert values["loudness_range"] == 4.9

    def test_an_empty_log_yields_nothing(self) -> None:
        assert parse_audio_measurements("") == {}

    def test_silence_is_called_out(self) -> None:
        result = build_audio_measurement({"mean_volume": -54.4, "max_volume": -30.2})
        assert result.is_effectively_silent is True
        assert any("nothing here worth keeping" in n for n in result.notes)

    def test_quiet_but_not_silent_is_described_as_such(self) -> None:
        result = build_audio_measurement({"mean_volume": -43.0, "max_volume": -12.0})
        assert result.is_effectively_silent is False
        assert any("room tone or handling noise" in n for n in result.notes)

    def test_clipping_is_flagged(self) -> None:
        result = build_audio_measurement({"mean_volume": -12.0, "max_volume": -0.1})
        assert any("clipping" in n for n in result.notes)

    def test_ordinary_speech_gets_no_complaints(self) -> None:
        result = build_audio_measurement(
            {"mean_volume": -18.0, "max_volume": -3.0, "integrated_lufs": -16.0}
        )
        assert result.notes == []
        assert result.is_effectively_silent is False
