"""Unit tests for ffprobe parsing, progress parsing, and encoder argument building."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from ffmpeg_mcp.binaries import parse_major_version
from ffmpeg_mcp.errors import InvalidParameterError
from ffmpeg_mcp.ffmpeg.encoding import audio_encode_args, output_args, video_encode_args
from ffmpeg_mcp.ffmpeg.probe import parse_frame_rate, parse_probe_document
from ffmpeg_mcp.ffmpeg.runner import build_progress_update, format_command, parse_progress_time
from ffmpeg_mcp.models import EncodeOptions

DOCUMENT = {
    "format": {
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "duration": "12.500000",
        "size": "1048576",
        "bit_rate": "671088",
    },
    "streams": [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "profile": "High",
            "pix_fmt": "yuv420p",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "30000/1001",
            "r_frame_rate": "30000/1001",
            "bit_rate": "600000",
            "duration": "12.500000",
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "48000",
            "channels": 2,
            "channel_layout": "stereo",
            "bit_rate": "128000",
            "tags": {"language": "eng"},
        },
        {
            "index": 2,
            "codec_type": "subtitle",
            "codec_name": "mov_text",
            "tags": {"language": "fr"},
        },
    ],
}


class TestProbeParsing:
    def test_format_level_fields(self) -> None:
        info = parse_probe_document(DOCUMENT, "/tmp/a.mp4")
        assert info.duration == 12.5
        assert info.size_bytes == 1048576
        assert info.format_name is not None and "mp4" in info.format_name

    def test_video_stream_fields(self) -> None:
        video = parse_probe_document(DOCUMENT, "/tmp/a.mp4").primary_video
        assert video is not None
        assert (video.codec, video.width, video.height) == ("h264", 1920, 1080)
        assert video.fps == pytest.approx(29.97, abs=0.01)

    def test_audio_stream_fields(self) -> None:
        audio = parse_probe_document(DOCUMENT, "/tmp/a.mp4").primary_audio
        assert audio is not None
        assert (audio.codec, audio.sample_rate, audio.channels) == ("aac", 48000, 2)
        assert audio.language == "eng"

    def test_subtitle_streams_are_captured(self) -> None:
        info = parse_probe_document(DOCUMENT, "/tmp/a.mp4")
        assert len(info.subtitle_streams) == 1
        assert info.subtitle_streams[0].language == "fr"

    def test_cover_art_is_not_treated_as_video(self) -> None:
        document = {
            "format": {"duration": "180.0"},
            "streams": [
                {"index": 0, "codec_type": "audio", "codec_name": "mp3", "channels": 2},
                {
                    "index": 1,
                    "codec_type": "video",
                    "codec_name": "mjpeg",
                    "disposition": {"attached_pic": 1},
                },
            ],
        }
        info = parse_probe_document(document, "/tmp/a.mp3")
        assert info.has_video is False
        assert info.has_audio is True

    def test_missing_container_duration_falls_back_to_a_stream(self) -> None:
        document = {
            "format": {},
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264", "duration": "7.0"}
            ],
        }
        assert parse_probe_document(document, "/tmp/a.mkv").duration == 7.0

    def test_na_values_become_none(self) -> None:
        document = {
            "format": {"duration": "N/A"},
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264", "bit_rate": "N/A"}
            ],
        }
        info = parse_probe_document(document, "/tmp/a.mkv")
        assert info.duration is None
        assert info.video_streams[0].bit_rate is None

    def test_rotation_from_side_data(self) -> None:
        document = {
            "format": {},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "side_data_list": [{"rotation": -90}],
                }
            ],
        }
        assert parse_probe_document(document, "/tmp/a.mov").video_streams[0].rotation == -90

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("30/1", 30.0), ("30000/1001", 29.97003), ("0/0", None), ("", None), (None, None)],
    )
    def test_frame_rate_parsing(self, value: object, expected: float | None) -> None:
        result = parse_frame_rate(value)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected, abs=0.001)


class TestProgressParsing:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("00:01:23.450000", 83.45),
            ("00:00:00.000000", 0.0),
            ("01:00:00.000000", 3600.0),
            ("N/A", None),
            ("", None),
        ],
    )
    def test_out_time_strings(self, value: str, expected: float | None) -> None:
        assert parse_progress_time(value) == (
            pytest.approx(expected) if expected is not None else None
        )

    def test_microsecond_field_is_preferred(self) -> None:
        update = build_progress_update(
            {"out_time_us": "2500000", "out_time": "00:00:99.0", "progress": "continue"}
        )
        assert update.out_time_seconds == pytest.approx(2.5)

    def test_negative_microseconds_are_ignored(self) -> None:
        # ffmpeg emits these while seeking; treating them as progress rewinds the bar.
        update = build_progress_update({"out_time_us": "-42", "out_time": "00:00:05.0"})
        assert update.out_time_seconds == pytest.approx(5.0)

    def test_out_time_ms_is_actually_microseconds(self) -> None:
        update = build_progress_update({"out_time_ms": "3000000"})
        assert update.out_time_seconds == pytest.approx(3.0)

    def test_frame_speed_and_size_are_captured(self) -> None:
        update = build_progress_update(
            {"frame": "150", "fps": "29.5", "speed": "1.8x", "total_size": "204800"}
        )
        assert update.frame == 150
        assert update.fps == pytest.approx(29.5)
        assert update.speed == pytest.approx(1.8)
        assert update.total_size == 204800

    def test_malformed_values_do_not_raise(self) -> None:
        update = build_progress_update({"frame": "abc", "speed": "N/A"})
        assert update.frame is None
        assert update.speed is None


class TestCommandFormatting:
    def test_arguments_with_spaces_are_quoted(self) -> None:
        assert format_command(["ffmpeg", "-i", "/tmp/my file.mp4"]) == (
            "ffmpeg -i '/tmp/my file.mp4'"
        )

    def test_the_audit_line_round_trips_back_to_the_original_argv(self) -> None:
        # The logged command must be re-runnable, so a filter graph containing
        # quotes and semicolons has to survive a shell round trip intact.
        argv = [
            "ffmpeg",
            "-filter_complex",
            "[0:v]drawtext=text=it\\'s:x='(w-tw)/2'[v];[v]null[out]",
            "/tmp/my file.mp4",
        ]
        assert shlex.split(format_command(argv)) == argv


class TestEncodeArguments:
    def test_crf_encoders_get_crf_and_preset(self) -> None:
        args = video_encode_args(EncodeOptions(video_codec="libx264", crf=18, preset="slow"))
        assert "-crf" in args and "18" in args
        assert "-preset" in args and "slow" in args

    def test_bitrate_overrides_crf(self) -> None:
        args = video_encode_args(EncodeOptions(video_codec="libx264", crf=18, video_bitrate="5M"))
        assert "-b:v" in args and "5M" in args
        assert "-crf" not in args

    def test_vp9_constant_quality_needs_zero_bitrate(self) -> None:
        args = video_encode_args(EncodeOptions(video_codec="libvpx-vp9", crf=32))
        assert args[args.index("-b:v") + 1] == "0"

    def test_videotoolbox_maps_crf_onto_its_quality_scale(self) -> None:
        args = video_encode_args(EncodeOptions(video_codec="h264_videotoolbox", crf=20))
        assert "-crf" not in args
        assert "-q:v" in args

    def test_preset_is_omitted_for_encoders_that_reject_it(self) -> None:
        assert "-preset" not in video_encode_args(
            EncodeOptions(video_codec="libvpx-vp9", preset="slow")
        )

    def test_copy_short_circuits(self) -> None:
        assert video_encode_args(EncodeOptions(video_codec="copy")) == ["-c:v", "copy"]
        assert audio_encode_args(EncodeOptions(audio_codec="copy")) == ["-c:a", "copy"]

    def test_lossless_audio_codecs_get_no_bitrate(self) -> None:
        assert "-b:a" not in audio_encode_args(EncodeOptions(audio_codec="flac"))

    def test_mp4_gets_faststart(self) -> None:
        args = output_args(EncodeOptions(), Path("/tmp/a.mp4"))
        assert "-movflags" in args and "+faststart" in args

    def test_audio_only_container_omits_video_arguments(self) -> None:
        args = output_args(EncodeOptions(), Path("/tmp/a.mp3"))
        assert "-c:v" not in args

    def test_extra_args_are_appended(self) -> None:
        args = output_args(EncodeOptions(extra_args=["-tune", "film"]), Path("/tmp/a.mp4"))
        assert args[-2:] == ["-tune", "film"]

    def test_an_output_with_no_streams_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            output_args(EncodeOptions(), Path("/tmp/a.mp4"), has_video=False, has_audio=False)


class TestVersionParsing:
    @pytest.mark.parametrize(
        ("banner", "expected"),
        [
            ("ffmpeg version 7.1 Copyright (c) 2000-2024", 7),
            ("ffmpeg version 6.1.1-3ubuntu5 Copyright", 6),
            ("ffmpeg version n7.1-24-g1a2b3c Copyright", 7),
            ("ffmpeg version 2024-01-01-git-abcdef Copyright", 2024),
            ("ffmpeg version N-113344-gabc Copyright", None),
        ],
    )
    def test_major_version_extraction(self, banner: str, expected: int | None) -> None:
        assert parse_major_version(banner) == expected
