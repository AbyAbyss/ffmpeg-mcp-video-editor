"""Phase 2 integration tests: grading, LUTs, curves, captions, and text overlays."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from ffmpeg_mcp.config import Settings
from ffmpeg_mcp.models import JobStatus

from .helpers import call_tool, output_path, probe_output, run_job, run_job_ok

pytestmark = [pytest.mark.integration]

IDENTITY_CUBE = """TITLE "Identity"
LUT_3D_SIZE 2
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
"""

# Swaps red and blue, so the effect is obvious in a pixel check.
SWAP_CUBE = """TITLE "Swap RB"
LUT_3D_SIZE 2
0.0 0.0 0.0
0.0 0.0 1.0
0.0 1.0 0.0
0.0 1.0 1.0
1.0 0.0 0.0
1.0 0.0 1.0
1.0 1.0 0.0
1.0 1.0 1.0
"""


def raw_frame(path: Path, at_seconds: float = 1.0) -> bytes:
    """Decode one frame to raw RGB24 bytes, so tests can inspect actual pixels."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-ss",
            str(at_seconds),
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )
    assert proc.stdout, f"no frame decoded from {path} at {at_seconds}s"
    return proc.stdout


def mean_channel_values(path: Path, at_seconds: float = 1.0) -> tuple[float, float, float]:
    """Average R, G and B across one frame."""
    data = raw_frame(path, at_seconds)
    reds, greens, blues = data[0::3], data[1::3], data[2::3]
    return (
        sum(reds) / len(reds),
        sum(greens) / len(greens),
        sum(blues) / len(blues),
    )


def mean_saturation(path: Path, at_seconds: float = 1.0) -> float:
    """Average per-pixel colourfulness, as max(R,G,B) - min(R,G,B).

    Channel *means* are a poor proxy for saturation: a vividly coloured frame
    can average out to three near-identical numbers. This measures each pixel.
    """
    data = raw_frame(path, at_seconds)
    total = 0
    for index in range(0, len(data) - 2, 3):
        pixel = data[index : index + 3]
        total += max(pixel) - min(pixel)
    return total / (len(data) / 3)


def frame_differs(first: Path, second: Path, at_seconds: float = 1.0) -> bool:
    """Whether two renders differ in actual pixel content at a given time."""
    return raw_frame(first, at_seconds) != raw_frame(second, at_seconds)


class TestColorGrade:
    async def test_brightness_lifts_the_image(self, settings: Settings, clip: Path) -> None:
        before = mean_channel_values(clip)
        record = await run_job_ok(
            "color_grade", {"input_path": str(clip), "brightness": 0.4}, settings
        )
        after = mean_channel_values(output_path(record))
        assert sum(after) > sum(before)

    async def test_zero_saturation_produces_a_grey_image(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "color_grade", {"input_path": str(clip), "saturation": 0.0}, settings
        )
        # Chroma subsampling leaves a small residual, so compare colourfulness
        # against the source rather than demanding exactly equal channels.
        assert mean_saturation(output_path(record)) < mean_saturation(clip) / 10

    async def test_warm_temperature_shifts_towards_red(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "color_grade", {"input_path": str(clip), "temperature": 80}, settings
        )
        before = mean_channel_values(clip)
        after = mean_channel_values(output_path(record))
        assert (after[0] - after[2]) > (before[0] - before[2])

    async def test_cool_temperature_shifts_towards_blue(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "color_grade", {"input_path": str(clip), "temperature": -80}, settings
        )
        before = mean_channel_values(clip)
        after = mean_channel_values(output_path(record))
        assert (after[0] - after[2]) < (before[0] - before[2])

    async def test_audio_and_duration_survive_the_grade(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "color_grade", {"input_path": str(clip), "contrast": 1.3}, settings
        )
        info = await probe_output(record, settings)
        assert info.has_audio is True
        assert info.duration == pytest.approx(3.0, abs=0.2)

    async def test_an_entirely_neutral_grade_is_rejected(
        self, settings: Settings, clip: Path
    ) -> None:
        with pytest.raises(Exception, match="non-neutral"):
            await call_tool("color_grade", {"input_path": str(clip)})

    async def test_only_the_requested_stages_reach_the_command(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "color_grade", {"input_path": str(clip), "contrast": 1.2}, settings
        )
        assert record.command is not None
        assert "eq=" in record.command
        assert "colorchannelmixer" not in record.command


class TestApplyLut:
    async def test_an_identity_lut_leaves_the_image_alone(
        self, settings: Settings, clip: Path
    ) -> None:
        lut = settings.workspace / "identity.cube"
        lut.write_text(IDENTITY_CUBE)
        record = await run_job_ok(
            "apply_lut", {"input_path": str(clip), "lut_path": str(lut)}, settings
        )
        before = mean_channel_values(clip)
        after = mean_channel_values(output_path(record))
        for channel_before, channel_after in zip(before, after, strict=True):
            assert abs(channel_before - channel_after) < 8

    async def test_a_channel_swapping_lut_changes_the_image(
        self, settings: Settings, clip: Path
    ) -> None:
        lut = settings.workspace / "swap.cube"
        lut.write_text(SWAP_CUBE)
        record = await run_job_ok(
            "apply_lut", {"input_path": str(clip), "lut_path": str(lut)}, settings
        )
        before = mean_channel_values(clip)
        after = mean_channel_values(output_path(record))
        assert abs(after[0] - before[2]) < abs(after[0] - before[0]) + 5

    async def test_partial_strength_renders_and_differs_from_full(
        self, settings: Settings, clip: Path
    ) -> None:
        lut = settings.workspace / "swap.cube"
        lut.write_text(SWAP_CUBE)
        full = await run_job_ok(
            "apply_lut", {"input_path": str(clip), "lut_path": str(lut)}, settings
        )
        half = await run_job_ok(
            "apply_lut",
            {"input_path": str(clip), "lut_path": str(lut), "strength": 0.5},
            settings,
        )
        assert frame_differs(output_path(full), output_path(half))
        assert half.command is not None and "blend" in half.command

    async def test_a_malformed_lut_is_rejected_before_queueing(
        self, settings: Settings, clip: Path
    ) -> None:
        from ffmpeg_mcp.errors import InvalidParameterError

        lut = settings.workspace / "bad.cube"
        lut.write_text("LUT_3D_SIZE 2\n0.0 0.0 0.0\n")  # too few rows
        with pytest.raises(InvalidParameterError, match="row count"):
            await call_tool("apply_lut", {"input_path": str(clip), "lut_path": str(lut)})

    async def test_a_non_cube_extension_is_rejected(self, settings: Settings, clip: Path) -> None:
        from ffmpeg_mcp.errors import InvalidParameterError

        lut = settings.workspace / "lut.txt"
        lut.write_text(IDENTITY_CUBE)
        with pytest.raises(InvalidParameterError, match=r"\.cube"):
            await call_tool("apply_lut", {"input_path": str(clip), "lut_path": str(lut)})

    async def test_the_lut_size_is_reported(self, settings: Settings, clip: Path) -> None:
        lut = settings.workspace / "identity.cube"
        lut.write_text(IDENTITY_CUBE)
        record = await run_job_ok(
            "apply_lut", {"input_path": str(clip), "lut_path": str(lut)}, settings
        )
        assert any("2x2x2" in note and "Identity" in note for note in record.result["notes"])


class TestCurves:
    async def test_a_preset_changes_the_image(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "apply_curves", {"input_path": str(clip), "preset": "vintage"}, settings
        )
        assert frame_differs(clip, output_path(record))

    async def test_custom_control_points_are_applied(self, settings: Settings, clip: Path) -> None:
        # A strong S-curve on the master channel.
        record = await run_job_ok(
            "apply_curves",
            {
                "input_path": str(clip),
                "master": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 0.25, "y": 0.1},
                    {"x": 0.75, "y": 0.9},
                    {"x": 1.0, "y": 1.0},
                ],
            },
            settings,
        )
        assert frame_differs(clip, output_path(record))

    async def test_points_may_be_given_out_of_order(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "apply_curves",
            {
                "input_path": str(clip),
                "master": [{"x": 1.0, "y": 1.0}, {"x": 0.0, "y": 0.2}],
            },
            settings,
        )
        assert record.command is not None and "0/0.2 1/1" in record.command

    async def test_an_unknown_preset_is_rejected(self, settings: Settings, clip: Path) -> None:
        with pytest.raises(Exception, match="unknown curve preset"):
            await call_tool("apply_curves", {"input_path": str(clip), "preset": "nope"})

    async def test_preset_and_points_together_are_rejected(
        self, settings: Settings, clip: Path
    ) -> None:
        with pytest.raises(Exception, match="not both"):
            await call_tool(
                "apply_curves",
                {
                    "input_path": str(clip),
                    "preset": "vintage",
                    "master": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
                },
            )


class TestBuildSrt:
    async def test_segments_become_a_valid_srt_file(self, settings: Settings) -> None:
        result = await call_tool(
            "build_srt",
            {
                "segments": [
                    {"start": 0.0, "end": 1.5, "text": "Hello there"},
                    {"start": 1.5, "end": 3.0, "text": "General Kenobi"},
                ],
                "output_path": str(settings.workspace / "subs.srt"),
            },
        )
        assert result["cue_count"] == 2
        assert result["duration"] == pytest.approx(3.0)
        content = Path(result["output_path"]).read_text()
        assert "00:00:00,000 --> 00:00:01,500" in content

    async def test_the_extension_is_corrected(self, settings: Settings) -> None:
        result = await call_tool(
            "build_srt",
            {
                "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
                "output_path": str(settings.workspace / "subs.txt"),
            },
        )
        assert result["output_path"].endswith(".srt")

    async def test_ffmpeg_accepts_the_generated_file(self, settings: Settings, clip: Path) -> None:
        # The real check that the output is well formed.
        built = await call_tool(
            "build_srt",
            {
                "segments": [
                    {"start": 0.2, "end": 1.0, "text": "Time: 12:30, [note]; it's 50%"},
                    {"start": 1.0, "end": 2.5, "text": "Second line here"},
                ],
                "output_path": str(settings.workspace / "tricky.srt"),
            },
        )
        record = await run_job_ok(
            "burn_captions",
            {"input_path": str(clip), "subtitle_path": built["output_path"]},
            settings,
        )
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(3.0, abs=0.2)


class TestBurnCaptions:
    async def test_captions_visibly_change_the_frame(self, settings: Settings, clip: Path) -> None:
        subtitle = settings.workspace / "subs.srt"
        subtitle.write_text("1\n00:00:00,000 --> 00:00:03,000\nHELLO WORLD\n", encoding="utf-8")
        record = await run_job_ok(
            "burn_captions", {"input_path": str(clip), "subtitle_path": str(subtitle)}, settings
        )
        assert frame_differs(clip, output_path(record))

    async def test_styling_reaches_the_filter_graph(self, settings: Settings, clip: Path) -> None:
        subtitle = settings.workspace / "subs.srt"
        subtitle.write_text("1\n00:00:00,000 --> 00:00:02,000\nStyled\n", encoding="utf-8")
        record = await run_job_ok(
            "burn_captions",
            {
                "input_path": str(clip),
                "subtitle_path": str(subtitle),
                "style": {
                    "font_size": 32,
                    "font_color": "#E8630A",
                    "position": "top-center",
                    "bold": True,
                },
            },
            settings,
        )
        assert record.command is not None
        assert "FontSize=32" in record.command
        assert "PrimaryColour=&H000A63E8" in record.command
        assert "Alignment=8" in record.command

    async def test_a_caption_full_of_filter_metacharacters_renders(
        self, settings: Settings, clip: Path
    ) -> None:
        # The case that would break a naively built filter graph.
        subtitle = settings.workspace / "tricky.srt"
        subtitle.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\nTime: 12:30, [note]; it's 50% \\ done\n",
            encoding="utf-8",
        )
        record = await run_job_ok(
            "burn_captions", {"input_path": str(clip), "subtitle_path": str(subtitle)}, settings
        )
        assert frame_differs(clip, output_path(record))

    async def test_a_subtitle_path_containing_commas_and_spaces_works(
        self, settings: Settings, clip: Path
    ) -> None:
        subtitle = settings.workspace / "my subs, v2.srt"
        subtitle.write_text("1\n00:00:00,000 --> 00:00:02,000\nPath test\n", encoding="utf-8")
        record = await run_job_ok(
            "burn_captions", {"input_path": str(clip), "subtitle_path": str(subtitle)}, settings
        )
        assert frame_differs(clip, output_path(record))

    async def test_an_unsupported_subtitle_format_is_rejected(
        self, settings: Settings, clip: Path
    ) -> None:
        from ffmpeg_mcp.errors import InvalidParameterError

        bad = settings.workspace / "subs.txt"
        bad.write_text("not a subtitle")
        with pytest.raises(InvalidParameterError, match="Unsupported subtitle"):
            await call_tool("burn_captions", {"input_path": str(clip), "subtitle_path": str(bad)})

    async def test_an_invalid_colour_is_rejected_before_queueing(
        self, settings: Settings, clip: Path
    ) -> None:
        from ffmpeg_mcp.errors import InvalidParameterError

        subtitle = settings.workspace / "subs.srt"
        subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        with pytest.raises(InvalidParameterError, match="hex value"):
            await call_tool(
                "burn_captions",
                {
                    "input_path": str(clip),
                    "subtitle_path": str(subtitle),
                    "style": {"font_color": "orange"},
                },
            )


class TestTextOverlay:
    async def test_a_title_changes_the_frame_within_its_window(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "text_overlay",
            {
                "input_path": str(clip),
                "items": [{"text": "MY TITLE", "start": 0.0, "end": 3.0, "font_size": 40}],
            },
            settings,
        )
        assert frame_differs(clip, output_path(record))

    async def test_the_overlay_only_shows_inside_its_time_window(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "text_overlay",
            {
                "input_path": str(clip),
                "items": [{"text": "LATE", "start": 2.0, "end": 3.0, "font_size": 60}],
            },
            settings,
        )
        rendered = output_path(record)
        # Unchanged before the window, changed inside it.
        assert mean_channel_values(rendered, 0.5) == pytest.approx(
            mean_channel_values(clip, 0.5), abs=2.0
        )
        assert mean_channel_values(rendered, 2.5) != pytest.approx(
            mean_channel_values(clip, 2.5), abs=0.5
        )

    async def test_several_overlays_render_in_one_pass(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "text_overlay",
            {
                "input_path": str(clip),
                "items": [
                    {"text": "TOP", "position": "top-center", "end": 1.5},
                    {"text": "BOTTOM", "position": "bottom-center", "start": 1.5},
                ],
            },
            settings,
        )
        assert record.command is not None
        assert record.command.count("drawtext") == 2
        assert record.command.count("-vf") == 1
        assert "Drew 2 overlay(s)." in record.result["notes"]

    async def test_text_with_filter_metacharacters_is_drawn_literally(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "text_overlay",
            {
                "input_path": str(clip),
                "items": [{"text": "Time: 12:30, [a]; it's 50% %{pts} \\ x", "end": 3.0}],
            },
            settings,
        )
        assert frame_differs(clip, output_path(record))
        # The text went to a sidecar file, not into the graph.
        assert record.command is not None
        assert "textfile=" in record.command
        assert "12:30" not in record.command

    async def test_fade_animation_produces_an_alpha_expression(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "text_overlay",
            {
                "input_path": str(clip),
                "items": [
                    {
                        "text": "FADE",
                        "start": 0.0,
                        "end": 3.0,
                        "animation": "fade",
                        "animation_duration": 0.5,
                    }
                ],
            },
            settings,
        )
        assert record.command is not None and "alpha=" in record.command

    async def test_slide_animation_produces_a_moving_coordinate(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "text_overlay",
            {
                "input_path": str(clip),
                "items": [{"text": "SLIDE", "start": 0.0, "end": 3.0, "animation": "slide-left"}],
            },
            settings,
        )
        assert record.command is not None and "if(lt(t," in record.command

    async def test_a_background_box_renders(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "text_overlay",
            {
                "input_path": str(clip),
                "items": [
                    {
                        "text": "LOWER THIRD",
                        "position": "lower-third",
                        "background_box": True,
                        "background_color": "#E8630A",
                        "background_opacity": 0.8,
                    }
                ],
            },
            settings,
        )
        assert record.command is not None and "boxcolor=0xE8630A@0.800" in record.command
        assert frame_differs(clip, output_path(record))

    async def test_explicit_coordinates_are_honoured(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "text_overlay",
            {"input_path": str(clip), "items": [{"text": "XY", "x": "12", "y": "34"}]},
            settings,
        )
        # The audit command is shell-quoted, so recover the filter graph from it.
        assert record.command is not None
        graph = shlex.split(record.command)[shlex.split(record.command).index("-vf") + 1]
        assert "x='12':y='34'" in graph

    async def test_an_audio_only_input_is_rejected_at_run_time(
        self, settings: Settings, clip: Path
    ) -> None:
        audio = await run_job_ok(
            "convert_format",
            {"input_path": str(clip), "container": "mp3", "audio_only": True},
            settings,
        )
        record = await run_job(
            "text_overlay",
            {"input_path": str(output_path(audio)), "items": [{"text": "hi"}]},
            settings,
        )
        assert record.status is JobStatus.FAILED
        assert record.error is not None and record.error.code == "invalid_parameter"
