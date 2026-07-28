"""Phase 5 integration tests: transitions, compositing, audio, resizing, timelines."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ffmpeg_mcp.config import Settings
from ffmpeg_mcp.models import JobStatus

from .helpers import call_tool, output_path, probe_output, run_job, run_job_ok
from .test_phase2 import frame_differs, raw_frame

pytestmark = [pytest.mark.integration]


def pixel_at(path: Path, at_seconds: float, x: int, y: int, frame_width: int) -> tuple[int, ...]:
    """One pixel's RGB from a decoded frame."""
    data = raw_frame(path, at_seconds)
    offset = (y * frame_width + x) * 3
    return tuple(data[offset : offset + 3])


def pixels_close(a: tuple[int, ...], b: tuple[int, ...], tolerance: int = 20) -> bool:
    """Whether two pixels match within lossy-encode rounding."""
    return all(abs(int(x) - int(y)) <= tolerance for x, y in zip(a, b, strict=True))


def mean_volume_db(path: Path) -> float:
    """Read a file's mean volume via ffmpeg's volumedetect filter."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    for line in proc.stderr.splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].strip().split()[0])
    raise AssertionError(f"volumedetect reported no mean_volume for {path}")


class TestAddTransition:
    async def test_the_result_is_shorter_by_the_overlap(
        self, settings: Settings, clip: Path, clip_alt: Path
    ) -> None:
        record = await run_job_ok(
            "add_transition",
            {"first_path": str(clip), "second_path": str(clip_alt), "duration": 1.0},
            settings,
        )
        info = await probe_output(record, settings)
        # 3s + 2s - 1s overlap.
        assert info.duration == pytest.approx(4.0, abs=0.3)

    async def test_the_second_clip_is_conformed_to_the_first(
        self, settings: Settings, clip: Path, clip_alt: Path
    ) -> None:
        record = await run_job_ok(
            "add_transition",
            {"first_path": str(clip), "second_path": str(clip_alt), "duration": 0.5},
            settings,
        )
        info = await probe_output(record, settings)
        assert (info.video_streams[0].width, info.video_streams[0].height) == (320, 240)

    @pytest.mark.parametrize("transition", ["fade", "wipeleft", "circleopen", "dissolve"])
    async def test_transition_styles_render(
        self, settings: Settings, clip: Path, clip_alt: Path, transition: str
    ) -> None:
        record = await run_job_ok(
            "add_transition",
            {
                "first_path": str(clip),
                "second_path": str(clip_alt),
                "transition": transition,
                "duration": 0.5,
            },
            settings,
        )
        assert output_path(record).exists()
        assert transition in " ".join(record.result["notes"])

    async def test_the_midpoint_of_a_fade_blends_both_clips(
        self, settings: Settings, clip: Path, clip_alt: Path
    ) -> None:
        record = await run_job_ok(
            "add_transition",
            {"first_path": str(clip), "second_path": str(clip_alt), "duration": 1.0},
            settings,
        )
        rendered = output_path(record)
        # At 2.5s the fade is halfway, so the frame matches neither source exactly.
        assert frame_differs(rendered, clip, at_seconds=2.5)

    async def test_audio_survives_the_join(
        self, settings: Settings, clip: Path, clip_alt: Path
    ) -> None:
        record = await run_job_ok(
            "add_transition",
            {"first_path": str(clip), "second_path": str(clip_alt)},
            settings,
        )
        info = await probe_output(record, settings)
        assert info.has_audio is True

    async def test_an_unknown_transition_is_rejected_before_queueing(
        self, settings: Settings, clip: Path, clip_alt: Path
    ) -> None:
        with pytest.raises(Exception, match="unknown transition"):
            await call_tool(
                "add_transition",
                {
                    "first_path": str(clip),
                    "second_path": str(clip_alt),
                    "transition": "teleport",
                },
            )

    async def test_an_over_long_transition_is_clamped_to_the_clips(
        self, settings: Settings, clip: Path, clip_alt: Path
    ) -> None:
        record = await run_job_ok(
            "add_transition",
            {"first_path": str(clip), "second_path": str(clip_alt), "duration": 30.0},
            settings,
        )
        info = await probe_output(record, settings)
        assert info.duration is not None and info.duration > 0


class TestOverlayMedia:
    async def test_a_watermark_changes_the_frame(
        self, settings: Settings, clip: Path, logo: Path
    ) -> None:
        record = await run_job_ok(
            "overlay_media",
            {"input_path": str(clip), "overlay_path": str(logo), "position": "top-right"},
            settings,
        )
        assert frame_differs(clip, output_path(record))

    async def test_the_overlay_lands_at_the_requested_corner(
        self, settings: Settings, clip: Path, logo: Path
    ) -> None:
        record = await run_job_ok(
            "overlay_media",
            {
                "input_path": str(clip),
                "overlay_path": str(logo),
                "x": 100,
                "y": 100,
                "width": 32,
            },
            settings,
        )
        # The base clip is red in its top-left corner, so sample where it is not,
        # and confirm the pixel both changed and became the logo's red.
        assert pixel_at(clip, 1.0, 110, 110, 320) != (255, 0, 0)
        assert pixel_at(output_path(record), 1.0, 110, 110, 320)[0] > 150
        assert pixel_at(output_path(record), 1.0, 110, 110, 320)[1] < 90
        # Outside the overlay the picture is untouched, allowing for re-encoding.
        assert pixels_close(
            pixel_at(output_path(record), 1.0, 200, 200, 320),
            pixel_at(clip, 1.0, 200, 200, 320),
        )

    async def test_an_overlay_window_limits_when_it_appears(
        self, settings: Settings, clip: Path, logo: Path
    ) -> None:
        record = await run_job_ok(
            "overlay_media",
            {
                "input_path": str(clip),
                "overlay_path": str(logo),
                "x": 100,
                "y": 100,
                "start": 2.0,
                "end": 3.0,
            },
            settings,
        )
        rendered = output_path(record)
        # Before the window the frame matches the source; inside it, it does not.
        assert pixels_close(
            pixel_at(rendered, 0.5, 110, 110, 320), pixel_at(clip, 0.5, 110, 110, 320)
        )
        assert not pixels_close(
            pixel_at(rendered, 2.5, 110, 110, 320), pixel_at(clip, 2.5, 110, 110, 320)
        )

    async def test_scaling_the_overlay(self, settings: Settings, clip: Path, logo: Path) -> None:
        record = await run_job_ok(
            "overlay_media",
            {"input_path": str(clip), "overlay_path": str(logo), "width": 16},
            settings,
        )
        assert record.command is not None and "scale=w=16" in record.command

    async def test_chroma_keying_renders(
        self, settings: Settings, clip: Path, green_clip: Path
    ) -> None:
        record = await run_job_ok(
            "overlay_media",
            {
                "input_path": str(clip),
                "overlay_path": str(green_clip),
                "chroma_key": "#00FF00",
            },
            settings,
        )
        assert output_path(record).exists()
        assert any("Keyed out" in note for note in record.result["notes"])

    async def test_audio_is_preserved(self, settings: Settings, clip: Path, logo: Path) -> None:
        record = await run_job_ok(
            "overlay_media", {"input_path": str(clip), "overlay_path": str(logo)}, settings
        )
        info = await probe_output(record, settings)
        assert info.has_audio is True


class TestResize:
    async def test_a_reel_preset_produces_a_vertical_video(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "resize_video", {"input_path": str(clip), "preset": "reel"}, settings
        )
        info = await probe_output(record, settings)
        assert (info.video_streams[0].width, info.video_streams[0].height) == (1080, 1920)

    @pytest.mark.parametrize(
        ("preset", "expected"),
        [
            ("tiktok", (1080, 1920)),
            ("youtube_short", (1080, 1920)),
            ("instagram_square", (1080, 1080)),
            ("youtube_720p", (1280, 720)),
        ],
    )
    async def test_presets_produce_their_documented_sizes(
        self, settings: Settings, clip: Path, preset: str, expected: tuple[int, int]
    ) -> None:
        record = await run_job_ok(
            "resize_video", {"input_path": str(clip), "preset": preset}, settings
        )
        info = await probe_output(record, settings)
        assert (info.video_streams[0].width, info.video_streams[0].height) == expected

    async def test_an_aspect_ratio_alone_reshapes_the_video(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "resize_video", {"input_path": str(clip), "aspect_ratio": "9:16"}, settings
        )
        info = await probe_output(record, settings)
        width, height = info.video_streams[0].width, info.video_streams[0].height
        assert width is not None and height is not None
        assert width / height == pytest.approx(9 / 16, abs=0.02)

    async def test_explicit_dimensions(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "resize_video", {"input_path": str(clip), "width": 640, "height": 360}, settings
        )
        info = await probe_output(record, settings)
        assert (info.video_streams[0].width, info.video_streams[0].height) == (640, 360)

    async def test_a_width_alone_preserves_the_aspect_ratio(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok("resize_video", {"input_path": str(clip), "width": 160}, settings)
        info = await probe_output(record, settings)
        assert (info.video_streams[0].width, info.video_streams[0].height) == (160, 120)

    async def test_contain_adds_solid_bars(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "resize_video",
            {
                "input_path": str(clip),
                "preset": "reel",
                "fit": "contain",
                "background_color": "white",
            },
            settings,
        )
        data = raw_frame(output_path(record), 1.0)
        # The very top row is a bar, so it should be the requested white.
        assert data[0] > 200 and data[1] > 200 and data[2] > 200

    async def test_cover_fills_the_frame_without_bars(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "resize_video",
            {"input_path": str(clip), "preset": "reel", "fit": "cover"},
            settings,
        )
        rendered = output_path(record)
        contained = await run_job_ok(
            "resize_video",
            {
                "input_path": str(clip),
                "preset": "reel",
                "fit": "contain",
                "background_color": "black",
                "output_path": str(settings.workspace / "contained.mp4"),
            },
            settings,
        )
        # A cover crop has picture where contain has bars, so the top rows differ.
        assert raw_frame(rendered, 1.0)[:3000] != raw_frame(output_path(contained), 1.0)[:3000]

    async def test_blur_fills_the_bars_with_picture_not_flat_colour(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "resize_video",
            {"input_path": str(clip), "preset": "reel", "fit": "blur"},
            settings,
        )
        data = raw_frame(output_path(record), 1.0)
        # Sample the top band, which would be flat black with fit='contain'.
        band = data[: 1080 * 200 * 3]
        assert sum(band) / len(band) > 12, "the blurred background band is flat"

    async def test_horizontal_focus_changes_which_part_survives(
        self, settings: Settings, clip: Path
    ) -> None:
        # A 4:3 source cropped to 1:1 overflows horizontally, so left vs right
        # is the axis that moves here.
        left = await run_job_ok(
            "resize_video",
            {
                "input_path": str(clip),
                "aspect_ratio": "1:1",
                "fit": "cover",
                "focus": "left",
                "output_path": str(settings.workspace / "left.mp4"),
            },
            settings,
        )
        right = await run_job_ok(
            "resize_video",
            {
                "input_path": str(clip),
                "aspect_ratio": "1:1",
                "fit": "cover",
                "focus": "right",
                "output_path": str(settings.workspace / "right.mp4"),
            },
            settings,
        )
        assert frame_differs(output_path(left), output_path(right))

    async def test_vertical_focus_changes_which_part_survives(
        self, settings: Settings, clip: Path
    ) -> None:
        # A 4:3 source cropped to 16:9 overflows vertically instead.
        top = await run_job_ok(
            "resize_video",
            {
                "input_path": str(clip),
                "width": 640,
                "height": 360,
                "fit": "cover",
                "focus": "top",
                "output_path": str(settings.workspace / "top.mp4"),
            },
            settings,
        )
        bottom = await run_job_ok(
            "resize_video",
            {
                "input_path": str(clip),
                "width": 640,
                "height": 360,
                "fit": "cover",
                "focus": "bottom",
                "output_path": str(settings.workspace / "bottom.mp4"),
            },
            settings,
        )
        assert frame_differs(output_path(top), output_path(bottom))

    async def test_stretch_distorts_rather_than_cropping(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "resize_video",
            {"input_path": str(clip), "preset": "reel", "fit": "stretch"},
            settings,
        )
        info = await probe_output(record, settings)
        assert (info.video_streams[0].width, info.video_streams[0].height) == (1080, 1920)
        assert record.command is not None
        assert "force_original_aspect_ratio" not in record.command

    async def test_a_shape_change_is_reported_in_the_notes(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "resize_video", {"input_path": str(clip), "preset": "reel"}, settings
        )
        assert any("cropped" in note for note in record.result["notes"])
        assert record.result["source_width"] == 320
        assert record.result["width"] == 1080

    async def test_audio_and_duration_survive(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "resize_video", {"input_path": str(clip), "preset": "reel"}, settings
        )
        info = await probe_output(record, settings)
        assert info.has_audio is True
        assert info.duration == pytest.approx(3.0, abs=0.3)

    async def test_an_unknown_preset_is_rejected_before_queueing(
        self, settings: Settings, clip: Path
    ) -> None:
        from ffmpeg_mcp.errors import InvalidParameterError

        with pytest.raises(InvalidParameterError, match="preset"):
            await call_tool("resize_video", {"input_path": str(clip), "preset": "myspace"})

    async def test_no_target_at_all_is_rejected(self, settings: Settings, clip: Path) -> None:
        with pytest.raises(Exception, match="preset"):
            await call_tool("resize_video", {"input_path": str(clip)})

    async def test_the_preset_list_is_available(self, settings: Settings) -> None:
        result = await call_tool("list_resolution_presets", {})
        assert result["presets"]["reel"] == [1080, 1920]
        assert set(result["fit_modes"]) == {"cover", "contain", "blur", "stretch"}
        assert "center" in result["focus_points"]


class TestAudioTools:
    async def test_fade_in_lowers_the_opening_level(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok("fade_audio", {"input_path": str(clip), "fade_in": 1.5}, settings)
        assert mean_volume_db(output_path(record)) < mean_volume_db(clip)

    async def test_fade_out_is_placed_from_the_end(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "fade_audio", {"input_path": str(clip), "fade_out": 1.0}, settings
        )
        assert record.command is not None
        assert "afade=t=out:st=2" in record.command

    async def test_video_is_stream_copied_during_an_audio_fade(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok("fade_audio", {"input_path": str(clip), "fade_in": 0.5}, settings)
        assert record.command is not None and "-c:v copy" in record.command
        info = await probe_output(record, settings)
        assert info.has_video is True

    async def test_a_fade_with_no_lengths_is_rejected(self, settings: Settings, clip: Path) -> None:
        with pytest.raises(Exception, match="fade_in or a fade_out"):
            await call_tool("fade_audio", {"input_path": str(clip)})

    async def test_normalisation_moves_the_level_towards_the_target(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "normalize_audio", {"input_path": str(clip), "target_lufs": -20.0}, settings
        )
        assert any("Normalised to -20" in note for note in record.result["notes"])
        info = await probe_output(record, settings)
        assert info.has_audio is True

    async def test_two_pass_normalisation_reports_the_measurement(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "normalize_audio",
            {"input_path": str(clip), "target_lufs": -16.0, "two_pass": True},
            settings,
        )
        assert any("Measured" in note for note in record.result["notes"])

    async def test_single_pass_normalisation_also_works(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "normalize_audio",
            {"input_path": str(clip), "two_pass": False},
            settings,
        )
        assert not any("Measured" in note for note in record.result["notes"])
        assert output_path(record).exists()

    async def test_normalising_a_file_with_no_audio_fails_clearly(
        self, settings: Settings, cuts_clip: Path
    ) -> None:
        record = await run_job("normalize_audio", {"input_path": str(cuts_clip)}, settings)
        assert record.status is JobStatus.FAILED
        assert record.error is not None
        assert record.error.code == "invalid_parameter"


class TestMixAudio:
    async def test_music_is_mixed_under_the_voice(
        self, settings: Settings, speech_clip: Path, clip: Path
    ) -> None:
        record = await run_job_ok(
            "mix_audio",
            {
                "voice_path": str(speech_clip),
                "tracks": [{"path": str(clip), "gain_db": -12.0}],
            },
            settings,
        )
        info = await probe_output(record, settings)
        assert info.has_audio is True
        assert any("Mixed 1 track" in note for note in record.result["notes"])

    async def test_the_video_is_carried_through(
        self, settings: Settings, speech_clip: Path, clip: Path
    ) -> None:
        record = await run_job_ok(
            "mix_audio",
            {"voice_path": str(speech_clip), "tracks": [{"path": str(clip)}]},
            settings,
        )
        info = await probe_output(record, settings)
        assert info.has_video is True

    async def test_ducking_engages_the_compressor(
        self, settings: Settings, speech_clip: Path, clip: Path
    ) -> None:
        record = await run_job_ok(
            "mix_audio",
            {
                "voice_path": str(speech_clip),
                "tracks": [{"path": str(clip), "duck": True}],
            },
            settings,
        )
        assert record.command is not None and "sidechaincompress" in record.command
        assert any("Ducking" in note for note in record.result["notes"])

    async def test_gain_changes_the_mixed_level(
        self, settings: Settings, speech_clip: Path, clip: Path
    ) -> None:
        loud = await run_job_ok(
            "mix_audio",
            {
                "voice_path": str(speech_clip),
                "tracks": [{"path": str(clip), "gain_db": 0.0}],
                "output_path": str(settings.workspace / "loud.mp4"),
            },
            settings,
        )
        quiet = await run_job_ok(
            "mix_audio",
            {
                "voice_path": str(speech_clip),
                "tracks": [{"path": str(clip), "gain_db": -30.0}],
                "output_path": str(settings.workspace / "quiet.mp4"),
            },
            settings,
        )
        assert mean_volume_db(output_path(quiet)) < mean_volume_db(output_path(loud))

    async def test_a_track_without_audio_is_rejected(
        self, settings: Settings, speech_clip: Path, cuts_clip: Path
    ) -> None:
        record = await run_job(
            "mix_audio",
            {"voice_path": str(speech_clip), "tracks": [{"path": str(cuts_clip)}]},
            settings,
        )
        assert record.status is JobStatus.FAILED
        assert record.error is not None
        assert record.error.code == "invalid_parameter"


class TestRenderTimeline:
    async def test_a_single_clip_timeline_renders(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "render_timeline",
            {
                "timeline": {
                    "width": 320,
                    "height": 240,
                    "fps": 30,
                    "clips": [{"source": str(clip), "in_point": 0.0, "out_point": 2.0}],
                }
            },
            settings,
        )
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(2.0, abs=0.2)
        assert (info.video_streams[0].width, info.video_streams[0].height) == (320, 240)

    async def test_clips_of_different_sizes_are_conformed_and_joined(
        self, settings: Settings, clip: Path, clip_alt: Path
    ) -> None:
        record = await run_job_ok(
            "render_timeline",
            {
                "timeline": {
                    "width": 480,
                    "height": 270,
                    "clips": [
                        {"source": str(clip), "out_point": 2.0},
                        {"source": str(clip_alt), "out_point": 1.5},
                    ],
                }
            },
            settings,
        )
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(3.5, abs=0.3)
        assert (info.video_streams[0].width, info.video_streams[0].height) == (480, 270)

    async def test_a_transition_shortens_the_render(
        self, settings: Settings, clip: Path, clip_alt: Path
    ) -> None:
        record = await run_job_ok(
            "render_timeline",
            {
                "timeline": {
                    "width": 320,
                    "height": 240,
                    "clips": [
                        {
                            "source": str(clip),
                            "out_point": 2.0,
                            "transition_to_next": {"type": "fade", "duration": 0.5},
                        },
                        {"source": str(clip_alt), "out_point": 2.0},
                    ],
                }
            },
            settings,
        )
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(3.5, abs=0.3)

    async def test_clip_speed_is_applied(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "render_timeline",
            {
                "timeline": {
                    "width": 320,
                    "height": 240,
                    "clips": [{"source": str(clip), "out_point": 3.0, "speed": 2.0}],
                }
            },
            settings,
        )
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(1.5, abs=0.2)

    async def test_text_overlays_render_and_stay_out_of_the_graph(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "render_timeline",
            {
                "timeline": {
                    "width": 320,
                    "height": 240,
                    "clips": [{"source": str(clip), "out_point": 3.0}],
                    "text_overlays": [
                        {
                            "text": "Time: 12:30, [x]; it's 50%",
                            "start": 0.0,
                            "end": 3.0,
                            "font_size": 20,
                        }
                    ],
                }
            },
            settings,
        )
        assert frame_differs(clip, output_path(record))
        assert record.command is not None and "12:30" not in record.command

    async def test_a_media_overlay_renders(
        self, settings: Settings, clip: Path, logo: Path
    ) -> None:
        record = await run_job_ok(
            "render_timeline",
            {
                "timeline": {
                    "width": 320,
                    "height": 240,
                    "clips": [{"source": str(clip), "out_point": 2.0}],
                    "media_overlays": [
                        {"source": str(logo), "position": "top-left", "margin": 0, "width": 32}
                    ],
                }
            },
            settings,
        )
        # Sample away from the base clip's own red corner, and require both a
        # change from the source and the logo's colour.
        assert pixels_close(pixel_at(output_path(record), 1.0, 10, 10, 320), (255, 0, 0))
        assert not pixels_close(
            pixel_at(output_path(record), 1.0, 20, 20, 320),
            pixel_at(clip, 1.0, 20, 20, 320),
        )

    async def test_an_extra_audio_track_is_mixed(
        self, settings: Settings, clip: Path, speech: Path
    ) -> None:
        record = await run_job_ok(
            "render_timeline",
            {
                "timeline": {
                    "width": 320,
                    "height": 240,
                    "clips": [{"source": str(clip), "out_point": 3.0}],
                    "audio_tracks": [
                        {"source": str(speech), "gain_db": -6.0, "duck_under_voice": True}
                    ],
                }
            },
            settings,
        )
        info = await probe_output(record, settings)
        assert info.has_audio is True
        assert record.command is not None and "sidechaincompress" in record.command

    async def test_burned_captions_render(self, settings: Settings, clip: Path) -> None:
        subtitle = settings.workspace / "subs.srt"
        subtitle.write_text("1\n00:00:00,000 --> 00:00:03,000\nTIMELINE CAPTION\n")
        record = await run_job_ok(
            "render_timeline",
            {
                "timeline": {
                    "width": 320,
                    "height": 240,
                    "clips": [{"source": str(clip), "out_point": 3.0}],
                    "captions": {"subtitle_path": str(subtitle), "font_size": 16},
                }
            },
            settings,
        )
        assert frame_differs(clip, output_path(record))

    async def test_a_silent_clip_is_padded_with_silence(
        self, settings: Settings, clip: Path, cuts_clip: Path
    ) -> None:
        record = await run_job_ok(
            "render_timeline",
            {
                "timeline": {
                    "width": 320,
                    "height": 240,
                    "clips": [
                        {"source": str(cuts_clip), "out_point": 2.0},
                        {"source": str(clip), "out_point": 2.0},
                    ],
                }
            },
            settings,
        )
        info = await probe_output(record, settings)
        assert info.has_audio is True
        assert info.duration == pytest.approx(4.0, abs=0.3)
        assert any("silence" in note for note in record.result["notes"])

    async def test_the_everything_timeline_renders(
        self, settings: Settings, clip: Path, clip_alt: Path, logo: Path, speech: Path
    ) -> None:
        """Clips, a transition, text, an overlay, captions, music, normalisation."""
        subtitle = settings.workspace / "full.srt"
        subtitle.write_text("1\n00:00:00,500 --> 00:00:02,000\nEverything at once\n")
        record = await run_job_ok(
            "render_timeline",
            {
                "timeline": {
                    "width": 480,
                    "height": 270,
                    "fps": 25,
                    "clips": [
                        {
                            "source": str(clip),
                            "out_point": 2.0,
                            "transition_to_next": {"type": "wipeleft", "duration": 0.5},
                        },
                        {"source": str(clip_alt), "out_point": 1.5, "speed": 1.5},
                    ],
                    "text_overlays": [
                        {"text": "TITLE", "start": 0.0, "end": 1.5, "fade": 0.3, "font_size": 24}
                    ],
                    "media_overlays": [{"source": str(logo), "width": 40, "opacity": 0.8}],
                    "audio_tracks": [
                        {
                            "source": str(speech),
                            "gain_db": -10.0,
                            "fade_in": 0.3,
                            "duck_under_voice": True,
                        }
                    ],
                    "captions": {"subtitle_path": str(subtitle), "font_size": 14},
                    "normalize_audio": True,
                    "target_lufs": -18.0,
                }
            },
            settings,
        )
        info = await probe_output(record, settings)
        assert info.has_video and info.has_audio
        assert (info.video_streams[0].width, info.video_streams[0].height) == (480, 270)
        assert info.duration == pytest.approx(2.5, abs=0.4)

    async def test_an_empty_timeline_is_rejected(self, settings: Settings) -> None:
        with pytest.raises(Exception, match="at least 1"):
            await call_tool("render_timeline", {"timeline": {"clips": []}})

    async def test_a_missing_source_is_rejected_before_queueing(self, settings: Settings) -> None:
        from ffmpeg_mcp.errors import InvalidPathError

        with pytest.raises(InvalidPathError):
            await call_tool(
                "render_timeline",
                {"timeline": {"clips": [{"source": str(settings.workspace / "nope.mp4")}]}},
            )


class TestInspectionTools:
    """The editing loop needs to look at and measure media, not just render it."""

    async def test_a_frame_can_be_extracted_and_is_a_real_image(
        self, settings: Settings, clip: Path
    ) -> None:
        result = await call_tool(
            "extract_frame", {"input_path": str(clip), "time": 1.5, "width": 160}
        )
        assert Path(result["output_path"]).exists()
        assert result["width"] == 160
        assert result["time"] == 1.5

    async def test_extracting_past_the_end_is_refused(self, settings: Settings, clip: Path) -> None:
        from ffmpeg_mcp.errors import InvalidParameterError

        with pytest.raises(InvalidParameterError, match="past the end"):
            await call_tool("extract_frame", {"input_path": str(clip), "time": 99.0})

    async def test_a_filmstrip_tiles_the_requested_frames(
        self, settings: Settings, clip: Path
    ) -> None:
        result = await call_tool(
            "extract_filmstrip",
            {"input_path": str(clip), "count": 4, "columns": 4, "tile_width": 80},
        )
        assert len(result["times"]) == 4
        assert result["rows"] == 1
        sheet = await call_tool("probe_media", {"input_path": result["output_path"]})
        # Four 80px tiles side by side.
        assert sheet["video_streams"][0]["width"] == pytest.approx(320, abs=8)

    async def test_a_filmstrip_wraps_onto_multiple_rows(
        self, settings: Settings, clip: Path
    ) -> None:
        result = await call_tool(
            "extract_filmstrip",
            {"input_path": str(clip), "count": 6, "columns": 3, "tile_width": 60},
        )
        assert result["rows"] == 2

    async def test_analysis_measures_brightness_and_colour(
        self, settings: Settings, clip: Path
    ) -> None:
        result = await call_tool("analyze_video", {"input_path": str(clip), "count": 4})
        assert len(result["frames"]) == 4
        assert 0 < result["luma_avg"] < 255
        assert result["is_greyscale"] is False  # the fixture is a colour test pattern

    async def test_analysis_detects_a_greyscale_render(
        self, settings: Settings, clip: Path
    ) -> None:
        # Grade the colour out, then confirm the analyser notices.
        grey = await run_job_ok(
            "color_grade", {"input_path": str(clip), "saturation": 0.0}, settings
        )
        result = await call_tool(
            "analyze_video", {"input_path": str(output_path(grey)), "count": 3}
        )
        assert result["is_greyscale"] is True
        assert any("greyscale" in n for n in result["notes"])

    async def test_analysis_detects_crushed_blacks(self, settings: Settings, clip: Path) -> None:
        dark = await run_job_ok(
            "color_grade",
            {"input_path": str(clip), "brightness": -0.85, "contrast": 1.5},
            settings,
        )
        result = await call_tool(
            "analyze_video", {"input_path": str(output_path(dark)), "count": 3}
        )
        assert result["crushed_blacks"] is True
        assert any("shadow detail" in n for n in result["notes"])

    async def test_audio_is_measured(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok("measure_audio", {"input_path": str(clip)}, settings)
        assert record.result["mean_volume_db"] is not None
        assert record.result["max_volume_db"] is not None

    async def test_measuring_a_silent_file_says_so(
        self, settings: Settings, cuts_clip: Path
    ) -> None:
        record = await run_job("measure_audio", {"input_path": str(cuts_clip)}, settings)
        assert record.status is JobStatus.FAILED
        assert record.error is not None
        assert record.error.code == "invalid_parameter"

    async def test_inspection_needs_a_video_stream(self, settings: Settings, speech: Path) -> None:
        from ffmpeg_mcp.errors import InvalidParameterError

        with pytest.raises(InvalidParameterError, match="no video"):
            await call_tool("extract_frame", {"input_path": str(speech), "time": 0.5})
