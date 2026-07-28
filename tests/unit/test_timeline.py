"""Unit tests for the timeline model and its filter-graph compiler."""

from __future__ import annotations

import pytest

from ffmpeg_mcp.errors import InvalidParameterError
from ffmpeg_mcp.models import AudioStreamInfo, MediaInfo, VideoStreamInfo
from ffmpeg_mcp.timeline import (
    Timeline,
    TimelineAudioTrack,
    TimelineCaptions,
    TimelineClip,
    TimelineMediaOverlay,
    TimelineTextOverlay,
    Transition,
    compile_timeline,
    overlay_position_expressions,
)
from ffmpeg_mcp.tools.composition import measured_loudnorm_filter, parse_loudnorm_json


def media(duration: float = 10.0, *, audio: bool = True, width: int = 1920) -> MediaInfo:
    """A probe result standing in for a real file."""
    return MediaInfo(
        path="/tmp/x.mp4",
        duration=duration,
        video_streams=[VideoStreamInfo(index=0, codec="h264", width=width, height=1080, fps=30.0)],
        audio_streams=(
            [AudioStreamInfo(index=1, codec="aac", sample_rate=48000, channels=2)] if audio else []
        ),
    )


def one_clip(**kwargs: object) -> Timeline:
    return Timeline(clips=[TimelineClip(source="/tmp/a.mp4", **kwargs)], **{})  # type: ignore[arg-type]


class TestClipDurations:
    def test_full_clip_duration(self) -> None:
        clip = TimelineClip(source="/tmp/a.mp4")
        assert clip.output_duration(media(10.0)) == 10.0

    def test_in_and_out_points(self) -> None:
        clip = TimelineClip(source="/tmp/a.mp4", in_point=2.0, out_point=7.0)
        assert clip.output_duration(media(10.0)) == 5.0

    def test_speed_shortens_the_output(self) -> None:
        clip = TimelineClip(source="/tmp/a.mp4", in_point=0.0, out_point=8.0, speed=2.0)
        assert clip.output_duration(media(10.0)) == 4.0

    def test_slow_motion_lengthens_the_output(self) -> None:
        clip = TimelineClip(source="/tmp/a.mp4", out_point=4.0, speed=0.5)
        assert clip.output_duration(media(10.0)) == 8.0

    def test_an_inverted_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="out_point must be greater"):
            TimelineClip(source="/tmp/a.mp4", in_point=5.0, out_point=2.0)


class TestTransitionValidation:
    def test_a_known_transition_is_accepted(self) -> None:
        assert Transition(type="wipeleft").type == "wipeleft"

    def test_an_unknown_transition_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown transition"):
            Transition(type="teleport")

    def test_a_zero_duration_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            Transition(duration=0)


class TestOverlayPositions:
    def test_overlay_uses_uppercase_base_dimensions(self) -> None:
        # W/H are the base frame and w/h the overlay — the reverse of drawtext.
        x, y = overlay_position_expressions("bottom-right", 20)
        assert x == "W-w-20"
        assert y == "H-h-20"

    def test_centre_position(self) -> None:
        assert overlay_position_expressions("center", 0) == ("(W-w)/2", "(H-h)/2")

    def test_an_unknown_position_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            overlay_position_expressions("nowhere", 10)


class TestCompileBasics:
    def test_a_single_clip_compiles(self) -> None:
        timeline = Timeline(clips=[TimelineClip(source="/tmp/a.mp4")], width=640, height=360)
        compiled = compile_timeline(timeline, [media(10.0)], [], [])
        assert compiled.inputs == [["-i", "/tmp/a.mp4"]]
        assert compiled.duration == 10.0
        assert compiled.video_label == "vout"
        assert compiled.audio_label == "aout"

    def test_the_clip_is_conformed_to_the_timeline_format(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")], width=640, height=360, fps=25
        )
        graph = compile_timeline(timeline, [media()], [], []).graph
        assert "scale=w=640:h=360" in graph
        assert "pad=w=640:h=360" in graph
        assert "fps=fps=25" in graph
        assert "setsar=1" in graph

    def test_in_and_out_points_become_a_trim(self) -> None:
        timeline = Timeline(clips=[TimelineClip(source="/tmp/a.mp4", in_point=1.5, out_point=4.0)])
        graph = compile_timeline(timeline, [media()], [], []).graph
        assert "trim=start=1.500000:end=4.000000" in graph
        assert "atrim=start=1.500000:end=4.000000" in graph

    def test_a_zero_length_clip_is_rejected(self) -> None:
        timeline = Timeline(clips=[TimelineClip(source="/tmp/a.mp4", in_point=5.0)])
        with pytest.raises(InvalidParameterError, match="zero length"):
            compile_timeline(timeline, [media(5.0)], [], [])

    def test_a_clip_without_video_is_rejected(self) -> None:
        timeline = Timeline(clips=[TimelineClip(source="/tmp/a.mp3")])
        audio_only = MediaInfo(
            path="/tmp/a.mp3",
            duration=5.0,
            audio_streams=[AudioStreamInfo(index=0, codec="mp3", channels=2)],
        )
        with pytest.raises(InvalidParameterError, match="video stream"):
            compile_timeline(timeline, [audio_only], [], [])

    def test_a_missing_probe_is_rejected(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4"), TimelineClip(source="/tmp/b.mp4")]
        )
        with pytest.raises(InvalidParameterError, match="every clip"):
            compile_timeline(timeline, [media()], [], [])

    def test_a_silent_clip_gets_synthesised_silence(self) -> None:
        # Otherwise a later concat would misalign the whole mix.
        timeline = Timeline(clips=[TimelineClip(source="/tmp/a.mp4")])
        compiled = compile_timeline(timeline, [media(audio=False)], [], [])
        assert "anullsrc" in compiled.graph
        assert any("no audio" in note for note in compiled.notes)

    def test_a_muted_clip_also_gets_silence(self) -> None:
        timeline = Timeline(clips=[TimelineClip(source="/tmp/a.mp4", mute=True)])
        assert "anullsrc" in compile_timeline(timeline, [media()], [], []).graph


class TestCompileJoins:
    def test_clips_without_transitions_are_concatenated(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4"), TimelineClip(source="/tmp/b.mp4")]
        )
        compiled = compile_timeline(timeline, [media(5.0), media(3.0)], [], [])
        assert "concat=n=2" in compiled.graph
        assert "xfade" not in compiled.graph
        assert compiled.duration == 8.0

    def test_a_transition_shortens_the_total_by_its_overlap(self) -> None:
        timeline = Timeline(
            clips=[
                TimelineClip(
                    source="/tmp/a.mp4", transition_to_next=Transition(type="fade", duration=1.0)
                ),
                TimelineClip(source="/tmp/b.mp4"),
            ]
        )
        compiled = compile_timeline(timeline, [media(5.0), media(3.0)], [], [])
        assert compiled.duration == pytest.approx(7.0)
        assert "xfade=transition=fade:duration=1:offset=4" in compiled.graph
        assert "acrossfade" in compiled.graph

    def test_transition_offsets_account_for_earlier_transitions(self) -> None:
        # The second transition's offset must be measured against the already
        # shortened running total, not the sum of raw clip lengths.
        transition = Transition(type="fade", duration=1.0)
        timeline = Timeline(
            clips=[
                TimelineClip(source="/tmp/a.mp4", transition_to_next=transition),
                TimelineClip(source="/tmp/b.mp4", transition_to_next=transition),
                TimelineClip(source="/tmp/c.mp4"),
            ]
        )
        compiled = compile_timeline(timeline, [media(5.0), media(5.0), media(5.0)], [], [])
        assert compiled.duration == pytest.approx(13.0)
        assert "offset=4" in compiled.graph  # first join: 5 - 1
        assert "offset=8" in compiled.graph  # second join: (5+5-1) - 1

    def test_mixed_cuts_and_transitions(self) -> None:
        timeline = Timeline(
            clips=[
                TimelineClip(source="/tmp/a.mp4"),
                TimelineClip(
                    source="/tmp/b.mp4", transition_to_next=Transition(type="fade", duration=0.5)
                ),
                TimelineClip(source="/tmp/c.mp4"),
            ]
        )
        compiled = compile_timeline(timeline, [media(2.0), media(2.0), media(2.0)], [], [])
        assert "concat=n=2" in compiled.graph
        assert "xfade" in compiled.graph
        assert compiled.duration == pytest.approx(5.5)

    def test_a_transition_longer_than_its_clips_is_clamped_not_broken(self) -> None:
        timeline = Timeline(
            clips=[
                TimelineClip(
                    source="/tmp/a.mp4", transition_to_next=Transition(type="fade", duration=10.0)
                ),
                TimelineClip(source="/tmp/b.mp4"),
            ]
        )
        compiled = compile_timeline(timeline, [media(2.0), media(2.0)], [], [])
        # Clamped to the shorter clip, so the render still succeeds.
        assert compiled.duration == pytest.approx(2.0)
        assert "duration=2" in compiled.graph

    def test_the_last_clips_transition_is_ignored(self) -> None:
        timeline = Timeline(
            clips=[
                TimelineClip(
                    source="/tmp/a.mp4", transition_to_next=Transition(type="fade", duration=1.0)
                )
            ]
        )
        compiled = compile_timeline(timeline, [media(5.0)], [], [])
        assert "xfade" not in compiled.graph
        assert compiled.duration == 5.0


class TestCompileOverlays:
    def test_a_text_overlay_becomes_drawtext_with_a_window(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")],
            text_overlays=[TimelineTextOverlay(text="Hello", start=1.0, end=3.0)],
        )
        graph = compile_timeline(timeline, [media()], [], []).graph
        assert "drawtext" in graph
        assert "enable='between(t,1,3)'" in graph

    def test_text_is_taken_from_a_sidecar_file_when_provided(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")],
            text_overlays=[TimelineTextOverlay(text="Time: 12:30, [x]", start=0.0, end=2.0)],
        )
        graph = compile_timeline(timeline, [media()], [], [], text_files=["/tmp/t0.txt"]).graph
        assert "textfile=/tmp/t0.txt" in graph
        assert "12:30" not in graph

    def test_a_fading_text_overlay_gets_an_alpha_expression(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")],
            text_overlays=[TimelineTextOverlay(text="Hi", start=0.0, end=4.0, fade=0.5)],
        )
        assert "alpha=" in compile_timeline(timeline, [media()], [], []).graph

    def test_a_media_overlay_adds_an_input_and_an_overlay_filter(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")],
            media_overlays=[TimelineMediaOverlay(source="/tmp/logo.png", width=120)],
        )
        compiled = compile_timeline(timeline, [media()], [None], [])
        assert compiled.inputs == [["-i", "/tmp/a.mp4"], ["-i", "/tmp/logo.png"]]
        assert "overlay=" in compiled.graph
        assert "scale=w=120" in compiled.graph

    def test_a_semi_transparent_overlay_gets_an_alpha_mixer(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")],
            media_overlays=[TimelineMediaOverlay(source="/tmp/logo.png", opacity=0.5)],
        )
        graph = compile_timeline(timeline, [media()], [None], []).graph
        assert "colorchannelmixer=aa=0.5000" in graph

    def test_captions_become_a_subtitles_filter(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")],
            captions=TimelineCaptions(subtitle_path="/tmp/subs.srt", font_size=30),
        )
        graph = compile_timeline(timeline, [media()], [], []).graph
        assert "subtitles=f=/tmp/subs.srt" in graph
        assert "FontSize=30" in graph


class TestCompileAudio:
    def test_an_extra_audio_track_is_mixed_in(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")],
            audio_tracks=[TimelineAudioTrack(source="/tmp/music.mp3", gain_db=-6.0)],
        )
        compiled = compile_timeline(timeline, [media()], [], [media(30.0)])
        assert compiled.inputs[-1] == ["-i", "/tmp/music.mp3"]
        assert "amix=inputs=2" in compiled.graph
        assert "volume=-6dB" in compiled.graph

    def test_a_delayed_track_gets_adelay(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")],
            audio_tracks=[TimelineAudioTrack(source="/tmp/music.mp3", start=2.5)],
        )
        assert "adelay=2500:all=1" in compile_timeline(timeline, [media()], [], [media(30.0)]).graph

    def test_track_fades_are_applied(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")],
            audio_tracks=[
                TimelineAudioTrack(source="/tmp/m.mp3", fade_in=1.0, fade_out=2.0, out_point=10.0)
            ],
        )
        graph = compile_timeline(timeline, [media()], [], [media(30.0)]).graph
        assert "afade=t=in:st=0:d=1" in graph
        assert "afade=t=out:st=8:d=2" in graph

    def test_ducking_splits_the_voice_and_adds_a_compressor(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")],
            audio_tracks=[TimelineAudioTrack(source="/tmp/music.mp3", duck_under_voice=True)],
        )
        compiled = compile_timeline(timeline, [media()], [], [media(30.0)])
        assert "asplit" in compiled.graph
        assert "sidechaincompress" in compiled.graph
        assert any("ducked" in note for note in compiled.notes)

    def test_two_ducked_tracks_each_get_their_own_key(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")],
            audio_tracks=[
                TimelineAudioTrack(source="/tmp/m1.mp3", duck_under_voice=True),
                TimelineAudioTrack(source="/tmp/m2.mp3", duck_under_voice=True),
            ],
        )
        graph = compile_timeline(timeline, [media()], [], [media(30.0), media(30.0)]).graph
        assert graph.count("sidechaincompress") == 2
        # Each compressor consumes its key stream, so it must be split again.
        assert graph.count("asplit") >= 2

    def test_normalisation_is_applied_to_the_final_mix(self) -> None:
        timeline = Timeline(
            clips=[TimelineClip(source="/tmp/a.mp4")], normalize_audio=True, target_lufs=-14.0
        )
        assert "loudnorm=I=-14" in compile_timeline(timeline, [media()], [], []).graph

    def test_no_normalisation_by_default(self) -> None:
        timeline = Timeline(clips=[TimelineClip(source="/tmp/a.mp4")])
        assert "loudnorm" not in compile_timeline(timeline, [media()], [], []).graph


class TestGraphStructure:
    def test_every_produced_label_is_consumed_or_mapped(self) -> None:
        """A dangling label is the classic cause of an 'unconnected output' failure."""
        import re

        timeline = Timeline(
            clips=[
                TimelineClip(
                    source="/tmp/a.mp4", transition_to_next=Transition(type="fade", duration=0.5)
                ),
                TimelineClip(source="/tmp/b.mp4"),
            ],
            text_overlays=[TimelineTextOverlay(text="Hi", end=2.0)],
            media_overlays=[TimelineMediaOverlay(source="/tmp/logo.png")],
            audio_tracks=[TimelineAudioTrack(source="/tmp/m.mp3", duck_under_voice=True)],
        )
        compiled = compile_timeline(timeline, [media(4.0), media(4.0)], [None], [media(30.0)])

        produced: list[str] = []
        consumed: list[str] = []
        for chain in compiled.graph.split(";"):
            labels = re.findall(r"\[([^\]]+)\]", chain)
            body_start = chain.find("]") + 1 if chain.startswith("[") else 0
            leading = re.findall(r"^(?:\[[^\]]+\])+", chain)
            n_in = len(re.findall(r"\[([^\]]+)\]", leading[0])) if leading else 0
            consumed.extend(labels[:n_in])
            produced.extend(labels[n_in:])
            assert body_start >= 0

        stream_refs = {c for c in consumed if ":" in c}
        dangling = set(produced) - set(consumed) - {"vout", "aout"}
        assert not dangling, f"labels produced but never used: {sorted(dangling)}"
        undefined = set(consumed) - set(produced) - stream_refs
        assert not undefined, f"labels used but never produced: {sorted(undefined)}"

    def test_the_graph_has_no_empty_chains(self) -> None:
        timeline = Timeline(clips=[TimelineClip(source="/tmp/a.mp4")])
        for chain in compile_timeline(timeline, [media()], [], []).graph.split(";"):
            assert chain.strip()


class TestLoudnormParsing:
    STDERR = """
[Parsed_loudnorm_0 @ 0x1] noise
{
    "input_i" : "-27.55",
    "input_tp" : "-9.15",
    "input_lra" : "3.30",
    "input_thresh" : "-37.80",
    "output_i" : "-16.02",
    "target_offset" : "0.02"
}
"""

    def test_measurements_are_extracted(self) -> None:
        parsed = parse_loudnorm_json(self.STDERR)
        assert parsed is not None
        assert parsed["input_i"] == "-27.55"

    def test_output_without_json_returns_none(self) -> None:
        assert parse_loudnorm_json("no json at all") is None

    def test_malformed_json_returns_none(self) -> None:
        assert parse_loudnorm_json("{ not json }") is None

    def test_the_second_pass_filter_carries_the_measurements(self) -> None:
        parsed = parse_loudnorm_json(self.STDERR)
        assert parsed is not None
        rendered = measured_loudnorm_filter(parsed, -16.0, -1.5, 11.0).render()
        assert "measured_I=-27.55" in rendered
        assert "measured_TP=-9.15" in rendered
        assert "measured_thresh=-37.80" in rendered
        assert "linear=true" in rendered
