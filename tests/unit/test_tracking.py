"""Unit tests for the pure tracking geometry: association, smoothing, crop paths."""

from __future__ import annotations

import itertools

import pytest

from ffmpeg_mcp.errors import InvalidParameterError
from ffmpeg_mcp.tools.vision import build_scenes, parse_scene_times
from ffmpeg_mcp.vision.detect import analysis_size
from ffmpeg_mcp.vision.tracking import (
    CropKeyframe,
    FaceBox,
    FaceTrack,
    FrameDetections,
    build_crop_path,
    build_tracks,
    clamp_crop_origin,
    crop_size_for_aspect,
    dedupe_keyframes,
    moving_average,
    parse_aspect_ratio,
    primary_track,
    render_sendcmd_script,
    smooth_path,
)


def box(x: float, y: float, size: float = 50, confidence: float = 0.9) -> FaceBox:
    return FaceBox(x=x, y=y, width=size, height=size, confidence=confidence)


class TestFaceBox:
    def test_centre(self) -> None:
        assert box(10, 20, 40).center == (30, 40)

    def test_area(self) -> None:
        assert FaceBox(0, 0, 10, 20, 0.5).area == 200

    def test_identical_boxes_have_iou_one(self) -> None:
        assert box(0, 0).iou(box(0, 0)) == pytest.approx(1.0)

    def test_disjoint_boxes_have_iou_zero(self) -> None:
        assert box(0, 0, 10).iou(box(100, 100, 10)) == 0.0

    def test_touching_boxes_have_iou_zero(self) -> None:
        assert box(0, 0, 10).iou(box(10, 0, 10)) == 0.0

    def test_half_overlap(self) -> None:
        # Two 10x10 boxes offset by 5 in x: overlap 50, union 150.
        assert box(0, 0, 10).iou(box(5, 0, 10)) == pytest.approx(50 / 150)

    def test_expansion_grows_about_the_centre(self) -> None:
        grown = FaceBox(40, 40, 20, 20, 0.9).expanded(2.0, 200, 200)
        assert (grown.x, grown.y, grown.width, grown.height) == (30, 30, 40, 40)
        assert grown.center == (50, 50)

    def test_expansion_is_clamped_to_the_frame(self) -> None:
        grown = FaceBox(0, 0, 20, 20, 0.9).expanded(4.0, 100, 100)
        assert grown.x == 0 and grown.y == 0
        assert grown.x + grown.width <= 100
        assert grown.y + grown.height <= 100


class TestTrackBuilding:
    def test_a_stationary_face_becomes_one_track(self) -> None:
        frames = [FrameDetections(time=i * 0.5, faces=[box(100, 100)]) for i in range(6)]
        tracks = build_tracks(frames)
        assert len(tracks) == 1
        assert len(tracks[0].boxes) == 6

    def test_a_drifting_face_stays_one_track(self) -> None:
        frames = [FrameDetections(time=i * 0.5, faces=[box(100 + i * 5, 100)]) for i in range(6)]
        assert len(build_tracks(frames)) == 1

    def test_two_separated_faces_become_two_tracks(self) -> None:
        frames = [
            FrameDetections(time=i * 0.5, faces=[box(50, 50), box(400, 300)]) for i in range(4)
        ]
        tracks = build_tracks(frames)
        assert len(tracks) == 2
        assert all(len(t.boxes) == 4 for t in tracks)

    def test_a_face_teleporting_starts_a_new_track(self) -> None:
        frames = [
            FrameDetections(time=0.0, faces=[box(0, 0)]),
            FrameDetections(time=0.5, faces=[box(500, 400)]),
        ]
        assert len(build_tracks(frames)) == 2

    def test_a_brief_dropout_is_bridged(self) -> None:
        # A single missed frame must not split one person into two tracks.
        frames = [
            FrameDetections(time=0.0, faces=[box(100, 100)]),
            FrameDetections(time=0.5, faces=[]),
            FrameDetections(time=1.0, faces=[box(100, 100)]),
        ]
        assert len(build_tracks(frames)) == 1

    def test_a_long_gap_is_not_bridged(self) -> None:
        frames = [
            FrameDetections(time=0.0, faces=[box(100, 100)]),
            FrameDetections(time=10.0, faces=[box(100, 100)]),
        ]
        assert len(build_tracks(frames, max_gap_seconds=1.0)) == 2

    def test_no_detections_yields_no_tracks(self) -> None:
        assert build_tracks([FrameDetections(time=0.0, faces=[])]) == []

    def test_track_bounding_box_covers_the_whole_path(self) -> None:
        frames = [
            FrameDetections(time=i * 0.5, faces=[box(100 + i * 5, 100, 50)]) for i in range(5)
        ]
        bounds = build_tracks(frames)[0].bounding_box()
        assert bounds.x == 100
        assert bounds.x + bounds.width == 100 + 4 * 5 + 50


class TestPrimaryTrack:
    def test_none_when_there_are_no_tracks(self) -> None:
        assert primary_track([], 640, 480) is None

    def test_the_bigger_more_central_face_wins(self) -> None:
        frames = [
            FrameDetections(time=i * 0.5, faces=[box(300, 220, 100), box(600, 20, 30)])
            for i in range(6)
        ]
        tracks = build_tracks(frames)
        chosen = primary_track(tracks, 640, 480)
        assert chosen is not None
        assert chosen.bounding_box().width >= 100

    def test_a_fleeting_face_loses_to_a_persistent_one(self) -> None:
        frames = [FrameDetections(time=i * 0.5, faces=[box(300, 220, 60)]) for i in range(10)]
        frames[0].faces.append(box(50, 50, 90))
        chosen = primary_track(build_tracks(frames), 640, 480)
        assert chosen is not None
        assert len(chosen.boxes) == 10


class TestSmoothing:
    def test_a_window_of_one_changes_nothing(self) -> None:
        assert moving_average([1.0, 5.0, 2.0], 1) == [1.0, 5.0, 2.0]

    def test_the_length_is_preserved(self) -> None:
        assert len(moving_average([1.0, 2.0, 3.0, 4.0, 5.0], 3)) == 5

    def test_a_spike_is_flattened(self) -> None:
        smoothed = moving_average([0.0, 0.0, 100.0, 0.0, 0.0], 5)
        assert max(smoothed) < 100.0

    def test_zero_smoothing_leaves_the_path_alone(self) -> None:
        values = [0.0, 50.0, 10.0, 60.0]
        assert smooth_path(values, smoothing=0.0) == values

    def test_smoothing_reduces_jitter(self) -> None:
        jittery = [0.0, 40.0, 0.0, 40.0, 0.0, 40.0, 0.0, 40.0]

        def total_variation(values: list[float]) -> float:
            return sum(abs(b - a) for a, b in itertools.pairwise(values))

        assert total_variation(smooth_path(jittery, smoothing=1.0)) < total_variation(jittery)

    def test_the_step_cap_limits_per_sample_movement(self) -> None:
        path = smooth_path([0.0, 1000.0], smoothing=0.0, max_step=10.0)
        assert path[1] - path[0] <= 10.0

    def test_smoothing_out_of_range_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            smooth_path([1.0, 2.0], smoothing=1.5)

    def test_an_empty_path_is_handled(self) -> None:
        assert smooth_path([], smoothing=0.5) == []


class TestAspectRatio:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("16:9", 16 / 9), ("9:16", 9 / 16), ("1:1", 1.0), ("1.5", 1.5)],
    )
    def test_parsing(self, value: str, expected: float) -> None:
        assert parse_aspect_ratio(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", ["16:9:4", "abc", "0:1", "-1:2", ""])
    def test_invalid_ratios_are_rejected(self, value: str) -> None:
        with pytest.raises(InvalidParameterError):
            parse_aspect_ratio(value)

    def test_vertical_crop_of_a_landscape_frame(self) -> None:
        width, height = crop_size_for_aspect(1920, 1080, 9 / 16)
        assert height == 1080
        assert width == pytest.approx(1080 * 9 / 16, abs=2)

    def test_square_crop_of_a_landscape_frame(self) -> None:
        assert crop_size_for_aspect(1920, 1080, 1.0) == (1080, 1080)

    def test_the_crop_always_fits_inside_the_frame(self) -> None:
        for ratio in (0.5, 1.0, 1.777, 2.35):
            width, height = crop_size_for_aspect(640, 480, ratio)
            assert width <= 640 and height <= 480

    def test_crop_dimensions_are_even_for_yuv420p(self) -> None:
        for ratio in (0.5625, 1.0, 1.777, 2.35):
            width, height = crop_size_for_aspect(1281, 721, ratio)
            assert width % 2 == 0 and height % 2 == 0


class TestClamping:
    def test_a_centred_crop(self) -> None:
        assert clamp_crop_origin(320, 200, 640) == 220

    def test_the_left_edge_is_clamped(self) -> None:
        assert clamp_crop_origin(10, 200, 640) == 0

    def test_the_right_edge_is_clamped(self) -> None:
        assert clamp_crop_origin(630, 200, 640) == 440

    def test_a_crop_as_wide_as_the_frame_pins_to_zero(self) -> None:
        assert clamp_crop_origin(320, 640, 640) == 0


class TestCropPath:
    @staticmethod
    def _track(positions: list[float]) -> FaceTrack:
        return FaceTrack(
            track_id=0,
            times=[i * 0.5 for i in range(len(positions))],
            boxes=[box(p, 200, 80) for p in positions],
        )

    def test_the_path_stays_inside_the_frame(self) -> None:
        path = build_crop_path(
            self._track([0.0, 100.0, 500.0, 560.0]),
            frame_width=640,
            frame_height=480,
            crop_width=270,
            crop_height=480,
            smoothing=0.5,
            duration=2.0,
        )
        for frame in path:
            assert 0 <= frame.x <= 640 - 270
            assert 0 <= frame.y <= 480 - 480

    def test_the_path_spans_the_whole_clip(self) -> None:
        path = build_crop_path(
            self._track([100.0, 200.0]),
            frame_width=640,
            frame_height=480,
            crop_width=270,
            crop_height=480,
            smoothing=0.5,
            duration=10.0,
        )
        assert path[0].time == 0.0
        assert path[-1].time == pytest.approx(10.0)

    def test_movement_is_gradual(self) -> None:
        # A hard jump in detections must not become a hard jump in the crop.
        path = build_crop_path(
            self._track([0.0, 0.0, 560.0, 560.0, 560.0, 560.0]),
            frame_width=640,
            frame_height=480,
            crop_width=270,
            crop_height=480,
            smoothing=0.6,
            duration=3.0,
        )
        steps = [abs(b.x - a.x) for a, b in itertools.pairwise(path)]
        assert max(steps) <= 270 * 0.06 + 1

    def test_an_empty_track_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            build_crop_path(
                FaceTrack(track_id=0),
                frame_width=640,
                frame_height=480,
                crop_width=100,
                crop_height=100,
                smoothing=0.5,
                duration=1.0,
            )


class TestKeyframes:
    def test_static_keyframes_collapse(self) -> None:
        frames = [CropKeyframe(t / 2, 100, 50) for t in range(10)]
        assert len(dedupe_keyframes(frames)) == 2  # first and last

    def test_moving_keyframes_are_kept(self) -> None:
        frames = [CropKeyframe(i / 2, i * 10, 0) for i in range(6)]
        assert len(dedupe_keyframes(frames)) == 6

    def test_the_last_keyframe_is_always_kept(self) -> None:
        frames = [CropKeyframe(0, 0, 0), CropKeyframe(1, 0, 0), CropKeyframe(2, 0, 0)]
        assert dedupe_keyframes(frames)[-1].time == 2

    def test_an_empty_list_is_handled(self) -> None:
        assert dedupe_keyframes([]) == []

    def test_the_sendcmd_script_format(self) -> None:
        script = render_sendcmd_script([CropKeyframe(0.0, 10, 20), CropKeyframe(1.5, 30, 40)])
        assert script == "0.000 crop x 10, crop y 20;\n1.500 crop x 30, crop y 40;\n"

    def test_the_target_filter_is_configurable(self) -> None:
        script = render_sendcmd_script([CropKeyframe(0.0, 1, 2)], "overlay")
        assert script.strip() == "0.000 overlay x 1, overlay y 2;"


class TestAnalysisSize:
    def test_large_frames_are_downscaled(self) -> None:
        assert analysis_size(1920, 1080, 640) == (640, 360)

    def test_small_frames_are_not_upscaled(self) -> None:
        assert analysis_size(320, 240, 640) == (320, 240)

    def test_dimensions_are_even(self) -> None:
        width, height = analysis_size(1921, 1081, 641)
        assert width % 2 == 0 and height % 2 == 0

    def test_zero_dimensions_are_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            analysis_size(0, 100)


class TestSceneParsing:
    def test_pts_times_are_extracted(self) -> None:
        stderr = (
            "[Parsed_showinfo_1 @ 0x1] n:0 pts:1234 pts_time:2.5 pos:1 fmt:yuv420p\n"
            "[Parsed_showinfo_1 @ 0x1] n:1 pts:5678 pts_time:4.25 pos:2 fmt:yuv420p\n"
        )
        assert parse_scene_times(stderr) == [2.5, 4.25]

    def test_output_without_matches_yields_nothing(self) -> None:
        assert parse_scene_times("no scene information here") == []

    def test_scenes_are_built_between_cuts(self) -> None:
        cuts, scenes = build_scenes([2.0, 4.0], duration=6.0, min_scene_seconds=0.5)
        assert cuts == [2.0, 4.0]
        assert [(s.start, s.end) for s in scenes] == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]

    def test_cuts_closer_than_the_minimum_are_dropped(self) -> None:
        cuts, _ = build_scenes([2.0, 2.1, 2.2, 5.0], duration=8.0, min_scene_seconds=0.5)
        assert cuts == [2.0, 5.0]

    def test_a_cut_at_zero_is_ignored(self) -> None:
        cuts, _ = build_scenes([0.0, 3.0], duration=6.0, min_scene_seconds=0.5)
        assert cuts == [3.0]

    def test_no_cuts_yields_a_single_scene(self) -> None:
        _, scenes = build_scenes([], duration=5.0, min_scene_seconds=0.5)
        assert len(scenes) == 1
        assert (scenes[0].start, scenes[0].end) == (0.0, 5.0)

    def test_cuts_are_sorted(self) -> None:
        cuts, _ = build_scenes([5.0, 2.0], duration=8.0, min_scene_seconds=0.5)
        assert cuts == [2.0, 5.0]

    def test_scene_durations_are_consistent(self) -> None:
        _, scenes = build_scenes([2.0, 4.0], duration=6.0, min_scene_seconds=0.5)
        for scene in scenes:
            assert scene.duration == pytest.approx(scene.end - scene.start)
