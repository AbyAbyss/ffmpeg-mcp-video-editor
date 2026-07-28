"""Phase 4 integration tests: face detection, reframing, blurring, scene cuts.

Fixtures use a synthetic drawn face that BlazeFace detects reliably, so these
tests need no photograph of a real person.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from ffmpeg_mcp.config import Settings
from ffmpeg_mcp.models import JobStatus

from .helpers import call_tool, output_path, probe_output, run_job, run_job_ok
from .test_phase2 import mean_saturation, raw_frame

pytestmark = [pytest.mark.integration]


def region_variance(
    path: Path, at_seconds: float, box: tuple[int, int, int, int], frame_width: int = 640
) -> float:
    """Variance of pixel values in a region — a proxy for how blurred it is."""
    data = raw_frame(path, at_seconds)
    x, y, width, height = box
    values: list[int] = []
    for row in range(y, y + height):
        start = (row * frame_width + x) * 3
        values.extend(data[start : start + width * 3])
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


class TestDetectFaces:
    async def test_a_face_is_found_in_every_sampled_frame(
        self, settings: Settings, face_clip: Path
    ) -> None:
        record = await run_job_ok(
            "detect_faces", {"input_path": str(face_clip), "sample_fps": 2.0}, settings
        )
        result = record.result
        assert result["frame_width"] == 640
        assert result["frame_height"] == 480
        assert result["sampled_frames"] >= 6
        detected = [f for f in result["frames"] if f["faces"]]
        assert len(detected) >= result["sampled_frames"] - 1

    async def test_boxes_are_reported_in_pixels_and_normalised(
        self, settings: Settings, face_clip: Path
    ) -> None:
        record = await run_job_ok("detect_faces", {"input_path": str(face_clip)}, settings)
        face = next(f["faces"][0] for f in record.result["frames"] if f["faces"])
        assert 0 < face["width"] <= 640
        assert 0 < face["height"] <= 480
        assert 0.0 <= face["confidence"] <= 1.0
        nx, ny, nw, nh = face["normalized"]
        assert nx == pytest.approx(face["x"] / 640, abs=0.01)
        assert nw == pytest.approx(face["width"] / 640, abs=0.01)
        assert all(0.0 <= v <= 1.0 for v in (nx, ny, nw, nh))

    async def test_the_moving_face_forms_one_track(
        self, settings: Settings, face_clip: Path
    ) -> None:
        record = await run_job_ok(
            "detect_faces", {"input_path": str(face_clip), "sample_fps": 4.0}, settings
        )
        tracks = record.result["tracks"]
        assert len(tracks) == 1
        assert tracks[0]["is_primary"] is True
        assert tracks[0]["detections"] >= 8

    async def test_the_track_follows_the_face_across_the_frame(
        self, settings: Settings, face_clip: Path
    ) -> None:
        record = await run_job_ok(
            "detect_faces", {"input_path": str(face_clip), "sample_fps": 4.0}, settings
        )
        centres = [
            f["faces"][0]["x"] + f["faces"][0]["width"] / 2
            for f in record.result["frames"]
            if f["faces"]
        ]
        # The fixture face drifts left to right, so the path must trend rightwards.
        assert centres[-1] - centres[0] > 200

    async def test_two_faces_produce_two_tracks_with_one_primary(
        self, settings: Settings, two_face_clip: Path
    ) -> None:
        record = await run_job_ok(
            "detect_faces", {"input_path": str(two_face_clip), "sample_fps": 3.0}, settings
        )
        tracks = record.result["tracks"]
        assert len(tracks) == 2
        assert sum(1 for t in tracks if t["is_primary"]) == 1
        assert record.result["max_faces_in_a_frame"] == 2

    async def test_the_larger_central_face_is_chosen_as_primary(
        self, settings: Settings, two_face_clip: Path
    ) -> None:
        record = await run_job_ok(
            "detect_faces", {"input_path": str(two_face_clip), "sample_fps": 3.0}, settings
        )
        primary = next(t for t in record.result["tracks"] if t["is_primary"])
        other = next(t for t in record.result["tracks"] if not t["is_primary"])
        assert primary["bounding_box"][2] > other["bounding_box"][2]

    async def test_a_clip_with_no_faces_reports_none(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok("detect_faces", {"input_path": str(clip)}, settings)
        assert record.result["tracks"] == []
        assert "No faces" in record.result["notes"][0]

    async def test_frames_can_be_omitted_for_a_compact_result(
        self, settings: Settings, face_clip: Path
    ) -> None:
        record = await run_job_ok(
            "detect_faces", {"input_path": str(face_clip), "include_frames": False}, settings
        )
        assert record.result["frames"] == []
        assert record.result["tracks"]

    async def test_the_frame_cap_is_reported(self, settings: Settings, face_clip: Path) -> None:
        record = await run_job_ok(
            "detect_faces",
            {"input_path": str(face_clip), "sample_fps": 4.0, "max_frames": 3},
            settings,
        )
        assert record.result["sampled_frames"] == 3
        assert any("cap" in note for note in record.result["notes"])


class TestTrackAndCrop:
    async def test_reframing_to_vertical(self, settings: Settings, face_clip: Path) -> None:
        record = await run_job_ok(
            "track_and_crop", {"input_path": str(face_clip), "aspect_ratio": "9:16"}, settings
        )
        info = await probe_output(record, settings)
        video = info.video_streams[0]
        assert video.width is not None and video.height is not None
        assert video.height == 480
        assert video.width / video.height == pytest.approx(9 / 16, abs=0.02)

    async def test_the_crop_actually_follows_the_face(
        self, settings: Settings, face_clip: Path
    ) -> None:
        # The fixture face moves left to right, so the generated crop path
        # must actually travel rather than sitting still.
        tracked = await run_job_ok(
            "track_and_crop",
            {"input_path": str(face_clip), "aspect_ratio": "1:1", "smoothing": 0.2},
            settings,
        )
        assert tracked.command is not None
        assert "sendcmd" in tracked.command
        script = next(
            p for p in (settings.jobs_dir / tracked.job_id).iterdir() if p.name == "crop_path.txt"
        )
        xs = [int(line.split()[3].rstrip(",")) for line in script.read_text().splitlines()]
        assert max(xs) - min(xs) > 100, "the crop never moved"

    async def test_the_crop_path_moves_gradually(self, settings: Settings, face_clip: Path) -> None:
        record = await run_job_ok(
            "track_and_crop",
            {"input_path": str(face_clip), "aspect_ratio": "9:16", "smoothing": 0.8},
            settings,
        )
        script = (settings.jobs_dir / record.job_id / "crop_path.txt").read_text()
        xs = [int(line.split()[3].rstrip(",")) for line in script.splitlines()]
        steps = [abs(b - a) for a, b in itertools.pairwise(xs)]
        assert max(steps) < 60, "the crop jumped rather than glided"

    async def test_a_specific_track_can_be_followed(
        self, settings: Settings, two_face_clip: Path
    ) -> None:
        detection = await run_job_ok(
            "detect_faces", {"input_path": str(two_face_clip), "sample_fps": 3.0}, settings
        )
        secondary = next(t for t in detection.result["tracks"] if not t["is_primary"])
        record = await run_job_ok(
            "track_and_crop",
            {
                "input_path": str(two_face_clip),
                "aspect_ratio": "1:1",
                "track_id": secondary["track_id"],
            },
            settings,
        )
        assert f"Followed track {secondary['track_id']}" in " ".join(record.result["notes"])

    async def test_an_unknown_track_id_fails_clearly(
        self, settings: Settings, face_clip: Path
    ) -> None:
        record = await run_job(
            "track_and_crop", {"input_path": str(face_clip), "track_id": 999}, settings
        )
        assert record.status is JobStatus.FAILED
        assert record.error is not None
        assert record.error.code == "invalid_parameter"

    async def test_a_faceless_clip_falls_back_to_a_centre_crop(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok(
            "track_and_crop", {"input_path": str(clip), "aspect_ratio": "1:1"}, settings
        )
        assert any("centred crop" in note for note in record.result["notes"])
        info = await probe_output(record, settings)
        assert info.video_streams[0].width == info.video_streams[0].height

    async def test_fallback_fail_errors_instead(self, settings: Settings, clip: Path) -> None:
        record = await run_job(
            "track_and_crop", {"input_path": str(clip), "fallback": "fail"}, settings
        )
        assert record.status is JobStatus.FAILED
        assert record.error is not None
        assert "No face was detected" in record.error.message

    async def test_the_output_can_be_scaled(self, settings: Settings, face_clip: Path) -> None:
        record = await run_job_ok(
            "track_and_crop",
            {"input_path": str(face_clip), "aspect_ratio": "9:16", "output_width": 108},
            settings,
        )
        info = await probe_output(record, settings)
        assert info.video_streams[0].width == 108
        assert info.video_streams[0].height == 192

    async def test_an_invalid_aspect_ratio_is_rejected_before_queueing(
        self, settings: Settings, face_clip: Path
    ) -> None:
        from ffmpeg_mcp.errors import InvalidParameterError

        with pytest.raises(InvalidParameterError):
            await call_tool(
                "track_and_crop", {"input_path": str(face_clip), "aspect_ratio": "banana"}
            )

    async def test_duration_is_preserved(self, settings: Settings, face_clip: Path) -> None:
        record = await run_job_ok("track_and_crop", {"input_path": str(face_clip)}, settings)
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(4.0, abs=0.4)


class TestBlurFaces:
    async def test_the_face_region_is_blurred(
        self, settings: Settings, two_face_clip: Path
    ) -> None:
        record = await run_job_ok(
            "blur_faces",
            {"input_path": str(two_face_clip), "sample_fps": 3.0, "blur_strength": 30},
            settings,
        )
        rendered = output_path(record)
        # The large face sits around (260, 250) with radius 105.
        region = (205, 205, 110, 110)
        before = region_variance(two_face_clip, 1.0, region)
        after = region_variance(rendered, 1.0, region)
        assert after < before / 2, "the face region was not visibly blurred"

    async def test_blurring_reports_the_tracks_it_covered(
        self, settings: Settings, two_face_clip: Path
    ) -> None:
        record = await run_job_ok(
            "blur_faces", {"input_path": str(two_face_clip), "sample_fps": 3.0}, settings
        )
        assert any("Blurred 2 tracked face" in note for note in record.result["notes"])

    async def test_excluding_the_primary_keeps_it_sharp(
        self, settings: Settings, two_face_clip: Path
    ) -> None:
        record = await run_job_ok(
            "blur_faces",
            {
                "input_path": str(two_face_clip),
                "sample_fps": 3.0,
                "exclude_primary": True,
                "blur_strength": 30,
            },
            settings,
        )
        rendered = output_path(record)
        primary_region = (205, 205, 110, 110)
        before = region_variance(two_face_clip, 1.0, primary_region)
        after = region_variance(rendered, 1.0, primary_region)
        assert after > before * 0.5, "the excluded primary face was blurred anyway"
        assert any("sharp" in note for note in record.result["notes"])

    async def test_a_moving_face_stays_covered(self, settings: Settings, face_clip: Path) -> None:
        record = await run_job_ok(
            "blur_faces",
            {"input_path": str(face_clip), "sample_fps": 5.0, "blur_strength": 30},
            settings,
        )
        rendered = output_path(record)
        # Sample near the end, where the face has moved right.
        detection = await run_job_ok(
            "detect_faces", {"input_path": str(face_clip), "sample_fps": 5.0}, settings
        )
        late = [f for f in detection.result["frames"] if f["time"] >= 3.0 and f["faces"]]
        assert late, "fixture face was not detected late in the clip"
        face = late[0]["faces"][0]
        region = (
            max(0, face["x"] + 10),
            max(0, face["y"] + 10),
            min(100, face["width"] - 20),
            min(100, face["height"] - 20),
        )
        before = region_variance(face_clip, late[0]["time"], region)
        after = region_variance(rendered, late[0]["time"], region)
        assert after < before / 2, "the blur did not follow the face"

    async def test_a_clip_with_no_faces_passes_through(
        self, settings: Settings, clip: Path
    ) -> None:
        record = await run_job_ok("blur_faces", {"input_path": str(clip)}, settings)
        assert any("No faces to blur" in note for note in record.result["notes"])
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(3.0, abs=0.3)

    async def test_audio_survives_blurring(self, settings: Settings, speech_clip: Path) -> None:
        record = await run_job_ok(
            "blur_faces", {"input_path": str(speech_clip), "sample_fps": 2.0}, settings
        )
        info = await probe_output(record, settings)
        assert info.has_audio is True


class TestDetectScenes:
    async def test_hard_cuts_are_found(self, settings: Settings, cuts_clip: Path) -> None:
        record = await run_job_ok(
            "detect_scenes", {"input_path": str(cuts_clip), "threshold": 0.3}, settings
        )
        cuts = record.result["cut_times"]
        assert len(cuts) == 2
        assert cuts[0] == pytest.approx(2.0, abs=0.3)
        assert cuts[1] == pytest.approx(4.0, abs=0.3)

    async def test_scenes_span_the_clip_without_gaps(
        self, settings: Settings, cuts_clip: Path
    ) -> None:
        record = await run_job_ok("detect_scenes", {"input_path": str(cuts_clip)}, settings)
        scenes = record.result["scenes"]
        assert scenes[0]["start"] == 0.0
        assert scenes[-1]["end"] == pytest.approx(record.result["duration"], abs=0.1)
        for previous, current in itertools.pairwise(scenes):
            assert previous["end"] == current["start"]

    async def test_a_single_shot_clip_reports_no_cuts(self, settings: Settings, clip: Path) -> None:
        record = await run_job_ok(
            "detect_scenes", {"input_path": str(clip), "threshold": 0.9}, settings
        )
        assert record.result["cut_times"] == []
        assert any("single shot" in note for note in record.result["notes"])

    async def test_scene_times_can_drive_a_trim(self, settings: Settings, cuts_clip: Path) -> None:
        # The point of the tool: its output feeds straight into phase 1.
        detection = await run_job_ok("detect_scenes", {"input_path": str(cuts_clip)}, settings)
        scene = detection.result["scenes"][1]
        record = await run_job_ok(
            "trim",
            {
                "input_path": str(cuts_clip),
                "start": scene["start"],
                "end": scene["end"],
                "mode": "reencode",
            },
            settings,
        )
        info = await probe_output(record, settings)
        assert info.duration == pytest.approx(scene["duration"], abs=0.3)
        # The middle shot is the blue one.
        assert mean_saturation(output_path(record)) > 50

    async def test_scene_detection_needs_a_video_stream(
        self, settings: Settings, speech: Path
    ) -> None:
        record = await run_job("detect_scenes", {"input_path": str(speech)}, settings)
        assert record.status is JobStatus.FAILED
        assert record.error is not None
        assert record.error.code == "invalid_parameter"
