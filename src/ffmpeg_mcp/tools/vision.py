"""Phase 4 tools: face detection, reframing, privacy blurring, and scene cuts.

Moving regions are driven with ffmpeg's ``sendcmd``, which feeds ``x``/``y``
commands to ``crop`` and ``overlay`` at set times. The alternative — building one
enormous nested ``if(...)`` expression per coordinate — grows with clip length
and eventually defeats the expression parser.

``detect_scenes`` needs no vision dependency at all; it uses ffmpeg's own scene
score, so it works without the optional extra installed.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from pydantic import Field

from ..errors import InvalidParameterError
from ..ffmpeg.encoding import output_args
from ..ffmpeg.filters import Filter, FilterChain, FilterGraph, gblur_filter
from ..ffmpeg.probe import probe
from ..ffmpeg.runner import run_ffmpeg
from ..jobs.worker import JobContext, JobOutcome, handler
from ..models import EncodeOptions, JobSubmission, MediaInfo, StrictModel
from ..paths import validate_input_file
from ..vision.detect import detect_faces_over_time
from ..vision.tracking import (
    FaceTrack,
    FrameDetections,
    build_crop_path,
    build_tracks,
    crop_size_for_aspect,
    dedupe_keyframes,
    parse_aspect_ratio,
    primary_track,
    render_sendcmd_script,
)
from .common import MediaJobArgs, finish_media_job, queue, resolve_io
from .registry import tool

MAX_BLUR_TRACKS = 12


class DetectedFace(StrictModel):
    """One face box at one moment, in source pixels."""

    x: int
    y: int
    width: int
    height: int
    confidence: float
    normalized: list[float] = Field(
        description="The same box as [x, y, width, height] in 0..1 frame coordinates."
    )


class FrameFaces(StrictModel):
    """All faces found in one sampled frame."""

    time: float = Field(description="Timestamp of the sampled frame, in seconds.")
    faces: list[DetectedFace] = Field(default_factory=list)


class TrackSummary(StrictModel):
    """One face followed across the clip."""

    track_id: int
    start: float
    end: float
    detections: int
    mean_confidence: float
    bounding_box: list[int] = Field(description="[x, y, width, height] covering the whole track.")
    is_primary: bool = False


class FaceDetectionResult(StrictModel):
    """Everything the face detector found."""

    frame_width: int
    frame_height: int
    sampled_frames: int
    sample_fps: float
    frames: list[FrameFaces] = Field(default_factory=list)
    tracks: list[TrackSummary] = Field(default_factory=list)
    max_faces_in_a_frame: int = 0
    notes: list[str] = Field(default_factory=list)


def _video_dimensions(info: MediaInfo, path: Path) -> tuple[int, int]:
    video = info.primary_video
    if video is None or not video.width or not video.height:
        raise InvalidParameterError("File has no readable video stream.", path=str(path))
    return video.width, video.height


def _to_detected(
    box_x: float, box_y: float, w: float, h: float, conf: float, frame_w: int, frame_h: int
) -> DetectedFace:
    return DetectedFace(
        x=round(box_x),
        y=round(box_y),
        width=round(w),
        height=round(h),
        confidence=round(conf, 4),
        normalized=[
            round(box_x / frame_w, 5),
            round(box_y / frame_h, 5),
            round(w / frame_w, 5),
            round(h / frame_h, 5),
        ],
    )


def _summarise_tracks(tracks: list[FaceTrack], primary: FaceTrack | None) -> list[TrackSummary]:
    summaries: list[TrackSummary] = []
    for track in tracks:
        box = track.bounding_box()
        summaries.append(
            TrackSummary(
                track_id=track.track_id,
                start=round(track.start, 3),
                end=round(track.end, 3),
                detections=len(track.boxes),
                mean_confidence=round(track.mean_confidence, 4),
                bounding_box=[
                    round(box.x),
                    round(box.y),
                    round(box.width),
                    round(box.height),
                ],
                is_primary=primary is not None and track.track_id == primary.track_id,
            )
        )
    return summaries


async def _detect(
    ctx: JobContext,
    source: Path,
    info: MediaInfo,
    *,
    sample_fps: float,
    min_confidence: float,
    analysis_width: int,
    max_frames: int,
    floor: float,
    ceiling: float,
) -> tuple[list[FrameDetections], int, int]:
    """Shared detection pass used by the three face tools."""
    frame_width, frame_height = _video_dimensions(info, source)
    expected = int((info.duration or 0.0) * sample_fps) or None
    loop = asyncio.get_running_loop()
    span = ceiling - floor

    def report(percent: float) -> None:
        asyncio.run_coroutine_threadsafe(ctx.report(floor + percent / 100.0 * span), loop)

    await ctx.report(floor, "Detecting faces")
    frames = await detect_faces_over_time(
        source,
        frame_width=frame_width,
        frame_height=frame_height,
        sample_fps=sample_fps,
        min_confidence=min_confidence,
        analysis_width=analysis_width,
        max_frames=max_frames,
        settings=ctx.settings,
        on_progress=report,
        should_cancel=ctx.cancelled,
        expected_frames=expected,
    )
    return frames, frame_width, frame_height


# --------------------------------------------------------------------------- #
# detect_faces
# --------------------------------------------------------------------------- #


class DetectFacesArgs(StrictModel):
    """Arguments for face detection."""

    input_path: str = Field(description="Video file to analyse.")
    sample_fps: float = Field(
        default=2.0,
        gt=0,
        le=60,
        description="Frames sampled per second. Higher is more precise and slower.",
    )
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    analysis_width: int = Field(
        default=640,
        ge=128,
        le=4096,
        description="Frames are downscaled to this width before detection, for speed.",
    )
    max_frames: int = Field(
        default=4000, ge=1, le=50000, description="Safety cap on sampled frames."
    )
    include_frames: bool = Field(
        default=True,
        description="Return every sampled frame. Turn off for a long clip to get only tracks.",
    )


@tool("detect_faces", title="Detect faces", phase=4)
async def detect_faces(args: DetectFacesArgs) -> JobSubmission:
    """Find faces in a video, sampled over time.

    Returns per-timestamp bounding boxes with confidence, in both source pixels
    and 0..1 normalised coordinates, plus 'tracks' — detections linked across
    frames into one entry per person, with the likely main subject flagged.

    Sampling at 2 fps is usually enough to follow a talking head; raise
    sample_fps for fast movement. Needs the 'vision' extra; the small detection
    model is downloaded and cached on first use.
    """
    validate_input_file(args.input_path)
    return queue("detect_faces", args)


@handler("detect_faces")
async def _run_detect_faces(ctx: JobContext) -> JobOutcome:
    args = DetectFacesArgs.model_validate(ctx.params)
    source = validate_input_file(args.input_path, ctx.settings)
    info = await probe(source, ctx.settings)
    frames, frame_width, frame_height = await _detect(
        ctx,
        source,
        info,
        sample_fps=args.sample_fps,
        min_confidence=args.min_confidence,
        analysis_width=args.analysis_width,
        max_frames=args.max_frames,
        floor=2.0,
        ceiling=95.0,
    )

    tracks = build_tracks(frames)
    primary = primary_track(tracks, frame_width, frame_height)
    notes: list[str] = []
    if not tracks:
        notes.append("No faces were detected at this confidence threshold.")
    if len(frames) >= args.max_frames:
        notes.append(
            f"Sampling stopped at the {args.max_frames}-frame cap; "
            "the later part of the clip was not analysed."
        )

    result = FaceDetectionResult(
        frame_width=frame_width,
        frame_height=frame_height,
        sampled_frames=len(frames),
        sample_fps=args.sample_fps,
        frames=[
            FrameFaces(
                time=round(frame.time, 3),
                faces=[
                    _to_detected(
                        box.x,
                        box.y,
                        box.width,
                        box.height,
                        box.confidence,
                        frame_width,
                        frame_height,
                    )
                    for box in frame.faces
                ],
            )
            for frame in frames
        ]
        if args.include_frames
        else [],
        tracks=_summarise_tracks(tracks, primary),
        max_faces_in_a_frame=max((len(f.faces) for f in frames), default=0),
        notes=notes,
    )
    await ctx.report(99.0, "Detection complete")
    return JobOutcome(result=result.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# track_and_crop
# --------------------------------------------------------------------------- #


class TrackAndCropArgs(MediaJobArgs):
    """Arguments for reframing a clip around a tracked face."""

    aspect_ratio: str = Field(
        default="9:16",
        description="Target shape, e.g. '9:16' for vertical, '1:1' square, '16:9' wide.",
    )
    output_width: int | None = Field(
        default=None,
        gt=0,
        le=16384,
        description="Scale the reframed result to this width. Height follows the aspect ratio.",
    )
    sample_fps: float = Field(default=4.0, gt=0, le=60)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    smoothing: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "How steadily the crop follows the face. 0 tracks exactly and looks "
            "jittery; 1 barely moves. The default glides."
        ),
    )
    track_id: int | None = Field(
        default=None,
        description="Follow a specific track from detect_faces. Defaults to the main subject.",
    )
    fallback: str = Field(
        default="center",
        description="What to do when no face is found: 'center' crops centrally, 'fail' errors.",
    )
    max_frames: int = Field(default=4000, ge=1, le=50000)
    encode: EncodeOptions = Field(default_factory=EncodeOptions)


@tool("track_and_crop", title="Reframe following a face", phase=4)
async def track_and_crop(args: TrackAndCropArgs) -> JobSubmission:
    """Reframe a clip to a new aspect ratio, following a face across the timeline.

    The classic use is turning a horizontal interview into a vertical clip that
    keeps the speaker in frame. The crop path is smoothed before rendering — a
    crop that snaps frame to frame looks worse than a slightly imperfect one that
    glides — and clamped so it never runs off the edge of the source.

    Call detect_faces first if you want to choose which person to follow, then
    pass its track_id. With no face found, this falls back to a centre crop
    unless fallback is 'fail'.
    """
    resolve_io(args, operation="reframed")
    parse_aspect_ratio(args.aspect_ratio)
    if args.fallback not in {"center", "fail"}:
        raise InvalidParameterError("fallback must be 'center' or 'fail'.", fallback=args.fallback)
    return queue("track_and_crop", args)


@handler("track_and_crop")
async def _run_track_and_crop(ctx: JobContext) -> JobOutcome:
    args = TrackAndCropArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="reframed", job_id=ctx.job_id)
    info = await probe(source, ctx.settings)
    frame_width, frame_height = _video_dimensions(info, source)
    ratio = parse_aspect_ratio(args.aspect_ratio)
    crop_width, crop_height = crop_size_for_aspect(frame_width, frame_height, ratio)

    frames, _, _ = await _detect(
        ctx,
        source,
        info,
        sample_fps=args.sample_fps,
        min_confidence=args.min_confidence,
        analysis_width=640,
        max_frames=args.max_frames,
        floor=2.0,
        ceiling=45.0,
    )
    tracks = build_tracks(frames)
    if args.track_id is not None:
        chosen = next((t for t in tracks if t.track_id == args.track_id), None)
        if chosen is None:
            raise InvalidParameterError(
                "No such track in this clip.",
                track_id=args.track_id,
                available=[t.track_id for t in tracks],
            )
    else:
        chosen = primary_track(tracks, frame_width, frame_height)

    notes: list[str] = []
    graph = FilterGraph()
    chain = graph.chain(["0:v:0"], ["vout"])

    if chosen is None:
        if args.fallback == "fail":
            raise InvalidParameterError(
                "No face was detected, and fallback is set to 'fail'.", path=str(source)
            )
        notes.append("No face detected; fell back to a centred crop.")
        chain.add(
            Filter(
                "crop",
                {
                    "w": crop_width,
                    "h": crop_height,
                    "x": (frame_width - crop_width) // 2,
                    "y": (frame_height - crop_height) // 2,
                },
            )
        )
    else:
        keyframes = dedupe_keyframes(
            build_crop_path(
                chosen,
                frame_width=frame_width,
                frame_height=frame_height,
                crop_width=crop_width,
                crop_height=crop_height,
                smoothing=args.smoothing,
                duration=info.duration or 0.0,
            )
        )
        script = ctx.workdir / "crop_path.txt"
        script.write_text(render_sendcmd_script(keyframes, "crop"), encoding="utf-8")
        chain.add(
            Filter("sendcmd", {}, raw_options=f"f={_escape(script)}"),
            Filter(
                "crop",
                {"w": crop_width, "h": crop_height, "x": keyframes[0].x, "y": keyframes[0].y},
            ),
        )
        notes.append(
            f"Followed track {chosen.track_id} across {len(chosen.boxes)} detections "
            f"using {len(keyframes)} smoothed keyframes."
        )

    if args.output_width:
        target_height = round(args.output_width / ratio)
        target_height -= target_height % 2
        chain.add(
            Filter("scale", {}, raw_options=f"w={args.output_width}:h={max(2, target_height)}")
        )
    chain.add(Filter("format", {}, raw_options="yuv420p"))

    argv = ["-i", str(source), "-filter_complex", graph.render(), "-map", "[vout]"]
    if info.has_audio:
        argv += ["-map", "0:a:0"]
    argv += output_args(args.encode, destination, has_video=True, has_audio=info.has_audio)
    argv += ["-y", str(destination)]
    result = await run_ffmpeg(
        argv,
        total_duration=info.duration,
        on_progress=ctx.make_progress_hook(46.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    notes.append(f"Reframed to {crop_width}x{crop_height} ({args.aspect_ratio}).")
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


def _escape(path: Path) -> str:
    from ..ffmpeg.filters import escape_path_for_filter

    return escape_path_for_filter(path)


# --------------------------------------------------------------------------- #
# blur_faces
# --------------------------------------------------------------------------- #


class BlurFacesArgs(MediaJobArgs):
    """Arguments for privacy blurring."""

    sample_fps: float = Field(default=4.0, gt=0, le=60)
    min_confidence: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Lower than the detection default, since a missed face is not blurred.",
    )
    blur_strength: float = Field(
        default=20.0, gt=0, le=200, description="Gaussian blur sigma. Higher is stronger."
    )
    expand: float = Field(
        default=1.6,
        ge=1.0,
        le=4.0,
        description="Grow each face box by this factor, so hair and chin are covered too.",
    )
    exclude_primary: bool = Field(
        default=False,
        description="Leave the main subject sharp and blur everyone else.",
    )
    exclude_track_ids: list[int] = Field(
        default_factory=list, description="Specific tracks from detect_faces to leave sharp."
    )
    max_frames: int = Field(default=4000, ge=1, le=50000)
    encode: EncodeOptions = Field(default_factory=EncodeOptions)


@tool("blur_faces", title="Blur faces", phase=4)
async def blur_faces(args: BlurFacesArgs) -> JobSubmission:
    """Blur every face in a clip, following each one as it moves.

    Each tracked face gets its own blurred region whose position is driven over
    time, so the blur stays on the person rather than covering a fixed rectangle.
    Boxes are expanded past the detected face by default so hair and chin are
    covered too.

    Set exclude_primary to keep the main subject sharp and blur everyone else —
    the usual requirement for street interviews. Because a missed detection means
    an unblurred face, review the output before publishing it.
    """
    resolve_io(args, operation="blurred")
    return queue("blur_faces", args)


@handler("blur_faces")
async def _run_blur_faces(ctx: JobContext) -> JobOutcome:
    args = BlurFacesArgs.model_validate(ctx.params)
    source, destination = resolve_io(args, operation="blurred", job_id=ctx.job_id)
    info = await probe(source, ctx.settings)
    frame_width, frame_height = _video_dimensions(info, source)

    frames, _, _ = await _detect(
        ctx,
        source,
        info,
        sample_fps=args.sample_fps,
        min_confidence=args.min_confidence,
        analysis_width=640,
        max_frames=args.max_frames,
        floor=2.0,
        ceiling=45.0,
    )
    tracks = build_tracks(frames)
    primary = primary_track(tracks, frame_width, frame_height)

    excluded = set(args.exclude_track_ids)
    if args.exclude_primary and primary is not None:
        excluded.add(primary.track_id)
    targets = [t for t in tracks if t.track_id not in excluded]

    notes: list[str] = []
    if len(targets) > MAX_BLUR_TRACKS:
        targets = sorted(targets, key=lambda t: t.mean_area, reverse=True)[:MAX_BLUR_TRACKS]
        notes.append(
            f"More than {MAX_BLUR_TRACKS} faces were tracked; blurred the "
            f"{MAX_BLUR_TRACKS} largest. Raise sample_fps or review the output."
        )

    if not targets:
        notes.append("No faces to blur; the video was re-encoded unchanged.")
        argv = ["-i", str(source)]
        argv += output_args(args.encode, destination, has_video=True, has_audio=info.has_audio)
        argv += ["-y", str(destination)]
    else:
        graph, _ = build_blur_graph(
            ctx.workdir,
            targets,
            frame_width=frame_width,
            frame_height=frame_height,
            expand=args.expand,
            blur_strength=args.blur_strength,
            duration=info.duration or 0.0,
        )
        argv = ["-i", str(source), "-filter_complex", graph, "-map", "[vout]"]
        if info.has_audio:
            argv += ["-map", "0:a:0"]
        argv += output_args(args.encode, destination, has_video=True, has_audio=info.has_audio)
        argv += ["-y", str(destination)]
        notes.append(f"Blurred {len(targets)} tracked face(s).")
        if excluded:
            notes.append(f"Left {len(excluded)} track(s) sharp: {sorted(excluded)}.")

    result = await run_ffmpeg(
        argv,
        total_duration=info.duration,
        on_progress=ctx.make_progress_hook(46.0, 99.0),
        cancel_check=ctx.cancelled,
        settings=ctx.settings,
    )
    return await finish_media_job(ctx, destination, notes=notes, command=result.command)


def build_blur_graph(
    workdir: Path,
    tracks: list[FaceTrack],
    *,
    frame_width: int,
    frame_height: int,
    expand: float,
    blur_strength: float,
    duration: float,
) -> tuple[str, list[Path]]:
    """Build the filter graph that blurs each tracked face as it moves.

    One crop/blur/overlay triple per track. Both the crop and the overlay are
    driven by ``sendcmd`` scripts, so the blurred patch follows the face instead
    of covering a static rectangle. The crop is sized to the track's largest
    detection so a single fixed size works for the whole track.

    Returns the rendered graph and the sendcmd scripts written to disk.
    """
    graph = FilterGraph()
    scripts: list[Path] = []

    split_labels = ["base"] + [f"src{i}" for i in range(len(tracks))]
    graph.chain(["0:v:0"], split_labels).add(
        Filter("split", {}, raw_options=str(len(split_labels)))
    )

    current = "base"
    for index, track in enumerate(tracks):
        expanded = [box.expanded(expand, frame_width, frame_height) for box in track.boxes]
        patch_width = min(frame_width, round(max(b.width for b in expanded)))
        patch_height = min(frame_height, round(max(b.height for b in expanded)))
        patch_width = max(2, patch_width - patch_width % 2)
        patch_height = max(2, patch_height - patch_height % 2)

        keyframes = dedupe_keyframes(
            build_crop_path(
                FaceTrack(track_id=track.track_id, times=track.times, boxes=expanded),
                frame_width=frame_width,
                frame_height=frame_height,
                crop_width=patch_width,
                crop_height=patch_height,
                smoothing=0.3,
                duration=duration,
            )
        )
        crop_script = workdir / f"blur_crop_{index}.txt"
        crop_script.write_text(render_sendcmd_script(keyframes, "crop"), encoding="utf-8")
        overlay_script = workdir / f"blur_overlay_{index}.txt"
        overlay_script.write_text(render_sendcmd_script(keyframes, "overlay"), encoding="utf-8")
        scripts += [crop_script, overlay_script]

        patch_label = f"blur{index}"
        graph.chain([f"src{index}"], [patch_label]).add(
            Filter("sendcmd", {}, raw_options=f"f={_escape(crop_script)}"),
            Filter(
                "crop",
                {
                    "w": patch_width,
                    "h": patch_height,
                    "x": keyframes[0].x,
                    "y": keyframes[0].y,
                },
            ),
            gblur_filter(blur_strength),
        )

        commanded = f"cmd{index}"
        graph.chain([current], [commanded]).add(
            Filter("sendcmd", {}, raw_options=f"f={_escape(overlay_script)}")
        )
        is_last = index == len(tracks) - 1
        output_label = "vout" if is_last else f"stage{index}"
        overlay_chain = graph.chain([commanded, patch_label], [output_label])
        overlay_chain.add(
            Filter(
                "overlay",
                {},
                raw_options=(
                    f"x={keyframes[0].x}:y={keyframes[0].y}:"
                    f"enable='between(t,{track.start:.3f},{track.end:.3f})'"
                ),
            )
        )
        if is_last:
            overlay_chain.add(Filter("format", {}, raw_options="yuv420p"))
        current = output_label

    return graph.render(), scripts


# --------------------------------------------------------------------------- #
# detect_scenes
# --------------------------------------------------------------------------- #

_PTS_TIME = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")


class DetectScenesArgs(StrictModel):
    """Arguments for scene-cut detection."""

    input_path: str = Field(description="Video file to analyse.")
    threshold: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description=(
            "Scene-change score to treat as a cut. Lower finds more cuts; 0.3 suits "
            "most edited footage, 0.4-0.5 suits noisy handheld video."
        ),
    )
    min_scene_seconds: float = Field(
        default=0.5, ge=0.0, le=600, description="Discard cuts closer together than this."
    )


class Scene(StrictModel):
    """One detected shot."""

    index: int
    start: float
    end: float
    duration: float


class SceneDetectionResult(StrictModel):
    """Detected cut points and the shots between them."""

    cut_times: list[float] = Field(description="Timestamps where a hard cut was detected.")
    scenes: list[Scene] = Field(default_factory=list)
    threshold: float
    duration: float | None = None
    notes: list[str] = Field(default_factory=list)


def parse_scene_times(stderr: str) -> list[float]:
    """Pull ``pts_time`` values out of ``showinfo`` output."""
    return [float(match) for match in _PTS_TIME.findall(stderr)]


def build_scenes(
    cut_times: list[float], duration: float | None, min_scene_seconds: float
) -> tuple[list[float], list[Scene]]:
    """Turn raw cut timestamps into filtered cuts and the shots between them."""
    kept: list[float] = []
    for time in sorted(cut_times):
        if time <= 0:
            continue
        if kept and time - kept[-1] < min_scene_seconds:
            continue
        kept.append(round(time, 3))

    boundaries = [0.0, *kept]
    if duration:
        boundaries.append(duration)
    elif kept:
        boundaries.append(kept[-1])

    scenes: list[Scene] = []
    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        if end - start < min_scene_seconds and index < len(boundaries) - 2:
            continue
        scenes.append(
            Scene(
                index=len(scenes),
                start=round(start, 3),
                end=round(end, 3),
                duration=round(end - start, 3),
            )
        )
    return kept, scenes


@tool("detect_scenes", title="Detect scene cuts", phase=4)
async def detect_scenes(args: DetectScenesArgs) -> JobSubmission:
    """Find hard cuts in a video and report the shots between them.

    Useful for chopping raw footage into clips: feed the returned scene start
    and end times straight into trim. Uses ffmpeg's own scene-change score, so
    unlike the other phase 4 tools this needs no vision dependency.

    Lower the threshold to catch softer cuts, raise it if handheld camera motion
    is being reported as cuts.
    """
    validate_input_file(args.input_path)
    return queue("detect_scenes", args)


@handler("detect_scenes")
async def _run_detect_scenes(ctx: JobContext) -> JobOutcome:
    args = DetectScenesArgs.model_validate(ctx.params)
    source = validate_input_file(args.input_path, ctx.settings)
    info = await probe(source, ctx.settings)
    if not info.has_video:
        raise InvalidParameterError("detect_scenes needs a video stream.", path=str(source))

    chain = FilterChain(
        filters=[
            Filter("select", {}, raw_options=f"'gt(scene,{args.threshold})'"),
            Filter("showinfo"),
        ]
    )
    argv = [
        "-i",
        str(source),
        "-vf",
        chain.render(),
        "-vsync",
        "vfr",
        "-f",
        "null",
        "-",
    ]
    # showinfo reports through stderr at info level, so the log has to be raised.
    result = await run_ffmpeg(
        argv,
        total_duration=info.duration,
        on_progress=ctx.make_progress_hook(1.0, 95.0),
        cancel_check=ctx.cancelled,
        loglevel="info",
        settings=ctx.settings,
    )
    cut_times, scenes = build_scenes(
        parse_scene_times(result.stderr), info.duration, args.min_scene_seconds
    )
    notes: list[str] = []
    if not cut_times:
        notes.append(
            "No cuts detected; the clip appears to be a single shot. "
            "Lower the threshold to catch softer transitions."
        )
    payload = SceneDetectionResult(
        cut_times=cut_times,
        scenes=scenes,
        threshold=args.threshold,
        duration=info.duration,
        notes=notes,
    )
    await ctx.report(99.0, "Scene detection complete")
    return JobOutcome(result=payload.model_dump(mode="json"), command=result.command)


__all__ = [
    "BlurFacesArgs",
    "DetectFacesArgs",
    "DetectScenesArgs",
    "FaceDetectionResult",
    "SceneDetectionResult",
    "TrackAndCropArgs",
    "build_blur_graph",
    "build_scenes",
    "parse_scene_times",
]
