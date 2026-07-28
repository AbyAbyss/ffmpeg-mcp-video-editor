"""Running MediaPipe face detection over frames streamed from ffmpeg.

Frames are decoded, resampled and downscaled by ffmpeg and piped in as raw
RGB24, so nothing here depends on OpenCV's video backend and a long clip never
lands on disk as thousands of images. Detection results are scaled back up to
source-pixel coordinates before they leave this module.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..errors import InvalidParameterError
from ..ffmpeg.runner import stream_raw_frames
from .models import ensure_face_model, require_vision
from .tracking import FaceBox, FrameDetections

log = logging.getLogger(__name__)

_detector_lock = threading.Lock()
_detectors: dict[tuple[str, int], Any] = {}

DEFAULT_ANALYSIS_WIDTH = 640


def analysis_size(
    width: int, height: int, target_width: int = DEFAULT_ANALYSIS_WIDTH
) -> tuple[int, int]:
    """Pick the size frames are analysed at.

    Detection is run on a downscaled copy: BlazeFace works on a small square
    anyway, and scaling first cuts decode and inference cost substantially on a
    4K source. Never upscales.
    """
    if width <= 0 or height <= 0:
        raise InvalidParameterError("Frame dimensions must be positive.")
    if width <= target_width:
        scaled_width, scaled_height = width, height
    else:
        scaled_width = target_width
        scaled_height = max(2, round(height * target_width / width))
    return scaled_width - scaled_width % 2, scaled_height - scaled_height % 2


def _get_detector(model_path: Path, min_confidence: float) -> Any:
    """Build (and cache) a MediaPipe face detector."""
    from mediapipe.tasks import python as mp_python

    key = (str(model_path), int(min_confidence * 100))
    with _detector_lock:
        detector = _detectors.get(key)
        if detector is None:
            options = mp_python.vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_python.vision.RunningMode.IMAGE,
                min_detection_confidence=min_confidence,
            )
            detector = mp_python.vision.FaceDetector.create_from_options(options)
            _detectors[key] = detector
    return detector


def clear_detector_cache() -> None:
    """Drop cached detectors. Used by tests."""
    with _detector_lock:
        _detectors.clear()


def detect_in_frame(
    detector: Any,
    frame: bytes,
    *,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    min_confidence: float,
) -> list[FaceBox]:
    """Run the detector on one raw RGB frame, returning source-space boxes."""
    import mediapipe as mp
    import numpy as np

    array = np.frombuffer(frame, dtype=np.uint8).reshape((height, width, 3))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(array))
    result = detector.detect(image)

    boxes: list[FaceBox] = []
    for detection in result.detections:
        score = float(detection.categories[0].score) if detection.categories else 0.0
        if score < min_confidence:
            continue
        box = detection.bounding_box
        boxes.append(
            FaceBox(
                x=float(box.origin_x) * scale_x,
                y=float(box.origin_y) * scale_y,
                width=float(box.width) * scale_x,
                height=float(box.height) * scale_y,
                confidence=score,
            )
        )
    return boxes


async def detect_faces_over_time(
    path: Path,
    *,
    frame_width: int,
    frame_height: int,
    sample_fps: float = 2.0,
    min_confidence: float = 0.5,
    analysis_width: int = DEFAULT_ANALYSIS_WIDTH,
    max_frames: int = 4000,
    settings: Settings | None = None,
    on_progress: Callable[[float], Any] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    expected_frames: int | None = None,
) -> list[FrameDetections]:
    """Sample a video and detect faces in each sampled frame.

    Returns one :class:`FrameDetections` per sampled frame, in time order, with
    boxes in source-pixel coordinates.
    """
    from ..errors import JobCancelledError

    require_vision()
    settings = settings or get_settings()
    if sample_fps <= 0:
        raise InvalidParameterError("sample_fps must be positive.", sample_fps=sample_fps)

    model_path = ensure_face_model(settings)
    detector = _get_detector(model_path, min_confidence)
    small_width, small_height = analysis_size(frame_width, frame_height, analysis_width)
    scale_x = frame_width / small_width
    scale_y = frame_height / small_height

    frames: list[FrameDetections] = []
    async for index, raw in stream_raw_frames(
        path,
        fps=sample_fps,
        width=small_width,
        height=small_height,
        max_frames=max_frames,
        settings=settings,
    ):
        if should_cancel is not None and should_cancel():
            raise JobCancelledError("Face detection cancelled.")
        faces = detect_in_frame(
            detector,
            raw,
            width=small_width,
            height=small_height,
            scale_x=scale_x,
            scale_y=scale_y,
            min_confidence=min_confidence,
        )
        frames.append(FrameDetections(time=index / sample_fps, faces=faces))
        if on_progress is not None and expected_frames:
            on_progress(min(99.0, (index + 1) / expected_frames * 100.0))
    return frames
