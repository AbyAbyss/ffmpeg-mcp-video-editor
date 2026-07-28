"""Pure geometry for face tracking: association, smoothing, and crop paths.

None of this touches MediaPipe, OpenCV or ffmpeg, so it is all unit tested
directly. The smoothing in particular matters more than detection accuracy: a
crop that snaps from frame to frame looks far worse than a slightly imperfect
one that moves steadily.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import InvalidParameterError


@dataclass(frozen=True)
class FaceBox:
    """One detected face in source-pixel coordinates."""

    x: float
    y: float
    width: float
    height: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def expanded(self, factor: float, frame_width: int, frame_height: int) -> FaceBox:
        """Grow the box by a factor about its centre, clamped to the frame."""
        centre_x, centre_y = self.center
        half_w = self.width * factor / 2
        half_h = self.height * factor / 2
        x0 = max(0.0, centre_x - half_w)
        y0 = max(0.0, centre_y - half_h)
        x1 = min(float(frame_width), centre_x + half_w)
        y1 = min(float(frame_height), centre_y + half_h)
        return FaceBox(x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0), self.confidence)

    def iou(self, other: FaceBox) -> float:
        """Intersection over union, used to link detections across frames."""
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x + self.width, other.x + other.width)
        bottom = min(self.y + self.height, other.y + other.height)
        if right <= left or bottom <= top:
            return 0.0
        overlap = (right - left) * (bottom - top)
        union = self.area + other.area - overlap
        return overlap / union if union > 0 else 0.0


@dataclass
class FrameDetections:
    """Every face found in one sampled frame."""

    time: float
    faces: list[FaceBox] = field(default_factory=list)


@dataclass
class FaceTrack:
    """One face followed across time."""

    track_id: int
    times: list[float] = field(default_factory=list)
    boxes: list[FaceBox] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.times[0]

    @property
    def end(self) -> float:
        return self.times[-1]

    @property
    def mean_area(self) -> float:
        return sum(b.area for b in self.boxes) / len(self.boxes)

    @property
    def mean_confidence(self) -> float:
        return sum(b.confidence for b in self.boxes) / len(self.boxes)

    def bounding_box(self) -> FaceBox:
        """The box that contains every detection in this track."""
        x0 = min(b.x for b in self.boxes)
        y0 = min(b.y for b in self.boxes)
        x1 = max(b.x + b.width for b in self.boxes)
        y1 = max(b.y + b.height for b in self.boxes)
        return FaceBox(x0, y0, x1 - x0, y1 - y0, self.mean_confidence)


def build_tracks(
    frames: list[FrameDetections],
    *,
    iou_threshold: float = 0.2,
    max_gap_seconds: float = 1.0,
) -> list[FaceTrack]:
    """Link per-frame detections into tracks by greedy IoU matching.

    A face that briefly drops out (a blink of the detector, a turned head) is
    reconnected as long as the gap is under ``max_gap_seconds``, so a single
    person does not become a dozen one-frame tracks.
    """
    tracks: list[FaceTrack] = []
    next_id = 0
    for frame in frames:
        unmatched = list(frame.faces)
        for track in tracks:
            if not unmatched or frame.time - track.end > max_gap_seconds:
                continue
            last = track.boxes[-1]
            best = max(unmatched, key=last.iou)
            if last.iou(best) >= iou_threshold:
                track.times.append(frame.time)
                track.boxes.append(best)
                unmatched.remove(best)
        for face in unmatched:
            tracks.append(FaceTrack(track_id=next_id, times=[frame.time], boxes=[face]))
            next_id += 1
    return tracks


def primary_track(tracks: list[FaceTrack], frame_width: int, frame_height: int) -> FaceTrack | None:
    """Pick the most likely subject: big, central, and present for a long time.

    Weighting all three avoids the two obvious failure modes — following a large
    face that appears for half a second, or a tiny one at the edge of frame.
    """
    if not tracks:
        return None
    total_span = max((t.end - t.start) for t in tracks) or 1.0
    frame_area = float(frame_width * frame_height)
    centre_x, centre_y = frame_width / 2, frame_height / 2
    max_distance = (centre_x**2 + centre_y**2) ** 0.5 or 1.0

    def score(track: FaceTrack) -> float:
        box = track.bounding_box()
        face_x, face_y = box.center
        distance = ((face_x - centre_x) ** 2 + (face_y - centre_y) ** 2) ** 0.5
        centrality = 1.0 - min(1.0, distance / max_distance)
        size = min(1.0, track.mean_area / frame_area * 8)
        persistence = (track.end - track.start) / total_span
        return float(0.4 * persistence + 0.35 * size + 0.25 * centrality)

    return max(tracks, key=score)


def moving_average(values: list[float], window: int) -> list[float]:
    """Centred moving average that keeps the list length, shrinking at the ends."""
    if window <= 1 or len(values) < 2:
        return list(values)
    half = window // 2
    smoothed: list[float] = []
    for index in range(len(values)):
        low = max(0, index - half)
        high = min(len(values), index + half + 1)
        window_values = values[low:high]
        smoothed.append(sum(window_values) / len(window_values))
    return smoothed


def smooth_path(
    values: list[float], *, smoothing: float, max_step: float | None = None
) -> list[float]:
    """Smooth a coordinate path.

    Args:
        values: Raw per-sample coordinates.
        smoothing: 0 leaves the path alone, 1 smooths as hard as possible.
        max_step: Optional per-sample movement cap, applied after smoothing to
            remove any residual jitter.

    Returns:
        A path the same length as ``values``.
    """
    if not 0.0 <= smoothing <= 1.0:
        raise InvalidParameterError("smoothing must be between 0 and 1.", smoothing=smoothing)
    if not values:
        return []
    window = 1 + round(smoothing * min(60, max(2, len(values) // 2)) * 2)
    smoothed = moving_average(values, window)
    if max_step is not None and max_step > 0:
        limited = [smoothed[0]]
        for value in smoothed[1:]:
            previous = limited[-1]
            delta = max(-max_step, min(max_step, value - previous))
            limited.append(previous + delta)
        smoothed = limited
    return smoothed


def parse_aspect_ratio(value: str) -> float:
    """Parse ``'9:16'`` or ``'1.777'`` into a width/height ratio."""
    text = value.strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            raise InvalidParameterError("Aspect ratio must look like '9:16'.", aspect_ratio=value)
        try:
            width, height = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise InvalidParameterError(
                "Aspect ratio parts must be numbers.", aspect_ratio=value
            ) from exc
    else:
        try:
            width, height = float(text), 1.0
        except ValueError as exc:
            raise InvalidParameterError(
                "Aspect ratio must look like '9:16' or a number.", aspect_ratio=value
            ) from exc
    if width <= 0 or height <= 0:
        raise InvalidParameterError("Aspect ratio must be positive.", aspect_ratio=value)
    return width / height


def crop_size_for_aspect(frame_width: int, frame_height: int, ratio: float) -> tuple[int, int]:
    """Largest even-sized crop of the given aspect ratio that fits in the frame."""
    if ratio <= 0:
        raise InvalidParameterError("Aspect ratio must be positive.", ratio=ratio)
    width = frame_width
    height = round(width / ratio)
    if height > frame_height:
        height = frame_height
        width = round(height * ratio)
    width = max(2, min(frame_width, width - width % 2))
    height = max(2, min(frame_height, height - height % 2))
    return width, height


def clamp_crop_origin(centre: float, crop_size: int, frame_size: int) -> float:
    """Convert a desired centre into a crop origin that stays inside the frame."""
    origin = centre - crop_size / 2
    return max(0.0, min(float(frame_size - crop_size), origin))


@dataclass
class CropKeyframe:
    """One point on a rendered crop path."""

    time: float
    x: int
    y: int


def build_crop_path(
    track: FaceTrack,
    *,
    frame_width: int,
    frame_height: int,
    crop_width: int,
    crop_height: int,
    smoothing: float,
    duration: float,
) -> list[CropKeyframe]:
    """Turn a face track into a smoothed, in-bounds crop path.

    The path is extended to cover the whole clip so the render never falls back
    to an undefined crop before the first or after the last detection.
    """
    if not track.boxes:
        raise InvalidParameterError("Cannot build a crop path from an empty track.")
    centres_x = [box.center[0] for box in track.boxes]
    centres_y = [box.center[1] for box in track.boxes]
    # Cap movement at a fraction of the crop size per sample so the frame glides.
    max_step_x = crop_width * 0.06
    max_step_y = crop_height * 0.06
    smooth_x = smooth_path(centres_x, smoothing=smoothing, max_step=max_step_x)
    smooth_y = smooth_path(centres_y, smoothing=smoothing, max_step=max_step_y)

    keyframes = [
        CropKeyframe(
            time=time,
            x=round(clamp_crop_origin(x, crop_width, frame_width)),
            y=round(clamp_crop_origin(y, crop_height, frame_height)),
        )
        for time, x, y in zip(track.times, smooth_x, smooth_y, strict=True)
    ]
    if keyframes[0].time > 0:
        keyframes.insert(0, CropKeyframe(0.0, keyframes[0].x, keyframes[0].y))
    if duration > 0 and keyframes[-1].time < duration:
        keyframes.append(CropKeyframe(duration, keyframes[-1].x, keyframes[-1].y))
    return keyframes


def dedupe_keyframes(keyframes: list[CropKeyframe], *, min_delta: int = 1) -> list[CropKeyframe]:
    """Drop keyframes that do not move, to keep the sendcmd script small."""
    if not keyframes:
        return []
    kept = [keyframes[0]]
    for frame in keyframes[1:]:
        last = kept[-1]
        if abs(frame.x - last.x) >= min_delta or abs(frame.y - last.y) >= min_delta:
            kept.append(frame)
    if kept[-1] is not keyframes[-1]:
        kept.append(keyframes[-1])
    return kept


def render_sendcmd_script(keyframes: list[CropKeyframe], target: str = "crop") -> str:
    """Render crop keyframes as an ffmpeg ``sendcmd`` script.

    ffmpeg's ``crop`` and ``overlay`` filters accept ``x``/``y`` as runtime
    commands, which is how a moving region is driven without building a huge
    nested expression that the parser would struggle with.
    """
    lines = [f"{frame.time:.3f} {target} x {frame.x}, {target} y {frame.y};" for frame in keyframes]
    return "\n".join(lines) + "\n"
