"""The structured timeline model and its filter-graph compiler.

This is the "do everything at once" entry point: one declarative description of
clips, transitions, overlays, captions and audio tracks, compiled into a single
``-filter_complex`` graph and rendered in one pass. The phase 6 timeline editor
produces exactly this structure, so the UI and a script driving the server hit
the same code path.

Compilation is deliberately pure — it takes the timeline plus each source's
probed metadata and returns the graph, the input list and the output maps — so
the graph shape is unit tested without running ffmpeg.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .errors import InvalidParameterError
from .ffmpeg.filters import (
    Filter,
    FilterGraph,
    acrossfade_filter,
    afade_filter,
    between_expr,
    drawtext_filter,
    escape_path_for_filter,
    fade_alpha_expr,
    pad_to_filter,
    scale_filter,
    setpts_filter,
    subtitles_filter,
    volume_filter,
    xfade_filter,
)
from .models import MediaInfo, StrictModel

# xfade transition names that ffmpeg ships. Restricting the set gives a clear
# error before the render instead of a filter-graph failure minutes in.
TRANSITIONS = (
    "fade",
    "fadeblack",
    "fadewhite",
    "wipeleft",
    "wiperight",
    "wipeup",
    "wipedown",
    "slideleft",
    "slideright",
    "slideup",
    "slidedown",
    "circlecrop",
    "rectcrop",
    "circleopen",
    "circleclose",
    "dissolve",
    "pixelize",
    "radial",
    "smoothleft",
    "smoothright",
    "smoothup",
    "smoothdown",
)

OverlayPosition = Literal[
    "top-left",
    "top-center",
    "top-right",
    "middle-left",
    "center",
    "middle-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
]

_OVERLAY_XY: dict[str, tuple[str, str]] = {
    "top-left": ("{m}", "{m}"),
    "top-center": ("(W-w)/2", "{m}"),
    "top-right": ("W-w-{m}", "{m}"),
    "middle-left": ("{m}", "(H-h)/2"),
    "center": ("(W-w)/2", "(H-h)/2"),
    "middle-right": ("W-w-{m}", "(H-h)/2"),
    "bottom-left": ("{m}", "H-h-{m}"),
    "bottom-center": ("(W-w)/2", "H-h-{m}"),
    "bottom-right": ("W-w-{m}", "H-h-{m}"),
}


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}


def is_still_image(source: str, info: MediaInfo | None) -> bool:
    """Whether an overlay source is a single still rather than a moving clip."""
    if Path(source).suffix.lower() in IMAGE_SUFFIXES:
        return True
    return info is None or info.duration is None


def overlay_eof_action(source: str, info: MediaInfo | None) -> str:
    """Pick ``overlay``'s eof_action for a source.

    A still image decodes to one frame at t=0, so 'pass' would make a watermark
    vanish immediately; it has to be held with 'repeat'. A video overlay that is
    shorter than the base should genuinely disappear when it ends, so it uses
    'pass' and does not freeze on its last frame.
    """
    return "repeat" if is_still_image(source, info) else "pass"


def overlay_position_expressions(position: str, margin: int) -> tuple[str, str]:
    """Return the (x, y) overlay expressions for a named position.

    ``W``/``H`` are the base frame's dimensions and ``w``/``h`` the overlay's —
    the opposite convention from drawtext, which is an easy thing to get wrong.
    """
    template = _OVERLAY_XY.get(position)
    if template is None:
        raise InvalidParameterError(
            "Unknown overlay position.", position=position, allowed=sorted(_OVERLAY_XY)
        )
    return template[0].format(m=margin), template[1].format(m=margin)


# --------------------------------------------------------------------------- #
# Timeline model
# --------------------------------------------------------------------------- #


class Transition(StrictModel):
    """A transition between one clip and the next."""

    type: str = Field(default="fade", description=f"One of: {', '.join(TRANSITIONS)}.")
    duration: float = Field(default=0.5, gt=0, le=30, description="Overlap length in seconds.")

    @model_validator(mode="after")
    def _check_type(self) -> Transition:
        if self.type not in TRANSITIONS:
            raise ValueError(f"unknown transition {self.type!r}")
        return self


class TimelineClip(StrictModel):
    """One clip on the video track."""

    source: str = Field(description="Path to the source media file.")
    in_point: float = Field(
        default=0.0, ge=0, description="Where to start reading the source, in seconds."
    )
    out_point: float | None = Field(
        default=None, description="Where to stop reading the source. Null means to its end."
    )
    speed: float = Field(
        default=1.0, gt=0.01, le=100, description="Playback multiplier for this clip."
    )
    transition_to_next: Transition | None = Field(
        default=None,
        description="Transition into the following clip. Ignored on the last clip.",
    )
    mute: bool = Field(default=False, description="Drop this clip's own audio.")

    @model_validator(mode="after")
    def _check_range(self) -> TimelineClip:
        if self.out_point is not None and self.out_point <= self.in_point:
            raise ValueError("out_point must be greater than in_point")
        return self

    def source_duration(self, info: MediaInfo) -> float:
        """How much of the source this clip consumes, before the speed change."""
        end = self.out_point if self.out_point is not None else (info.duration or 0.0)
        return max(0.0, end - self.in_point)

    def output_duration(self, info: MediaInfo) -> float:
        """How long this clip occupies on the timeline, after the speed change."""
        return self.source_duration(info) / self.speed


class TimelineTextOverlay(StrictModel):
    """A title or lower-third placed on the timeline."""

    text: str = Field(min_length=1)
    start: float = Field(default=0.0, ge=0)
    end: float = Field(gt=0)
    position: OverlayPosition = "bottom-center"
    margin: int = Field(default=40, ge=0, le=2000)
    font_size: int = Field(default=48, ge=6, le=400)
    font_color: str = Field(default="#FFFFFF")
    font_file: str | None = None
    fade: float = Field(
        default=0.0, ge=0, le=10, description="Fade in and out over this many seconds."
    )

    @model_validator(mode="after")
    def _check_window(self) -> TimelineTextOverlay:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class TimelineMediaOverlay(StrictModel):
    """A picture-in-picture, watermark, or logo placed on the timeline."""

    source: str = Field(description="Image or video to composite on top.")
    start: float = Field(default=0.0, ge=0)
    end: float | None = Field(default=None, description="Null means to the end of the timeline.")
    position: OverlayPosition = "top-right"
    margin: int = Field(default=24, ge=0, le=2000)
    width: int | None = Field(
        default=None, gt=0, le=16384, description="Scale the overlay to this width."
    )
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)


class TimelineAudioTrack(StrictModel):
    """An extra audio track — music, voiceover, or effects."""

    source: str
    start: float = Field(default=0.0, ge=0, description="Where it begins on the timeline.")
    in_point: float = Field(default=0.0, ge=0)
    out_point: float | None = None
    gain_db: float = Field(default=0.0, ge=-60, le=30)
    fade_in: float = Field(default=0.0, ge=0, le=60)
    fade_out: float = Field(default=0.0, ge=0, le=60)
    duck_under_voice: bool = Field(
        default=False,
        description="Automatically lower this track when the clips' own audio is loud.",
    )


class TimelineCaptions(StrictModel):
    """A subtitle file burned in across the whole timeline."""

    subtitle_path: str = Field(description="Path to an .srt or .ass file.")
    font_name: str = "Arial"
    font_size: int = Field(default=24, ge=6, le=400)
    font_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    position: str = "bottom-center"


class Timeline(StrictModel):
    """A complete edit: clips, transitions, overlays, captions, audio."""

    width: int = Field(default=1920, gt=0, le=16384)
    height: int = Field(default=1080, gt=0, le=16384)
    fps: float = Field(default=30.0, gt=0, le=240)
    background_color: str = Field(default="black")
    clips: list[TimelineClip] = Field(min_length=1, max_length=200)
    text_overlays: list[TimelineTextOverlay] = Field(default_factory=list, max_length=100)
    media_overlays: list[TimelineMediaOverlay] = Field(default_factory=list, max_length=50)
    audio_tracks: list[TimelineAudioTrack] = Field(default_factory=list, max_length=20)
    captions: TimelineCaptions | None = None
    normalize_audio: bool = Field(
        default=False, description="Apply EBU R128 loudness normalisation to the final mix."
    )
    target_lufs: float = Field(default=-16.0, ge=-70, le=-5)


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #


@dataclass
class CompiledTimeline:
    """The rendered graph plus everything needed to invoke ffmpeg."""

    inputs: list[list[str]] = field(default_factory=list)
    graph: str = ""
    video_label: str = "vout"
    audio_label: str | None = None
    duration: float = 0.0
    notes: list[str] = field(default_factory=list)


def _clip_video_chain(
    graph: FilterGraph,
    input_index: int,
    clip: TimelineClip,
    info: MediaInfo,
    timeline: Timeline,
    label: str,
) -> None:
    """Trim, retime, and conform one clip to the timeline's format."""
    chain = graph.chain([f"{input_index}:v:0"], [label])
    trim_options = f"start={clip.in_point:.6f}"
    if clip.out_point is not None:
        trim_options += f":end={clip.out_point:.6f}"
    chain.add(
        Filter("trim", {}, raw_options=trim_options),
        Filter("setpts", {}, raw_options="PTS-STARTPTS"),
    )
    if clip.speed != 1.0:
        chain.add(setpts_filter(clip.speed))
    chain.add(
        scale_filter(timeline.width, timeline.height, keep_aspect=True),
        pad_to_filter(timeline.width, timeline.height, timeline.background_color),
        Filter("setsar", {}, raw_options="1"),
        Filter("fps", {"fps": timeline.fps}),
        Filter("format", {}, raw_options="yuv420p"),
    )


def _clip_audio_chain(
    graph: FilterGraph,
    input_index: int,
    clip: TimelineClip,
    info: MediaInfo,
    duration: float,
    label: str,
    sample_rate: int,
) -> None:
    """Produce one audio stream per clip, synthesising silence where there is none.

    Every clip must contribute audio of the right length, or a later concat
    would misalign the whole mix.
    """
    if info.has_audio and not clip.mute:
        chain = graph.chain([f"{input_index}:a:0"], [label])
        trim_options = f"start={clip.in_point:.6f}"
        if clip.out_point is not None:
            trim_options += f":end={clip.out_point:.6f}"
        chain.add(
            Filter("atrim", {}, raw_options=trim_options),
            Filter("asetpts", {}, raw_options="PTS-STARTPTS"),
        )
        if clip.speed != 1.0:
            from .ffmpeg.filters import atempo_chain

            chain.extend(atempo_chain(clip.speed))
        chain.add(
            Filter(
                "aformat",
                {},
                raw_options=(f"sample_fmts=fltp:sample_rates={sample_rate}:channel_layouts=stereo"),
            ),
            Filter("apad", {}, raw_options=f"whole_dur={duration:.6f}"),
            Filter("atrim", {}, raw_options=f"duration={duration:.6f}"),
            Filter("asetpts", {}, raw_options="PTS-STARTPTS"),
        )
    else:
        chain = graph.chain([], [label])
        chain.add(
            Filter(
                "anullsrc",
                {},
                raw_options=f"channel_layout=stereo:sample_rate={sample_rate}:d={duration:.6f}",
            )
        )


def _join_clips(
    graph: FilterGraph,
    video_labels: list[str],
    audio_labels: list[str],
    clips: list[TimelineClip],
    durations: list[float],
) -> tuple[str, str, float]:
    """Join clips pairwise, using a transition where one is requested.

    Accumulating pairwise keeps the timing arithmetic honest: a transition
    overlaps the two clips, so the running total shortens by its duration, and
    the next transition's offset depends on that updated total.
    """
    video = video_labels[0]
    audio = audio_labels[0]
    total = durations[0]

    for index in range(1, len(video_labels)):
        transition = clips[index - 1].transition_to_next
        next_video, next_audio = video_labels[index], audio_labels[index]
        out_video = f"vj{index}"
        out_audio = f"aj{index}"

        if transition is not None:
            overlap = min(transition.duration, total, durations[index])
            if overlap <= 0:
                raise InvalidParameterError(
                    "Transition is longer than the clips it joins.",
                    clip_index=index - 1,
                    transition_duration=transition.duration,
                )
            offset = max(0.0, total - overlap)
            graph.chain([video, next_video], [out_video]).add(
                xfade_filter(transition.type, overlap, offset)
            )
            graph.chain([audio, next_audio], [out_audio]).add(acrossfade_filter(overlap))
            total = total + durations[index] - overlap
        else:
            graph.chain([video, next_video], [out_video]).add(
                Filter("concat", {"n": 2, "v": 1, "a": 0})
            )
            graph.chain([audio, next_audio], [out_audio]).add(
                Filter("concat", {"n": 2, "v": 0, "a": 1})
            )
            total += durations[index]
        video, audio = out_video, out_audio
    return video, audio, total


def compile_timeline(
    timeline: Timeline,
    clip_infos: list[MediaInfo],
    overlay_infos: list[MediaInfo | None],
    audio_infos: list[MediaInfo],
    *,
    text_files: list[str] | None = None,
    sample_rate: int = 48000,
) -> CompiledTimeline:
    """Compile a timeline into a single ffmpeg filter graph.

    Args:
        timeline: The edit to render.
        clip_infos: Probed metadata for each clip, in the same order.
        overlay_infos: Probed metadata for each media overlay; None for stills.
        audio_infos: Probed metadata for each extra audio track.
        text_files: Sidecar files holding each text overlay's text, so the text
            itself never enters the filter graph.
        sample_rate: Sample rate the whole mix is conformed to.

    Returns:
        The inputs, graph, output labels and total duration.
    """
    if len(clip_infos) != len(timeline.clips):
        raise InvalidParameterError("A probe result is required for every clip.")

    graph = FilterGraph()
    inputs: list[list[str]] = []
    notes: list[str] = []

    durations = [
        clip.output_duration(info) for clip, info in zip(timeline.clips, clip_infos, strict=True)
    ]
    for index, duration in enumerate(durations):
        if duration <= 0:
            raise InvalidParameterError(
                "Clip has zero length; check its in and out points.", clip_index=index
            )

    video_labels: list[str] = []
    audio_labels: list[str] = []
    for index, (clip, info) in enumerate(zip(timeline.clips, clip_infos, strict=True)):
        inputs.append(["-i", clip.source])
        if not info.has_video:
            raise InvalidParameterError(
                "Timeline clips must have a video stream.", clip_index=index, source=clip.source
            )
        _clip_video_chain(graph, index, clip, info, timeline, f"cv{index}")
        _clip_audio_chain(graph, index, clip, info, durations[index], f"ca{index}", sample_rate)
        video_labels.append(f"cv{index}")
        audio_labels.append(f"ca{index}")
        if not info.has_audio:
            notes.append(f"Clip {index} has no audio; silence was inserted.")

    video, audio, total = _join_clips(graph, video_labels, audio_labels, timeline.clips, durations)

    # -- media overlays ---------------------------------------------------- #
    next_input = len(timeline.clips)
    for index, overlay in enumerate(timeline.media_overlays):
        inputs.append(["-i", overlay.source])
        overlay_info = overlay_infos[index] if index < len(overlay_infos) else None
        is_still = is_still_image(overlay.source, overlay_info)
        prepared = f"ov{index}"
        chain = graph.chain([f"{next_input}:v:0"], [prepared])
        if overlay.width:
            chain.add(scale_filter(overlay.width, None))
        if overlay.opacity < 1.0:
            chain.add(
                Filter("format", {}, raw_options="rgba"),
                Filter("colorchannelmixer", {}, raw_options=f"aa={overlay.opacity:.4f}"),
            )
        end = overlay.end if overlay.end is not None else total
        x_expr, y_expr = overlay_position_expressions(overlay.position, overlay.margin)
        eof_action = overlay_eof_action(overlay.source, overlay_info)
        out_label = f"vo{index}"
        graph.chain([video, prepared], [out_label]).add(
            Filter(
                "overlay",
                {},
                raw_options=(
                    f"x={x_expr}:y={y_expr}:eof_action={eof_action}:"
                    f"enable='{between_expr(overlay.start, end)}'"
                ),
            )
        )
        video = out_label
        next_input += 1
        if is_still:
            notes.append(f"Overlay {index} treated as a still image.")

    # -- text overlays and captions ---------------------------------------- #
    text_filters: list[Filter] = []
    for index, text in enumerate(timeline.text_overlays):
        x_expr, y_expr = _text_position(text)
        alpha = (
            fade_alpha_expr(text.start, text.end, text.fade, text.fade) if text.fade > 0 else None
        )
        source_file = text_files[index] if text_files and index < len(text_files) else None
        text_filters.append(
            drawtext_filter(
                textfile=source_file,
                text=None if source_file else text.text,
                fontfile=text.font_file,
                fontsize=text.font_size,
                fontcolor=_hex_colour(text.font_color),
                x=x_expr,
                y=y_expr,
                alpha_expr=alpha,
                enable_expr=between_expr(text.start, text.end),
            )
        )
    if timeline.captions:
        text_filters.append(
            subtitles_filter(
                timeline.captions.subtitle_path,
                force_style={
                    "FontName": timeline.captions.font_name,
                    "FontSize": timeline.captions.font_size,
                },
            )
        )
    if text_filters:
        graph.chain([video], ["vtext"]).extend(text_filters)
        video = "vtext"

    # -- extra audio tracks ------------------------------------------------- #
    music_labels: list[str] = []
    for index, track in enumerate(timeline.audio_tracks):
        inputs.append(["-i", track.source])
        label = f"mt{index}"
        chain = graph.chain([f"{next_input}:a:0"], [label])
        trim_options = f"start={track.in_point:.6f}"
        if track.out_point is not None:
            trim_options += f":end={track.out_point:.6f}"
        chain.add(
            Filter("atrim", {}, raw_options=trim_options),
            Filter("asetpts", {}, raw_options="PTS-STARTPTS"),
            Filter(
                "aformat",
                {},
                raw_options=f"sample_fmts=fltp:sample_rates={sample_rate}:channel_layouts=stereo",
            ),
        )
        if track.gain_db != 0.0:
            chain.add(volume_filter(track.gain_db))
        if track.fade_in > 0:
            chain.add(afade_filter("in", 0.0, track.fade_in))
        if track.fade_out > 0:
            track_length = _audio_track_length(track, audio_infos, index, total)
            chain.add(afade_filter("out", max(0.0, track_length - track.fade_out), track.fade_out))
        if track.start > 0:
            chain.add(Filter("adelay", {}, raw_options=f"{int(track.start * 1000)}:all=1"))
        music_labels.append(label)
        next_input += 1

    ducked = [t for t in timeline.audio_tracks if t.duck_under_voice]
    if ducked and music_labels:
        audio = _apply_ducking(graph, audio, music_labels, timeline, notes)
    elif music_labels:
        graph.chain([audio, *music_labels], ["amixed"]).add(
            Filter(
                "amix",
                {},
                raw_options=(
                    f"inputs={1 + len(music_labels)}:duration=first:"
                    "dropout_transition=0:normalize=0"
                ),
            )
        )
        audio = "amixed"

    final_audio_chain = graph.chain([audio], ["aout"])
    if timeline.normalize_audio:
        from .ffmpeg.filters import loudnorm_filter

        final_audio_chain.add(loudnorm_filter(timeline.target_lufs, -1.5, 11.0))
    final_audio_chain.add(
        Filter(
            "aformat",
            {},
            raw_options=f"sample_fmts=fltp:sample_rates={sample_rate}:channel_layouts=stereo",
        )
    )

    graph.chain([video], ["vout"]).add(Filter("format", {}, raw_options="yuv420p"))

    return CompiledTimeline(
        inputs=inputs,
        graph=graph.render(),
        video_label="vout",
        audio_label="aout",
        duration=total,
        notes=notes,
    )


def _apply_ducking(
    graph: FilterGraph,
    voice_label: str,
    music_labels: list[str],
    timeline: Timeline,
    notes: list[str],
) -> str:
    """Duck the music tracks under the clips' own audio, then mix.

    ``sidechaincompress`` needs the voice as a second input to key off, so the
    voice is split first: one copy drives the compressor, the other is mixed in.
    """
    graph.chain([voice_label], ["voicemix", "voicekey"]).add(Filter("asplit", {}, raw_options="2"))
    ducked_labels: list[str] = []
    key = "voicekey"
    for index, label in enumerate(music_labels):
        track = timeline.audio_tracks[index]
        if not track.duck_under_voice:
            ducked_labels.append(label)
            continue
        out_label = f"duck{index}"
        if index < len(music_labels) - 1:
            # The key stream is consumed by each compressor, so split off a copy.
            graph.chain([key], [f"key{index}", f"keyrest{index}"]).add(
                Filter("asplit", {}, raw_options="2")
            )
            this_key, key = f"key{index}", f"keyrest{index}"
        else:
            this_key = key
        graph.chain([label, this_key], [out_label]).add(
            Filter(
                "sidechaincompress",
                {},
                raw_options="threshold=0.03:ratio=8:attack=20:release=400:makeup=1",
            )
        )
        ducked_labels.append(out_label)
    notes.append("Background audio is ducked under the clips' own audio.")
    graph.chain(["voicemix", *ducked_labels], ["amixed"]).add(
        Filter(
            "amix",
            {},
            raw_options=(
                f"inputs={1 + len(ducked_labels)}:duration=first:dropout_transition=0:normalize=0"
            ),
        )
    )
    return "amixed"


def _audio_track_length(
    track: TimelineAudioTrack, infos: list[MediaInfo], index: int, total: float
) -> float:
    if track.out_point is not None:
        return track.out_point - track.in_point
    if index < len(infos) and infos[index].duration:
        return max(0.0, (infos[index].duration or 0.0) - track.in_point)
    return total


def _text_position(text: TimelineTextOverlay) -> tuple[str, str]:
    from .tools.captions import position_expressions

    return position_expressions(text.position, text.margin)


def _hex_colour(colour: str) -> str:
    from .tools.captions import _hex_to_drawtext_colour

    return _hex_to_drawtext_colour(colour)


__all__ = [
    "TRANSITIONS",
    "CompiledTimeline",
    "Timeline",
    "TimelineAudioTrack",
    "TimelineCaptions",
    "TimelineClip",
    "TimelineMediaOverlay",
    "TimelineTextOverlay",
    "Transition",
    "compile_timeline",
    "escape_path_for_filter",
    "is_still_image",
    "overlay_eof_action",
    "overlay_position_expressions",
]
