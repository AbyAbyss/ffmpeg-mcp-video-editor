"""faster-whisper engine wrapper.

Model loading is slow (seconds to minutes on first use, since the weights are
downloaded), so loaded models are cached per (name, device, compute type) for
the life of the process. Transcription itself is CPU-bound and blocking, so
callers run it through ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import JobCancelledError, MissingDependencyError
from .models import Segment, WordTiming

log = logging.getLogger(__name__)

_models: dict[tuple[str, str, str], Any] = {}
_lock = threading.Lock()

# Whisper accepts these as source languages; the list is only used to give a
# clear error before a long job starts rather than after.
KNOWN_LANGUAGES = {
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
    "yue",
}

MODEL_SIZES = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
)


@dataclass
class TranscriptionOutcome:
    """Everything a transcription run produces."""

    segments: list[Segment] = field(default_factory=list)
    words: list[WordTiming] = field(default_factory=list)
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None

    @property
    def text(self) -> str:
        """The full transcript as one string."""
        return " ".join(s.text.strip() for s in self.segments).strip()


def _require_faster_whisper() -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MissingDependencyError(
            "Transcription needs the 'whisper' extra.",
            install="uv sync --extra whisper",
        ) from exc
    return WhisperModel


def load_model(name: str, compute_type: str = "int8", device: str = "auto") -> Any:
    """Load a Whisper model, reusing an already-loaded one where possible."""
    WhisperModel = _require_faster_whisper()
    key = (name, device, compute_type)
    with _lock:
        model = _models.get(key)
        if model is None:
            log.info("Loading Whisper model %s (%s, %s)", name, device, compute_type)
            model = WhisperModel(name, device=device, compute_type=compute_type)
            _models[key] = model
    return model


def clear_model_cache() -> None:
    """Drop cached models. Used by tests and to release memory."""
    with _lock:
        _models.clear()


def transcribe_file(
    audio_path: Path,
    *,
    model_name: str = "base",
    compute_type: str = "int8",
    device: str = "auto",
    language: str | None = None,
    task: str = "transcribe",
    word_timestamps: bool = False,
    vad_filter: bool = True,
    beam_size: int = 5,
    initial_prompt: str | None = None,
    on_progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> TranscriptionOutcome:
    """Transcribe (or translate) an audio file. Blocking; call from a thread.

    Args:
        audio_path: Audio or video file to read.
        model_name: A Whisper size name such as 'base' or 'large-v3'.
        language: Source language code, or None to auto-detect.
        task: 'transcribe' keeps the source language; 'translate' outputs English.
        word_timestamps: Also return per-word timings.
        on_progress: Called with a 0-100 percentage as segments arrive.
        should_cancel: Polled between segments; stops the run when it returns True.

    Returns:
        The segments, optional word timings, and detected language.
    """
    model = load_model(model_name, compute_type=compute_type, device=device)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        task=task,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        beam_size=beam_size,
        initial_prompt=initial_prompt,
    )

    total = float(getattr(info, "duration", 0.0) or 0.0)
    outcome = TranscriptionOutcome(
        language=getattr(info, "language", None),
        language_probability=getattr(info, "language_probability", None),
        duration=total or None,
    )

    # faster-whisper returns a generator: work only happens as it is consumed,
    # which is what makes incremental progress and cancellation possible.
    for segment in segments_iter:
        if should_cancel is not None and should_cancel():
            raise JobCancelledError("Transcription cancelled.")
        text = (segment.text or "").strip()
        if text:
            outcome.segments.append(
                Segment(start=float(segment.start), end=float(segment.end), text=text)
            )
        for word in getattr(segment, "words", None) or []:
            word_text = (word.word or "").strip()
            if word_text:
                outcome.words.append(
                    WordTiming(
                        start=float(word.start),
                        end=float(word.end),
                        word=word_text,
                        probability=getattr(word, "probability", None),
                    )
                )
        if on_progress and total > 0:
            on_progress(min(99.0, float(segment.end) / total * 100.0))
    return outcome
