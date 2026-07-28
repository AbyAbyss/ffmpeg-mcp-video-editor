"""Downloading and caching the MediaPipe face-detection model.

MediaPipe's Tasks API needs a ``.tflite`` bundle that ships separately from the
Python wheel, and OpenCV's wheel no longer bundles Haar cascades, so a small
model download is unavoidable. It follows the same policy as the ffmpeg binary
manager: cached under the workspace, and hash-pinned on first download since
Google publishes no digest alongside it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import httpx

from ..config import Settings, get_settings
from ..errors import ChecksumMismatchError, MissingDependencyError

log = logging.getLogger(__name__)

FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
FACE_DETECTOR_FILENAME = "blaze_face_short_range.tflite"
# Verified at build time; a mismatch means the published asset changed.
FACE_DETECTOR_SHA256 = "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"
_MAX_MODEL_BYTES = 64 * 1024 * 1024


def models_dir(settings: Settings | None = None) -> Path:
    """Where downloaded vision models are cached."""
    settings = settings or get_settings()
    path = settings.workspace / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_face_model(settings: Settings | None = None) -> Path:
    """Return the path to the face-detection model, downloading it if needed."""
    settings = settings or get_settings()
    directory = models_dir(settings)
    target = directory / FACE_DETECTOR_FILENAME
    if target.exists() and target.stat().st_size > 0:
        return target

    log.info("Downloading face detection model from %s", FACE_DETECTOR_URL)
    temporary = target.with_suffix(".partial")
    try:
        with (
            httpx.Client() as client,
            client.stream(
                "GET", FACE_DETECTOR_URL, follow_redirects=True, timeout=120.0
            ) as response,
        ):
            response.raise_for_status()
            written = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(256 * 1024):
                    written += len(chunk)
                    if written > _MAX_MODEL_BYTES:
                        raise ChecksumMismatchError(
                            "Face model download is implausibly large.", url=FACE_DETECTOR_URL
                        )
                    handle.write(chunk)
        digest = _sha256(temporary)
        if digest != FACE_DETECTOR_SHA256:
            raise ChecksumMismatchError(
                "Face detection model failed checksum verification.",
                url=FACE_DETECTOR_URL,
                expected=FACE_DETECTOR_SHA256,
                actual=digest,
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    (directory / "manifest.json").write_text(
        json.dumps({FACE_DETECTOR_URL: FACE_DETECTOR_SHA256}, indent=2)
    )
    return target


def require_vision() -> None:
    """Raise a clear error if the vision extra is not installed."""
    import importlib.util

    missing = [
        name
        for name, spec in (
            ("mediapipe", importlib.util.find_spec("mediapipe")),
            ("opencv-python", importlib.util.find_spec("cv2")),
            ("numpy", importlib.util.find_spec("numpy")),
        )
        if spec is None
    ]
    if missing:
        raise MissingDependencyError(
            "Face detection needs the 'vision' extra.",
            missing=missing,
            install="uv sync --extra vision",
        )
