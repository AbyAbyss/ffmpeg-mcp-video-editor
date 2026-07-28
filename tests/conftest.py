"""Shared test fixtures.

Unit tests never touch ffmpeg. Integration tests use tiny generated clips so the
suite stays fast; they are marked ``integration`` and can be deselected.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from ffmpeg_mcp.binaries import reset_binary_cache
from ffmpeg_mcp.config import Settings, set_settings
from ffmpeg_mcp.jobs.store import JobStore, reset_store

FIXTURE_DIR = Path(__file__).parent / "fixtures"

ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="needs a real ffmpeg binary"
)


def make_settings(workspace: Path, **overrides: object) -> Settings:
    """Build a Settings object pointed at a throwaway workspace."""
    defaults: dict[str, object] = {
        "workspace": workspace,
        "allowed_roots": [workspace],
        "max_input_bytes": 16 * 1024**3,
        "max_job_seconds": 300,
        "retention_hours": 24,
        "worker_concurrency": 2,
        "ffmpeg_path": None,
        "ffprobe_path": None,
        "auto_download_ffmpeg": False,
        "whisper_model": "tiny",
        "whisper_compute_type": "int8",
        "log_level": "WARNING",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An isolated workspace directory."""
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def settings(workspace: Path) -> Iterator[Settings]:
    """Process settings pointed at an isolated workspace, restored afterwards."""
    configured = make_settings(workspace)
    configured.ensure_dirs()
    set_settings(configured)
    reset_store()
    reset_binary_cache()
    yield configured
    set_settings(None)
    reset_store()
    reset_binary_cache()


@pytest.fixture
def store(settings: Settings) -> JobStore:
    """A job store on the isolated workspace."""
    return JobStore(settings)


def _generate(path: Path, args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *args, "-y", str(path)],
        check=True,
        timeout=120,
    )


@pytest.fixture(scope="session")
def sample_video() -> Path:
    """A 3-second 320x240 30fps clip with a sine tone, generated once per session."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs a real ffmpeg binary")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "sample.mp4"
    if not path.exists():
        # fmt: off
        _generate(
            path,
            [
                "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=3",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
            ],
        )
        # fmt: on
    return path


@pytest.fixture(scope="session")
def sample_video_alt() -> Path:
    """A second clip at a different resolution and frame rate, for concat tests."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs a real ffmpeg binary")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "sample_alt.mp4"
    if not path.exists():
        # fmt: off
        _generate(
            path,
            [
                "-f", "lavfi", "-i", "smptebars=size=640x360:rate=25:duration=2",
                "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
            ],
        )
        # fmt: on
    return path


@pytest.fixture(scope="session")
def sample_image() -> Path:
    """A small PNG, for overlay and watermark tests."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs a real ffmpeg binary")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "logo.png"
    if not path.exists():
        _generate(
            path, ["-f", "lavfi", "-i", "color=c=red:size=64x64:duration=1", "-frames:v", "1"]
        )
    return path


@pytest.fixture(scope="session")
def sample_speech() -> Path:
    """A short spoken-word-ish audio file. Falls back to a tone when no TTS exists.

    Whisper on a pure tone returns no segments, which is still a valid result to
    assert on; tests that need real words say so explicitly.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs a real ffmpeg binary")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "speech.wav"
    if not path.exists():
        say = shutil.which("say")
        if say:
            aiff = path.with_suffix(".aiff")
            subprocess.run(
                [say, "-o", str(aiff), "Testing one two three. This is a caption test."],
                check=True,
                timeout=60,
            )
            _generate(path, ["-i", str(aiff), "-ar", "16000", "-ac", "1"])
            aiff.unlink(missing_ok=True)
        else:
            _generate(
                path,
                ["-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-ar", "16000", "-ac", "1"],
            )
    return path


def pytest_configure(config: pytest.Config) -> None:
    """Register the integration marker (also declared in pyproject)."""
    config.addinivalue_line("markers", "integration: needs a real ffmpeg binary")


@pytest.fixture
def clip(settings: Settings, sample_video: Path) -> Path:
    """The sample clip copied inside the isolated workspace."""
    destination = settings.workspace / "clip.mp4"
    shutil.copy(sample_video, destination)
    return destination


@pytest.fixture
def clip_alt(settings: Settings, sample_video_alt: Path) -> Path:
    """The second sample clip copied inside the isolated workspace."""
    destination = settings.workspace / "clip_alt.mp4"
    shutil.copy(sample_video_alt, destination)
    return destination


@pytest.fixture
def logo(settings: Settings, sample_image: Path) -> Path:
    """The sample PNG copied inside the isolated workspace."""
    destination = settings.workspace / "logo.png"
    shutil.copy(sample_image, destination)
    return destination


@pytest.fixture
def speech(settings: Settings, sample_speech: Path) -> Path:
    """The sample speech audio copied inside the isolated workspace."""
    destination = settings.workspace / "speech.wav"
    shutil.copy(sample_speech, destination)
    return destination


@pytest.fixture(scope="session")
def sample_speech_video(sample_speech: Path) -> Path:
    """A video with real spoken audio, for auto-captioning tests."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "speech_video.mp4"
    if not path.exists():
        # fmt: off
        _generate(
            path,
            [
                "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15",
                "-i", str(sample_speech),
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
            ],
        )
        # fmt: on
    return path


@pytest.fixture
def speech_clip(settings: Settings, sample_speech_video: Path) -> Path:
    """The spoken-audio video copied inside the isolated workspace."""
    destination = settings.workspace / "speech_clip.mp4"
    shutil.copy(sample_speech_video, destination)
    return destination


def _draw_synthetic_face(image, centre_x: int, centre_y: int, radius: int) -> None:
    """Draw a crude but detectable face.

    BlazeFace finds this reliably (around 0.85 confidence), which is what makes
    face fixtures possible without shipping a photograph of a real person.
    """
    import cv2

    skin = (150, 190, 225)
    cv2.ellipse(image, (centre_x, centre_y), (int(radius * 0.78), radius), 0, 0, 360, skin, -1)
    for side in (-0.32, 0.32):
        eye_x = int(centre_x + side * radius)
        eye_y = int(centre_y - 0.18 * radius)
        cv2.ellipse(
            image,
            (eye_x, eye_y),
            (int(radius * 0.16), int(radius * 0.10)),
            0,
            0,
            360,
            (255, 255, 255),
            -1,
        )
        cv2.circle(image, (eye_x, eye_y), int(radius * 0.06), (40, 30, 25), -1)
        cv2.ellipse(
            image,
            (eye_x, int(centre_y - 0.36 * radius)),
            (int(radius * 0.20), int(radius * 0.09)),
            0,
            180,
            360,
            (60, 45, 40),
            3,
        )
    cv2.line(
        image,
        (centre_x, int(centre_y - 0.05 * radius)),
        (centre_x, int(centre_y + 0.16 * radius)),
        (110, 150, 190),
        3,
    )
    cv2.ellipse(
        image,
        (centre_x, int(centre_y + 0.36 * radius)),
        (int(radius * 0.28), int(radius * 0.14)),
        0,
        0,
        180,
        (70, 60, 120),
        -1,
    )


@pytest.fixture(scope="session")
def face_video() -> Path:
    """A 4-second clip with one synthetic face drifting left to right.

    Movement is what makes the tracking and smoothing tests meaningful; a static
    face would pass even with the crop path wired up wrongly.
    """
    pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs a real ffmpeg binary")
    import cv2
    import numpy as np

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "face_moving.mp4"
    if path.exists():
        return path

    width, height, fps, seconds = 640, 480, 15, 4
    frames_dir = FIXTURE_DIR / "_face_frames"
    frames_dir.mkdir(exist_ok=True)
    total = fps * seconds
    for index in range(total):
        image = np.full((height, width, 3), 210, np.uint8)
        progress = index / max(1, total - 1)
        _draw_synthetic_face(image, int(140 + progress * 360), 240, 95)
        cv2.imwrite(str(frames_dir / f"f{index:04d}.png"), image)
    # fmt: off
    _generate(
        path,
        [
            "-framerate", str(fps), "-i", str(frames_dir / "f%04d.png"),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        ],
    )
    # fmt: on
    shutil.rmtree(frames_dir, ignore_errors=True)
    return path


@pytest.fixture(scope="session")
def two_face_video() -> Path:
    """A clip with two faces: a large central one and a small one at the edge."""
    pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs a real ffmpeg binary")
    import cv2
    import numpy as np

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "two_faces.mp4"
    if path.exists():
        return path

    width, height, fps, seconds = 640, 480, 15, 3
    frames_dir = FIXTURE_DIR / "_two_face_frames"
    frames_dir.mkdir(exist_ok=True)
    total = fps * seconds
    for index in range(total):
        image = np.full((height, width, 3), 205, np.uint8)
        _draw_synthetic_face(image, 260, 250, 105)
        _draw_synthetic_face(image, 530, 130, 78)
        cv2.imwrite(str(frames_dir / f"f{index:04d}.png"), image)
    # fmt: off
    _generate(
        path,
        [
            "-framerate", str(fps), "-i", str(frames_dir / "f%04d.png"),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        ],
    )
    # fmt: on
    shutil.rmtree(frames_dir, ignore_errors=True)
    return path


@pytest.fixture
def face_clip(settings: Settings, face_video: Path) -> Path:
    """The moving-face clip copied inside the isolated workspace."""
    destination = settings.workspace / "face.mp4"
    shutil.copy(face_video, destination)
    return destination


@pytest.fixture
def two_face_clip(settings: Settings, two_face_video: Path) -> Path:
    """The two-face clip copied inside the isolated workspace."""
    destination = settings.workspace / "two_faces.mp4"
    shutil.copy(two_face_video, destination)
    return destination


@pytest.fixture(scope="session")
def cuts_video() -> Path:
    """Three visually distinct shots joined hard, for scene detection."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs a real ffmpeg binary")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "cuts.mp4"
    if not path.exists():
        # fmt: off
        _generate(
            path,
            [
                "-f", "lavfi", "-i", "color=c=red:s=320x240:r=15:d=2",
                "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=15:d=2",
                "-f", "lavfi", "-i", "color=c=white:s=320x240:r=15:d=2",
                "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
                "-map", "[v]",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            ],
        )
        # fmt: on
    return path


@pytest.fixture
def cuts_clip(settings: Settings, cuts_video: Path) -> Path:
    """The multi-shot clip copied inside the isolated workspace."""
    destination = settings.workspace / "cuts.mp4"
    shutil.copy(cuts_video, destination)
    return destination


@pytest.fixture(scope="session")
def green_screen_video() -> Path:
    """A green background with a red square, for chroma-key tests."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs a real ffmpeg binary")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "green_screen.mp4"
    if not path.exists():
        # fmt: off
        _generate(
            path,
            [
                "-f", "lavfi", "-i", "color=c=0x00FF00:s=320x240:r=15:d=3",
                "-f", "lavfi", "-i", "color=c=red:s=80x80:r=15:d=3",
                "-filter_complex", "[0:v][1:v]overlay=x=120:y=80[v]",
                "-map", "[v]",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            ],
        )
        # fmt: on
    return path


@pytest.fixture
def green_clip(settings: Settings, green_screen_video: Path) -> Path:
    """The green-screen clip copied inside the isolated workspace."""
    destination = settings.workspace / "green.mp4"
    shutil.copy(green_screen_video, destination)
    return destination
