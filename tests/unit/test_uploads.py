"""Unit tests for upload filename handling and size limits.

An upload is the only place a name chosen entirely by the client becomes a
filesystem path, so the traversal and sanitisation cases are covered directly
rather than only through the HTTP layer.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from ffmpeg_mcp.config import Settings
from ffmpeg_mcp.errors import FileTooLargeError, InvalidParameterError
from ffmpeg_mcp.ui.uploads import safe_upload_name, store_upload, unique_path

from ..conftest import make_settings


class FakeUpload:
    """Stands in for Starlette's UploadFile."""

    def __init__(self, filename: str | None, data: bytes) -> None:
        self.filename = filename
        self.file = io.BytesIO(data)


class TestSafeUploadName:
    def test_an_ordinary_name_is_kept(self) -> None:
        assert safe_upload_name("holiday.mp4") == "holiday.mp4"

    @pytest.mark.parametrize(
        "attack",
        [
            "../../etc/passwd.mp4",
            "..\\..\\windows\\system32\\evil.mp4",
            "/etc/cron.d/evil.mp4",
            "....//....//evil.mp4",
        ],
    )
    def test_traversal_is_reduced_to_a_bare_filename(self, attack: str) -> None:
        result = safe_upload_name(attack)
        assert "/" not in result and "\\" not in result
        assert not result.startswith(".")
        assert Path(result).name == result

    def test_an_absolute_path_keeps_only_its_last_component(self) -> None:
        assert safe_upload_name("/var/tmp/clip.mp4") == "clip.mp4"

    def test_null_bytes_are_removed(self) -> None:
        assert "\x00" not in safe_upload_name("bad\x00name.mp4")

    def test_awkward_characters_are_replaced(self) -> None:
        result = safe_upload_name("my clip; rm -rf $HOME (final).mp4")
        assert result.endswith(".mp4")
        assert all(char.isalnum() or char in "._-" for char in result)

    def test_a_very_long_name_is_truncated(self) -> None:
        result = safe_upload_name("a" * 500 + ".mp4")
        assert len(result) <= 90
        assert result.endswith(".mp4")

    def test_a_name_that_sanitises_to_nothing_still_gets_one(self) -> None:
        assert safe_upload_name("....mp4").endswith(".mp4")

    @pytest.mark.parametrize(
        "name", ["evil.sh", "payload.py", "thing.exe", "archive.zip", "noextension"]
    )
    def test_unsupported_extensions_are_rejected(self, name: str) -> None:
        with pytest.raises(InvalidParameterError, match="Unsupported upload type"):
            safe_upload_name(name)

    def test_an_uppercase_extension_is_accepted_and_normalised(self) -> None:
        # Normalising avoids case-sensitivity surprises between filesystems.
        assert safe_upload_name("CLIP.MP4") == "CLIP.mp4"

    @pytest.mark.parametrize("name", [None, "", "   "])
    def test_a_missing_filename_is_rejected(self, name: str | None) -> None:
        with pytest.raises(InvalidParameterError):
            safe_upload_name(name)

    def test_a_double_extension_is_judged_on_the_last_one(self) -> None:
        # 'clip.sh.mp4' really is an mp4 as far as the filesystem is concerned.
        assert safe_upload_name("clip.sh.mp4").endswith(".mp4")
        with pytest.raises(InvalidParameterError):
            safe_upload_name("clip.mp4.sh")


class TestUniquePath:
    def test_a_free_name_is_used_as_is(self, workspace: Path) -> None:
        assert unique_path(workspace, "a.mp4") == workspace / "a.mp4"

    def test_a_collision_gets_a_suffix(self, workspace: Path) -> None:
        (workspace / "a.mp4").write_bytes(b"x")
        assert unique_path(workspace, "a.mp4") == workspace / "a_1.mp4"

    def test_repeated_collisions_keep_counting(self, workspace: Path) -> None:
        for name in ("a.mp4", "a_1.mp4", "a_2.mp4"):
            (workspace / name).write_bytes(b"x")
        assert unique_path(workspace, "a.mp4") == workspace / "a_3.mp4"


class TestStoreUpload:
    def test_a_file_lands_in_the_uploads_directory(self, settings: Settings) -> None:
        path = store_upload(FakeUpload("clip.mp4", b"video-bytes"), settings)
        assert path.parent == settings.uploads_dir
        assert path.read_bytes() == b"video-bytes"

    def test_the_stored_path_stays_inside_the_allowlist(self, settings: Settings) -> None:
        from ffmpeg_mcp.paths import resolve_within_roots

        path = store_upload(FakeUpload("../../escape.mp4", b"x"), settings)
        # Must not raise: the sanitised name cannot leave the workspace.
        assert resolve_within_roots(path, settings) == path
        assert path.parent == settings.uploads_dir

    def test_an_oversized_upload_is_rejected_and_leaves_nothing_behind(
        self, workspace: Path
    ) -> None:
        configured = make_settings(workspace, max_input_bytes=16)
        configured.ensure_dirs()
        with pytest.raises(FileTooLargeError):
            store_upload(FakeUpload("big.mp4", b"0" * 1024), configured)
        assert list(configured.uploads_dir.iterdir()) == []

    def test_an_empty_upload_is_rejected(self, settings: Settings) -> None:
        with pytest.raises(InvalidParameterError, match="empty"):
            store_upload(FakeUpload("nothing.mp4", b""), settings)
        assert list(settings.uploads_dir.iterdir()) == []

    def test_a_rejected_extension_writes_nothing(self, settings: Settings) -> None:
        with pytest.raises(InvalidParameterError):
            store_upload(FakeUpload("evil.sh", b"#!/bin/sh"), settings)
        assert list(settings.uploads_dir.iterdir()) == []

    def test_two_uploads_of_the_same_name_both_survive(self, settings: Settings) -> None:
        first = store_upload(FakeUpload("clip.mp4", b"one"), settings)
        second = store_upload(FakeUpload("clip.mp4", b"two"), settings)
        assert first != second
        assert first.read_bytes() == b"one"
        assert second.read_bytes() == b"two"
