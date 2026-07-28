"""Unit tests for path validation and the allowlist."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ffmpeg_mcp.config import Settings
from ffmpeg_mcp.errors import FileTooLargeError, InvalidPathError
from ffmpeg_mcp.paths import resolve_within_roots, validate_input_file, validate_output_path

from ..conftest import make_settings


class TestAllowlist:
    def test_path_inside_root_is_accepted(self, settings: Settings) -> None:
        target = settings.workspace / "clip.mp4"
        assert resolve_within_roots(target, settings) == target

    def test_relative_path_resolves_against_the_workspace(self, settings: Settings) -> None:
        assert resolve_within_roots("clip.mp4", settings) == settings.workspace / "clip.mp4"

    def test_path_outside_root_is_rejected(self, settings: Settings) -> None:
        with pytest.raises(InvalidPathError):
            resolve_within_roots("/etc/passwd", settings)

    def test_dot_dot_traversal_is_rejected(self, settings: Settings) -> None:
        with pytest.raises(InvalidPathError):
            resolve_within_roots(settings.workspace / ".." / ".." / "etc" / "passwd", settings)

    def test_symlink_escaping_the_root_is_rejected(self, settings: Settings) -> None:
        # The check must follow the link before comparing, or this passes wrongly.
        link = settings.workspace / "escape"
        link.symlink_to("/etc")
        with pytest.raises(InvalidPathError):
            resolve_within_roots(link / "passwd", settings)

    def test_symlink_inside_the_root_is_allowed(self, settings: Settings) -> None:
        inner = settings.workspace / "real"
        inner.mkdir()
        link = settings.workspace / "link"
        link.symlink_to(inner)
        assert resolve_within_roots(link / "a.mp4", settings) == inner / "a.mp4"

    def test_a_sibling_directory_with_a_shared_prefix_is_rejected(self, tmp_path: Path) -> None:
        # '/tmp/ws' must not authorise '/tmp/ws-evil'.
        root = tmp_path / "ws"
        root.mkdir()
        (tmp_path / "ws-evil").mkdir()
        configured = make_settings(root)
        with pytest.raises(InvalidPathError):
            resolve_within_roots(tmp_path / "ws-evil" / "x.mp4", configured)

    def test_multiple_roots_are_all_honoured(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        configured = make_settings(first, allowed_roots=[first, second])
        assert resolve_within_roots(second / "x.mp4", configured) == second / "x.mp4"


class TestInputValidation:
    def test_missing_file_is_rejected(self, settings: Settings) -> None:
        with pytest.raises(InvalidPathError):
            validate_input_file(settings.workspace / "nope.mp4", settings)

    def test_directory_is_not_a_valid_input(self, settings: Settings) -> None:
        with pytest.raises(InvalidPathError):
            validate_input_file(settings.workspace, settings)

    def test_existing_file_is_accepted(self, settings: Settings) -> None:
        path = settings.workspace / "clip.mp4"
        path.write_bytes(b"data")
        assert validate_input_file(path, settings) == path

    def test_oversized_file_is_rejected(self, workspace: Path) -> None:
        configured = make_settings(workspace, max_input_bytes=4)
        path = workspace / "big.mp4"
        path.write_bytes(b"0123456789")
        with pytest.raises(FileTooLargeError):
            validate_input_file(path, configured)


class TestOutputValidation:
    def test_none_falls_back_to_the_job_directory(self, settings: Settings) -> None:
        out = validate_output_path(None, suggested_name="a.mp4", job_id="job1", settings=settings)
        assert out == settings.jobs_dir / "job1" / "a.mp4"
        assert out.parent.is_dir()

    def test_parent_directory_is_created(self, settings: Settings) -> None:
        out = validate_output_path(
            settings.workspace / "deep" / "nested" / "a.mp4",
            suggested_name="a.mp4",
            settings=settings,
        )
        assert out.parent.is_dir()

    def test_directory_target_gets_the_suggested_name(self, settings: Settings) -> None:
        target = settings.workspace / "outdir"
        target.mkdir()
        out = validate_output_path(target, suggested_name="named.mp4", settings=settings)
        assert out == target / "named.mp4"

    def test_output_outside_the_allowlist_is_rejected(self, settings: Settings) -> None:
        with pytest.raises(InvalidPathError):
            validate_output_path(
                Path(os.sep) / "etc" / "evil.mp4", suggested_name="a.mp4", settings=settings
            )


class TestUnreachablePathMessages:
    """A tool error is read by the calling model, so it has to point somewhere.

    An attachment lives in the assistant's sandbox, not on the host. Saying only
    "outside the allowed roots" sends the model hunting for a misconfiguration
    that does not exist.
    """

    @pytest.mark.parametrize(
        "path",
        [
            "/mnt/user-data/uploads/IMG_1779.MOV",
            "/mnt/outputs/render.mp4",
            "/mnt/skills/public/thing",
            "/home/claude/clip.mp4",
            "/tmp/outputs/out.mp4",
        ],
    )
    def test_a_sandbox_path_says_it_is_not_on_this_machine(
        self, settings: Settings, path: str
    ) -> None:
        with pytest.raises(InvalidPathError) as info:
            resolve_within_roots(path, settings)
        assert info.value.details["reason"] == "assistant_sandbox_path"
        assert "not on this machine" in info.value.message
        assert "real path" in info.value.message

    def test_an_ordinary_outside_path_keeps_the_allowlist_message(self, settings: Settings) -> None:
        with pytest.raises(InvalidPathError) as info:
            resolve_within_roots("/etc/passwd", settings)
        assert info.value.details["reason"] == "outside_allowed_roots"
        assert "allowed roots" in info.value.message
        assert "FFMPEG_MCP_ALLOWED_ROOTS" in info.value.details["hint"]

    def test_a_nonexistent_outside_path_says_so_too(self, settings: Settings) -> None:
        with pytest.raises(InvalidPathError) as info:
            resolve_within_roots("/nowhere/at/all/clip.mp4", settings)
        assert "does not exist on this machine" in info.value.message

    def test_a_real_path_that_merely_sits_outside_does_not_claim_to_be_missing(
        self, settings: Settings
    ) -> None:
        with pytest.raises(InvalidPathError) as info:
            resolve_within_roots("/etc", settings)
        assert "does not exist" not in info.value.message

    def test_a_directory_merely_named_like_the_sandbox_inside_a_root_is_fine(
        self, workspace: Path
    ) -> None:
        # The check must not fire on a legitimate folder that happens to match.
        from ..conftest import make_settings

        configured = make_settings(workspace)
        target = workspace / "mnt" / "user-data"
        target.mkdir(parents=True)
        assert resolve_within_roots(target, configured) == target


class TestFontPaths:
    """Fonts live in read-only OS directories no allowlist would contain.

    Requiring ALLOWED_ROOTS to include /System just to draw text would open
    those roots to every other tool, so fonts get their own narrow rule.
    """

    def test_a_system_font_is_allowed(self, settings: Settings) -> None:
        from ffmpeg_mcp.paths import validate_font_file

        candidates = [
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/Library/Fonts/Arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
        present = next((c for c in candidates if c.is_file()), None)
        if present is None:
            pytest.skip("no system font present to test with")
        assert validate_font_file(present, settings) == present

    def test_a_font_inside_the_workspace_is_allowed(self, settings: Settings) -> None:
        from ffmpeg_mcp.paths import validate_font_file

        font = settings.workspace / "brand.ttf"
        font.write_bytes(b"not really a font, but a real file")
        assert validate_font_file(font, settings) == font

    def test_a_non_font_extension_is_refused(self, settings: Settings) -> None:
        from ffmpeg_mcp.paths import validate_font_file

        bad = settings.workspace / "payload.sh"
        bad.write_text("#!/bin/sh")
        with pytest.raises(InvalidPathError, match="ttf"):
            validate_font_file(bad, settings)

    def test_the_font_rule_does_not_open_the_rest_of_the_system(self, settings: Settings) -> None:
        # A font extension must not become a way to read arbitrary locations.
        from ffmpeg_mcp.paths import validate_font_file

        with pytest.raises(InvalidPathError):
            validate_font_file("/etc/passwd", settings)
        with pytest.raises(InvalidPathError):
            validate_font_file("/etc/shadow.ttf", settings)

    def test_a_missing_font_is_refused(self, settings: Settings) -> None:
        from ffmpeg_mcp.paths import validate_font_file

        with pytest.raises(InvalidPathError, match="does not exist"):
            validate_font_file(settings.workspace / "nope.ttf", settings)
