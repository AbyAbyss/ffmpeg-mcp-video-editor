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
