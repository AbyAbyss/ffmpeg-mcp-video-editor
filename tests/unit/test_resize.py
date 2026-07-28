"""Unit tests for resolution and aspect-ratio conversion."""

from __future__ import annotations

import pytest

from ffmpeg_mcp.errors import InvalidParameterError
from ffmpeg_mcp.tools.resize import (
    RESOLUTION_PRESETS,
    build_resize_graph,
    crop_offsets,
    resolve_target_size,
)


def target(**kwargs: object) -> tuple[int, int]:
    defaults: dict[str, object] = {
        "source_width": 1920,
        "source_height": 1080,
        "preset": None,
        "width": None,
        "height": None,
        "aspect_ratio": None,
    }
    defaults.update(kwargs)
    return resolve_target_size(**defaults)  # type: ignore[arg-type]


class TestPresets:
    @pytest.mark.parametrize(
        ("preset", "expected"),
        [
            ("reel", (1080, 1920)),
            ("tiktok", (1080, 1920)),
            ("youtube_short", (1080, 1920)),
            ("story", (1080, 1920)),
            ("youtube_1080p", (1920, 1080)),
            ("youtube_4k", (3840, 2160)),
            ("instagram_square", (1080, 1080)),
            ("instagram_portrait", (1080, 1350)),
        ],
    )
    def test_named_presets(self, preset: str, expected: tuple[int, int]) -> None:
        assert target(preset=preset) == expected

    def test_preset_names_are_case_insensitive(self) -> None:
        assert target(preset="Reel") == target(preset="reel")

    def test_an_unknown_preset_is_rejected_with_the_available_list(self) -> None:
        with pytest.raises(InvalidParameterError) as info:
            target(preset="myspace")
        assert "available" in info.value.details

    def test_a_preset_overrides_explicit_dimensions(self) -> None:
        assert target(preset="reel", width=640, height=480) == (1080, 1920)

    def test_every_preset_is_even_dimensioned(self) -> None:
        for width, height in RESOLUTION_PRESETS.values():
            assert width % 2 == 0 and height % 2 == 0


class TestExplicitDimensions:
    def test_both_dimensions(self) -> None:
        assert target(width=1280, height=720) == (1280, 720)

    def test_width_alone_preserves_the_source_ratio(self) -> None:
        assert target(width=960) == (960, 540)

    def test_height_alone_preserves_the_source_ratio(self) -> None:
        assert target(height=540) == (960, 540)

    def test_odd_dimensions_are_rounded_down_to_even(self) -> None:
        assert target(width=1281, height=721) == (1280, 720)

    def test_nothing_at_all_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError, match="preset"):
            target()


class TestAspectRatio:
    def test_ratio_with_a_width(self) -> None:
        assert target(aspect_ratio="9:16", width=1080) == (1080, 1920)

    def test_ratio_with_a_height(self) -> None:
        assert target(aspect_ratio="9:16", height=1920) == (1080, 1920)

    def test_ratio_alone_preserves_roughly_the_source_area(self) -> None:
        width, height = target(aspect_ratio="9:16")
        assert width / height == pytest.approx(9 / 16, abs=0.01)
        assert width * height == pytest.approx(1920 * 1080, rel=0.02)

    def test_a_square_ratio(self) -> None:
        width, height = target(aspect_ratio="1:1")
        assert width == height

    def test_an_invalid_ratio_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            target(aspect_ratio="not-a-ratio")

    def test_the_result_is_always_even(self) -> None:
        for ratio in ("9:16", "16:9", "1:1", "4:5", "2.35"):
            width, height = target(aspect_ratio=ratio)
            assert width % 2 == 0 and height % 2 == 0


class TestCropOffsets:
    def test_centre_focus_splits_the_overflow(self) -> None:
        x, y = crop_offsets(1920, 1080, 608, 1080, "center")
        assert x == "(iw-608)*0.5"
        assert y == "(ih-1080)*0.5"

    def test_top_focus_keeps_the_top(self) -> None:
        _, y = crop_offsets(1080, 1920, 1080, 1920, "top")
        assert y == "(ih-1920)*0.0"

    def test_bottom_focus_keeps_the_bottom(self) -> None:
        _, y = crop_offsets(1080, 1920, 1080, 1920, "bottom")
        assert y == "(ih-1920)*1.0"

    def test_an_unknown_focus_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            crop_offsets(100, 100, 50, 50, "diagonally")


class TestResizeGraph:
    def test_cover_scales_up_then_crops(self) -> None:
        graph = build_resize_graph(target_width=1080, target_height=1920, fit="cover")
        assert "force_original_aspect_ratio=increase" in graph
        assert "crop=w=1080:h=1920" in graph
        assert "pad=" not in graph

    def test_contain_scales_down_then_pads(self) -> None:
        graph = build_resize_graph(target_width=1080, target_height=1920, fit="contain")
        assert "force_original_aspect_ratio=decrease" in graph
        assert "pad=w=1080:h=1920" in graph
        assert "crop=" not in graph

    def test_contain_honours_the_background_colour(self) -> None:
        graph = build_resize_graph(
            target_width=100, target_height=100, fit="contain", background_color="white"
        )
        assert "color=white" in graph

    def test_stretch_scales_without_preserving_the_ratio(self) -> None:
        graph = build_resize_graph(target_width=1080, target_height=1920, fit="stretch")
        assert "scale=w=1080:h=1920" in graph
        assert "force_original_aspect_ratio" not in graph
        assert "crop=" not in graph and "pad=" not in graph

    def test_blur_composites_a_blurred_background_behind_the_fitted_picture(self) -> None:
        graph = build_resize_graph(
            target_width=1080, target_height=1920, fit="blur", blur_strength=30
        )
        assert "split=2" in graph
        assert "gblur=sigma=30" in graph
        assert "force_original_aspect_ratio=increase" in graph  # background fills
        assert "force_original_aspect_ratio=decrease" in graph  # foreground fits
        assert "overlay=x=(W-w)/2:y=(H-h)/2" in graph

    @pytest.mark.parametrize("fit", ["cover", "contain", "blur", "stretch"])
    def test_every_mode_produces_the_expected_output_label(self, fit: str) -> None:
        graph = build_resize_graph(target_width=640, target_height=640, fit=fit)
        assert graph.rstrip().endswith("[vout]")

    @pytest.mark.parametrize("fit", ["cover", "contain", "blur", "stretch"])
    def test_every_mode_ends_in_a_yuv420p_conversion(self, fit: str) -> None:
        assert "format=yuv420p" in build_resize_graph(target_width=640, target_height=640, fit=fit)

    def test_an_unknown_fit_mode_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError, match="fit mode"):
            build_resize_graph(target_width=100, target_height=100, fit="squish")
