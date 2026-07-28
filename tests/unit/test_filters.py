"""Unit tests for filter-graph construction and escaping."""

from __future__ import annotations

import pytest

from ffmpeg_mcp.errors import InvalidParameterError
from ffmpeg_mcp.ffmpeg.filters import (
    Filter,
    FilterChain,
    FilterGraph,
    atempo_chain,
    between_expr,
    curves_filter,
    drawtext_filter,
    eq_filter,
    escape_filter_value,
    fade_alpha_expr,
    format_curve_points,
    rotate_filter,
    scale_filter,
    setpts_filter,
    subtitles_filter,
    temperature_filter,
    xfade_filter,
)


class TestEscaping:
    def test_plain_text_is_unchanged(self) -> None:
        assert escape_filter_value("hello world") == "hello world"

    def test_colon_is_escaped_twice(self) -> None:
        # Level 1 turns ':' into '\:', level 2 doubles the backslash.
        assert escape_filter_value("12:30") == r"12\\:30"

    @pytest.mark.parametrize("char", ["[", "]", ",", ";"])
    def test_structural_characters_are_escaped(self, char: str) -> None:
        assert escape_filter_value(f"a{char}b") == f"a\\{char}b"

    def test_quote_is_escaped(self) -> None:
        assert escape_filter_value("it's") == r"it\\\'s"

    def test_backslash_is_escaped(self) -> None:
        assert escape_filter_value("a\\b") == "a\\\\\\\\b"

    def test_windows_path_colon_and_separators(self) -> None:
        escaped = escape_filter_value(r"C:\Users\a\subs.srt")
        assert ":" not in escaped.replace("\\:", "")
        assert escaped.startswith("C\\\\:")

    def test_caption_with_every_special_character_survives(self) -> None:
        raw = "Time: 12:30, [note]; it's 50% \\ done"
        escaped = escape_filter_value(raw)
        # No structural character is left bare: each is preceded by a backslash.
        for index, char in enumerate(escaped):
            if char in "[],;":
                assert escaped[index - 1] == "\\"


class TestFilterRendering:
    def test_filter_with_options(self) -> None:
        assert Filter("scale", {"w": 1280, "h": 720}).render() == "scale=w=1280:h=720"

    def test_filter_without_options(self) -> None:
        assert Filter("hflip").render() == "hflip"

    def test_float_options_are_trimmed(self) -> None:
        assert Filter("eq", {"contrast": 1.50}).render() == "eq=contrast=1.5"

    def test_boolean_options_render_as_digits(self) -> None:
        assert Filter("box", {"enabled": True}).render() == "box=enabled=1"

    def test_string_option_values_are_escaped(self) -> None:
        rendered = Filter("drawtext", {"text": "a:b"}).render()
        assert rendered == r"drawtext=text=a\\:b"

    def test_chain_joins_with_commas_and_labels(self) -> None:
        chain = FilterChain(inputs=["0:v"], outputs=["v0"])
        chain.add(Filter("hflip"), Filter("vflip"))
        assert chain.render() == "[0:v]hflip,vflip[v0]"

    def test_empty_chain_renders_null(self) -> None:
        assert FilterChain(inputs=["0:v"], outputs=["v0"]).render() == "[0:v]null[v0]"

    def test_graph_joins_chains_with_semicolons(self) -> None:
        graph = FilterGraph()
        graph.chain(["0:v"], ["a"]).add(Filter("hflip"))
        graph.chain(["a"], ["b"]).add(Filter("vflip"))
        assert graph.render() == "[0:v]hflip[a];[a]vflip[b]"


class TestGeometry:
    def test_scale_derives_missing_dimension_as_even(self) -> None:
        assert scale_filter(1280, None).render() == "scale=w=1280:h=-2"

    def test_scale_requires_a_dimension(self) -> None:
        with pytest.raises(InvalidParameterError):
            scale_filter(None, None)

    def test_scale_both_dimensions_fits_inside_box(self) -> None:
        assert "force_original_aspect_ratio=decrease" in scale_filter(640, 480).render()

    def test_scale_both_dimensions_stretches_when_aspect_not_kept(self) -> None:
        assert scale_filter(640, 480, keep_aspect=False).render() == "scale=w=640:h=480"

    @pytest.mark.parametrize(("degrees", "count"), [(0, 0), (90, 1), (180, 2), (270, 1), (360, 0)])
    def test_rotation_uses_transpose_steps(self, degrees: int, count: int) -> None:
        assert len(rotate_filter(degrees)) == count

    def test_rotation_rejects_odd_angles(self) -> None:
        with pytest.raises(InvalidParameterError):
            rotate_filter(45)


class TestSpeed:
    def test_unity_speed_needs_no_atempo(self) -> None:
        assert atempo_chain(1.0) == []

    @pytest.mark.parametrize("factor", [0.5, 0.75, 1.5, 2.0, 4.0, 8.0, 0.25, 0.1, 16.0])
    def test_atempo_steps_stay_in_range_and_multiply_back(self, factor: float) -> None:
        steps = atempo_chain(factor)
        product = 1.0
        for step in steps:
            value = float(step.options["tempo"])
            assert 0.5 <= value <= 2.0
            product *= value
        assert product == pytest.approx(factor, rel=1e-6)

    def test_atempo_rejects_zero(self) -> None:
        with pytest.raises(InvalidParameterError):
            atempo_chain(0)

    def test_setpts_inverts_the_speed_factor(self) -> None:
        assert setpts_filter(2.0).render() == "setpts=0.5*PTS"


class TestColour:
    def test_neutral_eq_is_omitted(self) -> None:
        assert eq_filter() is None

    def test_eq_renders_all_parameters(self) -> None:
        rendered = eq_filter(brightness=0.1, contrast=1.2).render()  # type: ignore[union-attr]
        assert "brightness=0.1" in rendered and "contrast=1.2" in rendered

    def test_neutral_temperature_is_omitted(self) -> None:
        assert temperature_filter(0) is None

    def test_warm_temperature_lifts_red_and_drops_blue(self) -> None:
        rendered = temperature_filter(100).render()  # type: ignore[union-attr]
        assert "rr=1.3" in rendered and "bb=0.7" in rendered

    def test_temperature_out_of_range_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            temperature_filter(500)

    def test_curve_points_are_sorted_and_formatted(self) -> None:
        assert format_curve_points([(1, 1), (0, 0), (0.5, 0.6)]) == "0/0 0.5/0.6 1/1"

    def test_curve_points_must_be_in_unit_square(self) -> None:
        with pytest.raises(InvalidParameterError):
            format_curve_points([(0, 0), (1.5, 1)])

    def test_curve_needs_two_points(self) -> None:
        with pytest.raises(InvalidParameterError):
            format_curve_points([(0, 0)])

    def test_duplicate_x_values_are_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            format_curve_points([(0.5, 0.1), (0.5, 0.9)])

    def test_curves_accepts_a_preset(self) -> None:
        assert curves_filter(preset="vintage").render() == "curves=preset=vintage"

    def test_curves_needs_a_preset_or_points(self) -> None:
        with pytest.raises(InvalidParameterError):
            curves_filter()


class TestText:
    def test_drawtext_disables_expansion(self) -> None:
        rendered = drawtext_filter(text="hello")
        assert "expansion=none" in rendered.render()

    def test_drawtext_escapes_the_text(self) -> None:
        rendered = drawtext_filter(text="Chapter 1: Start").render()
        assert r"Chapter 1\\: Start" in rendered

    def test_drawtext_requires_exactly_one_text_source(self) -> None:
        with pytest.raises(InvalidParameterError):
            drawtext_filter()
        with pytest.raises(InvalidParameterError):
            drawtext_filter(text="a", textfile="/tmp/a.txt")

    def test_subtitles_escapes_the_path(self) -> None:
        rendered = subtitles_filter("/tmp/my subs, v2.srt").render()
        assert r"my subs\, v2.srt" in rendered

    def test_force_style_commas_are_escaped(self) -> None:
        rendered = subtitles_filter(
            "/tmp/a.srt", force_style={"FontName": "Arial", "FontSize": 24}
        ).render()
        assert r"force_style=FontName=Arial\,FontSize=24" in rendered

    def test_between_expression(self) -> None:
        assert between_expr(1, 2.5) == "between(t,1,2.5)"

    def test_between_rejects_inverted_range(self) -> None:
        with pytest.raises(InvalidParameterError):
            between_expr(3, 1)

    def test_fade_alpha_is_balanced(self) -> None:
        expr = fade_alpha_expr(0, 4, 0.5, 0.5)
        assert expr.count("(") == expr.count(")")
        assert expr.startswith("if(lt(t,0.5)")

    def test_fade_alpha_without_fades_is_opaque(self) -> None:
        assert fade_alpha_expr(0, 4, 0, 0) == "1"

    def test_fade_durations_are_clamped_to_half_the_window(self) -> None:
        expr = fade_alpha_expr(0, 2, 10, 10)
        assert "/1" in expr  # clamped to 1.0 second each


class TestTransitions:
    def test_xfade_renders_transition_and_timing(self) -> None:
        assert xfade_filter("fade", 1.0, 4.0).render() == (
            "xfade=transition=fade:duration=1:offset=4"
        )

    def test_xfade_rejects_zero_duration(self) -> None:
        with pytest.raises(InvalidParameterError):
            xfade_filter("fade", 0, 1)
