"""Unit tests for SRT generation, LUT parsing, and caption styling."""

from __future__ import annotations

import pytest

from ffmpeg_mcp.errors import InvalidParameterError
from ffmpeg_mcp.models import Segment
from ffmpeg_mcp.subtitles import (
    build_srt,
    format_ass_timestamp,
    format_srt_timestamp,
    normalise_segments,
    parse_cube,
    parse_srt,
    wrap_caption_text,
)
from ffmpeg_mcp.tools.captions import (
    CaptionStyle,
    TextOverlayItem,
    build_force_style,
    position_expressions,
    to_ass_colour,
)


class TestTimestamps:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00,000"),
            (1.5, "00:00:01,500"),
            (61.25, "00:01:01,250"),
            (3661.001, "01:01:01,001"),
            (-5.0, "00:00:00,000"),
        ],
    )
    def test_srt_timestamps(self, seconds: float, expected: str) -> None:
        assert format_srt_timestamp(seconds) == expected

    def test_srt_rounds_rather_than_truncates(self) -> None:
        assert format_srt_timestamp(1.9999) == "00:00:02,000"

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0.0, "0:00:00.00"), (61.25, "0:01:01.25"), (3600.0, "1:00:00.00")],
    )
    def test_ass_timestamps(self, seconds: float, expected: str) -> None:
        assert format_ass_timestamp(seconds) == expected


class TestSegmentNormalisation:
    def test_segments_are_sorted(self) -> None:
        result = normalise_segments(
            [Segment(start=2, end=3, text="b"), Segment(start=0, end=1, text="a")]
        )
        assert [s.text for s in result] == ["a", "b"]

    def test_empty_text_is_dropped(self) -> None:
        result = normalise_segments(
            [Segment(start=0, end=1, text="   "), Segment(start=1, end=2, text="ok")]
        )
        assert [s.text for s in result] == ["ok"]

    def test_overlapping_cues_are_truncated(self) -> None:
        # Without this, two captions render on top of each other.
        result = normalise_segments(
            [Segment(start=0, end=5, text="first"), Segment(start=2, end=6, text="second")]
        )
        assert result[0].end == pytest.approx(2.0)
        assert result[1].start == pytest.approx(2.0)

    def test_zero_length_cues_get_a_minimum_duration(self) -> None:
        result = normalise_segments([Segment(start=1.0, end=1.0, text="blink")])
        assert result[0].end > result[0].start

    def test_text_is_stripped(self) -> None:
        result = normalise_segments([Segment(start=0, end=1, text="  hi  ")])
        assert result[0].text == "hi"


class TestWrapping:
    def test_short_text_is_untouched(self) -> None:
        assert wrap_caption_text("hello there", 40) == "hello there"

    def test_wrapping_happens_at_word_boundaries(self) -> None:
        wrapped = wrap_caption_text("one two three four five six", 10)
        assert all(len(line) <= 10 for line in wrapped.splitlines()[:-1])
        assert " ".join(wrapped.split()) == "one two three four five six"

    def test_a_word_longer_than_the_limit_is_kept_whole(self) -> None:
        assert wrap_caption_text("supercalifragilistic", 5) == "supercalifragilistic"

    def test_overflow_folds_into_the_last_allowed_line(self) -> None:
        # No words may be silently dropped.
        wrapped = wrap_caption_text("a b c d e f g h i j", 3, max_lines=2)
        assert len(wrapped.splitlines()) == 2
        assert " ".join(wrapped.split()) == "a b c d e f g h i j"

    def test_a_zero_width_limit_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            wrap_caption_text("hi", 0)


class TestBuildSrt:
    def test_a_basic_document(self) -> None:
        content = build_srt(
            [Segment(start=0, end=1.5, text="Hello"), Segment(start=1.5, end=3, text="World")],
            max_chars_per_line=None,
        )
        assert content == (
            "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n2\n00:00:01,500 --> 00:00:03,000\nWorld\n"
        )

    def test_cues_are_numbered_from_one_after_sorting(self) -> None:
        content = build_srt(
            [Segment(start=5, end=6, text="second"), Segment(start=0, end=1, text="first")],
            max_chars_per_line=None,
        )
        assert content.startswith("1\n00:00:00,000")
        assert "\n2\n00:00:05,000" in content

    def test_the_document_round_trips_through_the_parser(self) -> None:
        segments = [
            Segment(start=0, end=2, text="Line one"),
            Segment(start=2.5, end=4, text="Line two"),
        ]
        parsed = parse_srt(build_srt(segments, max_chars_per_line=None))
        assert [(round(s.start, 2), round(s.end, 2), s.text) for s in parsed] == [
            (0.0, 2.0, "Line one"),
            (2.5, 4.0, "Line two"),
        ]

    def test_structural_characters_survive_a_round_trip(self) -> None:
        # These are exactly the characters that would break a filter graph.
        text = "Time: 12:30, [note]; it's 50% \\ done"
        parsed = parse_srt(build_srt([Segment(start=0, end=1, text=text)], max_chars_per_line=None))
        assert parsed[0].text == text

    def test_multiline_cues_round_trip(self) -> None:
        parsed = parse_srt(
            build_srt([Segment(start=0, end=1, text="first\nsecond")], max_chars_per_line=None)
        )
        assert parsed[0].text == "first\nsecond"

    def test_wrapping_is_applied_when_requested(self) -> None:
        content = build_srt(
            [Segment(start=0, end=2, text="one two three four five six seven")],
            max_chars_per_line=12,
            max_lines=4,
        )
        body = content.splitlines()[2:]
        assert len(body) > 1, "long text was not wrapped"
        assert all(len(line) <= 12 for line in body)

    def test_wrapping_never_drops_words_even_when_it_must_overflow(self) -> None:
        # With too few lines available the last one overflows rather than
        # truncating; losing caption text silently would be worse.
        content = build_srt(
            [Segment(start=0, end=2, text="one two three four five six seven")],
            max_chars_per_line=12,
            max_lines=2,
        )
        body = "\n".join(content.splitlines()[2:])
        assert len(body.splitlines()) == 2
        assert body.split() == ["one", "two", "three", "four", "five", "six", "seven"]


class TestParseSrt:
    def test_dot_separated_milliseconds_are_accepted(self) -> None:
        parsed = parse_srt("1\n00:00:01.000 --> 00:00:02.000\nhi\n")
        assert parsed[0].start == pytest.approx(1.0)

    def test_blocks_without_timing_are_skipped(self) -> None:
        assert parse_srt("garbage\n\nmore garbage") == []

    def test_an_empty_document_yields_nothing(self) -> None:
        assert parse_srt("") == []


class TestAssColour:
    @pytest.mark.parametrize(
        ("hex_colour", "expected"),
        [
            ("#FFFFFF", "&H00FFFFFF"),
            ("#000000", "&H00000000"),
            # Red in RGB becomes 0000FF in ASS's BGR ordering.
            ("#FF0000", "&H000000FF"),
            ("#0000FF", "&H00FF0000"),
            ("#E8630A", "&H000A63E8"),
            ("E8630A", "&H000A63E8"),
        ],
    )
    def test_rgb_is_reordered_to_bgr(self, hex_colour: str, expected: str) -> None:
        assert to_ass_colour(hex_colour) == expected

    def test_alpha_is_inverted(self) -> None:
        # ASS alpha is transparency: fully opaque input becomes 00.
        assert to_ass_colour("#FFFFFFFF") == "&H00FFFFFF"
        assert to_ass_colour("#FFFFFF00") == "&HFFFFFFFF"

    @pytest.mark.parametrize("bad", ["red", "#FFF", "#GGGGGG", "", "#1234567"])
    def test_invalid_colours_are_rejected(self, bad: str) -> None:
        with pytest.raises(InvalidParameterError):
            to_ass_colour(bad)


class TestCaptionStyle:
    def test_force_style_includes_the_expected_keys(self) -> None:
        overrides = build_force_style(CaptionStyle())
        assert overrides["FontName"] == "Arial"
        assert overrides["PrimaryColour"] == "&H00FFFFFF"
        assert overrides["Alignment"] == 2

    @pytest.mark.parametrize(
        ("position", "alignment"),
        [("bottom-center", 2), ("top-center", 8), ("center", 5), ("bottom-right", 3)],
    )
    def test_positions_map_to_ass_alignment_codes(self, position: str, alignment: int) -> None:
        overrides = build_force_style(CaptionStyle(position=position))  # type: ignore[arg-type]
        assert overrides["Alignment"] == alignment

    def test_bold_and_italic_use_the_ass_convention(self) -> None:
        overrides = build_force_style(CaptionStyle(bold=True, italic=True))
        assert overrides["Bold"] == -1 and overrides["Italic"] == -1

    def test_background_box_switches_border_style(self) -> None:
        assert build_force_style(CaptionStyle(background_box=True))["BorderStyle"] == 3


class TestPositions:
    @pytest.mark.parametrize(
        "position",
        [
            "top-left",
            "top-center",
            "top-right",
            "middle-left",
            "center",
            "middle-right",
            "bottom-left",
            "bottom-center",
            "bottom-right",
            "lower-third",
        ],
    )
    def test_every_named_position_produces_expressions(self, position: str) -> None:
        x, y = position_expressions(position, 40)
        assert x and y
        assert "{" not in x and "{" not in y  # the margin placeholder was filled

    def test_the_margin_is_substituted(self) -> None:
        x, _ = position_expressions("top-left", 25)
        assert x == "25"

    def test_an_unknown_position_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            position_expressions("middle-of-nowhere", 40)


class TestOverlayValidation:
    def test_an_inverted_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="end must be greater"):
            TextOverlayItem(text="hi", start=5, end=2)

    def test_x_without_y_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="both x and y"):
            TextOverlayItem(text="hi", x="10")

    def test_explicit_coordinates_are_accepted_together(self) -> None:
        item = TextOverlayItem(text="hi", x="10", y="20")
        assert (item.x, item.y) == ("10", "20")


CUBE_2 = """
TITLE "Test LUT"
LUT_3D_SIZE 2
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
"""


class TestCubeParsing:
    def test_a_valid_cube_is_described(self) -> None:
        info = parse_cube(CUBE_2)
        assert info.size == 2
        assert info.dimensions == 3
        assert info.entries == 8
        assert info.title == "Test LUT"

    def test_comments_and_blank_lines_are_ignored(self) -> None:
        text = "# a comment\n\n" + CUBE_2
        assert parse_cube(text).entries == 8

    def test_a_missing_size_declaration_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError, match="LUT_3D_SIZE"):
            parse_cube("0.0 0.0 0.0\n1.0 1.0 1.0\n")

    def test_a_row_count_mismatch_is_rejected(self) -> None:
        # ffmpeg's own error for this case gives no hint what is wrong.
        truncated = CUBE_2.rsplit("\n", 2)[0]
        with pytest.raises(InvalidParameterError, match="row count"):
            parse_cube(truncated)

    def test_a_non_numeric_row_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError, match="not numeric"):
            parse_cube("LUT_3D_SIZE 2\n" + "a b c\n")

    def test_a_row_with_the_wrong_column_count_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError, match="three float"):
            parse_cube("LUT_3D_SIZE 2\n0.0 0.0\n")

    def test_an_implausible_size_is_rejected(self) -> None:
        with pytest.raises(InvalidParameterError, match="out of range"):
            parse_cube("LUT_3D_SIZE 9999\n")

    def test_a_one_dimensional_lut_is_recognised(self) -> None:
        info = parse_cube("LUT_1D_SIZE 2\n0.0 0.0 0.0\n1.0 1.0 1.0\n")
        assert info.dimensions == 1 and info.entries == 2


class TestOverlayFilterBuilding:
    """The drawtext builder, where an unquoted coordinate once broke the graph."""

    @staticmethod
    def _render(**kwargs: object) -> str:
        from pathlib import Path

        from ffmpeg_mcp.tools.captions import build_overlay_filter

        item = TextOverlayItem(text="hi", **kwargs)  # type: ignore[arg-type]
        return build_overlay_filter(item, Path("/tmp/t.txt"), 10.0, None).render()

    def test_coordinates_are_quoted(self) -> None:
        # Unquoted, an animated x expression's commas read as filter separators.
        rendered = self._render(x="12", y="34")
        assert "x='12':y='34'" in rendered

    def test_a_slide_animation_produces_a_comma_bearing_expression_inside_quotes(
        self,
    ) -> None:
        rendered = self._render(start=0.0, end=3.0, animation="slide-left")
        x_expr = rendered.split("x='", 1)[1].split("'", 1)[0]
        assert "," in x_expr and x_expr.startswith("if(lt(t,")

    def test_the_enable_window_matches_the_item(self) -> None:
        assert "enable='between(t,1,4)'" in self._render(start=1.0, end=4.0)

    def test_an_open_ended_item_runs_to_the_media_duration(self) -> None:
        assert "enable='between(t,0,10)'" in self._render(start=0.0, end=None)

    def test_fade_sets_a_balanced_alpha_expression(self) -> None:
        rendered = self._render(start=0.0, end=4.0, animation="fade")
        alpha = rendered.split("alpha='", 1)[1].split("'", 1)[0]
        assert alpha.count("(") == alpha.count(")")

    def test_expansion_is_disabled_so_percent_sequences_stay_literal(self) -> None:
        assert "expansion=none" in self._render()

    def test_the_text_never_appears_in_the_graph(self) -> None:
        from pathlib import Path

        from ffmpeg_mcp.tools.captions import build_overlay_filter

        item = TextOverlayItem(text="Time: 12:30, [x]; it's 50%")
        rendered = build_overlay_filter(item, Path("/tmp/t.txt"), 10.0, None).render()
        assert "12:30" not in rendered
        assert "textfile=" in rendered
