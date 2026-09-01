"""Pure format helpers for the bot candidates panel (no Streamlit runtime)."""

from __future__ import annotations

import pytest

from streamlit_utils.play_candidates import (
    format_analysis_dataframe_caption,
    format_delta,
    format_model_p,
    format_temperature,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.423, "42.3%"),
        (0.0, "0.0%"),
        (1.0, "100.0%"),
    ],
)
def test_format_model_p(value: float, expected: str) -> None:
    assert format_model_p(value) == expected


def test_format_delta_two_decimals() -> None:
    assert format_delta(-0.421) == "-0.42"
    assert format_delta(0.0) == "0.00"


def test_format_temperature_two_decimals() -> None:
    assert format_temperature(0.0) == "0.00"
    assert format_temperature(0.75) == "0.75"
    assert format_temperature(1.5) == "1.50"


def test_format_analysis_dataframe_caption_line_break_before_gap() -> None:
    from chess_teacher.bots.move_analysis import BotMoveAnalysis

    analysis = BotMoveAnalysis(
        fen_before="start",
        played_uci="e2e4",
        played_san="e4",
        rows=(),
        temperature_used=0.5,
    )
    caption = format_analysis_dataframe_caption(analysis)
    assert caption.startswith("Temperature: 0.50")
    assert "\n\nGap = pawns below the engine's best line." in caption
