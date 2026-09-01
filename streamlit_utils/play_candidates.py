"""Render the play-page bot move candidates panel (baseline only)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chess_teacher.bots.move_analysis import BotMoveAnalysis

_PLAYED_ROW_BG = "#f5e6c8"  # soft amber/cream; theme-friendly, not purple glow
_PANEL_VISIBLE_KEY = "play_bot_candidates_visible"


def format_model_p(p: float) -> str:
    return f"{p * 100.0:.1f}%"


def format_delta(v: float) -> str:
    return f"{v:.2f}"


def format_temperature(t: float) -> str:
    return f"{t:.2f}"


def _analysis_dataframe(analysis: BotMoveAnalysis):
    rows = analysis.rows
    frame = pd.DataFrame({
        "Move": [f"▸ {r.san}" if r.is_played else r.san for r in rows],
        "Model P": [format_model_p(r.model_p) for r in rows],
        "Gap": [format_delta(r.delta_vs_best) for r in rows],
    })
    played_flags = [r.is_played for r in rows]

    def _highlight_played(row: pd.Series) -> list[str]:
        if played_flags[int(row.name)]:
            return [f"background-color: {_PLAYED_ROW_BG}"] * len(row)
        return [""] * len(row)

    return frame.style.apply(_highlight_played, axis=1)


def _meta_caption(analysis: BotMoveAnalysis) -> str:
    return f"Temperature: {format_temperature(analysis.temperature_used)}"


def format_analysis_dataframe_caption(analysis: BotMoveAnalysis) -> str:
    return f"{_meta_caption(analysis)}\n\nGap = pawns below the engine's best line."


def render_bot_candidates_panel(
    analysis: BotMoveAnalysis | None,
    *,
    show_dataframe: bool,
    show_no_analysis: bool,
    panel_visible: bool,
) -> None:
    """Right-column panel: header, optional dataframe, placeholder captions."""
    st.markdown("**Bot candidates**")
    visible = st.toggle(
        "Show candidates",
        value=panel_visible,
        key=_PANEL_VISIBLE_KEY,
        help="Hide to give the board more room.",
    )

    if not visible:
        return

    if show_dataframe and analysis is not None and analysis.rows:
        st.caption(format_analysis_dataframe_caption(analysis))
        height = min(46 * len(analysis.rows) + 38, 240)
        st.dataframe(
            _analysis_dataframe(analysis),
            hide_index=True,
            width="stretch",
            height=height,
        )
        return
    if show_no_analysis and analysis is not None:
        st.caption(_meta_caption(analysis))
        st.caption("No candidate analysis for this move.")
        return
    st.caption("Candidates appear after the bot moves.")
