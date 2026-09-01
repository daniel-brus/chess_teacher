"""Streamlit loading feedback for play-page I/O (database, storage, TensorFlow)."""

from __future__ import annotations

import streamlit as st

from chess_teacher.bots import (
    ChessBot,
    OpponentCategory,
    category_for_preset_key,
    list_baseline_presets,
)
from chess_teacher.utils.db.client import DatabaseClient
from streamlit_utils.play_game import create_bot

MSG_BASELINE_PRESETS = "Loading baseline models from database…"
MSG_STOCKFISH_ENGINE = "Starting Stockfish engine…"
MSG_BASELINE_PREPARE = "Preparing baseline opponent…"
MSG_OPPONENT_PREPARE = "Preparing opponent…"
MSG_OPPONENT_READY = "Opponent ready"
MSG_BASELINE_THINKING = "Analyzing position (Stockfish + neural model)…"


def initial_bot_loading_message(preset_key: str) -> str:
    category = category_for_preset_key(preset_key)
    if category == OpponentCategory.BASELINE:
        return MSG_BASELINE_PREPARE
    if category == OpponentCategory.STOCKFISH:
        return MSG_STOCKFISH_ENGINE
    return MSG_OPPONENT_PREPARE


def bot_thinking_message(preset_key: str, *, label: str) -> str:
    if category_for_preset_key(preset_key) == OpponentCategory.BASELINE:
        return f"{label} — {MSG_BASELINE_THINKING}"
    return f"{label} is thinking…"


def list_baseline_presets_with_feedback(db_client: DatabaseClient):
    with st.spinner(MSG_BASELINE_PRESETS):
        return list_baseline_presets(db_client)


def create_bot_with_feedback(
    preset_key: str,
    *,
    baseline_temperature: float | None = None,
    db_client: DatabaseClient,
) -> ChessBot:
    category = category_for_preset_key(preset_key)
    if category != OpponentCategory.BASELINE:
        with st.spinner(initial_bot_loading_message(preset_key)):
            return create_bot(
                preset_key,
                baseline_temperature=baseline_temperature,
                db_client=db_client,
            )

    with st.status(initial_bot_loading_message(preset_key), expanded=True) as status:

        def on_progress(message: str) -> None:
            status.update(label=message)

        bot = create_bot(
            preset_key,
            baseline_temperature=baseline_temperature,
            db_client=db_client,
            on_progress=on_progress,
        )
        status.update(label=MSG_OPPONENT_READY, state="complete", expanded=False)
    return bot
