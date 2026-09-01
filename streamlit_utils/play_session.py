"""Serialize and restore in-progress play games per authenticated user."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess
import streamlit as st

from chess_teacher.bots.move_analysis import BotMoveAnalysis, BotMoveCandidateRow
from streamlit_utils.play_game import PlayGameState, state_fen

_SNAPSHOT_VERSION = 1
_SESSION_KEY_PREFIX = "play_saved_session"


@dataclass(frozen=True, slots=True)
class PlaySessionSnapshot:
    state: PlayGameState
    bot_analysis: BotMoveAnalysis | None = None
    win_celebrated_instance: int | None = None
    candidates_visible: bool = True


def play_session_storage_key(user_id: str) -> str:
    return f"{_SESSION_KEY_PREFIX}:{user_id}"


def snapshot_play_state(state: PlayGameState) -> dict[str, Any]:
    return {
        "version": _SNAPSHOT_VERSION,
        "fen": state_fen(state),
        "user_color": "white" if state.user_color == chess.WHITE else "black",
        "preset_key": state.preset_key,
        "last_move_uci": state.last_move_uci,
        "instance_id": int(state.instance_id),
        "pending_bot_move": bool(state.pending_bot_move),
        "resigned": bool(state.resigned),
        "baseline_temperature": float(state.baseline_temperature),
    }


def restore_play_state(data: dict[str, Any]) -> PlayGameState | None:
    if data.get("version") != _SNAPSHOT_VERSION:
        return None
    fen = data.get("fen")
    preset_key = data.get("preset_key")
    if not isinstance(fen, str) or not isinstance(preset_key, str):
        return None
    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    raw_color = data.get("user_color")
    if raw_color == "white":
        user_color = chess.WHITE
    elif raw_color == "black":
        user_color = chess.BLACK
    else:
        return None

    last_move_uci = data.get("last_move_uci")
    if last_move_uci is not None and not isinstance(last_move_uci, str):
        return None

    try:
        instance_id = int(data.get("instance_id", 0))
        pending_bot_move = bool(data.get("pending_bot_move", False))
        resigned = bool(data.get("resigned", False))
        baseline_temperature = float(data.get("baseline_temperature", 0.0))
    except (TypeError, ValueError):
        return None

    return PlayGameState(
        board=board,
        user_color=user_color,
        preset_key=preset_key,
        last_move_uci=last_move_uci,
        instance_id=instance_id,
        pending_bot_move=pending_bot_move,
        resigned=resigned,
        baseline_temperature=baseline_temperature,
    )


def snapshot_bot_analysis(analysis: BotMoveAnalysis | None) -> dict[str, Any] | None:
    if analysis is None:
        return None
    return {
        "fen_before": analysis.fen_before,
        "played_uci": analysis.played_uci,
        "played_san": analysis.played_san,
        "temperature_used": float(analysis.temperature_used),
        "rows": [
            {
                "uci": row.uci,
                "san": row.san,
                "model_p": float(row.model_p),
                "sf_eval_stm": float(row.sf_eval_stm),
                "delta_vs_best": float(row.delta_vs_best),
                "is_played": bool(row.is_played),
            }
            for row in analysis.rows
        ],
    }


def restore_bot_analysis(data: dict[str, Any] | None) -> BotMoveAnalysis | None:
    if data is None:
        return None
    fen_before = data.get("fen_before")
    played_uci = data.get("played_uci")
    played_san = data.get("played_san")
    if not all(isinstance(value, str) for value in (fen_before, played_uci, played_san)):
        return None
    try:
        temperature_used = float(data.get("temperature_used", 0.0))
    except (TypeError, ValueError):
        return None

    raw_rows = data.get("rows")
    if not isinstance(raw_rows, list):
        return None

    rows: list[BotMoveCandidateRow] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            return None
        try:
            rows.append(
                BotMoveCandidateRow(
                    uci=str(raw["uci"]),
                    san=str(raw["san"]),
                    model_p=float(raw["model_p"]),
                    sf_eval_stm=float(raw["sf_eval_stm"]),
                    delta_vs_best=float(raw["delta_vs_best"]),
                    is_played=bool(raw["is_played"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            return None

    return BotMoveAnalysis(
        fen_before=fen_before,
        played_uci=played_uci,
        played_san=played_san,
        rows=tuple(rows),
        temperature_used=temperature_used,
    )


def snapshot_play_session(
    *,
    state: PlayGameState,
    bot_analysis: BotMoveAnalysis | None = None,
    win_celebrated_instance: int | None = None,
    candidates_visible: bool = True,
) -> dict[str, Any]:
    return {
        "version": _SNAPSHOT_VERSION,
        "state": snapshot_play_state(state),
        "bot_analysis": snapshot_bot_analysis(bot_analysis),
        "win_celebrated_instance": win_celebrated_instance,
        "candidates_visible": bool(candidates_visible),
    }


def restore_play_session(data: dict[str, Any] | None) -> PlaySessionSnapshot | None:
    if not isinstance(data, dict) or data.get("version") != _SNAPSHOT_VERSION:
        return None
    raw_state = data.get("state")
    if not isinstance(raw_state, dict):
        return None
    state = restore_play_state(raw_state)
    if state is None:
        return None

    bot_analysis = restore_bot_analysis(data.get("bot_analysis"))
    if data.get("bot_analysis") is not None and bot_analysis is None:
        return None

    raw_win = data.get("win_celebrated_instance")
    win_celebrated_instance: int | None
    if raw_win is None:
        win_celebrated_instance = None
    else:
        try:
            win_celebrated_instance = int(raw_win)
        except (TypeError, ValueError):
            return None

    return PlaySessionSnapshot(
        state=state,
        bot_analysis=bot_analysis,
        win_celebrated_instance=win_celebrated_instance,
        candidates_visible=bool(data.get("candidates_visible", True)),
    )


def save_play_session_for_user(
    user_id: str,
    *,
    state: PlayGameState | None,
    bot_analysis: BotMoveAnalysis | None = None,
    win_celebrated_instance: int | None = None,
    candidates_visible: bool = True,
) -> None:
    key = play_session_storage_key(user_id)
    if state is None:
        st.session_state.pop(key, None)
        return
    st.session_state[key] = snapshot_play_session(
        state=state,
        bot_analysis=bot_analysis,
        win_celebrated_instance=win_celebrated_instance,
        candidates_visible=candidates_visible,
    )


def load_play_session_for_user(user_id: str) -> PlaySessionSnapshot | None:
    raw = st.session_state.get(play_session_storage_key(user_id))
    if not isinstance(raw, dict):
        return None
    return restore_play_session(raw)
