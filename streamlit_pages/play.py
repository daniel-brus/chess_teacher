from __future__ import annotations

import streamlit as st

from chess_teacher.utils.chess_bots import BOT_PRESETS, ChessBot
from streamlit_components.chess_board import chess_board
from streamlit_utils.layout import ingest_css
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view, log_user_action
from streamlit_utils.play_game import (
    PlayGameState,
    apply_bot_move,
    apply_legal_move,
    close_bot,
    create_bot,
    game_status_message,
    is_user_turn,
    move_from_board_event,
    orientation_for_user,
    start_new_game,
    user_color_label,
)

configure_page("Play")
user = require_authenticated_user()
log_page_view("Play", user)

st.title("Play a game of chess")

_PLAY_STATE_KEY = "play_game_state"
_PLAY_BOT_KEY = "play_game_bot"
_PLAY_WIN_CELEBRATED_KEY = "play_win_celebrated_instance"

_PLAY_STATUS_CSS = """
div[class*="st-key-play_board_status"] {
    min-height: 1.75rem;
    margin: 0.25rem 0 0.5rem;
}
div[class*="st-key-play_board_status"] [data-testid="stMarkdownContainer"] p {
    margin: 0;
    line-height: 1.75rem;
}
"""


def _get_state() -> PlayGameState | None:
    state = st.session_state.get(_PLAY_STATE_KEY)
    return state if isinstance(state, PlayGameState) else None


def _set_state(state: PlayGameState | None) -> None:
    st.session_state[_PLAY_STATE_KEY] = state


def _get_bot() -> ChessBot | None:
    bot = st.session_state.get(_PLAY_BOT_KEY)
    return bot if isinstance(bot, ChessBot) else None


def _set_bot(bot: ChessBot | None) -> None:
    st.session_state[_PLAY_BOT_KEY] = bot


def _reset_game() -> None:
    close_bot(_get_bot())
    _set_bot(None)
    _set_state(None)
    st.session_state.pop(_PLAY_WIN_CELEBRATED_KEY, None)


def _user_won(state: PlayGameState) -> bool:
    if state.resigned or not state.board.is_game_over():
        return False
    outcome = state.board.outcome()
    return outcome is not None and outcome.winner == state.user_color


def _maybe_celebrate_win(state: PlayGameState) -> None:
    if not _user_won(state):
        return
    if st.session_state.get(_PLAY_WIN_CELEBRATED_KEY) == state.instance_id:
        return
    st.balloons()
    st.session_state[_PLAY_WIN_CELEBRATED_KEY] = state.instance_id


def _ensure_bot(state: PlayGameState) -> ChessBot:
    bot = _get_bot()
    if bot is None or st.session_state.get("play_game_bot_preset") != state.preset_key:
        close_bot(bot)
        bot = create_bot(state.preset_key)
        _set_bot(bot)
        st.session_state["play_game_bot_preset"] = state.preset_key
    return bot


def _render_setup() -> None:
    st.markdown("Choose your color and opponent, then start a new game.")
    preset_labels = {preset.key: f"{preset.label} — {preset.description}" for preset in BOT_PRESETS}
    preset_keys = [preset.key for preset in BOT_PRESETS]

    with st.form("play_game_setup"):
        color_choice = st.selectbox("Your color", ["White", "Black", "Random"])
        preset_key = st.selectbox(
            "Opponent",
            preset_keys,
            format_func=lambda key: preset_labels[key],
            index=2,
        )
        submitted = st.form_submit_button("Start game", type="primary", width="stretch")

    if submitted:
        state = start_new_game(color_choice, preset_key)
        _set_state(state)
        _ensure_bot(state)
        log_user_action(
            f"Started play game color={color_choice} preset={preset_key}",
            user,
        )
        st.rerun()


def _render_active_game(state: PlayGameState) -> None:
    preset = next(p for p in BOT_PRESETS if p.key == state.preset_key)
    st.caption(
        f"You are **{user_color_label(state.user_color)}** vs **{preset.label}** ({preset.description})"
    )

    status = game_status_message(state)
    bot = _ensure_bot(state)
    thinking = state.pending_bot_move and not state.board.is_game_over() and not state.resigned

    ingest_css(_PLAY_STATUS_CSS)
    with st.container(key="play_board_status"):
        if thinking:
            st.markdown(f"*{preset.label} is thinking…*")
        elif status:
            st.markdown(f"**{status}**")
        else:
            st.markdown("&#8203;", unsafe_allow_html=True)

    _maybe_celebrate_win(state)

    if thinking:
        state = apply_bot_move(state, bot)
        _set_state(state)
        st.rerun()

    board_event = chess_board(
        state.board,
        key="play_chess_board",
        orientation=orientation_for_user(state.user_color),
        disabled=not is_user_turn(state),
        last_move_uci=state.last_move_uci,
        instance_id=state.instance_id,
        height=520,
    )

    move = move_from_board_event(state.board, board_event) if board_event else None
    if move is not None and is_user_turn(state):
        try:
            state = apply_legal_move(state, move)
            _set_state(state)
            log_user_action(f"Play page user move={move.uci()}", user)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    cols = st.columns(2)
    with cols[0]:
        if st.button("New game", width="stretch"):
            log_user_action("Play page new game requested", user)
            _reset_game()
            st.rerun()
    with cols[1]:
        if st.button(
            "Resign", width="stretch", disabled=state.resigned or state.board.is_game_over()
        ):
            log_user_action("Play page resignation", user)
            resigned_state = PlayGameState(
                board=state.board.copy(),
                user_color=state.user_color,
                preset_key=state.preset_key,
                last_move_uci=state.last_move_uci,
                instance_id=state.instance_id,
                pending_bot_move=False,
                resigned=True,
            )
            _set_state(resigned_state)
            st.rerun()


def _render_page() -> None:
    state = _get_state()
    if state is None:
        _render_setup()
        return
    _render_active_game(state)


_render_page()
