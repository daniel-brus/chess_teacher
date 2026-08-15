from __future__ import annotations

import threading
from datetime import timedelta
from typing import Any

import streamlit as st

from chess_teacher.utils.chess_bots import (
    OPPONENT_CATEGORY_LABELS,
    STOCKFISH_DEPTH_DEFAULT,
    STOCKFISH_DEPTH_MAX,
    STOCKFISH_DEPTH_MIN,
    ChessBot,
    OpponentCategory,
    get_bot_preset,
    list_baseline_presets,
    list_other_presets,
    list_play_presets,
    stockfish_preset_key,
)
from chess_teacher.utils.db.client import get_db_client
from streamlit_components.chess_board import chess_board
from streamlit_utils.layout import ingest_css
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view, log_user_action
from streamlit_utils.play_game import (
    PlayGameState,
    apply_uci_move,
    apply_user_board_event,
    bot_job_matches_state,
    choose_bot_move_uci,
    close_bot,
    create_bot,
    game_status_message,
    is_bot_thinking,
    is_game_finished,
    is_user_turn,
    orientation_for_user,
    resign_game,
    start_new_game,
    state_fen,
    user_color_label,
    user_won,
)

configure_page("Play")
user = require_authenticated_user()
log_page_view("Play", user)
db_client = get_db_client()

st.title("Play a game of chess")

_PLAY_STATE_KEY = "play_game_state"
_PLAY_BOT_KEY = "play_game_bot"
_PLAY_BOT_LOCK_KEY = "play_bot_lock"
_PLAY_BOT_JOB_KEY = "play_bot_job"
_PLAY_WIN_CELEBRATED_KEY = "play_win_celebrated_instance"
_PLAY_SETUP_COLOR_KEY = "play_setup_color"
_PLAY_SETUP_CATEGORY_KEY = "play_setup_category"
_PLAY_SETUP_STOCKFISH_DEPTH_KEY = "play_setup_stockfish_depth"
_PLAY_SETUP_BASELINE_KEY = "play_setup_baseline"
_PLAY_SETUP_OTHER_KEY = "play_setup_other"
_BOT_POLL_INTERVAL = timedelta(milliseconds=400)

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


def _bot_lock() -> threading.Lock:
    if _PLAY_BOT_LOCK_KEY not in st.session_state:
        st.session_state[_PLAY_BOT_LOCK_KEY] = threading.Lock()
    return st.session_state[_PLAY_BOT_LOCK_KEY]


def _clear_bot_job() -> None:
    st.session_state.pop(_PLAY_BOT_JOB_KEY, None)


def _reset_game() -> None:
    close_bot(_get_bot())
    _set_bot(None)
    _set_state(None)
    _clear_bot_job()
    st.session_state.pop(_PLAY_WIN_CELEBRATED_KEY, None)


def _maybe_celebrate_win(state: PlayGameState) -> None:
    if not user_won(state):
        return
    if st.session_state.get(_PLAY_WIN_CELEBRATED_KEY) == state.instance_id:
        return
    st.balloons()
    st.session_state[_PLAY_WIN_CELEBRATED_KEY] = state.instance_id


def _ensure_bot(state: PlayGameState) -> ChessBot:
    bot = _get_bot()
    if bot is None or st.session_state.get("play_game_bot_preset") != state.preset_key:
        close_bot(bot)
        bot = create_bot(state.preset_key, db_client=db_client)
        _set_bot(bot)
        st.session_state["play_game_bot_preset"] = state.preset_key
    return bot


def _bot_worker(job: dict[str, Any], bot: ChessBot, lock: threading.Lock) -> None:
    """Run engine search off the Streamlit script thread to avoid the page dim overlay."""
    try:
        with lock:
            job["uci"] = choose_bot_move_uci(bot, str(job["fen"]))
    except Exception as exc:
        job["error"] = str(exc)
    finally:
        job["done"] = True


def _ensure_bot_job(state: PlayGameState, bot: ChessBot) -> dict[str, Any]:
    fen = state_fen(state)
    job = st.session_state.get(_PLAY_BOT_JOB_KEY)
    if isinstance(job, dict) and bot_job_matches_state(job, state):
        return job

    job = {
        "instance_id": state.instance_id,
        "fen": fen,
        "done": False,
        "uci": None,
        "error": None,
    }
    st.session_state[_PLAY_BOT_JOB_KEY] = job
    thread = threading.Thread(
        target=_bot_worker,
        args=(job, bot, _bot_lock()),
        daemon=True,
        name="play-bot-move",
    )
    thread.start()
    return job


@st.fragment(run_every=_BOT_POLL_INTERVAL)
def _poll_bot_job() -> None:
    """Poll background search; full-app rerun only when a move is ready."""
    state = _get_state()
    if state is None:
        return

    job = st.session_state.get(_PLAY_BOT_JOB_KEY)
    if not isinstance(job, dict) or not job.get("done"):
        return
    if not bot_job_matches_state(job, state):
        return

    error = job.get("error")
    if error:
        _clear_bot_job()
        st.error(f"Bot failed to move: {error}")
        return

    uci = job.get("uci")
    if not isinstance(uci, str) or not uci:
        _clear_bot_job()
        return

    try:
        next_state = apply_uci_move(state, uci)
    except ValueError as exc:
        _clear_bot_job()
        st.error(str(exc))
        return

    _clear_bot_job()
    _set_state(next_state)
    st.rerun()


def _resolve_preset(preset_key: str):
    """Look up a play preset; refresh from DB so baseline keys stay valid."""
    try:
        return get_bot_preset(preset_key, db_client=db_client)
    except KeyError:
        return next(
            (p for p in list_play_presets(db_client) if p.key == preset_key),
            None,
        )


def _init_setup_defaults() -> None:
    if st.session_state.get(_PLAY_SETUP_COLOR_KEY) not in {"White", "Black", "Random"}:
        st.session_state[_PLAY_SETUP_COLOR_KEY] = "White"

    raw_category = st.session_state.get(_PLAY_SETUP_CATEGORY_KEY)
    valid_categories = {c.value for c in OpponentCategory}
    if raw_category not in valid_categories:
        st.session_state[_PLAY_SETUP_CATEGORY_KEY] = OpponentCategory.STOCKFISH.value

    depth = st.session_state.get(_PLAY_SETUP_STOCKFISH_DEPTH_KEY)
    if not isinstance(depth, int) or not (STOCKFISH_DEPTH_MIN <= depth <= STOCKFISH_DEPTH_MAX):
        st.session_state[_PLAY_SETUP_STOCKFISH_DEPTH_KEY] = STOCKFISH_DEPTH_DEFAULT

    other_presets = list_other_presets()
    other_keys = [p.key for p in other_presets]
    if st.session_state.get(_PLAY_SETUP_OTHER_KEY) not in other_keys:
        st.session_state[_PLAY_SETUP_OTHER_KEY] = other_keys[0] if other_keys else "random"


def _resolve_setup_preset_key(category: OpponentCategory) -> tuple[str | None, str | None]:
    """Return ``(preset_key, error_message)`` from current setup widgets."""
    if category == OpponentCategory.STOCKFISH:
        depth = int(st.session_state[_PLAY_SETUP_STOCKFISH_DEPTH_KEY])
        return stockfish_preset_key(depth), None

    if category == OpponentCategory.BASELINE:
        baseline_presets = list_baseline_presets(db_client)
        if not baseline_presets:
            return None, "No promoted baseline models available yet."
        key = st.session_state.get(_PLAY_SETUP_BASELINE_KEY)
        keys = [p.key for p in baseline_presets]
        if key not in keys:
            key = keys[0]
        return key, None

    if category == OpponentCategory.PERSONAL:
        return None, "Personal bots are not available yet."

    if category == OpponentCategory.OTHER:
        key = st.session_state.get(_PLAY_SETUP_OTHER_KEY, "random")
        return key, None

    return None, f"Unknown opponent category: {category!r}"


def _render_category_options(category: OpponentCategory) -> None:
    if category == OpponentCategory.STOCKFISH:
        st.slider(
            "Stockfish depth",
            min_value=STOCKFISH_DEPTH_MIN,
            max_value=STOCKFISH_DEPTH_MAX,
            key=_PLAY_SETUP_STOCKFISH_DEPTH_KEY,
            help="Higher depth thinks longer and plays stronger.",
        )
        return

    if category == OpponentCategory.BASELINE:
        baseline_presets = list_baseline_presets(db_client)
        if not baseline_presets:
            st.info("No promoted baseline models yet. Train and promote a policy first.")
            return
        labels = {p.key: f"{p.label} — {p.description}" for p in baseline_presets}
        keys = [p.key for p in baseline_presets]
        if st.session_state.get(_PLAY_SETUP_BASELINE_KEY) not in keys:
            st.session_state[_PLAY_SETUP_BASELINE_KEY] = keys[0]
        st.selectbox(
            "Baseline version",
            keys,
            format_func=lambda key: labels.get(key, key),
            key=_PLAY_SETUP_BASELINE_KEY,
        )
        return

    if category == OpponentCategory.PERSONAL:
        st.info("Personal (user-finetuned) bots are coming later.")
        return

    other_presets = list_other_presets()
    labels = {p.key: f"{p.label} — {p.description}" for p in other_presets}
    keys = [p.key for p in other_presets]
    st.selectbox(
        "Opponent",
        keys,
        format_func=lambda key: labels.get(key, key),
        key=_PLAY_SETUP_OTHER_KEY,
    )


def _render_setup() -> None:
    st.markdown("Choose your color and opponent, then start a new game.")
    _init_setup_defaults()

    category_values = [c.value for c in OpponentCategory]
    st.radio(
        "Opponent type",
        category_values,
        format_func=lambda value: OPPONENT_CATEGORY_LABELS[OpponentCategory(value)],
        horizontal=True,
        key=_PLAY_SETUP_CATEGORY_KEY,
    )
    category = OpponentCategory(st.session_state[_PLAY_SETUP_CATEGORY_KEY])
    _render_category_options(category)

    color_choice = st.selectbox(
        "Your color",
        ["White", "Black", "Random"],
        key=_PLAY_SETUP_COLOR_KEY,
    )

    preset_key, setup_error = _resolve_setup_preset_key(category)
    start_disabled = preset_key is None
    if setup_error and category != OpponentCategory.PERSONAL:
        st.warning(setup_error)

    if st.button(
        "Start game",
        type="primary",
        width="stretch",
        disabled=start_disabled,
    ):
        if preset_key is None:
            st.error(setup_error or "Choose a valid opponent.")
            return
        try:
            get_bot_preset(preset_key, db_client=db_client)
        except KeyError:
            st.error(f"Unknown opponent preset: {preset_key!r}")
            return
        state = start_new_game(color_choice, preset_key)
        _set_state(state)
        _clear_bot_job()
        _ensure_bot(state)
        log_user_action(
            f"Started play game color={color_choice} preset={preset_key}",
            user,
        )
        st.rerun()


def _render_active_game(state: PlayGameState) -> None:
    preset = _resolve_preset(state.preset_key)
    label = preset.label if preset else state.preset_key
    description = preset.description if preset else ""
    st.caption(
        f"You are **{user_color_label(state.user_color)}** vs **{label}**"
        + (f" ({description})" if description else "")
    )

    status = game_status_message(state)
    bot = _ensure_bot(state)
    thinking = is_bot_thinking(state)

    ingest_css(_PLAY_STATUS_CSS)
    with st.container(key="play_board_status"):
        if thinking:
            st.markdown(f"*{label} is thinking…*")
        elif status:
            st.markdown(f"**{status}**")
        else:
            st.markdown("&#8203;", unsafe_allow_html=True)

    _maybe_celebrate_win(state)

    if thinking:
        _ensure_bot_job(state, bot)

    board_event = chess_board(
        state_fen(state),
        key="play_chess_board",
        orientation=orientation_for_user(state.user_color),
        disabled=not is_user_turn(state),
        last_move_uci=state.last_move_uci,
        instance_id=state.instance_id,
        height=520,
    )

    try:
        applied = apply_user_board_event(state, board_event)
    except ValueError as exc:
        st.error(str(exc))
        applied = None
    if applied is not None:
        state, move_uci = applied
        _set_state(state)
        _clear_bot_job()
        log_user_action(f"Play page user move={move_uci}", user)
        st.rerun()

    cols = st.columns(2)
    with cols[0]:
        if st.button("New game", width="stretch"):
            log_user_action("Play page new game requested", user)
            _reset_game()
            st.rerun()
    with cols[1]:
        if st.button("Resign", width="stretch", disabled=is_game_finished(state)):
            log_user_action("Play page resignation", user)
            _clear_bot_job()
            _set_state(resign_game(state))
            st.rerun()

    # Poll after board/buttons. run_every (not scope=fragment) — first fragment
    # invocation during a full script run cannot use scope="fragment".
    if thinking:
        _poll_bot_job()


def _render_page() -> None:
    state = _get_state()
    if state is None:
        _render_setup()
        return
    _render_active_game(state)


_render_page()
