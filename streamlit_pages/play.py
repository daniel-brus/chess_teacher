from __future__ import annotations

import streamlit as st

from chess_teacher.bots import (
    BASELINE_TEMPERATURE_DEFAULT,
    BASELINE_TEMPERATURE_MAX,
    BASELINE_TEMPERATURE_MIN,
    BASELINE_TEMPERATURE_STEP,
    OPPONENT_CATEGORY_LABELS,
    STOCKFISH_DEPTH_DEFAULT,
    STOCKFISH_DEPTH_MAX,
    STOCKFISH_DEPTH_MIN,
    ChessBot,
    OpponentCategory,
    category_for_preset_key,
    get_bot_preset,
    list_other_presets,
    list_play_presets,
    stockfish_preset_key,
)
from chess_teacher.bots.move_analysis import BotMoveAnalysis
from chess_teacher.utils.db.client import get_db_client
from streamlit_components.chess_board import chess_board
from streamlit_utils.layout import ingest_css
from streamlit_utils.login import require_authenticated_user
from streamlit_utils.page_config import configure_page
from streamlit_utils.page_logging import log_page_view, log_user_action
from streamlit_utils.play_candidates import render_bot_candidates_panel
from streamlit_utils.play_game import (
    PlayGameState,
    apply_bot_move,
    apply_user_board_event,
    close_bot,
    game_status_message,
    is_bot_thinking,
    is_game_finished,
    is_user_turn,
    orientation_for_user,
    resign_game,
    start_new_game,
    state_fen,
    take_bot_move_analysis,
    user_color_label,
    user_won,
)
from streamlit_utils.play_loading import (
    bot_thinking_message,
    create_bot_with_feedback,
    list_baseline_presets_with_feedback,
)
from streamlit_utils.play_session import load_play_session_for_user, save_play_session_for_user

configure_page("Play")
user = require_authenticated_user()
log_page_view("Play", user)
db_client = get_db_client()

st.title("Play a game of chess")

_PLAY_STATE_KEY = "play_game_state"
_PLAY_BOT_KEY = "play_game_bot"
_PLAY_WIN_CELEBRATED_KEY = "play_win_celebrated_instance"
_PLAY_SETUP_COLOR_KEY = "play_setup_color"
_PLAY_SETUP_CATEGORY_KEY = "play_setup_category"
_PLAY_SETUP_STOCKFISH_DEPTH_KEY = "play_setup_stockfish_depth"
_PLAY_SETUP_BASELINE_KEY = "play_setup_baseline"
_PLAY_SETUP_BASELINE_TEMPERATURE_KEY = "play_setup_baseline_temperature"
_PLAY_SETUP_OTHER_KEY = "play_setup_other"
_BOT_ANALYSIS_KEY = "play_bot_move_analysis"
_BOT_CANDIDATES_VISIBLE_KEY = "play_bot_candidates_visible"
_PLAY_SETUP_BASELINE_PRESETS_KEY = "play_setup_baseline_presets"

_PLAY_STATUS_CSS = """
div[class*="st-key-play_board_status"],
div[class*="st-key-play_bottom_status"] {
    min-height: 2.5rem;
}
div[class*="st-key-play_board_status"] {
    height: 2.5rem;
    max-height: 2.5rem;
    overflow: hidden;
    margin: 0.25rem 0 0.5rem;
}
div[class*="st-key-play_bottom_status"] {
    margin: 0.5rem 0 0;
}
div[class*="st-key-play_board_status"] [data-testid="stMarkdownContainer"] p,
div[class*="st-key-play_bottom_status"] [data-testid="stMarkdownContainer"] p {
    margin: 0;
    line-height: 1.25rem;
}
div[class*="st-key-play_board_col"] [data-testid="stCustomComponentV1"] {
    width: 100%;
    display: flex;
    justify-content: center;
}
"""


def _get_state() -> PlayGameState | None:
    state = st.session_state.get(_PLAY_STATE_KEY)
    if isinstance(state, PlayGameState):
        return state
    if state is not None:
        st.session_state.pop(_PLAY_STATE_KEY, None)
    return None


def _sync_play_session(state: PlayGameState | None = None) -> None:
    """Persist the active play session for this user (survives page navigation)."""
    active = state if state is not None else _get_state()
    save_play_session_for_user(
        user.user_id,
        state=active,
        bot_analysis=_get_bot_analysis(),
        win_celebrated_instance=st.session_state.get(_PLAY_WIN_CELEBRATED_KEY),
        candidates_visible=_candidates_panel_visible(),
    )


def _hydrate_session_from_cache() -> None:
    if _get_state() is not None:
        return
    snapshot = load_play_session_for_user(user.user_id)
    if snapshot is None:
        return
    st.session_state[_PLAY_STATE_KEY] = snapshot.state
    if snapshot.bot_analysis is not None:
        st.session_state[_BOT_ANALYSIS_KEY] = snapshot.bot_analysis
    else:
        st.session_state.pop(_BOT_ANALYSIS_KEY, None)
    if snapshot.win_celebrated_instance is not None:
        st.session_state[_PLAY_WIN_CELEBRATED_KEY] = snapshot.win_celebrated_instance
    else:
        st.session_state.pop(_PLAY_WIN_CELEBRATED_KEY, None)
    st.session_state[_BOT_CANDIDATES_VISIBLE_KEY] = snapshot.candidates_visible
    st.session_state.setdefault("play_game_bot_preset", snapshot.state.preset_key)
    bot = _get_bot()
    if bot is None or not _bot_matches_state(bot, snapshot.state):
        close_bot(bot)
        st.session_state.pop(_PLAY_BOT_KEY, None)
        st.session_state.pop("play_game_bot_preset", None)


def _set_state(state: PlayGameState | None) -> None:
    st.session_state[_PLAY_STATE_KEY] = state
    _sync_play_session(state)


def _get_bot() -> ChessBot | None:
    bot = st.session_state.get(_PLAY_BOT_KEY)
    return bot if isinstance(bot, ChessBot) else None


def _set_bot(bot: ChessBot | None) -> None:
    st.session_state[_PLAY_BOT_KEY] = bot


def _clear_bot_analysis() -> None:
    st.session_state.pop(_BOT_ANALYSIS_KEY, None)
    _sync_play_session()


def _candidates_panel_visible() -> bool:
    return bool(st.session_state.get(_BOT_CANDIDATES_VISIBLE_KEY, True))


def _get_bot_analysis() -> BotMoveAnalysis | None:
    analysis = st.session_state.get(_BOT_ANALYSIS_KEY)
    return analysis if isinstance(analysis, BotMoveAnalysis) else None


def _baseline_presets_for_setup():
    cached = st.session_state.get(_PLAY_SETUP_BASELINE_PRESETS_KEY)
    if cached is not None:
        return cached
    presets = list_baseline_presets_with_feedback(db_client)
    st.session_state[_PLAY_SETUP_BASELINE_PRESETS_KEY] = presets
    return presets


def _reset_game() -> None:
    close_bot(_get_bot())
    _set_bot(None)
    _set_state(None)
    st.session_state.pop(_PLAY_WIN_CELEBRATED_KEY, None)
    st.session_state.pop(_PLAY_SETUP_BASELINE_PRESETS_KEY, None)
    _clear_bot_analysis()


def _maybe_celebrate_win(state: PlayGameState) -> None:
    if not user_won(state):
        return
    if st.session_state.get(_PLAY_WIN_CELEBRATED_KEY) == state.instance_id:
        return
    st.balloons()
    st.session_state[_PLAY_WIN_CELEBRATED_KEY] = state.instance_id


def _bot_matches_state(bot: ChessBot, state: PlayGameState) -> bool:
    if st.session_state.get("play_game_bot_preset") != state.preset_key:
        return False
    if category_for_preset_key(state.preset_key) == OpponentCategory.BASELINE:
        temperature = getattr(bot, "temperature", None)
        if temperature is None:
            return False
        if float(temperature) != float(state.baseline_temperature):
            return False
    return True


def _ensure_bot(state: PlayGameState) -> ChessBot:
    bot = _get_bot()
    if bot is None or not _bot_matches_state(bot, state):
        close_bot(bot)
        bot = create_bot_with_feedback(
            state.preset_key,
            db_client=db_client,
            baseline_temperature=state.baseline_temperature,
        )
        _set_bot(bot)
        st.session_state["play_game_bot_preset"] = state.preset_key
    return bot


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

    temp = st.session_state.get(_PLAY_SETUP_BASELINE_TEMPERATURE_KEY)
    if not isinstance(temp, (int, float)) or not (
        BASELINE_TEMPERATURE_MIN <= float(temp) <= BASELINE_TEMPERATURE_MAX
    ):
        st.session_state[_PLAY_SETUP_BASELINE_TEMPERATURE_KEY] = BASELINE_TEMPERATURE_DEFAULT


def _resolve_setup_preset_key(category: OpponentCategory) -> tuple[str | None, str | None]:
    """Return ``(preset_key, error_message)`` from current setup widgets."""
    if category == OpponentCategory.STOCKFISH:
        depth = int(st.session_state[_PLAY_SETUP_STOCKFISH_DEPTH_KEY])
        return stockfish_preset_key(depth), None

    if category == OpponentCategory.BASELINE:
        baseline_presets = _baseline_presets_for_setup()
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
        baseline_presets = _baseline_presets_for_setup()
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
        st.slider(
            "Temperature",
            min_value=BASELINE_TEMPERATURE_MIN,
            max_value=BASELINE_TEMPERATURE_MAX,
            step=BASELINE_TEMPERATURE_STEP,
            key=_PLAY_SETUP_BASELINE_TEMPERATURE_KEY,
            help="0 = always play the top move; higher values add randomness.",
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
        state = start_new_game(
            color_choice,
            preset_key,
            baseline_temperature=(
                float(st.session_state[_PLAY_SETUP_BASELINE_TEMPERATURE_KEY])
                if category == OpponentCategory.BASELINE
                else BASELINE_TEMPERATURE_DEFAULT
            ),
        )
        _set_state(state)
        st.session_state.pop(_PLAY_SETUP_BASELINE_PRESETS_KEY, None)
        _ensure_bot(state)
        log_user_action(
            f"Started play game color={color_choice} preset={preset_key}",
            user,
        )
        st.rerun()


def _run_bot_turn(state: PlayGameState, bot: ChessBot) -> PlayGameState:
    """Block until the bot moves (script thread). Used for opening-as-black and after user."""
    new_state = apply_bot_move(state, bot)
    # Only attach analysis when a bot ply actually landed (avoid stale instance payload).
    if new_state.last_move_uci != state.last_move_uci:
        analysis = take_bot_move_analysis(bot)
        if analysis is None or analysis.played_uci == new_state.last_move_uci:
            st.session_state[_BOT_ANALYSIS_KEY] = analysis
            _sync_play_session(new_state)
    return new_state


def _is_baseline_game(state: PlayGameState) -> bool:
    return category_for_preset_key(state.preset_key) == OpponentCategory.BASELINE


def _candidates_show_flags(
    state: PlayGameState,
    analysis: BotMoveAnalysis | None,
) -> tuple[bool, bool]:
    """Return ``(show_dataframe, show_no_analysis)`` for the baseline panel."""
    if not _is_baseline_game(state) or analysis is None:
        return False, False
    if is_bot_thinking(state):
        return False, False
    if analysis.played_uci != state.last_move_uci:
        return False, False
    if len(analysis.rows) > 0:
        return True, False
    return False, True


def _render_board(state: PlayGameState):
    return chess_board(
        state_fen(state),
        key="play_chess_board",
        orientation=orientation_for_user(state.user_color),
        disabled=not is_user_turn(state),
        last_move_uci=state.last_move_uci,
        instance_id=state.instance_id,
        height=520,
    )


def _render_outcome_banner(status: str | None) -> None:
    """Top slot: win / loss / draw / resignation only (fixed height)."""
    with st.container(key="play_board_status"):
        if status:
            st.markdown(f"**{status}**")
        else:
            st.markdown("&#8203;", unsafe_allow_html=True)


def _render_bottom_bar(
    *,
    thinking: bool,
    preset_key: str,
    label: str,
    state: PlayGameState,
    align_with_panel: bool,
) -> None:
    """Bottom row: thinking message (left) and game controls (right)."""
    if align_with_panel:
        msg_col, btn_col = st.columns([2.0, 1.0], gap="medium", vertical_alignment="center")
    else:
        msg_col, btn_col = st.columns([3.0, 1.0], gap="medium", vertical_alignment="center")

    with msg_col:
        with st.container(key="play_bottom_status"):
            if thinking:
                st.markdown(f"*{bot_thinking_message(preset_key, label=label)}*")
            else:
                st.markdown("&#8203;", unsafe_allow_html=True)

    with btn_col:
        new_col, resign_col = st.columns(2, gap="small")
        with new_col:
            if st.button("New game", width="stretch"):
                log_user_action("Play page new game requested", user)
                _reset_game()
                st.rerun()
        with resign_col:
            if st.button("Resign", width="stretch", disabled=is_game_finished(state)):
                log_user_action("Play page resignation", user)
                _clear_bot_analysis()
                _set_state(resign_game(state))
                st.rerun()


def _render_match_caption(state: PlayGameState, *, label: str, description: str) -> None:
    st.caption(
        f"You are **{user_color_label(state.user_color)}** vs **{label}**"
        + (f" ({description})" if description else "")
    )


def _render_active_game(state: PlayGameState) -> None:
    preset = _resolve_preset(state.preset_key)
    label = preset.label if preset else state.preset_key
    description = preset.description if preset else ""

    bot = _ensure_bot(state)

    thinking = is_bot_thinking(state)
    if thinking:
        _clear_bot_analysis()

    status = game_status_message(state)
    baseline = _is_baseline_game(state)
    ingest_css(_PLAY_STATUS_CSS)

    _render_outcome_banner(status)

    if not baseline:
        _render_match_caption(state, label=label, description=description)

    if not thinking:
        _maybe_celebrate_win(state)

    analysis = _get_bot_analysis()
    show_df, show_no = _candidates_show_flags(state, analysis)

    if baseline:
        board_col, panel_col = st.columns([2.0, 1.0], gap="medium", vertical_alignment="top")
        with board_col:
            with st.container(key="play_board_col"):
                board_event = _render_board(state)
        with panel_col:
            _render_match_caption(state, label=label, description=description)
            render_bot_candidates_panel(
                analysis,
                show_dataframe=show_df,
                show_no_analysis=show_no,
                panel_visible=_candidates_panel_visible(),
            )
    else:
        board_event = _render_board(state)

    _render_bottom_bar(
        thinking=thinking,
        preset_key=state.preset_key,
        label=label,
        state=state,
        align_with_panel=baseline,
    )

    if thinking:
        state = _run_bot_turn(state, bot)
        _set_state(state)
        st.rerun()
        return

    try:
        applied = apply_user_board_event(state, board_event)
    except ValueError as exc:
        st.error(str(exc))
        applied = None
    if applied is not None:
        state, move_uci = applied
        _set_state(state)
        _clear_bot_analysis()
        log_user_action(f"Play page user move={move_uci}", user)
        # Rerun before bot so thinking paint clears panel (no stale rows under status).
        st.rerun()

    _sync_play_session(state)


def _render_page() -> None:
    _hydrate_session_from_cache()
    state = _get_state()
    if state is None:
        _render_setup()
        return
    _render_active_game(state)


_render_page()
