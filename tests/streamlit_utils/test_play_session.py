from __future__ import annotations

import chess
import pytest

from chess_teacher.bots.move_analysis import BotMoveAnalysis, BotMoveCandidateRow
from streamlit_utils.play_game import start_new_game
from streamlit_utils.play_session import (
    load_play_session_for_user,
    play_session_storage_key,
    restore_bot_analysis,
    restore_play_session,
    restore_play_state,
    save_play_session_for_user,
    snapshot_bot_analysis,
    snapshot_play_session,
    snapshot_play_state,
)


class _SessionState(dict):
    def pop(self, key, default=None):  # type: ignore[no-untyped-def]
        if key in self:
            return super().pop(key)
        return default


@pytest.fixture
def streamlit_session(monkeypatch: pytest.MonkeyPatch) -> _SessionState:
    session = _SessionState()
    monkeypatch.setattr("streamlit_utils.play_session.st.session_state", session)
    return session


def test_play_session_storage_key_is_user_scoped() -> None:
    assert play_session_storage_key("abc") == "play_saved_session:abc"
    assert play_session_storage_key("abc") != play_session_storage_key("xyz")


def test_snapshot_roundtrip_midgame() -> None:
    state = start_new_game("White", "random", baseline_temperature=0.75)
    state = state.__class__(
        board=chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        user_color=state.user_color,
        preset_key=state.preset_key,
        last_move_uci="e7e5",
        instance_id=3,
        pending_bot_move=False,
        baseline_temperature=0.75,
    )

    restored = restore_play_state(snapshot_play_state(state))
    assert restored is not None
    assert restored.board.fen() == state.board.fen()
    assert restored.user_color == chess.WHITE
    assert restored.preset_key == "random"
    assert restored.last_move_uci == "e7e5"
    assert restored.instance_id == 3
    assert restored.pending_bot_move is False
    assert restored.baseline_temperature == 0.75


def test_snapshot_roundtrip_bot_analysis() -> None:
    analysis = BotMoveAnalysis(
        fen_before=chess.STARTING_FEN,
        played_uci="e2e4",
        played_san="e4",
        rows=(
            BotMoveCandidateRow(
                uci="e2e4",
                san="e4",
                model_p=0.42,
                sf_eval_stm=0.1,
                delta_vs_best=0.0,
                is_played=True,
            ),
        ),
        temperature_used=1.25,
    )

    restored = restore_bot_analysis(snapshot_bot_analysis(analysis))
    assert restored == analysis


def test_restore_play_session_includes_aux_fields() -> None:
    state = start_new_game("Black", "stockfish:3")
    payload = snapshot_play_session(
        state=state,
        bot_analysis=None,
        win_celebrated_instance=7,
        candidates_visible=False,
    )

    restored = restore_play_session(payload)
    assert restored is not None
    assert restored.state.preset_key == "stockfish:3"
    assert restored.win_celebrated_instance == 7
    assert restored.candidates_visible is False


def test_restore_play_state_rejects_bad_payload() -> None:
    good = snapshot_play_state(start_new_game("White", "random"))
    assert restore_play_state({**good, "version": 99}) is None
    assert restore_play_state({**good, "fen": "not-a-fen"}) is None
    assert restore_play_state({**good, "user_color": "green"}) is None


def test_restore_bot_analysis_rejects_malformed_rows() -> None:
    assert (
        restore_bot_analysis({"fen_before": "x", "played_uci": "e2e4", "played_san": "e4"}) is None
    )
    assert (
        restore_bot_analysis({
            "fen_before": chess.STARTING_FEN,
            "played_uci": "e2e4",
            "played_san": "e4",
            "temperature_used": 0.0,
            "rows": [{"uci": "e2e4"}],
        })
        is None
    )


def test_full_play_session_roundtrip_with_analysis() -> None:
    state = start_new_game("White", "baseline:v1", baseline_temperature=0.8)
    analysis = BotMoveAnalysis(
        fen_before=chess.STARTING_FEN,
        played_uci="e2e4",
        played_san="e4",
        rows=(),
        temperature_used=0.8,
    )
    payload = snapshot_play_session(
        state=state,
        bot_analysis=analysis,
        win_celebrated_instance=2,
        candidates_visible=True,
    )

    restored = restore_play_session(payload)
    assert restored is not None
    assert restored.state.baseline_temperature == pytest.approx(0.8)
    assert restored.bot_analysis == analysis
    assert restored.win_celebrated_instance == 2


def test_save_and_load_play_session_for_user(streamlit_session: _SessionState) -> None:
    state = start_new_game("White", "random")
    save_play_session_for_user(
        "user-42",
        state=state,
        win_celebrated_instance=1,
        candidates_visible=False,
    )

    loaded = load_play_session_for_user("user-42")
    assert loaded is not None
    assert loaded.state.preset_key == "random"
    assert loaded.win_celebrated_instance == 1
    assert loaded.candidates_visible is False
    assert play_session_storage_key("user-42") in streamlit_session

    save_play_session_for_user("user-42", state=None)
    assert load_play_session_for_user("user-42") is None
    assert play_session_storage_key("user-42") not in streamlit_session


def test_play_sessions_are_isolated_per_user(streamlit_session: _SessionState) -> None:
    save_play_session_for_user("alice", state=start_new_game("White", "random"))
    save_play_session_for_user("bob", state=start_new_game("Black", "stockfish:3"))

    alice = load_play_session_for_user("alice")
    bob = load_play_session_for_user("bob")
    assert alice is not None and bob is not None
    assert alice.state.user_color == chess.WHITE
    assert bob.state.user_color == chess.BLACK
