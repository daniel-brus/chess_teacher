"""Unit tests for bot move-candidate panel payloads (no Streamlit / Stockfish / TF)."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_teacher.bots.move_analysis import (
    BotMoveAnalysis,
    _display_softmax_probs,
    build_bot_move_analysis,
    empty_bot_move_analysis,
)

# After 1.e4 — Black to move.
_AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def _black_reply_evals(
    ucis: list[str], *, best_uci: str, best_white: float = -0.2
) -> dict[str, float]:
    """White-POV after-move scores; ``best_uci`` is SF-best for Black (lowest white eval)."""
    out = {uci: 0.1 for uci in ucis}
    out[best_uci] = best_white
    return out


def test_display_softmax_probs_masks_illegal_and_sums_to_one() -> None:
    logits = np.array([3.0, 1.0, 2.0, 0.0])
    mask = np.array([1.0, 0.0, 1.0, 1.0])  # index 1 illegal
    probs = _display_softmax_probs(logits, mask, n=4)
    assert probs[1] == pytest.approx(0.0)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] > probs[2] > probs[3]


def test_display_softmax_probs_all_masked_returns_zeros() -> None:
    logits = np.array([1.0, 2.0])
    mask = np.zeros(2)
    probs = _display_softmax_probs(logits, mask, n=2)
    assert probs.tolist() == [0.0, 0.0]


def test_build_pins_played_first_even_when_not_top_by_p() -> None:
    board = chess.Board(_AFTER_E4)
    ucis = ["e7e5", "c7c5", "g8f6"]
    logits = np.array([3.0, 1.0, 0.5])  # e7e5 highest P
    mask = np.ones(3)
    evals = _black_reply_evals(ucis, best_uci="e7e5")
    played = "g8f6"  # third by logit

    analysis = build_bot_move_analysis(
        ucis,
        logits,
        mask,
        evals,
        board,
        played,
        temperature_used=0.8,
    )

    assert len(analysis.rows) == 3
    assert analysis.rows[0].uci == played
    assert analysis.rows[0].is_played is True
    assert analysis.rows[0].san == "Nf6"
    assert all(not r.is_played for r in analysis.rows[1:])
    # Display P uses T=1 softmax, not sampling temperature.
    assert analysis.rows[0].model_p < analysis.rows[1].model_p


def test_build_limits_to_played_plus_four_extras() -> None:
    board = chess.Board(_AFTER_E4)
    ucis = [m.uci() for m in list(board.legal_moves)[:8]]
    logits = np.arange(len(ucis), dtype=np.float64)
    mask = np.ones(len(ucis))
    evals = _black_reply_evals(ucis, best_uci=ucis[0])
    played = ucis[-1]

    analysis = build_bot_move_analysis(
        ucis,
        logits,
        mask,
        evals,
        board,
        played,
        temperature_used=0.0,
    )

    assert len(analysis.rows) == 5
    assert analysis.rows[0].uci == played
    extra_ucis = {r.uci for r in analysis.rows[1:]}
    assert played not in extra_ucis
    assert len(extra_ucis) == 4


def test_build_san_from_pre_move_board_not_after_push() -> None:
    board = chess.Board(_AFTER_E4)
    ucis = ["e7e5", "c7c5"]
    logits = np.array([1.0, 0.0])
    mask = np.ones(2)
    evals = _black_reply_evals(ucis, best_uci="e7e5")

    analysis = build_bot_move_analysis(
        ucis,
        logits,
        mask,
        evals,
        board,
        "e7e5",
        temperature_used=0.0,
    )

    assert analysis.played_san == "e5"
    assert analysis.rows[0].san == "e5"
    assert analysis.rows[1].san == "c5"


def test_build_delta_vs_best_zero_for_sf_best() -> None:
    board = chess.Board(_AFTER_E4)
    ucis = ["e7e5", "c7c5"]
    logits = np.array([0.0, 1.0])
    mask = np.ones(2)
    evals = _black_reply_evals(ucis, best_uci="e7e5", best_white=-0.3)

    analysis = build_bot_move_analysis(
        ucis,
        logits,
        mask,
        evals,
        board,
        "c7c5",
        temperature_used=0.0,
    )

    by_uci = {r.uci: r for r in analysis.rows}
    assert by_uci["e7e5"].delta_vs_best == pytest.approx(0.0)
    assert by_uci["c7c5"].delta_vs_best < 0.0


def test_build_temperature_used_stored_display_p_unchanged() -> None:
    board = chess.Board(_AFTER_E4)
    ucis = ["e7e5", "c7c5"]
    logits = np.array([2.0, 0.0])
    mask = np.ones(2)
    evals = _black_reply_evals(ucis, best_uci="e7e5")

    low_t = build_bot_move_analysis(ucis, logits, mask, evals, board, "e7e5", temperature_used=0.0)
    high_t = build_bot_move_analysis(ucis, logits, mask, evals, board, "e7e5", temperature_used=1.5)

    assert low_t.temperature_used == pytest.approx(0.0)
    assert high_t.temperature_used == pytest.approx(1.5)
    assert low_t.rows[0].model_p == pytest.approx(high_t.rows[0].model_p)


def test_build_resilient_played_row_when_played_missing_sf_in_loop() -> None:
    board = chess.Board(_AFTER_E4)
    ucis = ["e7e5", "c7c5"]
    logits = np.array([1.0, 0.0])
    mask = np.ones(2)
    # Omit played UCI from evals — loop skips it; resilient path still pins played.
    evals = {"c7c5": 0.0}

    analysis = build_bot_move_analysis(
        ucis,
        logits,
        mask,
        evals,
        board,
        "e7e5",
        temperature_used=0.0,
    )

    assert len(analysis.rows) >= 1
    assert analysis.rows[0].uci == "e7e5"
    assert analysis.rows[0].is_played is True
    assert analysis.rows[0].san == "e5"


def test_build_empty_ucis_returns_no_rows() -> None:
    board = chess.Board(_AFTER_E4)
    analysis = build_bot_move_analysis(
        [],
        np.array([]),
        np.array([]),
        {},
        board,
        "e7e5",
        temperature_used=0.5,
    )
    assert analysis.rows == ()
    assert analysis.played_uci == "e7e5"
    assert analysis.temperature_used == pytest.approx(0.5)


def test_empty_bot_move_analysis_records_played_san_and_temperature() -> None:
    board = chess.Board(_AFTER_E4)
    played = chess.Move.from_uci("e7e5")
    analysis = empty_bot_move_analysis(board, played, temperature_used=0.75)

    assert isinstance(analysis, BotMoveAnalysis)
    assert analysis.rows == ()
    assert analysis.played_uci == "e7e5"
    assert analysis.played_san == "e5"
    assert analysis.temperature_used == pytest.approx(0.75)
    assert analysis.fen_before == board.fen(en_passant="fen")
