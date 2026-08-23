"""Unit tests for candidate_style move features (no Stockfish / TF)."""

from __future__ import annotations

import json

import chess
import numpy as np
import pytest

from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_KEYS,
    CANDIDATE_MOVE_FEAT_VERSION,
    MAX_CANDIDATES,
    MOVE_FEAT_DIM,
    build_candidate_payload,
    candidate_move_rows,
    live_candidate_tensors,
    pack_candidate_tensors,
    parse_candidate_evaluations,
    white_to_user_pov,
)

_START = chess.STARTING_FEN
# After 1.e4, Black to move (user = Black).
_AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def _evals_for_fen(fen: str, *, score: float = 0.25) -> dict[str, float]:
    board = chess.Board(fen)
    return {m.uci(): score for m in board.legal_moves}


def test_move_feat_dim_tracks_keys() -> None:
    assert MOVE_FEAT_DIM == len(CANDIDATE_MOVE_FEAT_KEYS)
    assert len(CANDIDATE_MOVE_FEAT_KEYS) == len(set(CANDIDATE_MOVE_FEAT_KEYS))
    assert CANDIDATE_MOVE_FEAT_VERSION >= 3


def test_white_to_user_pov_flips_for_black() -> None:
    assert white_to_user_pov(1.5, color_is_white=True) == pytest.approx(1.5)
    assert white_to_user_pov(1.5, color_is_white=False) == pytest.approx(-1.5)


def test_parse_and_build_candidate_payload_roundtrip() -> None:
    payload = build_candidate_payload({"e2e4": 0.3, "d2d4": 0.1}, depth=12, num_nodes=50_000)
    raw = json.dumps(payload)
    parsed = parse_candidate_evaluations(raw)
    assert parsed is not None
    assert parsed["evals_white_pov"]["e2e4"] == pytest.approx(0.3)
    assert parsed["depth"] == 12
    assert parsed["num_nodes"] == 50_000


@pytest.mark.parametrize("raw", [None, "", "{}", '{"evals_white_pov": {}}', "not-json"])
def test_parse_candidate_evaluations_rejects_bad(raw: object) -> None:
    assert parse_candidate_evaluations(raw) is None


def test_candidate_move_rows_delta_vs_best_and_sort() -> None:
    # White POV scores; Black to move → user POV = -white.
    rows = candidate_move_rows(
        {"a7a6": 0.0, "e7e5": -0.4, "b8c6": 0.2},
        color_is_white=False,
    )
    ucis = [u for u, _, _ in rows]
    assert ucis == sorted(ucis)
    by_uci = {u: (ev, d) for u, ev, d in rows}
    # Best for Black = most positive user POV = -min(white) = -(-0.4) = 0.4 for e7e5
    assert by_uci["e7e5"][0] == pytest.approx(0.4)
    assert by_uci["e7e5"][1] == pytest.approx(0.0)
    assert by_uci["a7a6"][1] < 0.0


def test_pack_candidate_tensors_shape_mask_label() -> None:
    fen = _START
    board = chess.Board(fen)
    user = "e2e4"
    evals = _evals_for_fen(fen)
    evals[user] = 0.5
    packed = pack_candidate_tensors(
        evals,
        fen_before=fen,
        color_is_white=True,
        user_move_uci=user,
        legal_ucis=tuple(m.uci() for m in board.legal_moves),
        evaluation_before_white=0.1,
        opponent_move_was_capture=False,
    )
    assert packed is not None
    feats, mask, label = packed
    assert feats.shape == (MAX_CANDIDATES, MOVE_FEAT_DIM)
    assert mask.shape == (MAX_CANDIDATES,)
    assert float(mask.sum()) == pytest.approx(float(board.legal_moves.count()))
    assert 0 <= label < MAX_CANDIDATES
    assert mask[label] == pytest.approx(1.0)
    assert np.isfinite(feats[mask > 0.5]).all()


def test_pack_missing_user_move_returns_none() -> None:
    fen = _START
    evals = {"d2d4": 0.1}  # omit e2e4
    assert (
        pack_candidate_tensors(
            evals,
            fen_before=fen,
            color_is_white=True,
            user_move_uci="e2e4",
        )
        is None
    )


def test_pack_sets_capture_and_recapture_flags() -> None:
    # Position where exd5 is a capture for White; opponent just captured.
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    user = "e4d5"
    evals = _evals_for_fen(fen)
    packed = pack_candidate_tensors(
        evals,
        fen_before=fen,
        color_is_white=True,
        user_move_uci=user,
        opponent_move_was_capture=True,
        evaluation_before_white=0.0,
    )
    assert packed is not None
    feats, _mask, label = packed
    capture_i = CANDIDATE_MOVE_FEAT_KEYS.index("is_capture")
    recapture_i = CANDIDATE_MOVE_FEAT_KEYS.index("is_recapture")
    assert feats[label, capture_i] == pytest.approx(1.0)
    assert feats[label, recapture_i] == pytest.approx(1.0)


def test_pack_evaluation_delta_uses_before() -> None:
    fen = _START
    user = "e2e4"
    evals = _evals_for_fen(fen, score=0.4)
    evals[user] = 0.4
    packed = pack_candidate_tensors(
        evals,
        fen_before=fen,
        color_is_white=True,
        user_move_uci=user,
        evaluation_before_white=0.1,
    )
    assert packed is not None
    feats, _mask, label = packed
    # Indices: after, delta, delta_vs_best
    after_i = CANDIDATE_MOVE_FEAT_KEYS.index("evaluation_after_user_pov")
    delta_i = CANDIDATE_MOVE_FEAT_KEYS.index("evaluation_delta_user_pov")
    expected_delta = float(np.tanh(0.3 / 5.0))
    assert feats[label, after_i] == pytest.approx(float(np.tanh(0.4 / 5.0)), abs=1e-5)
    assert feats[label, delta_i] == pytest.approx(expected_delta, abs=1e-5)


def test_live_candidate_tensors_reuses_evals_no_engine() -> None:
    board = chess.Board(_AFTER_E4)
    evals = _evals_for_fen(_AFTER_E4)

    # engine unused when evals provided; pass a sentinel object that would fail if called.
    class _NoEngine:
        pass

    ucis, feats, mask = live_candidate_tensors(
        _NoEngine(),  # type: ignore[arg-type]
        board,
        evals=evals,
        opponent_move_was_capture=False,
        evaluation_before_white=0.2,
    )
    assert len(ucis) == int(mask.sum())
    assert feats.shape == (MAX_CANDIDATES, MOVE_FEAT_DIM)
    assert set(ucis) == {m.uci() for m in board.legal_moves}
