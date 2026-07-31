"""Sanity tests for fixed-vocab move encoding."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_teacher.pipelines.neural_network.move_encoding import (
    POLICY_VOCAB_SIZE,
    MoveEncoder,
    select_move_from_logits,
)


@pytest.mark.parametrize(
    "uci",
    [
        "e2e4",
        "g1f3",
        "e1g1",  # castling encoded as king two-step
        "a7a8q",
        "a7a8n",
        "b7a8r",
        "e7e8b",
    ],
)
def test_encode_decode_roundtrip_geometry(uci: str) -> None:
    idx = MoveEncoder.encode(uci)
    assert 0 <= idx < POLICY_VOCAB_SIZE
    move = chess.Move.from_uci(uci)
    decoded = MoveEncoder.decode(idx)
    assert decoded.from_square == move.from_square
    assert decoded.to_square == move.to_square
    if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        assert decoded.promotion == move.promotion


def test_mask_from_startpos_covers_all_legal() -> None:
    board = chess.Board()
    mask = MoveEncoder.mask_from_board(board)
    assert mask.dtype == np.bool_
    assert mask.shape == (POLICY_VOCAB_SIZE,)
    for move in board.legal_moves:
        assert mask[MoveEncoder.encode(move)]


def test_select_move_never_illegal() -> None:
    board = chess.Board()
    rng = np.random.default_rng(0)
    logits = rng.normal(size=POLICY_VOCAB_SIZE)
    for _ in range(20):
        move = select_move_from_logits(logits, board, temperature=0.0)
        assert move in board.legal_moves
        logits = rng.normal(size=POLICY_VOCAB_SIZE)
