"""TrainingDatum → pack_candidate_tensors wiring (context flags / eval before)."""

from __future__ import annotations

from datetime import UTC, datetime

import chess
import pytest

from chess_teacher.pipelines.neural_network.candidate_eval import (
    MOVE_FEAT_DIM,
    build_candidate_payload,
)
from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatumBuilder
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.moves import Move, MoveCharacteristics
from chess_teacher.utils.chess_utils import Color, Reason, Result

_FEN = chess.STARTING_FEN


def _game(*, color: Color = Color.WHITE) -> Game:
    return Game(
        game_id="g1",
        platform_game_id="p1",
        account_id="a1",
        raw_pgn="x",
        cleaned_pgn="x",
        color=color,
        result=Result.WIN,
        reason=Reason.RESIGNATION,
        end_time=datetime.now(UTC),
    )


def _move(*, fen_before: str = _FEN, move_uci: str = "e2e4") -> Move:
    board = chess.Board(fen_before)
    board.push(chess.Move.from_uci(move_uci))
    return Move(
        move_id="m1",
        game_id="g1",
        account_id="a1",
        move_nr=1,
        ply=1,
        move_san="e4",
        move_uci=move_uci,
        fen_before=fen_before,
        fen_after=board.fen(en_passant="fen"),
    )


def test_candidate_style_target_packs_v3_with_eval_before_and_recapture_context() -> None:
    fen = _FEN
    board = chess.Board(fen)
    evals = {m.uci(): 0.2 for m in board.legal_moves}
    evals["e2e4"] = 0.4
    chars = MoveCharacteristics(
        move_id="m1",
        game_id="g1",
        account_id="a1",
        evaluation_before=0.1,
        opponent_move_was_capture=True,
        candidate_evaluations=build_candidate_payload(evals),
    )
    datum = TrainingDatumBuilder.from_db_rows(_move(), chars, _game())
    packed = datum.candidate_style_target()
    assert packed is not None
    feats, mask, label = packed
    assert feats.shape[1] == MOVE_FEAT_DIM
    assert float(mask.sum()) >= 1.0
    assert mask[label] == pytest.approx(1.0)
    assert datum.features.get("evaluation_before_user_pov") == pytest.approx(0.1)
    assert datum.features.get("opponent_move_was_capture") is True


def test_candidate_style_target_none_without_evals() -> None:
    chars = MoveCharacteristics(
        move_id="m1",
        game_id="g1",
        account_id="a1",
        candidate_evaluations=None,
    )
    datum = TrainingDatumBuilder.from_db_rows(_move(), chars, _game())
    assert datum.candidate_style_target() is None
