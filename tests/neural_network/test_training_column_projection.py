"""Training hydration should project columns (no PGN / unused move fields)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import chess

from chess_teacher.pipelines.neural_network.create_training_set import (
    _CHARS_TRAINING_COLUMNS,
    _GAME_TRAINING_COLUMNS,
    _MOVE_TRAINING_COLUMNS,
    TrainingDataStore,
    TrainingDatumBuilder,
)
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.moves import Move, MoveCharacteristics
from chess_teacher.utils.chess_utils import Color


def test_training_column_sets_exclude_pgn_and_unused_text() -> None:
    assert _GAME_TRAINING_COLUMNS == ["game_id", "color"]
    assert "raw_pgn" not in _GAME_TRAINING_COLUMNS
    assert "cleaned_pgn" not in _GAME_TRAINING_COLUMNS
    assert "previous_opponent_move_san" not in _MOVE_TRAINING_COLUMNS
    assert "previous_opponent_move_san" not in _CHARS_TRAINING_COLUMNS


def test_from_db_rows_uses_color_without_full_game() -> None:
    move = Move(
        move_id="m1",
        game_id="g1",
        account_id="a1",
        move_nr=1,
        ply=1,
        move_san="e4",
        move_uci="e2e4",
        fen_before=chess.STARTING_FEN,
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    )
    chars = MoveCharacteristics(move_id="m1", game_id="g1", account_id="a1")

    datum = TrainingDatumBuilder.from_db_rows(move, chars, color=Color.WHITE, game_id="g1")

    assert datum.color == Color.WHITE
    assert datum.move_uci == "e2e4"


def test_datums_from_moves_projects_game_and_chars_reads() -> None:
    move = Move(
        move_id="m1",
        game_id="g1",
        account_id="a1",
        move_nr=1,
        ply=1,
        move_san="e4",
        move_uci="e2e4",
        fen_before=chess.STARTING_FEN,
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    )
    chars = MoveCharacteristics(move_id="m1", game_id="g1", account_id="a1")

    db = MagicMock()
    db.read.return_value = [{"game_id": "g1", "color": Color.WHITE.value}]
    store = TrainingDataStore(db)

    with (
        patch.object(MoveCharacteristics, "fetch_all_from_db", return_value=[chars]) as fetch_chars,
        patch.object(Game, "fetch_all_from_db") as fetch_games,
    ):
        datums = store._datums_from_moves([move])

    fetch_chars.assert_called_once()
    assert fetch_chars.call_args.kwargs["columns"] == _CHARS_TRAINING_COLUMNS
    fetch_games.assert_not_called()
    db.read.assert_called_once()
    assert db.read.call_args.kwargs["columns"] == _GAME_TRAINING_COLUMNS
    assert len(datums) == 1
    assert datums[0].color == Color.WHITE
