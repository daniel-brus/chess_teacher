from unittest.mock import MagicMock

import chess
import polars as pl

from chess_teacher.ingestion.move_extraction import (
    ExtractUserMovesTransformation,
    FilterGamesAlreadyInMovesTransformation,
    extract_user_moves,
    tokenize_cleaned_movetext,
)
from chess_teacher.ingestion.moves import Move
from chess_teacher.utils.chess_utils import Color

SAMPLE_PGN = "1. e4 e5 2. Nf3 Nc6 3. d3"


def test_extract_user_moves_white() -> None:
    rows = extract_user_moves(
        game_id="game-1",
        cleaned_pgn=SAMPLE_PGN,
        color=Color.WHITE,
    )
    assert len(rows) == 3
    assert rows[0]["move_nr"] == 1
    assert rows[0]["ply"] == 1
    assert rows[0]["move_san"] == "e4"
    assert rows[0]["move_uci"] == "e2e4"
    assert rows[1]["move_nr"] == 2
    assert rows[1]["ply"] == 3
    assert rows[1]["move_san"] == "Nf3"
    assert rows[2]["move_nr"] == 3
    assert rows[2]["move_san"] == "d3"
    assert rows[0]["fen_before"] == chess.STARTING_FEN
    assert rows[0]["fen_after"] == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_extract_user_moves_black() -> None:
    rows = extract_user_moves(
        game_id="game-1",
        cleaned_pgn=SAMPLE_PGN,
        color=Color.BLACK,
    )
    assert len(rows) == 2
    assert rows[0]["move_nr"] == 1
    assert rows[0]["ply"] == 2
    assert rows[0]["move_san"] == "e5"
    assert rows[0]["move_uci"] == "e7e5"
    assert rows[1]["move_nr"] == 2
    assert rows[1]["ply"] == 4
    assert rows[1]["move_san"] == "Nc6"


def test_extract_user_moves_empty_pgn() -> None:
    assert (
        extract_user_moves(
            game_id="game-1",
            cleaned_pgn="",
            color=Color.WHITE,
        )
        == []
    )


def test_extract_user_moves_skips_non_standard_variant() -> None:
    assert (
        extract_user_moves(
            game_id="game-1",
            cleaned_pgn=SAMPLE_PGN,
            color=Color.WHITE,
            variant="chess960",
        )
        == []
    )


def test_move_generate_id_is_stable() -> None:
    move_id = Move.generate_id({"game_id": "abc", "move_nr": 2})
    assert move_id == Move.generate_id({"game_id": "abc", "move_nr": 2})
    assert move_id != Move.generate_id({"game_id": "abc", "move_nr": 3})


def test_tokenize_cleaned_movetext() -> None:
    assert tokenize_cleaned_movetext(SAMPLE_PGN) == ["e4", "e5", "Nf3", "Nc6", "d3"]


def test_extract_user_moves_transformation_includes_account_id() -> None:
    df = pl.DataFrame({
        "game_id": ["game-1"],
        "account_id": ["acct-1"],
        "cleaned_pgn": [SAMPLE_PGN],
        "color": [Color.WHITE.value],
        "variant": ["standard"],
    })
    result = ExtractUserMovesTransformation().transform(df)
    assert result.height == 3
    assert result["account_id"].unique().to_list() == ["acct-1"]


def test_filter_games_already_in_moves_skips_existing() -> None:
    df = pl.DataFrame({
        "game_id": ["game-1", "game-2", "game-3"],
        "account_id": ["acct-1", "acct-1", "acct-1"],
        "cleaned_pgn": [SAMPLE_PGN, SAMPLE_PGN, SAMPLE_PGN],
        "color": [Color.WHITE.value, Color.WHITE.value, Color.WHITE.value],
        "variant": ["standard", "standard", "standard"],
    })
    db_client = MagicMock()
    db_client.table_exists.return_value = True
    db_client.engine.execute_parameterized_query.return_value = [
        {"game_id": "game-1"},
        {"game_id": "game-2"},
    ]

    result = FilterGamesAlreadyInMovesTransformation(db_client=db_client).transform(df)

    assert result.height == 1
    assert result["game_id"].to_list() == ["game-3"]


def test_filter_games_already_in_moves_keeps_all_when_table_missing() -> None:
    df = pl.DataFrame({
        "game_id": ["game-1"],
        "account_id": ["acct-1"],
        "cleaned_pgn": [SAMPLE_PGN],
        "color": [Color.WHITE.value],
        "variant": ["standard"],
    })
    db_client = MagicMock()
    db_client.table_exists.return_value = False

    result = FilterGamesAlreadyInMovesTransformation(db_client=db_client).transform(df)

    assert result.height == 1
    db_client.engine.execute_parameterized_query.assert_not_called()
