import chess
import polars as pl

from chess_teacher.pipelines.preprocessing.move_extraction import (
    ExtractUserMovesTransformation,
    extract_user_moves,
    tokenize_cleaned_movetext,
)
from chess_teacher.pipelines.preprocessing.moves import Move
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
    assert rows[0]["previous_opponent_move_san"] is None
    assert rows[0]["opponent_move_was_capture"] is False
    assert rows[1]["move_nr"] == 2
    assert rows[1]["ply"] == 3
    assert rows[1]["move_san"] == "Nf3"
    assert rows[1]["previous_opponent_move_san"] == "e5"
    assert rows[1]["previous_opponent_move_uci"] == "e7e5"
    assert rows[1]["opponent_move_was_capture"] is False
    assert rows[2]["move_nr"] == 3
    assert rows[2]["move_san"] == "d3"
    assert rows[2]["previous_opponent_move_san"] == "Nc6"
    assert rows[0]["fen_before"] == chess.STARTING_FEN
    assert rows[0]["fen_after"] == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_extract_user_moves_opponent_capture_flag() -> None:
    rows = extract_user_moves(
        game_id="game-1",
        cleaned_pgn="1. e4 e5 2. d4 exd4 3. Nf3",
        color=Color.WHITE,
    )
    assert rows[2]["previous_opponent_move_san"] == "exd4"
    assert rows[2]["opponent_move_was_capture"] is True


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
    assert rows[0]["previous_opponent_move_san"] == "e4"
    assert rows[0]["opponent_move_was_capture"] is False
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
