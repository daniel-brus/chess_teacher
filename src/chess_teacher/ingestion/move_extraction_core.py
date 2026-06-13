from __future__ import annotations

import re
from io import StringIO

import chess
import chess.pgn

from chess_teacher.utils.chess_utils import Color

_RESULT_SUFFIX_RE = re.compile(r"\s*(?:1-0|0-1|1/2-1/2|\*)\s*$")
_BLACK_MOVE_NUMBER_RE = re.compile(r"\b\d+\.{2,3}\s*")
_WHITE_MOVE_NUMBER_RE = re.compile(r"\b\d+\.\s")


def tokenize_cleaned_movetext(cleaned_pgn: str) -> list[str]:
    """Split cleaned movetext into SAN tokens (no PGN parser)."""
    body = cleaned_pgn.strip()
    body = _RESULT_SUFFIX_RE.sub("", body).strip()
    body = _BLACK_MOVE_NUMBER_RE.sub("", body)
    body = _WHITE_MOVE_NUMBER_RE.sub("", body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return []
    return body.split(" ")


def _rows_from_sans(
    *,
    game_id: str,
    sans: list[str],
    user_turn: chess.Color,
) -> list[dict[str, object]]:
    board = chess.Board()
    ply = 0
    move_nr = 0
    rows: list[dict[str, object]] = []

    for san in sans:
        ply += 1
        if board.turn == user_turn:
            move_nr += 1
            fen_before = board.fen()
            move = board.push_san(san)
            rows.append({
                "game_id": game_id,
                "move_nr": move_nr,
                "ply": ply,
                "move_san": san,
                "move_uci": move.uci(),
                "fen_before": fen_before,
                "fen_after": board.fen(),
            })
        else:
            board.push_san(san)

    return rows


def _rows_from_pgn_parser(
    *,
    game_id: str,
    cleaned_pgn: str,
    user_turn: chess.Color,
) -> list[dict[str, object]]:
    try:
        game = chess.pgn.read_game(StringIO(f'[Event "?"]\n\n{cleaned_pgn}'))
    except (ValueError, chess.InvalidMoveError):
        return []
    if game is None:
        return []

    board = game.board()
    node: chess.pgn.Game | chess.pgn.ChildNode = game
    ply = 0
    move_nr = 0
    rows: list[dict[str, object]] = []

    while node.variations:
        ply += 1
        next_node = node.variation(0)
        move = next_node.move
        if move is None:
            break
        if board.turn == user_turn:
            move_nr += 1
            fen_before = board.fen()
            move_san = board.san(move)
            board.push(move)
            rows.append({
                "game_id": game_id,
                "move_nr": move_nr,
                "ply": ply,
                "move_san": move_san,
                "move_uci": move.uci(),
                "fen_before": fen_before,
                "fen_after": board.fen(),
            })
        else:
            board.push(move)
        node = next_node

    return rows


def extract_user_moves(
    *,
    game_id: str,
    cleaned_pgn: str,
    color: Color,
    variant: str = "standard",
) -> list[dict[str, object]]:
    """Extract one row per user move from cleaned movetext."""
    if variant != "standard":
        return []
    if not cleaned_pgn or not cleaned_pgn.strip():
        return []

    user_turn = chess.WHITE if color == Color.WHITE else chess.BLACK
    sans = tokenize_cleaned_movetext(cleaned_pgn)
    if sans:
        try:
            return _rows_from_sans(game_id=game_id, sans=sans, user_turn=user_turn)
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            pass

    return _rows_from_pgn_parser(
        game_id=game_id,
        cleaned_pgn=cleaned_pgn,
        user_turn=user_turn,
    )
