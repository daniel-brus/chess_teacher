from __future__ import annotations

import chess

from chess_teacher.utils.chess_utils.constants import PIECE_VALUES
from chess_teacher.utils.chess_utils.fen_metrics import _attacks_by_piece, _board_from_fen


def move_is_promotion(move: chess.Move) -> bool:
    return move.promotion is not None


def move_is_en_passant(board: chess.Board, move: chess.Move) -> bool:
    return board.is_en_passant(move)


def move_is_capture(board: chess.Board, move: chess.Move) -> bool:
    return board.is_capture(move)


def move_is_castle(move: chess.Move) -> bool:
    return (
        chess.square_rank(move.from_square) == chess.square_rank(move.to_square)
        and abs(chess.square_file(move.from_square) - chess.square_file(move.to_square)) == 2
    )


def move_gave_check(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    in_check = board.is_check()
    board.pop()
    return in_check


def _qualifies_as_fork_target(
    board: chess.Board,
    target_square: chess.Square,
    *,
    mover_color: chess.Color,
    mover_value: int,
) -> bool:
    """True when a newly attacked square counts toward a fork (loose, up-exchange, or king)."""
    target_piece = board.piece_at(target_square)
    if target_piece is None or target_piece.color == mover_color:
        return False
    if target_piece.piece_type == chess.KING:
        return True
    target_value = PIECE_VALUES.get(target_piece.piece_type, 0)
    if target_value > mover_value:
        return True
    attackers = len(board.attackers(mover_color, target_square))
    defenders = len(board.attackers(target_piece.color, target_square))
    return attackers > defenders


def move_created_fork(fen_before: str, fen_after: str, move_uci: str) -> bool:
    """True when the moved piece newly attacks two+ qualifying enemy targets."""
    del fen_after
    board = _board_from_fen(fen_before)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return False

    mover_color = board.turn
    before_attacks = _attacks_by_piece(board, move.from_square)
    board.push(move)
    mover_piece = board.piece_at(move.to_square)
    if mover_piece is None:
        return False
    mover_value = PIECE_VALUES.get(mover_piece.piece_type, 0)
    after_attacks = _attacks_by_piece(board, move.to_square)

    newly_attacked = after_attacks - before_attacks
    fork_targets = sum(
        1
        for target in newly_attacked
        if _qualifies_as_fork_target(
            board, target, mover_color=mover_color, mover_value=mover_value
        )
    )
    return fork_targets >= 2
