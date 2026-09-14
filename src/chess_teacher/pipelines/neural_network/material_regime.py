"""Cheap FEN material-regime flags (oversample / slice / Play fallback).

Board planes already show pieces; these scalars help **weighting** and
**metrics slices**, not a substitute for data or SF-regularized loss.
"""

from __future__ import annotations

import chess


def piece_count(fen: str) -> int:
    return len(chess.Board(fen).piece_map())


def pawn_count(fen: str) -> int:
    board = chess.Board(fen)
    return len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK))


def only_kings_and_pawns(fen: str) -> bool:
    board = chess.Board(fen)
    for piece in board.piece_map().values():
        if piece.piece_type not in (chess.KING, chess.PAWN):
            return False
    return True


def material_regime_flags(fen: str) -> dict[str, float | bool | int]:
    """Dict of regime cues for logging / future sample weights."""
    n = piece_count(fen)
    pawns = pawn_count(fen)
    kap = only_kings_and_pawns(fen)
    return {
        "piece_count": n,
        "pawn_count": pawns,
        "only_kings_and_pawns": kap,
        "pieces_le_10": n <= 10,
        "pieces_le_6": n <= 6,
    }
