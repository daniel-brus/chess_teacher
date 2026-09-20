"""Normalize a FEN to the cache position key (EPD)."""

from __future__ import annotations

import chess


def position_key(fen: str) -> str:
    """Board identity: placement, side to move, castling, en passant."""
    return chess.Board(fen).epd()
