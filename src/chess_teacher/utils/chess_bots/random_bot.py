from __future__ import annotations

import random

import chess

from chess_teacher.utils.chess_bots.base import ChessBot


class RandomBot(ChessBot):
    """Pick a uniformly random legal move."""

    name = "Random"

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("Cannot choose a move when there are no legal moves.")
        return self._rng.choice(moves)
