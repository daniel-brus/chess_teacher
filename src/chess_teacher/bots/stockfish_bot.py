from __future__ import annotations

import chess

from chess_teacher.bots.base import ChessBot
from chess_teacher.utils.chess_utils import StockfishEngine


class StockfishBot(ChessBot):
    """Stockfish-backed bot at a fixed search depth."""

    name = "Stockfish"

    def __init__(
        self,
        depth: int,
        *,
        path: str | None = None,
        threads: int | None = None,
        hash_mb: int | None = None,
    ) -> None:
        self.depth = depth
        self._engine = StockfishEngine(
            depth=depth,
            path=path,
            threads=threads,
            hash_mb=hash_mb,
        )
        self._engine.__enter__()

    def choose_move(self, board: chess.Board) -> chess.Move:
        return self._engine.choose_move(board)

    def close(self) -> None:
        self._engine.__exit__(None, None, None)
