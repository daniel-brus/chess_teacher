from __future__ import annotations

from abc import ABC, abstractmethod

import chess


class ChessBot(ABC):
    """Abstract opponent that picks one legal move for the side to move."""

    name: str

    @abstractmethod
    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return a legal move for ``board.turn``."""

    def close(self) -> None:
        """Release any resources held by the bot."""

    def __enter__(self) -> ChessBot:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
