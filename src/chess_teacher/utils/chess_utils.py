from __future__ import annotations

from contextlib import AbstractContextManager
from enum import StrEnum

import chess
from stockfish import Stockfish

from chess_teacher.utils.env_utils import get_env_variable, get_stockfish_path
from chess_teacher.utils.process_utils import WORKER_NO_OP_LOGGER, is_parent_process

_DEFAULT_STOCKFISH_THREADS = 1
_DEFAULT_STOCKFISH_HASH_MB = 32

PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def material_balance_white_pov(fen: str) -> float:
    """Return white material minus black material using standard piece values."""
    board = chess.Board(fen)
    white = 0
    black = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white += value
        else:
            black += value
    return float(white - black)


_logger: object | None = None


def _log():
    global _logger
    if not is_parent_process():
        return WORKER_NO_OP_LOGGER
    if _logger is None:
        from chess_teacher.utils.logging import get_logger

        _logger = get_logger()
    return _logger


class Color(StrEnum):
    WHITE = "white"
    BLACK = "black"


class Result(StrEnum):
    WIN = "win"
    DRAW = "draw"
    LOSS = "loss"
    NO_RESULT = "no_result"


class Reason(StrEnum):
    # win/loss reasons:
    CHECKMATE = "checkmate"
    RESIGNATION = "resignation"
    TIMEOUT = "timeout"
    # draw reasons:
    STALEMATE = "stalemate"
    INSUFFICIENT_MATERIAL = "insufficient_material"
    TIMEOUT_INSUFFICIENT_MATERIAL = "timeout_insufficient_material"
    THREEFOLD_REPETITION = "threefold_repetition"
    AGREED_DRAW = "agreed_draw"
    FIFTY_MOVE_RULE = "fifty_move_rule"
    # ambiguous reasons:
    ABANDONED = "abandoned"
    OTHER = "other"


def evaluation_to_white_pov_pawns(evaluation: dict[str, str | int], fen: str) -> float:
    """
    Convert a Stockfish evaluation dict to pawns from white's perspective.

    Mate scores map to ``±(100 - mate)`` pawns so deltas remain comparable.
    """
    board = chess.Board(fen)
    stm_is_white = board.turn == chess.WHITE
    eval_type = evaluation["type"]
    value = int(evaluation["value"])

    if eval_type == "cp":
        cp = value if stm_is_white else -value
        return cp / 100.0

    if eval_type == "mate":
        mate = value if stm_is_white else -value
        if mate == 0:
            return 0.0
        return (100.0 - abs(mate)) * (1.0 if mate > 0 else -1.0)

    raise ValueError(f"Unsupported Stockfish evaluation type: {eval_type!r}")


def game_over_white_pov_pawns(board: chess.Board) -> float:
    """Mate-style score from white's perspective when the position is already decided."""
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0.0
    return 100.0 if outcome.winner == chess.WHITE else -100.0


def stockfish_uci_parameters(
    *,
    threads: int | None = None,
    hash_mb: int | None = None,
) -> dict[str, int]:
    """Per-engine UCI options for pooled evaluation (one Stockfish .exe per process)."""
    resolved_threads = threads or int(
        get_env_variable("STOCKFISH_THREADS_PER_ENGINE", str(_DEFAULT_STOCKFISH_THREADS))
    )
    resolved_hash = hash_mb or int(
        get_env_variable("STOCKFISH_HASH_MB", str(_DEFAULT_STOCKFISH_HASH_MB))
    )
    return {
        "Threads": max(1, resolved_threads),
        "Hash": max(1, resolved_hash),
    }


class StockfishEngine(AbstractContextManager["StockfishEngine"]):
    """Thin wrapper around the ``stockfish`` package with white-POV eval helpers."""

    def __init__(
        self,
        *,
        depth: int = 20,
        path: str | None = None,
        threads: int | None = None,
        hash_mb: int | None = None,
    ) -> None:
        self.depth = depth
        self.path = path or get_stockfish_path()
        self._uci_parameters = stockfish_uci_parameters(threads=threads, hash_mb=hash_mb)
        self._engine: Stockfish | None = None

    def __enter__(self) -> StockfishEngine:
        self._engine = Stockfish(
            path=self.path,
            depth=self.depth,
            parameters=self._uci_parameters,
        )
        return self

    def __exit__(self, *args: object) -> None:
        if self._engine is not None:
            try:
                self._engine.send_quit_command()
            except OSError:
                pass
            self._engine = None

    def evaluate_white_pov_pawns(self, fen: str) -> float | None:
        """Evaluate a single FEN; return ``None`` when the FEN string is unusable."""
        try:
            board = chess.Board(fen)
        except ValueError:
            _log().warning("Invalid FEN for Stockfish evaluation: %s", fen)
            return None
        if board.is_game_over():
            return game_over_white_pov_pawns(board)

        engine = self._require_engine()
        if not engine.is_fen_valid(fen):
            _log().warning("Stockfish rejected FEN: %s", fen)
            return None
        engine.set_fen_position(fen)
        return evaluation_to_white_pov_pawns(engine.get_evaluation(), fen)

    def _require_engine(self) -> Stockfish:
        if self._engine is None:
            raise RuntimeError("StockfishEngine is not active; use it as a context manager.")
        return self._engine
