from __future__ import annotations

from contextlib import AbstractContextManager

import chess
from stockfish import Stockfish  # type: ignore[import-untyped]

from chess_teacher.utils.env_utils import get_env_variable, get_stockfish_path
from chess_teacher.utils.process_utils import WorkerSafeLogger

_DEFAULT_STOCKFISH_THREADS = 1
_DEFAULT_STOCKFISH_HASH_MB = 32

_logger = WorkerSafeLogger(__name__)


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
        _logger.info(
            "Starting Stockfish engine path=%s depth=%s threads=%s hash_mb=%s",
            self.path,
            self.depth,
            self._uci_parameters["Threads"],
            self._uci_parameters["Hash"],
        )
        self._engine = Stockfish(
            path=self.path,
            depth=self.depth,
            parameters=self._uci_parameters,
        )
        return self

    def __exit__(self, *args: object) -> None:
        if self._engine is not None:
            _logger.info("Stopping Stockfish engine path=%s", self.path)
            try:
                self._engine.send_quit_command()
            except OSError:
                _logger.warning(
                    "Stockfish quit command failed path=%s",
                    self.path,
                    exc_info=True,
                )
            self._engine = None

    def evaluate_white_pov_pawns(self, fen: str) -> float | None:
        """Evaluate a single FEN; return ``None`` when the FEN string is unusable."""
        try:
            board = chess.Board(fen)
        except ValueError:
            _logger.warning("Invalid FEN for Stockfish evaluation: %s", fen)
            return None
        if board.is_game_over():
            return game_over_white_pov_pawns(board)

        engine = self._require_engine()
        if not engine.is_fen_valid(fen):
            _logger.warning("Stockfish rejected FEN: %s", fen)
            return None
        engine.set_fen_position(fen)
        score = evaluation_to_white_pov_pawns(engine.get_evaluation(), fen)
        _logger.debug(
            "Stockfish evaluation depth=%s score=%s fen=%s",
            self.depth,
            score,
            fen,
        )
        return score

    def _require_engine(self) -> Stockfish:
        if self._engine is None:
            raise RuntimeError("StockfishEngine is not active; use it as a context manager.")
        return self._engine

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return Stockfish's best move for ``board``."""
        if board.is_game_over():
            raise ValueError("Cannot choose a move in a finished game.")
        fen = board.fen(en_passant="fen")
        engine = self._require_engine()
        if not engine.is_fen_valid(fen):
            raise ValueError(f"Stockfish rejected FEN: {fen!r}")
        engine.set_fen_position(fen)
        best_uci = engine.get_best_move()
        if best_uci is None:
            raise RuntimeError("Stockfish returned no legal move.")
        move = chess.Move.from_uci(best_uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"Stockfish move {best_uci!r} is not legal in python-chess.")
        return move
