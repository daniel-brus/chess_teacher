from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from chess_teacher.pipelines.preprocessing.fen_characteristic import FenCharacteristicTransformation
from chess_teacher.utils.chess_utils import StockfishEngine
from chess_teacher.utils.env_utils import get_stockfish_path
from chess_teacher.utils.exception_utils import TransformationError


class StockfishEvaluationTransformation(FenCharacteristicTransformation):
    """Stockfish centipawn/mate evaluation in pawns from white's perspective."""

    characteristic_name = "evaluation"

    def __init__(
        self,
        *,
        depth: int = 20,
        stockfish_path: str | None = None,
        log_progress_percent: int | None = 5,
        n_workers: int | None = None,
    ) -> None:
        super().__init__(log_progress_percent=log_progress_percent, n_workers=n_workers)
        self.depth = depth
        self.stockfish_path = stockfish_path or get_stockfish_path()
        self._engine: StockfishEngine | None = None
        self._worker_init_kwargs = {
            "depth": self.depth,
            "stockfish_path": self.stockfish_path,
            "n_workers": 1,
        }

    @contextmanager
    def _evaluation_context(self) -> Iterator[None]:
        with StockfishEngine(depth=self.depth, path=self.stockfish_path) as engine:
            self._engine = engine
            try:
                yield
            finally:
                self._engine = None

    def evaluate(self, fen: str, *, row: dict[str, object]) -> float:
        del row
        if self._engine is None:
            raise TransformationError("Stockfish engine is not active outside transform().")
        score = self._engine.evaluate_white_pov_pawns(fen)
        if score is None:
            raise TransformationError(f"Invalid FEN for Stockfish evaluation: {fen!r}")
        return score
