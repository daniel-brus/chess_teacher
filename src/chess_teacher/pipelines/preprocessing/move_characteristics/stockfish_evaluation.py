from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import polars as pl

from chess_teacher.pipelines.fen_eval_cache.keys import position_key
from chess_teacher.pipelines.fen_eval_cache.service import (
    DEFAULT_EVAL_DEPTH,
    FenEvalRequest,
    get_position_eval_service,
)
from chess_teacher.pipelines.neural_network.candidate_eval import CANDIDATE_STOCKFISH_NODES
from chess_teacher.pipelines.preprocessing.fen_characteristic import FenCharacteristicTransformation
from chess_teacher.utils.env_utils import get_stockfish_path
from chess_teacher.utils.exception_utils import TransformationError


class StockfishEvaluationTransformation(FenCharacteristicTransformation):
    """Stockfish centipawn/mate evaluation in pawns from white's perspective."""

    characteristic_name = "evaluation"

    def __init__(
        self,
        *,
        depth: int = DEFAULT_EVAL_DEPTH,
        stockfish_path: str | None = None,
        log_progress_percent: int | None = 5,
        n_workers: int | None = None,
    ) -> None:
        super().__init__(log_progress_percent=log_progress_percent, n_workers=n_workers)
        self.depth = depth
        self.stockfish_path = stockfish_path or get_stockfish_path()
        self._fen_ply: dict[str, int | None] = {}
        self._worker_init_kwargs = {
            "depth": self.depth,
            "stockfish_path": self.stockfish_path,
            "n_workers": 1,
        }

    @contextmanager
    def _evaluation_context(self) -> Iterator[None]:
        yield

    def evaluate(self, fen: str, *, row: dict[str, object]) -> float:
        result = get_position_eval_service().evaluate(
            fen,
            _resolve_ply(row, fen, self._fen_ply),
            eval_depth=self.depth,
            candidate_nodes=CANDIDATE_STOCKFISH_NODES,
        )
        return result.eval_white_pov

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        self._fen_ply = _min_ply_by_epd(df)
        return super().transform(df)

    def _evaluate_unique_fens(self, unique_fens: list[str]) -> dict[str, float]:
        requests = [
            FenEvalRequest(
                fen,
                ply=self._fen_ply.get(_safe_epd(fen)),
                eval_depth=self.depth,
                candidate_nodes=CANDIDATE_STOCKFISH_NODES,
            )
            for fen in unique_fens
        ]
        results = get_position_eval_service().evaluate_many(requests)
        scores: dict[str, float] = {}
        for fen in unique_fens:
            epd = _safe_epd(fen)
            if epd is None or epd not in results:
                raise TransformationError(f"Invalid FEN for Stockfish evaluation: {fen!r}")
            scores[fen] = results[epd].eval_white_pov
        return scores


def _safe_epd(fen: str) -> str | None:
    try:
        return position_key(fen)
    except ValueError:
        return None


def _resolve_ply(row: dict[str, object], fen: str, fen_ply: dict[str, int | None]) -> int | None:
    ply_raw = row.get("ply")
    if ply_raw is not None:
        return int(ply_raw)
    epd = _safe_epd(fen)
    if epd is None:
        return None
    return fen_ply.get(epd)


def _min_ply_by_epd(df: pl.DataFrame) -> dict[str, int | None]:
    if "ply" not in df.columns:
        return {}
    plies = [int(ply) for ply in df["ply"].to_list()]
    fens = [str(fen) for fen in df["fen_before"].to_list()] + [
        str(fen) for fen in df["fen_after"].to_list()
    ]
    fen_plies = plies + plies
    out: dict[str, int | None] = {}
    for fen, ply in zip(fens, fen_plies, strict=True):
        epd = _safe_epd(fen)
        if epd is None:
            continue
        current = out.get(epd)
        out[epd] = ply if current is None else min(current, ply)
    return out
