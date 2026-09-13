"""MultiPV-all-legal candidate evaluations for ``move_characteristics``.

Last transform in :class:`~chess_teacher.pipelines.preprocessing.pipeline_steps.EnrichExpensiveMoveCharacteristicsStep`
(after played-move Stockfish eval). Stockfish work goes through ``PositionEvalService``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import polars as pl

from chess_teacher.pipelines.fen_eval_cache.keys import position_key
from chess_teacher.pipelines.fen_eval_cache.service import FenEvalRequest, get_position_eval_service
from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_STOCKFISH_DEPTH,
    CANDIDATE_STOCKFISH_NODES,
)
from chess_teacher.pipelines.preprocessing.fen_checkpoint import (
    CHECKPOINT_MERGE,
    checkpoint_percent_from_env,
)
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.metadata_utils import TableMetadata
from chess_teacher.utils.pipeline_utils.dataframe_transformation import DataFrameTransformation
from chess_teacher.utils.process_utils import WorkerSafeLogger, snapshot_host_pressure

_logger = WorkerSafeLogger(__name__)

_DEFAULT_LOG_PROGRESS_PERCENT = 5


@dataclass(frozen=True, slots=True)
class _MoveRef:
    move_id: str
    game_id: str
    account_id: str


class CandidateEvaluationsTransformation(DataFrameTransformation):
    """Fill ``candidate_evaluations`` from unique ``fen_before`` MultiPV searches."""

    def __init__(
        self,
        *,
        depth: int = CANDIDATE_STOCKFISH_DEPTH,
        num_nodes: int | None = CANDIDATE_STOCKFISH_NODES,
        log_progress_percent: int | None = _DEFAULT_LOG_PROGRESS_PERCENT,
        checkpoint_percent: int | None = None,
        n_workers: int | None = None,
    ) -> None:
        del n_workers
        if log_progress_percent is not None and not 1 <= log_progress_percent <= 100:
            raise ValueError("log_progress_percent must be between 1 and 100, or None")
        if checkpoint_percent == 0:
            resolved_checkpoint_percent: int | None = None
        elif checkpoint_percent is None:
            resolved_checkpoint_percent = checkpoint_percent_from_env(
                "CANDIDATE_EVAL_CHECKPOINT_PERCENT"
            )
        elif not 1 <= checkpoint_percent <= 100:
            raise ValueError("checkpoint_percent must be between 1 and 100, 0, or None")
        else:
            resolved_checkpoint_percent = checkpoint_percent
        self.depth = depth
        self.num_nodes = CANDIDATE_STOCKFISH_NODES if num_nodes is None else num_nodes
        self.log_progress_percent = log_progress_percent
        self.checkpoint_percent = resolved_checkpoint_percent
        self._db_client: DatabaseClient | None = None
        self._table_metadata: TableMetadata | None = None
        self._fen_to_moves: dict[str, list[_MoveRef]] = {}
        self._checkpointed_fens: set[str] = set()

    def bind_checkpoint(
        self,
        *,
        db_client: DatabaseClient,
        table_metadata: TableMetadata,
    ) -> None:
        self._db_client = db_client
        self._table_metadata = table_metadata

    def _checkpoint_enabled(self) -> bool:
        return self._db_client is not None and self.checkpoint_percent is not None

    def _rows_for_fen_payloads(
        self,
        fen_payloads: dict[str, dict[str, Any] | None],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for fen, payload in fen_payloads.items():
            if payload is None:
                continue
            for move_ref in self._fen_to_moves.get(fen, []):
                rows.append({
                    "move_id": move_ref.move_id,
                    "game_id": move_ref.game_id,
                    "account_id": move_ref.account_id,
                    "candidate_evaluations": payload,
                })
        return rows

    def _maybe_checkpoint_batch(self, fen_payloads: dict[str, dict[str, Any] | None]) -> None:
        if not self._checkpoint_enabled() or self._table_metadata is None:
            return
        new_payloads = {
            fen: payload
            for fen, payload in fen_payloads.items()
            if fen not in self._checkpointed_fens and payload is not None
        }
        if not new_payloads:
            return
        rows = self._rows_for_fen_payloads(new_payloads)
        if not rows:
            return
        df = pl.DataFrame(rows)
        assert self._db_client is not None
        self._db_client.merge(df, self._table_metadata, strategy=CHECKPOINT_MERGE)
        self._checkpointed_fens.update(new_payloads)
        _logger.info(
            "%s: checkpointed %d move row(s) for %d FEN(s).",
            type(self).__name__,
            len(rows),
            len(new_payloads),
        )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df.with_columns(pl.lit(None).alias("candidate_evaluations"))

        move_ids = df["move_id"].cast(pl.Utf8).to_list()
        game_ids = df["game_id"].cast(pl.Utf8).to_list()
        account_ids = df["account_id"].cast(pl.Utf8).to_list()
        fens_before = [str(fen) for fen in df["fen_before"].cast(pl.Utf8).to_list()]
        plies = (
            [int(ply) for ply in df["ply"].to_list()] if "ply" in df.columns else [None] * df.height
        )

        fen_to_moves: dict[str, list[_MoveRef]] = defaultdict(list)
        requests_by_epd: dict[str, FenEvalRequest] = {}
        for move_id, game_id, account_id, fen, ply in zip(
            move_ids,
            game_ids,
            account_ids,
            fens_before,
            plies,
            strict=True,
        ):
            fen_to_moves[fen].append(_MoveRef(move_id, game_id, account_id))
            try:
                epd = position_key(fen)
            except ValueError:
                continue
            current = requests_by_epd.get(epd)
            if current is None or (ply is not None and (current.ply is None or ply < current.ply)):
                requests_by_epd[epd] = FenEvalRequest(
                    fen,
                    ply=ply,
                    eval_depth=self.depth,
                    candidate_nodes=int(self.num_nodes),
                )

        self._fen_to_moves = dict(fen_to_moves)
        self._checkpointed_fens = set()

        started = snapshot_host_pressure()
        _logger.info(
            "%s: evaluating %d unique FEN(s) from %d row(s). %s",
            type(self).__name__,
            len(requests_by_epd),
            df.height,
            started.format_fields(),
        )
        results = get_position_eval_service().evaluate_many(list(requests_by_epd.values()))
        fen_payloads: dict[str, dict[str, Any] | None] = {}
        for fen in fen_to_moves:
            try:
                epd = position_key(fen)
            except ValueError:
                fen_payloads[fen] = None
                continue
            result = results.get(epd)
            fen_payloads[fen] = None if result is None else result.candidates
        self._maybe_checkpoint_batch(fen_payloads)

        candidate_evaluations = [fen_payloads.get(fen) for fen in fens_before]
        return df.with_columns(
            pl.Series("candidate_evaluations", candidate_evaluations, dtype=pl.Object),
        )
