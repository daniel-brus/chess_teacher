"""MultiPV-all-legal candidate evaluations for ``move_characteristics``.

Last transform in :class:`~chess_teacher.pipelines.preprocessing.pipeline_steps.EnrichMoveCharacteristicsStep`
(after played-move Stockfish eval). Kept separate from
:class:`~chess_teacher.pipelines.preprocessing.move_characteristics.stockfish_evaluation.StockfishEvaluationTransformation`:
that scores ``fen_before`` / ``fen_after`` at fixed depth; this runs MultiPV on
``fen_before`` only and stores ``{uci: eval_white_pov}`` JSONB (deduped by FEN).
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import polars as pl

from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_STOCKFISH_DEPTH,
    CANDIDATE_STOCKFISH_NODES,
    build_candidate_payload,
    evaluate_all_legal_after,
)
from chess_teacher.pipelines.preprocessing.fen_characteristic import (
    FenCharacteristicTransformation,
    _advance_fen_progress,
    _default_fen_eval_workers,
    _WorkerFenProgressTracker,
)
from chess_teacher.utils.chess_utils import StockfishEngine
from chess_teacher.utils.db.client import DatabaseClient, MergeStrategy
from chess_teacher.utils.env_utils import get_optional_env_variable
from chess_teacher.utils.metadata_utils import TableMetadata
from chess_teacher.utils.pipeline_utils.dataframe_transformation import DataFrameTransformation
from chess_teacher.utils.process_utils import WorkerSafeLogger, is_parent_process

_logger = WorkerSafeLogger(__name__)

_DEFAULT_LOG_PROGRESS_PERCENT = 5
_DEFAULT_CHECKPOINT_PERCENT = 10
_MIN_PARALLEL_FENS = 100

_CHECKPOINT_MERGE = MergeStrategy(
    when_matched="update",
    when_not_matched_by_target="ignore",
    when_not_matched_by_source="ignore",
)


def _checkpoint_percent_from_env() -> int | None:
    raw = get_optional_env_variable("CANDIDATE_EVAL_CHECKPOINT_PERCENT")
    if raw is None:
        return _DEFAULT_CHECKPOINT_PERCENT
    try:
        value = int(raw)
    except ValueError:
        _logger.warning(
            "Invalid CANDIDATE_EVAL_CHECKPOINT_PERCENT=%r; using default %s",
            raw,
            _DEFAULT_CHECKPOINT_PERCENT,
        )
        return _DEFAULT_CHECKPOINT_PERCENT
    if value <= 0:
        return None
    return value


@dataclass(frozen=True, slots=True)
class _MoveRef:
    move_id: str
    game_id: str
    account_id: str


class _FenPoolRunner(FenCharacteristicTransformation):
    """Reuse fen-characteristic process-pool helpers without scalar evaluate()."""

    characteristic_name = "_fen_pool"

    def evaluate(self, fen: str, *, row: dict[str, object]) -> float:
        del fen, row
        raise RuntimeError("_FenPoolRunner is only used for _run_fen_pool().")


def _evaluate_candidate_fen_chunk_worker(
    worker_index: int,
    chunk: list[str],
    *,
    depth: int,
    num_nodes: int | None,
    progress_shm_name: str,
) -> dict[str, dict[str, Any] | None]:
    os.environ.setdefault("ENVIRONMENT", "AGENT")
    tracker = _WorkerFenProgressTracker(progress_shm_name, worker_index)
    payloads: dict[str, dict[str, Any] | None] = {}
    try:
        with StockfishEngine(depth=depth) as engine:
            for index, fen in enumerate(chunk):
                evals = evaluate_all_legal_after(engine, fen, num_nodes=num_nodes)
                payloads[fen] = (
                    build_candidate_payload(evals, depth=depth, num_nodes=num_nodes)
                    if evals
                    else None
                )
                tracker.maybe_update(index + 1)
        tracker.finalize(len(chunk))
        return payloads
    finally:
        tracker.close()


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
        if log_progress_percent is not None and not 1 <= log_progress_percent <= 100:
            raise ValueError("log_progress_percent must be between 1 and 100, or None")
        if checkpoint_percent is not None and not 1 <= checkpoint_percent <= 100:
            raise ValueError("checkpoint_percent must be between 1 and 100, or None")
        self.depth = depth
        self.num_nodes = num_nodes
        self.log_progress_percent = log_progress_percent
        self.checkpoint_percent = (
            _checkpoint_percent_from_env() if checkpoint_percent is None else checkpoint_percent
        )
        self.n_workers = n_workers if n_workers is not None else _default_fen_eval_workers()
        self._pool = _FenPoolRunner(
            log_progress_percent=log_progress_percent,
            n_workers=self.n_workers,
        )
        self._db_client: DatabaseClient | None = None
        self._table_metadata: TableMetadata | None = None
        self._merge_strategy: MergeStrategy | None = None
        self._fen_to_moves: dict[str, list[_MoveRef]] = {}
        self._checkpointed_fens: set[str] = set()

    def bind_checkpoint(
        self,
        *,
        db_client: DatabaseClient,
        table_metadata: TableMetadata,
    ) -> None:
        """Enable periodic partial MERGE of ``candidate_evaluations`` during long runs."""
        self._db_client = db_client
        self._table_metadata = table_metadata
        self._merge_strategy = _CHECKPOINT_MERGE

    def _report_fen_progress(self, completed: int, total: int) -> None:
        if not is_parent_process():
            return
        percent = (completed * 100) // total if total > 0 else 100
        _logger.info(
            "%s: %d / %d FENs evaluated (%d%%).",
            type(self).__name__,
            completed,
            total,
            percent,
        )

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
        if (
            self._db_client is None
            or self._table_metadata is None
            or self._merge_strategy is None
            or self.checkpoint_percent is None
        ):
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
        self._db_client.merge(df, self._table_metadata, strategy=self._merge_strategy)
        self._checkpointed_fens.update(new_payloads)
        _logger.info(
            "%s: checkpointed %d move row(s) for %d FEN(s).",
            type(self).__name__,
            len(rows),
            len(new_payloads),
        )

    def _evaluate_unique_fens(self, unique_fens: list[str]) -> dict[str, dict[str, Any] | None]:
        total_fens = len(unique_fens)
        use_parallel = self.n_workers > 1 and total_fens >= _MIN_PARALLEL_FENS
        on_batch: Callable[[dict[str, dict[str, Any] | None]], None] | None = None
        if self._db_client is not None:
            on_batch = self._maybe_checkpoint_batch

        if use_parallel:
            _logger.info(
                "%s: parallel candidate eval (%d unique FEN(s), %d worker(s)).",
                type(self).__name__,
                total_fens,
                self.n_workers,
            )

            def submit_chunk(
                executor: ProcessPoolExecutor,
                worker_index: int,
                chunk: list[str],
                progress_shm_name: str,
            ) -> Future[dict[str, dict[str, Any] | None]]:
                return executor.submit(
                    _evaluate_candidate_fen_chunk_worker,
                    worker_index,
                    chunk,
                    depth=self.depth,
                    num_nodes=self.num_nodes,
                    progress_shm_name=progress_shm_name,
                )

            return self._pool._run_fen_pool(unique_fens, submit_chunk, on_scores_batch=on_batch)

        _logger.info(
            "%s: serial candidate eval (%d unique FEN(s)).",
            type(self).__name__,
            total_fens,
        )
        payloads: dict[str, dict[str, Any] | None] = {}
        last_logged_percent = 0
        report: Callable[[int, int], None] | None = (
            self._report_fen_progress if self.log_progress_percent is not None else None
        )
        pending_batch: dict[str, dict[str, Any] | None] = {}
        last_checkpoint_percent = 0
        with StockfishEngine(depth=self.depth) as engine:
            for index, fen in enumerate(unique_fens):
                evals = evaluate_all_legal_after(engine, fen, num_nodes=self.num_nodes)
                payload = (
                    build_candidate_payload(
                        evals,
                        depth=self.depth,
                        num_nodes=self.num_nodes,
                    )
                    if evals
                    else None
                )
                payloads[fen] = payload
                pending_batch[fen] = payload
                completed = index + 1
                if on_batch is not None and self.checkpoint_percent is not None:
                    current_percent = (completed * 100) // total_fens
                    if (
                        current_percent >= last_checkpoint_percent + self.checkpoint_percent
                        or completed >= total_fens
                    ):
                        while current_percent >= last_checkpoint_percent + self.checkpoint_percent:
                            last_checkpoint_percent += self.checkpoint_percent
                        on_batch(pending_batch)
                        pending_batch = {}
                last_logged_percent = _advance_fen_progress(
                    completed=completed,
                    total=total_fens,
                    progress_percent=self.log_progress_percent,
                    last_logged_percent=last_logged_percent,
                    report=report,
                )
        if on_batch is not None and pending_batch:
            on_batch(pending_batch)
        return payloads

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return df.with_columns(pl.lit(None).alias("candidate_evaluations"))

        move_ids = df["move_id"].cast(pl.Utf8).to_list()
        game_ids = df["game_id"].cast(pl.Utf8).to_list()
        account_ids = df["account_id"].cast(pl.Utf8).to_list()
        fens_before = df["fen_before"].cast(pl.Utf8).to_list()

        fen_to_moves: dict[str, list[_MoveRef]] = defaultdict(list)
        for move_id, game_id, account_id, fen in zip(
            move_ids,
            game_ids,
            account_ids,
            fens_before,
            strict=True,
        ):
            fen_to_moves[fen].append(_MoveRef(move_id, game_id, account_id))

        self._fen_to_moves = dict(fen_to_moves)
        self._checkpointed_fens = set()

        unique_fens = list(fen_to_moves.keys())
        fen_payloads = self._evaluate_unique_fens(unique_fens)

        candidate_evaluations = [fen_payloads.get(fen) for fen in fens_before]
        return df.with_columns(pl.Series("candidate_evaluations", candidate_evaluations))
