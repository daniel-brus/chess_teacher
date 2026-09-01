"""FEN-based move characteristics with optional parallel evaluation.

Kept separate from ``transformations.py`` so ProcessPool workers on Windows do
not import the full preprocessing stack (SQLAlchemy, logging buffer, etc.).
"""

from __future__ import annotations

import os
import struct
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, TypeVar, cast

import polars as pl

from chess_teacher.pipelines.preprocessing.fen_checkpoint import (
    CHECKPOINT_MERGE,
    checkpoint_percent_from_env,
)
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.env_utils import get_optional_env_variable
from chess_teacher.utils.exception_utils import TransformationError
from chess_teacher.utils.metadata_utils import TableMetadata
from chess_teacher.utils.pipeline_utils.dataframe_transformation import DataFrameTransformation
from chess_teacher.utils.process_utils import WorkerSafeLogger, is_parent_process

_MIN_PARALLEL_FENS = 100
_DEFAULT_LOG_PROGRESS_PERCENT = 5
_PROGRESS_INT_BYTES = struct.calcsize("i")
_PROGRESS_UPDATE_EVERY = 1

TScore = TypeVar("TScore")

_logger = WorkerSafeLogger(__name__)


@dataclass(frozen=True, slots=True)
class _CheckpointMove:
    move_id: str
    game_id: str
    account_id: str
    fen_before: str
    fen_after: str


def _default_fen_eval_workers() -> int:
    if env := get_optional_env_variable("STOCKFISH_WORKERS"):
        return max(1, int(env))
    if hasattr(os, "sched_getaffinity"):
        cpu_count = len(os.sched_getaffinity(0))
    else:
        cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _split_fen_list[TItem](fens: list[TItem], n_chunks: int) -> list[list[TItem]]:
    if n_chunks <= 1 or not fens:
        return [fens]
    n_chunks = min(n_chunks, len(fens))
    chunk_size = (len(fens) + n_chunks - 1) // n_chunks
    return [fens[index : index + chunk_size] for index in range(0, len(fens), chunk_size)]


def _completed_percent(completed: int, total: int) -> int:
    if total <= 0:
        return 100
    if completed >= total:
        return 100
    return (completed * 100) // total


def _advance_fen_progress(
    *,
    completed: int,
    total: int,
    progress_percent: int | None,
    last_logged_percent: int,
    report: Callable[[int, int], None] | None,
) -> int:
    """Emit aggregate progress at each ``progress_percent`` step; return updated last logged percent."""
    if progress_percent is None or report is None or total <= 0:
        return last_logged_percent

    current_percent = _completed_percent(completed, total)
    next_threshold = last_logged_percent + progress_percent
    while current_percent >= next_threshold and next_threshold <= 100:
        report(completed, total)
        last_logged_percent = next_threshold
        next_threshold += progress_percent

    if completed >= total and last_logged_percent < 100:
        report(completed, total)
        return 100
    return last_logged_percent


class _WorkerFenProgressTracker:
    """Write completed FEN counts for one worker into shared memory."""

    def __init__(self, progress_shm_name: str, worker_index: int) -> None:
        self._shm = shared_memory.SharedMemory(name=progress_shm_name)
        if self._shm.buf is None:
            raise RuntimeError("Shared memory progress buffer is unavailable.")
        self._buf: memoryview = self._shm.buf
        self._offset = worker_index * _PROGRESS_INT_BYTES
        self._last_written = 0

    def maybe_update(self, completed: int) -> None:
        if completed <= self._last_written:
            return
        if completed - self._last_written < _PROGRESS_UPDATE_EVERY and completed != 0:
            return
        struct.pack_into("i", self._buf, self._offset, completed)
        self._last_written = completed

    def finalize(self, completed: int) -> None:
        if completed > self._last_written:
            struct.pack_into("i", self._buf, self._offset, completed)
            self._last_written = completed

    def close(self) -> None:
        self._shm.close()


def _sum_shared_fen_progress(progress_buf: memoryview, n_workers: int) -> int:
    return sum(
        struct.unpack_from("i", progress_buf, index * _PROGRESS_INT_BYTES)[0]
        for index in range(n_workers)
    )


class FenCharacteristicTransformation(DataFrameTransformation, ABC):
    """
    Derive ``{name}_after`` and ``{name}_delta`` from ``fen_before`` and ``fen_after``.

    ``_after`` is the evaluation of ``fen_after``; ``_delta`` is after minus before.
    """

    characteristic_name: str

    def __init__(
        self,
        *,
        log_progress_percent: int | None = _DEFAULT_LOG_PROGRESS_PERCENT,
        n_workers: int | None = None,
        checkpoint_percent: int | None = None,
    ) -> None:
        if log_progress_percent is not None and not 1 <= log_progress_percent <= 100:
            raise ValueError("log_progress_percent must be between 1 and 100, or None")
        if n_workers is not None and n_workers <= 0:
            raise ValueError("n_workers must be a positive int or None")
        if checkpoint_percent is not None and not 1 <= checkpoint_percent <= 100:
            raise ValueError("checkpoint_percent must be between 1 and 100, or None")
        self.log_progress_percent = log_progress_percent
        self.n_workers = n_workers if n_workers is not None else _default_fen_eval_workers()
        self.checkpoint_percent = (
            checkpoint_percent_from_env("FEN_EVAL_CHECKPOINT_PERCENT")
            if checkpoint_percent is None
            else checkpoint_percent
        )
        self._worker_init_kwargs: dict[str, object] = {}
        self._db_client: DatabaseClient | None = None
        self._table_metadata: TableMetadata | None = None
        self._fen_scores: dict[str, float] = {}
        self._checkpoint_moves: list[_CheckpointMove] = []

    def bind_checkpoint(
        self,
        *,
        db_client: DatabaseClient,
        table_metadata: TableMetadata,
    ) -> None:
        self._db_client = db_client
        self._table_metadata = table_metadata

    def _checkpoint_enabled(self) -> bool:
        return (
            self._db_client is not None
            and self._table_metadata is not None
            and self.checkpoint_percent is not None
        )

    def _prepare_checkpoint_rows(self, df: pl.DataFrame) -> None:
        self._fen_scores = {}
        required = ("move_id", "game_id", "account_id", "fen_before", "fen_after")
        if any(column not in df.columns for column in required):
            self._checkpoint_moves = []
            return
        move_ids = df["move_id"].cast(pl.Utf8).to_list()
        game_ids = df["game_id"].cast(pl.Utf8).to_list()
        account_ids = df["account_id"].cast(pl.Utf8).to_list()
        fens_before = df["fen_before"].cast(pl.Utf8).to_list()
        fens_after = df["fen_after"].cast(pl.Utf8).to_list()
        self._checkpoint_moves = [
            _CheckpointMove(move_id, game_id, account_id, fen_before, fen_after)
            for move_id, game_id, account_id, fen_before, fen_after in zip(
                move_ids,
                game_ids,
                account_ids,
                fens_before,
                fens_after,
                strict=True,
            )
        ]

    def _maybe_checkpoint_fen_batch(self, new_scores: dict[str, float]) -> None:
        if not self._checkpoint_enabled() or not new_scores:
            return
        self._fen_scores.update(new_scores)
        touched_fens = set(new_scores)
        rows: list[dict[str, object]] = []
        for move in self._checkpoint_moves:
            if move.fen_before not in touched_fens and move.fen_after not in touched_fens:
                continue
            before = self._fen_scores.get(move.fen_before)
            after = self._fen_scores.get(move.fen_after)
            if before is None and after is None:
                continue
            row: dict[str, object] = {
                "move_id": move.move_id,
                "game_id": move.game_id,
                "account_id": move.account_id,
            }
            if before is not None:
                row[self.before_column()] = before
            if after is not None:
                row[self.after_column()] = after
            if before is not None and after is not None:
                row[self.delta_column()] = after - before
            rows.append(row)
        if not rows:
            return
        assert self._db_client is not None and self._table_metadata is not None
        self._db_client.merge(pl.DataFrame(rows), self._table_metadata, strategy=CHECKPOINT_MERGE)
        _logger.info(
            "%s: checkpointed %d move row(s) for %d FEN score(s).",
            type(self).__name__,
            len(rows),
            len(new_scores),
        )

    def before_column(self) -> str:
        return f"{self.characteristic_name}_before"

    def after_column(self) -> str:
        return f"{self.characteristic_name}_after"

    def delta_column(self) -> str:
        return f"{self.characteristic_name}_delta"

    @abstractmethod
    def evaluate(self, fen: str, *, row: dict[str, object]) -> float:
        """Map a FEN to a numeric characteristic value."""

    def _score_fen(self, fen: str) -> float:
        return self.evaluate(fen, row={})

    @contextmanager
    def _evaluation_context(self) -> Iterator[None]:
        """Optional setup for batched evaluation (e.g. a reused Stockfish engine)."""
        yield

    def _report_fen_progress(self, completed: int, total: int) -> None:
        if not is_parent_process():
            return
        percent = _completed_percent(completed, total)
        _logger.info(
            "%s: %d / %d FENs evaluated (%d%%).",
            type(self).__name__,
            completed,
            total,
            percent,
        )

    def _collect_unique_fens(self, df: pl.DataFrame) -> tuple[list[str], list[str], list[str]]:
        fen_before = df["fen_before"].cast(pl.Utf8).to_list()
        fen_after = df["fen_after"].cast(pl.Utf8).to_list()
        unique_fens = pl.concat([df["fen_before"], df["fen_after"]]).unique().to_list()
        return unique_fens, [str(fen) for fen in fen_before], [str(fen) for fen in fen_after]

    def _evaluate_fens_serial(self, unique_fens: list[str]) -> dict[str, float]:
        on_batch = self._maybe_checkpoint_fen_batch if self._checkpoint_enabled() else None
        return self._evaluate_fens_serial_with(unique_fens, self._score_fen, on_scores_batch=on_batch)

    def _evaluate_fens_serial_with(
        self,
        unique_fens: list[str],
        score_fn: Callable[[str], TScore],
        *,
        on_scores_batch: Callable[[dict[str, TScore]], None] | None = None,
    ) -> dict[str, TScore]:
        scores: dict[str, TScore] = {}
        total_fens = len(unique_fens)
        last_logged_percent = 0
        report: Callable[[int, int], None] | None = (
            self._report_fen_progress if self.log_progress_percent is not None else None
        )
        pending_batch: dict[str, TScore] = {}
        last_checkpoint_percent = 0
        with self._evaluation_context():
            for index, fen in enumerate(unique_fens):
                score = score_fn(fen)
                scores[fen] = score
                if on_scores_batch is not None and self.checkpoint_percent is not None:
                    pending_batch[fen] = score
                    completed = index + 1
                    current_percent = (completed * 100) // total_fens if total_fens else 100
                    if (
                        current_percent >= last_checkpoint_percent + self.checkpoint_percent
                        or completed >= total_fens
                    ):
                        while current_percent >= last_checkpoint_percent + self.checkpoint_percent:
                            last_checkpoint_percent += self.checkpoint_percent
                        on_scores_batch(pending_batch)  # type: ignore[arg-type]
                        pending_batch = {}
                last_logged_percent = _advance_fen_progress(
                    completed=index + 1,
                    total=total_fens,
                    progress_percent=self.log_progress_percent,
                    last_logged_percent=last_logged_percent,
                    report=report,
                )
        if on_scores_batch is not None and pending_batch:
            on_scores_batch(pending_batch)  # type: ignore[arg-type]
        return scores

    def _run_fen_pool(
        self,
        unique_fens: list[str],
        submit_chunk: Callable[
            [ProcessPoolExecutor, int, list[str], str], Future[dict[str, TScore]]
        ],
        *,
        on_scores_batch: Callable[[dict[str, TScore]], None] | None = None,
    ) -> dict[str, TScore]:
        chunks = _split_fen_list(unique_fens, self.n_workers)
        n_chunks = len(chunks)
        chunk_size = len(chunks[0]) if chunks else 0
        progress_shm = shared_memory.SharedMemory(create=True, size=n_chunks * _PROGRESS_INT_BYTES)
        progress_shm_name = progress_shm.name
        if progress_shm.buf is None:
            progress_shm.close()
            progress_shm.unlink()
            raise RuntimeError("Shared memory progress buffer is unavailable.")
        progress_buf: memoryview = progress_shm.buf

        _logger.info(
            "%s: starting ProcessPoolExecutor with %d worker(s) for %d unique FEN(s) "
            "(%d FENs per chunk).",
            type(self).__name__,
            n_chunks,
            len(unique_fens),
            chunk_size,
        )

        try:
            scores: dict[str, TScore] = {}
            total_fens = len(unique_fens)
            with ProcessPoolExecutor(max_workers=n_chunks) as executor:
                futures = [
                    submit_chunk(executor, worker_index, chunk, progress_shm_name)
                    for worker_index, chunk in enumerate(chunks)
                ]
                pending = set(futures)
                completed_chunks = 0
                last_heartbeat = time.monotonic()
                while pending:
                    done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
                    completed_fens = _sum_shared_fen_progress(progress_buf, n_chunks)
                    for future in done:
                        chunk_scores = future.result()
                        scores.update(chunk_scores)
                        if on_scores_batch is not None:
                            on_scores_batch(chunk_scores)
                        completed_chunks += 1
                        _logger.info(
                            "%s: worker chunk finished (%d / %d chunks complete, %d scores so far).",
                            type(self).__name__,
                            completed_chunks,
                            n_chunks,
                            len(scores),
                        )
                    now = time.monotonic()
                    if now - last_heartbeat >= 60.0:
                        percent = _completed_percent(completed_fens, total_fens)
                        _logger.info(
                            "%s: %d / %d FENs evaluated (%d%%), "
                            "%d / %d chunk(s) complete, %d pending.",
                            type(self).__name__,
                            completed_fens,
                            total_fens,
                            percent,
                            completed_chunks,
                            n_chunks,
                            len(pending),
                        )
                        last_heartbeat = now

            _logger.info(
                "%s: ProcessPoolExecutor finished (%d unique FEN scores).",
                type(self).__name__,
                len(scores),
            )
            return scores
        finally:
            progress_shm.close()
            progress_shm.unlink()

    def _evaluate_fens_parallel(
        self,
        unique_fens: list[str],
        *,
        on_scores_batch: Callable[[dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        transformation_cls = type(self)

        def submit_chunk(
            executor: ProcessPoolExecutor,
            worker_index: int,
            chunk: list[str],
            progress_shm_name: str,
        ) -> Future[dict[str, float]]:
            return executor.submit(
                _evaluate_fen_chunk_worker,
                worker_index,
                transformation_cls,
                self._worker_init_kwargs,
                chunk,
                progress_shm_name,
            )

        return self._run_fen_pool(unique_fens, submit_chunk, on_scores_batch=on_scores_batch)

    def _evaluate_unique_fens(self, unique_fens: list[str]) -> dict[str, float]:
        on_batch = self._maybe_checkpoint_fen_batch if self._checkpoint_enabled() else None
        total_fens = len(unique_fens)
        use_parallel = self.n_workers > 1 and total_fens >= _MIN_PARALLEL_FENS

        if use_parallel:
            _logger.info(
                "%s: using parallel evaluation (%d unique FEN(s), %d ProcessPool worker(s), "
                "heartbeat every 60s).",
                type(self).__name__,
                total_fens,
                self.n_workers,
            )
            return self._evaluate_fens_parallel(unique_fens, on_scores_batch=on_batch)

        _logger.info(
            "%s: using serial evaluation (%d unique FEN(s), progress every %s%%).",
            type(self).__name__,
            total_fens,
            self.log_progress_percent if self.log_progress_percent is not None else "off",
        )
        return self._evaluate_fens_serial(unique_fens)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = [column for column in ("fen_before", "fen_after") if column not in df.columns]
        if missing:
            _logger.log_and_raise(
                TransformationError(f"FenCharacteristicTransformation requires columns {missing}.")
            )
        if df.height == 0:
            return df

        if self._db_client is not None:
            self._prepare_checkpoint_rows(df)

        after_column = self.after_column()
        delta_column = self.delta_column()
        before_column = self.before_column()

        try:
            unique_fens, before_fens, after_fens = self._collect_unique_fens(df)
            _logger.info(
                "%s: collected %d unique FEN(s) from %d row(s).",
                type(self).__name__,
                len(unique_fens),
                df.height,
            )
            scores = self._evaluate_unique_fens(unique_fens)
        except TransformationError:
            raise
        except Exception as e:
            _logger.log_and_raise(
                TransformationError(
                    f"Failed to compute {self.characteristic_name} characteristic: {e}"
                )
            )

        before_values = [scores[fen] for fen in before_fens]
        after_values = [scores[fen] for fen in after_fens]
        delta_values = [
            after - before for before, after in zip(before_values, after_values, strict=True)
        ]

        return df.with_columns(
            pl.Series(before_column, before_values, dtype=pl.Float64),
            pl.Series(after_column, after_values, dtype=pl.Float64),
            pl.Series(delta_column, delta_values, dtype=pl.Float64),
        )


class DualSidedFenCharacteristicTransformation(FenCharacteristicTransformation):
    """
    Derive white/black ``{name}_after`` and ``{name}_delta`` from ``fen_before`` / ``fen_after``.

    Reuses FEN deduplication, serial/parallel evaluation, and progress reporting from
    ``FenCharacteristicTransformation``.
    """

    @abstractmethod
    def evaluate_sides(self, fen: str, *, row: dict[str, object]) -> tuple[float, float]:
        """Return ``(white_value, black_value)`` for one FEN."""

    def evaluate(self, fen: str, *, row: dict[str, object]) -> float:
        del fen, row
        raise NotImplementedError(f"{type(self).__name__} uses evaluate_sides(), not evaluate().")

    def _score_fen_sides(self, fen: str) -> tuple[float, float]:
        return self.evaluate_sides(fen, row={})

    def white_after_column(self) -> str:
        return f"white_{self.characteristic_name}_after"

    def white_delta_column(self) -> str:
        return f"white_{self.characteristic_name}_delta"

    def black_after_column(self) -> str:
        return f"black_{self.characteristic_name}_after"

    def black_delta_column(self) -> str:
        return f"black_{self.characteristic_name}_delta"

    def _evaluate_fens_parallel_sides(
        self, unique_fens: list[str]
    ) -> dict[str, tuple[float, float]]:
        transformation_cls = type(self)

        def submit_chunk(
            executor: ProcessPoolExecutor,
            worker_index: int,
            chunk: list[str],
            progress_shm_name: str,
        ) -> Future[dict[str, tuple[float, float]]]:
            return executor.submit(
                _evaluate_dual_sided_fen_chunk_worker,
                worker_index,
                transformation_cls,
                self._worker_init_kwargs,
                chunk,
                progress_shm_name,
            )

        return self._run_fen_pool(unique_fens, submit_chunk)

    def _evaluate_unique_fens_sides(self, unique_fens: list[str]) -> dict[str, tuple[float, float]]:
        total_fens = len(unique_fens)
        use_parallel = self.n_workers > 1 and total_fens >= _MIN_PARALLEL_FENS

        if use_parallel:
            _logger.info(
                "%s: using parallel evaluation (%d unique FEN(s), %d ProcessPool worker(s), "
                "heartbeat every 60s).",
                type(self).__name__,
                total_fens,
                self.n_workers,
            )
            return self._evaluate_fens_parallel_sides(unique_fens)

        _logger.info(
            "%s: using serial evaluation (%d unique FEN(s), progress every %s%%).",
            type(self).__name__,
            total_fens,
            self.log_progress_percent if self.log_progress_percent is not None else "off",
        )
        return self._evaluate_fens_serial_with(unique_fens, self._score_fen_sides)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = [column for column in ("fen_before", "fen_after") if column not in df.columns]
        if missing:
            _logger.log_and_raise(
                TransformationError(
                    f"DualSidedFenCharacteristicTransformation requires columns {missing}."
                )
            )
        if df.height == 0:
            return df

        try:
            unique_fens, before_fens, after_fens = self._collect_unique_fens(df)
            _logger.info(
                "%s: collected %d unique FEN(s) from %d row(s).",
                type(self).__name__,
                len(unique_fens),
                df.height,
            )
            scores = self._evaluate_unique_fens_sides(unique_fens)
        except TransformationError:
            raise
        except Exception as e:
            _logger.log_and_raise(
                TransformationError(
                    f"Failed to compute {self.characteristic_name} characteristic: {e}"
                )
            )

        white_before = [scores[fen][0] for fen in before_fens]
        white_after = [scores[fen][0] for fen in after_fens]
        black_before = [scores[fen][1] for fen in before_fens]
        black_after = [scores[fen][1] for fen in after_fens]

        return df.with_columns(
            pl.Series(self.white_after_column(), white_after, dtype=pl.Float64),
            pl.Series(
                self.white_delta_column(),
                [after - before for before, after in zip(white_before, white_after, strict=True)],
                dtype=pl.Float64,
            ),
            pl.Series(self.black_after_column(), black_after, dtype=pl.Float64),
            pl.Series(
                self.black_delta_column(),
                [after - before for before, after in zip(black_before, black_after, strict=True)],
                dtype=pl.Float64,
            ),
        )


def _evaluate_fen_chunk_worker(
    worker_index: int,
    transformation_cls: type[FenCharacteristicTransformation],
    init_kwargs: dict[str, object],
    fens: list[str],
    progress_shm_name: str,
) -> dict[str, float]:
    instance = transformation_cls(**cast(Any, init_kwargs))
    scores: dict[str, float] = {}
    total_fens = len(fens)
    progress_tracker = _WorkerFenProgressTracker(progress_shm_name, worker_index)
    try:
        with instance._evaluation_context():
            for index, fen in enumerate(fens):
                scores[fen] = instance._score_fen(fen)
                progress_tracker.maybe_update(index + 1)
    finally:
        progress_tracker.finalize(total_fens)
        progress_tracker.close()
    return scores


def _evaluate_dual_sided_fen_chunk_worker(
    worker_index: int,
    transformation_cls: type[DualSidedFenCharacteristicTransformation],
    init_kwargs: dict[str, object],
    fens: list[str],
    progress_shm_name: str,
) -> dict[str, tuple[float, float]]:
    instance = transformation_cls(**cast(Any, init_kwargs))
    scores: dict[str, tuple[float, float]] = {}
    total_fens = len(fens)
    progress_tracker = _WorkerFenProgressTracker(progress_shm_name, worker_index)
    try:
        with instance._evaluation_context():
            for index, fen in enumerate(fens):
                scores[fen] = instance._score_fen_sides(fen)
                progress_tracker.maybe_update(index + 1)
    finally:
        progress_tracker.finalize(total_fens)
        progress_tracker.close()
    return scores
