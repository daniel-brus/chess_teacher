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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import polars as pl

from chess_teacher.utils.exception_utils import TransformationError
from chess_teacher.utils.pipeline_utils.dataframe_transformation import DataFrameTransformation
from chess_teacher.utils.process_utils import (
    WORKER_NO_OP_LOGGER,
    _WorkerNoOpLogger,
    is_parent_process,
)

if TYPE_CHECKING:
    from chess_teacher.utils.logging.logger import EnhancedLogger

_MIN_PARALLEL_FENS = 100
_DEFAULT_LOG_PROGRESS_PERCENT = 5
_PROGRESS_INT_BYTES = struct.calcsize("i")
_PROGRESS_UPDATE_EVERY = 25

_logger: EnhancedLogger | _WorkerNoOpLogger | None = None


def _log() -> EnhancedLogger | _WorkerNoOpLogger:
    global _logger
    if not is_parent_process():
        return WORKER_NO_OP_LOGGER
    if _logger is None:
        from chess_teacher.utils.logging import get_logger

        _logger = get_logger(__name__)
    return _logger


def _default_fen_eval_workers() -> int:
    if env := os.getenv("STOCKFISH_WORKERS"):
        return max(1, int(env))
    if hasattr(os, "sched_getaffinity"):
        cpu_count = len(os.sched_getaffinity(0))
    else:
        cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _split_fen_list(fens: list[str], n_chunks: int) -> list[list[str]]:
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
        self._offset = worker_index * _PROGRESS_INT_BYTES
        self._last_written = 0

    def maybe_update(self, completed: int) -> None:
        if completed <= self._last_written:
            return
        if completed - self._last_written < _PROGRESS_UPDATE_EVERY and completed != 0:
            return
        struct.pack_into("i", self._shm.buf, self._offset, completed)
        self._last_written = completed

    def finalize(self, completed: int) -> None:
        if completed > self._last_written:
            struct.pack_into("i", self._shm.buf, self._offset, completed)
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
    ) -> None:
        if log_progress_percent is not None and not 1 <= log_progress_percent <= 100:
            raise ValueError("log_progress_percent must be between 1 and 100, or None")
        if n_workers is not None and n_workers <= 0:
            raise ValueError("n_workers must be a positive int or None")
        self.log_progress_percent = log_progress_percent
        self.n_workers = n_workers if n_workers is not None else _default_fen_eval_workers()
        self._worker_init_kwargs: dict[str, object] = {}

    def after_column(self) -> str:
        return f"{self.characteristic_name}_after"

    def delta_column(self) -> str:
        return f"{self.characteristic_name}_delta"

    @abstractmethod
    def evaluate(self, fen: str, *, row: dict[str, object]) -> float:
        """Map a FEN to a numeric characteristic value."""

    @contextmanager
    def _evaluation_context(self) -> Iterator[None]:
        """Optional setup for batched evaluation (e.g. a reused Stockfish engine)."""
        yield

    def _report_fen_progress(self, completed: int, total: int) -> None:
        if not is_parent_process():
            return
        percent = _completed_percent(completed, total)
        _log().info(
            "%s: %d / %d FENs evaluated (%d%%).",
            type(self).__name__,
            completed,
            total,
            percent,
        )

    def _evaluate_fens_serial(self, unique_fens: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        total_fens = len(unique_fens)
        last_logged_percent = 0
        report: Callable[[int, int], None] | None = (
            self._report_fen_progress if self.log_progress_percent is not None else None
        )
        with self._evaluation_context():
            for index, fen in enumerate(unique_fens):
                scores[fen] = self.evaluate(fen, row={})
                last_logged_percent = _advance_fen_progress(
                    completed=index + 1,
                    total=total_fens,
                    progress_percent=self.log_progress_percent,
                    last_logged_percent=last_logged_percent,
                    report=report,
                )
        return scores

    def _evaluate_fens_parallel(self, unique_fens: list[str]) -> dict[str, float]:
        chunks = _split_fen_list(unique_fens, self.n_workers)
        transformation_cls = type(self)
        n_chunks = len(chunks)
        chunk_size = len(chunks[0]) if chunks else 0
        progress_shm = shared_memory.SharedMemory(create=True, size=n_chunks * _PROGRESS_INT_BYTES)
        progress_shm_name = progress_shm.name

        _log().info(
            "%s: starting ProcessPoolExecutor with %d worker(s) for %d unique FEN(s) "
            "(%d FENs per chunk).",
            type(self).__name__,
            n_chunks,
            len(unique_fens),
            chunk_size,
        )

        try:
            scores: dict[str, float] = {}
            total_fens = len(unique_fens)
            with ProcessPoolExecutor(max_workers=n_chunks) as executor:
                futures = [
                    executor.submit(
                        _evaluate_fen_chunk_worker,
                        worker_index,
                        transformation_cls,
                        self._worker_init_kwargs,
                        chunk,
                        progress_shm_name,
                    )
                    for worker_index, chunk in enumerate(chunks)
                ]
                pending = set(futures)
                completed_chunks = 0
                last_heartbeat = time.monotonic()
                while pending:
                    done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
                    completed_fens = _sum_shared_fen_progress(progress_shm.buf, n_chunks)
                    for future in done:
                        scores.update(future.result())
                        completed_chunks += 1
                        _log().info(
                            "%s: worker chunk finished (%d / %d chunks complete, %d scores so far).",
                            type(self).__name__,
                            completed_chunks,
                            n_chunks,
                            len(scores),
                        )
                    now = time.monotonic()
                    if now - last_heartbeat >= 60.0:
                        percent = _completed_percent(completed_fens, total_fens)
                        _log().info(
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

            _log().info(
                "%s: ProcessPoolExecutor finished (%d unique FEN scores).",
                type(self).__name__,
                len(scores),
            )
            return scores
        finally:
            progress_shm.close()
            progress_shm.unlink()

    def _evaluate_unique_fens(self, unique_fens: list[str]) -> dict[str, float]:
        total_fens = len(unique_fens)
        use_parallel = self.n_workers > 1 and total_fens >= _MIN_PARALLEL_FENS

        if use_parallel:
            _log().info(
                "%s: using parallel evaluation (%d unique FEN(s), %d ProcessPool worker(s), "
                "heartbeat every 60s).",
                type(self).__name__,
                total_fens,
                self.n_workers,
            )
            return self._evaluate_fens_parallel(unique_fens)

        _log().info(
            "%s: using serial evaluation (%d unique FEN(s), progress every %s%%).",
            type(self).__name__,
            total_fens,
            self.log_progress_percent if self.log_progress_percent is not None else "off",
        )
        return self._evaluate_fens_serial(unique_fens)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = [column for column in ("fen_before", "fen_after") if column not in df.columns]
        if missing:
            _log().log_and_raise(
                TransformationError(f"FenCharacteristicTransformation requires columns {missing}.")
            )
        if df.height == 0:
            return df

        after_column = self.after_column()
        delta_column = self.delta_column()
        fen_before = df["fen_before"].cast(pl.Utf8)
        fen_after = df["fen_after"].cast(pl.Utf8)

        try:
            unique_fens = pl.concat([fen_before, fen_after]).unique().to_list()
            _log().info(
                "%s: collected %d unique FEN(s) from %d row(s).",
                type(self).__name__,
                len(unique_fens),
                df.height,
            )
            scores = self._evaluate_unique_fens(unique_fens)
        except TransformationError:
            raise
        except Exception as e:
            _log().log_and_raise(
                TransformationError(
                    f"Failed to compute {self.characteristic_name} characteristic: {e}"
                )
            )

        before_values = [scores[str(fen)] for fen in fen_before.to_list()]
        after_values = [scores[str(fen)] for fen in fen_after.to_list()]
        delta_values = [
            after - before for before, after in zip(before_values, after_values, strict=True)
        ]

        return df.with_columns(
            pl.Series(after_column, after_values, dtype=pl.Float64),
            pl.Series(delta_column, delta_values, dtype=pl.Float64),
        )


def _evaluate_fen_chunk_worker(
    worker_index: int,
    transformation_cls: type[FenCharacteristicTransformation],
    init_kwargs: dict[str, object],
    fens: list[str],
    progress_shm_name: str,
) -> dict[str, float]:
    instance = transformation_cls(**init_kwargs)
    scores: dict[str, float] = {}
    total_fens = len(fens)
    progress_tracker = _WorkerFenProgressTracker(progress_shm_name, worker_index)
    try:
        with instance._evaluation_context():
            for index, fen in enumerate(fens):
                scores[fen] = instance.evaluate(fen, row={})
                progress_tracker.maybe_update(index + 1)
    finally:
        progress_tracker.finalize(total_fens)
        progress_tracker.close()
    return scores
