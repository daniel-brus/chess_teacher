"""Single door for Stockfish scalar + MultiPV evals."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from chess_teacher.pipelines.fen_eval_cache.keys import position_key
from chess_teacher.pipelines.fen_eval_cache.payload import build_candidate_payload, payload_evals
from chess_teacher.pipelines.fen_eval_cache.store import (
    EvalStore,
    MemoryEvalStore,
    PostgresEvalStore,
    StoredEval,
)
from chess_teacher.utils.chess_utils import StockfishEngine
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.env_utils import get_optional_env_variable
from chess_teacher.utils.logging import get_logger

logger = get_logger()

DEFAULT_ENGINE_NAME = "stockfish"
DEFAULT_EVAL_DEPTH = 12
DEFAULT_CANDIDATE_NODES = 50_000
DEFAULT_MAX_PLY = 32
DEFAULT_CAPACITY = 10_000
_UNSET = object()
_service: PositionEvalService | object | None = _UNSET


@dataclass(frozen=True, slots=True)
class FenEvalRequest:
    fen: str
    ply: int | None = None
    eval_depth: int = DEFAULT_EVAL_DEPTH
    candidate_nodes: int = 0


@dataclass(frozen=True, slots=True)
class PositionEval:
    epd: str
    eval_white_pov: float
    eval_depth: int
    candidates: dict[str, Any]
    candidate_nodes: int

    @property
    def candidate_evals(self) -> dict[str, float]:
        return payload_evals(self.candidates)


def cache_max_ply() -> int:
    raw = get_optional_env_variable("FEN_EVAL_CACHE_MAX_PLY")
    if not raw:
        return DEFAULT_MAX_PLY
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid FEN_EVAL_CACHE_MAX_PLY=%r; using %s", raw, DEFAULT_MAX_PLY)
        return DEFAULT_MAX_PLY


def cache_capacity() -> int:
    raw = get_optional_env_variable("FEN_EVAL_CACHE_CAPACITY")
    if not raw:
        return DEFAULT_CAPACITY
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid FEN_EVAL_CACHE_CAPACITY=%r; using %s", raw, DEFAULT_CAPACITY)
        return DEFAULT_CAPACITY


def cache_disabled() -> bool:
    raw = (get_optional_env_variable("FEN_EVAL_CACHE") or "readwrite").strip().lower()
    return raw in {"off", "0", "false", "no"}


class PositionEvalService:
    """Look up / compute / remember EPD evals. Application code must use this."""

    def __init__(
        self,
        *,
        store: EvalStore | None = None,
        compute_scalar: Callable[[str], float | None] | None = None,
        compute_candidates: Callable[[str, int], dict[str, float]] | None = None,
        max_ply: int | None = None,
        capacity: int | None = None,
        engine_name: str = DEFAULT_ENGINE_NAME,
        n_workers: int = 1,
        stockfish_path: str | None = None,
    ) -> None:
        self._memory = MemoryEvalStore()
        self._store = store
        self._compute_scalar = compute_scalar
        self._compute_candidates = compute_candidates
        self.max_ply = cache_max_ply() if max_ply is None else max_ply
        self.capacity = cache_capacity() if capacity is None else capacity
        self.engine_name = engine_name
        self.n_workers = max(1, n_workers)
        self.stockfish_path = stockfish_path

    def evaluate(
        self,
        fen: str,
        ply: int | None = None,
        *,
        eval_depth: int = DEFAULT_EVAL_DEPTH,
        candidate_nodes: int = DEFAULT_CANDIDATE_NODES,
    ) -> PositionEval:
        """Pass a FEN (or EPD); we strip clocks to the EPD key.

        ``ply=None`` still computes, but does not insert (same as ply > cap).
        Depth / nodes default to the pipeline budget.
        """
        results = self.evaluate_many([
            FenEvalRequest(fen, ply=ply, eval_depth=eval_depth, candidate_nodes=candidate_nodes)
        ])
        epd = position_key(fen)
        result = results.get(epd)
        if result is None:
            raise ValueError(f"No evaluation for FEN: {fen!r}")
        return result

    def evaluate_many(self, requests: Sequence[FenEvalRequest]) -> dict[str, PositionEval]:
        grouped = _group_requests(requests)
        if not grouped:
            return {}

        epds = list(grouped)
        memory_hits = self._memory.get_many(epds)
        durable_hits: dict[str, StoredEval] = {}
        missing_for_durable = [epd for epd in epds if epd not in memory_hits]
        store = self._require_store()
        if store is not None and missing_for_durable:
            try:
                durable_hits = store.get_many(missing_for_durable)
            except Exception:
                logger.warning("Position eval cache read failed; computing", exc_info=True)
                durable_hits = {}

        out: dict[str, PositionEval] = {}
        touch_epds: list[str] = []
        to_compute: list[_ComputeJob] = []

        for epd, job in grouped.items():
            stored = memory_hits.get(epd) or durable_hits.get(epd)
            if stored is None:
                to_compute.append(job)
                continue
            need_scalar = stored.eval_depth < job.eval_depth
            need_candidates = stored.candidate_nodes < job.candidate_nodes
            if not need_scalar and not need_candidates:
                result = _from_stored(stored)
                out[epd] = result
                touch_epds.append(epd)
                continue
            to_compute.append(
                _ComputeJob(
                    epd=epd,
                    fen=job.fen,
                    ply=job.ply,
                    eval_depth=job.eval_depth,
                    candidate_nodes=job.candidate_nodes,
                    need_scalar=need_scalar,
                    need_candidates=need_candidates,
                    existing=stored,
                )
            )

        if touch_epds:
            self._memory.touch(touch_epds)
            if store is not None:
                try:
                    store.touch(touch_epds)
                except Exception:
                    logger.warning("Position eval cache touch failed", exc_info=True)

        computed = self._compute_jobs(to_compute)
        writes: list[StoredEval] = []
        for job in to_compute:
            computed_result = computed.get(job.epd)
            if computed_result is None:
                continue
            out[job.epd] = computed_result
            stored = _to_stored(computed_result, engine=self.engine_name)
            self._memory.upsert_many([stored])
            if _should_persist(job.ply, self.max_ply, existing=job.existing is not None):
                writes.append(stored)

        if writes and store is not None:
            try:
                store.upsert_many(writes)
                store.evict(self.capacity)
            except Exception:
                logger.warning("Position eval cache write failed", exc_info=True)
        self._memory.evict(self.capacity)
        return out

    def _require_store(self) -> EvalStore | None:
        if cache_disabled():
            return None
        if self._store is not None:
            return self._store
        try:
            self._store = PostgresEvalStore(get_db_client())
        except Exception:
            logger.warning(
                "Position eval cache durable store unavailable; memory only",
                exc_info=True,
            )
            self._store = None
        return self._store

    def _compute_jobs(self, jobs: list[_ComputeJob]) -> dict[str, PositionEval]:
        if not jobs:
            return {}
        if self.n_workers > 1 and len(jobs) >= 2 and self._compute_scalar is None:
            return self._compute_jobs_parallel(jobs)
        return self._compute_jobs_serial(jobs)

    def _compute_jobs_serial(self, jobs: list[_ComputeJob]) -> dict[str, PositionEval]:
        out: dict[str, PositionEval] = {}
        if self._compute_scalar is not None or self._compute_candidates is not None:
            for job in jobs:
                result = self._compute_one_injected(job)
                if result is not None:
                    out[job.epd] = result
            return out

        with StockfishEngine(depth=_max_depth(jobs), path=self.stockfish_path) as engine:
            for job in jobs:
                result = _compute_with_engine(engine, job)
                if result is not None:
                    out[job.epd] = result
        return out

    def _compute_jobs_parallel(self, jobs: list[_ComputeJob]) -> dict[str, PositionEval]:
        chunks = _split_jobs(jobs, self.n_workers)
        out: dict[str, PositionEval] = {}
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [
                pool.submit(
                    _compute_job_chunk_worker,
                    chunk,
                    self.stockfish_path,
                )
                for chunk in chunks
            ]
            for future in futures:
                out.update(future.result())
        return out

    def _compute_one_injected(self, job: _ComputeJob) -> PositionEval | None:
        existing = job.existing
        scalar = existing.eval_white_pov if existing is not None and not job.need_scalar else None
        depth = (
            existing.eval_depth if existing is not None and not job.need_scalar else job.eval_depth
        )
        payload = existing.candidates if existing is not None and not job.need_candidates else None
        nodes = (
            existing.candidate_nodes
            if existing is not None and not job.need_candidates
            else job.candidate_nodes
        )
        if job.need_scalar:
            if self._compute_scalar is None:
                return None
            computed = self._compute_scalar(job.fen)
            if computed is None:
                return None
            scalar = computed
            depth = job.eval_depth
        if job.need_candidates:
            if self._compute_candidates is None:
                return None
            evals = self._compute_candidates(job.fen, job.candidate_nodes)
            payload = build_candidate_payload(
                evals, depth=job.eval_depth, num_nodes=job.candidate_nodes
            )
            nodes = job.candidate_nodes
        if scalar is None or payload is None:
            return None
        return PositionEval(
            epd=job.epd,
            eval_white_pov=float(scalar),
            eval_depth=int(depth),
            candidates=payload,
            candidate_nodes=int(nodes),
        )


@dataclass(frozen=True, slots=True)
class _ComputeJob:
    epd: str
    fen: str
    ply: int | None
    eval_depth: int
    candidate_nodes: int
    need_scalar: bool = True
    need_candidates: bool = True
    existing: StoredEval | None = None


def _group_requests(requests: Sequence[FenEvalRequest]) -> dict[str, _ComputeJob]:
    """One job per EPD. Ply is the minimum seen (store if any ply <= cap)."""
    grouped: dict[str, _ComputeJob] = {}
    for request in requests:
        try:
            epd = position_key(request.fen)
        except ValueError:
            logger.warning("Invalid FEN for position eval: %s", request.fen)
            continue
        current = grouped.get(epd)
        ply = request.ply
        if current is not None:
            if ply is None:
                ply = current.ply
            elif current.ply is not None:
                ply = min(ply, current.ply)
            grouped[epd] = _ComputeJob(
                epd=epd,
                fen=current.fen,
                ply=ply,
                eval_depth=max(current.eval_depth, request.eval_depth),
                candidate_nodes=max(current.candidate_nodes, request.candidate_nodes),
            )
        else:
            grouped[epd] = _ComputeJob(
                epd=epd,
                fen=request.fen,
                ply=ply,
                eval_depth=request.eval_depth,
                candidate_nodes=request.candidate_nodes,
            )
    return grouped


def _should_persist(ply: int | None, max_ply: int, *, existing: bool) -> bool:
    """Insert a new row if this EPD was seen at ply <= cap (min ply of the batch)."""
    if existing:
        return True
    if ply is None:
        return False
    return ply <= max_ply


def _from_stored(row: StoredEval) -> PositionEval:
    return PositionEval(
        epd=row.epd,
        eval_white_pov=row.eval_white_pov,
        eval_depth=row.eval_depth,
        candidates=row.candidates,
        candidate_nodes=row.candidate_nodes,
    )


def _to_stored(result: PositionEval, *, engine: str) -> StoredEval:
    return StoredEval(
        epd=result.epd,
        engine=engine,
        eval_white_pov=result.eval_white_pov,
        eval_depth=result.eval_depth,
        candidates=result.candidates,
        candidate_nodes=result.candidate_nodes,
    )


def _max_depth(jobs: Sequence[_ComputeJob]) -> int:
    return max((job.eval_depth for job in jobs), default=DEFAULT_EVAL_DEPTH)


def _split_jobs(jobs: list[_ComputeJob], n_chunks: int) -> list[list[_ComputeJob]]:
    if n_chunks <= 1 or not jobs:
        return [jobs]
    n_chunks = min(n_chunks, len(jobs))
    size = (len(jobs) + n_chunks - 1) // n_chunks
    return [jobs[index : index + size] for index in range(0, len(jobs), size)]


def _compute_with_engine(engine: StockfishEngine, job: _ComputeJob) -> PositionEval | None:
    existing = job.existing
    scalar = existing.eval_white_pov if existing is not None and not job.need_scalar else None
    depth = existing.eval_depth if existing is not None and not job.need_scalar else job.eval_depth
    payload = existing.candidates if existing is not None and not job.need_candidates else None
    nodes = (
        existing.candidate_nodes
        if existing is not None and not job.need_candidates
        else job.candidate_nodes
    )
    if job.need_scalar:
        computed = engine.evaluate_white_pov_pawns(job.fen)
        if computed is None:
            return None
        scalar = computed
        depth = job.eval_depth
    if job.need_candidates:
        evals = engine.evaluate_all_legal_moves_white_pov(job.fen, num_nodes=job.candidate_nodes)
        payload = build_candidate_payload(
            evals, depth=job.eval_depth, num_nodes=job.candidate_nodes
        )
        nodes = job.candidate_nodes
    if scalar is None or payload is None:
        return None
    return PositionEval(
        epd=job.epd,
        eval_white_pov=float(scalar),
        eval_depth=int(depth),
        candidates=payload,
        candidate_nodes=int(nodes),
    )


def _compute_job_chunk_worker(
    jobs: list[_ComputeJob],
    stockfish_path: str | None,
) -> dict[str, PositionEval]:
    out: dict[str, PositionEval] = {}
    with StockfishEngine(depth=_max_depth(jobs), path=stockfish_path) as engine:
        for job in jobs:
            result = _compute_with_engine(engine, job)
            if result is not None:
                out[job.epd] = result
    return out


def _default_workers() -> int:
    if env := get_optional_env_variable("STOCKFISH_WORKERS"):
        return max(1, int(env))
    if hasattr(os, "sched_getaffinity"):
        cpu_count = len(os.sched_getaffinity(0))
    else:
        cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def get_position_eval_service() -> PositionEvalService:
    global _service
    if _service is not _UNSET and _service is not None:
        return _service  # type: ignore[return-value]
    _service = PositionEvalService(n_workers=_default_workers())
    return _service


def reset_position_eval_service_for_tests(service: PositionEvalService | None = None) -> None:
    global _service
    _service = service if service is not None else _UNSET
