"""Backfill ``move_characteristics.candidate_evaluations`` (MultiPV-all @ node budget).

Resumable: skips rows where ``candidate_evaluations IS NOT NULL``.

Reuses the fen-characteristic process-pool + %-progress helpers (same pattern as
``StockfishEvaluationTransformation``), but cannot subclass that transform:
those map unique FENs → scalar floats into before/after columns; this job maps
each ``move_id`` → MultiPV ``{uci: score}`` JSONB update.

Run::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/ops/backfill_candidate_evals.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import shared_memory
from typing import Any

from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_STOCKFISH_DEPTH,
    CANDIDATE_STOCKFISH_NODES,
    build_candidate_payload,
    evaluate_all_legal_after,
)
from chess_teacher.pipelines.preprocessing.fen_characteristic import (
    _PROGRESS_INT_BYTES,
    _advance_fen_progress,
    _completed_percent,
    _default_fen_eval_workers,
    _split_fen_list,
    _sum_shared_fen_progress,
    _WorkerFenProgressTracker,
)
from chess_teacher.pipelines.preprocessing.moves import MoveCharacteristics
from chess_teacher.utils.chess_utils import StockfishEngine
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.general_utils import quote_literal
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import (
    WorkerSafeLogger,
    log_script_runtime_context,
    run_script_main,
)

logger = get_logger()
_worker_logger = WorkerSafeLogger(__name__)

_DEFAULT_LOG_PROGRESS_PERCENT = 5


def _pending_rows(*, limit: int | None) -> list[dict[str, str]]:
    db = get_db_client()
    db.ensure_metadata(MoveCharacteristics.get_metadata())
    sql = """
        SELECT m.move_id AS move_id, m.fen_before AS fen_before
        FROM games.moves m
        INNER JOIN games.move_characteristics mc ON mc.move_id = m.move_id
        WHERE mc.candidate_evaluations IS NULL
        ORDER BY m.game_id ASC, m.move_nr ASC
    """
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    rows = db.engine.execute_parameterized_query(sql, {})
    return [{"move_id": str(r["move_id"]), "fen_before": str(r["fen_before"])} for r in rows]


def _count_pending() -> int:
    db = get_db_client()
    rows = db.engine.execute_parameterized_query(
        """
        SELECT COUNT(*) AS n
        FROM games.move_characteristics
        WHERE candidate_evaluations IS NULL
        """,
        {},
    )
    return int(rows[0]["n"]) if rows else 0


def _persist_payload(move_id: str, payload: dict[str, Any]) -> None:
    db = get_db_client()
    payload_sql = quote_literal(json.dumps(payload))
    mid_sql = quote_literal(move_id)
    sql = (
        "UPDATE games.move_characteristics\n"
        f"SET candidate_evaluations = {payload_sql}::jsonb\n"
        f"WHERE move_id = {mid_sql};"
    )
    db.engine.execute_write(sql, {})


def _report_progress(completed: int, total: int) -> None:
    percent = _completed_percent(completed, total)
    logger.info(
        "CandidateEvalBackfill: %d / %d moves evaluated (%d%%).",
        completed,
        total,
        percent,
    )


def _process_rows(
    rows: list[dict[str, str]],
    *,
    depth: int,
    num_nodes: int,
    log_progress_percent: int | None,
    progress_tracker: _WorkerFenProgressTracker | None = None,
    report: Any | None = None,
) -> dict[str, int]:
    """Evaluate + persist one chunk; optional SHM + percent progress."""
    done = 0
    failed = 0
    total = len(rows)
    last_logged_percent = 0
    with StockfishEngine(depth=depth) as engine:
        for index, row in enumerate(rows):
            move_id = row["move_id"]
            try:
                evals = evaluate_all_legal_after(
                    engine,
                    row["fen_before"],
                    num_nodes=num_nodes,
                )
                if not evals:
                    _worker_logger.warning(
                        "CandidateEvalBackfill: empty MultiPV for move_id=%s",
                        move_id,
                    )
                    failed += 1
                else:
                    _persist_payload(
                        move_id,
                        build_candidate_payload(
                            evals,
                            depth=depth,
                            num_nodes=num_nodes,
                        ),
                    )
                    done += 1
            except Exception:
                _worker_logger.exception(
                    "CandidateEvalBackfill: failed move_id=%s",
                    move_id,
                )
                failed += 1

            completed = index + 1
            if progress_tracker is not None:
                progress_tracker.maybe_update(completed)
            if report is not None:
                last_logged_percent = _advance_fen_progress(
                    completed=completed,
                    total=total,
                    progress_percent=log_progress_percent,
                    last_logged_percent=last_logged_percent,
                    report=report,
                )
    if progress_tracker is not None:
        progress_tracker.finalize(total)
    return {"done": done, "failed": failed}


def _worker_chunk(
    worker_index: int,
    rows: list[dict[str, str]],
    depth: int,
    num_nodes: int,
    progress_shm_name: str,
) -> dict[str, int]:
    os.environ.setdefault("ENVIRONMENT", "AGENT")
    tracker = _WorkerFenProgressTracker(progress_shm_name, worker_index)
    try:
        return _process_rows(
            rows,
            depth=depth,
            num_nodes=num_nodes,
            log_progress_percent=None,
            progress_tracker=tracker,
            report=None,
        )
    finally:
        tracker.close()


def backfill(
    *,
    depth: int,
    num_nodes: int,
    limit: int | None,
    workers: int,
    log_progress_percent: int | None = _DEFAULT_LOG_PROGRESS_PERCENT,
) -> int:
    pending_total = _count_pending()
    rows = _pending_rows(limit=limit)
    total = len(rows)
    logger.info(
        "CandidateEvalBackfill: start pending_total=%s this_run=%s "
        "depth=%s nodes=%s workers=%s method=multipv_nodes progress_%%=%s",
        pending_total,
        total,
        depth,
        num_nodes,
        workers,
        log_progress_percent if log_progress_percent is not None else "off",
    )
    if not rows:
        logger.info("CandidateEvalBackfill: nothing to backfill.")
        return 0

    t0 = time.monotonic()
    done = 0
    failed = 0

    use_pool = workers > 1 and total >= 2
    if not use_pool:
        logger.info("CandidateEvalBackfill: serial path (workers=%s, n=%s).", workers, total)
        result = _process_rows(
            rows,
            depth=depth,
            num_nodes=num_nodes,
            log_progress_percent=log_progress_percent,
            progress_tracker=None,
            report=_report_progress if log_progress_percent is not None else None,
        )
        done = result["done"]
        failed = result["failed"]
    else:
        chunks = _split_fen_list(rows, workers)
        n_chunks = len(chunks)
        chunk_size = len(chunks[0]) if chunks else 0
        progress_shm = shared_memory.SharedMemory(create=True, size=n_chunks * _PROGRESS_INT_BYTES)
        progress_shm_name = progress_shm.name
        if progress_shm.buf is None:
            progress_shm.close()
            progress_shm.unlink()
            raise RuntimeError("Shared memory progress buffer is unavailable.")
        progress_buf: memoryview = progress_shm.buf

        logger.info(
            "CandidateEvalBackfill: starting ProcessPoolExecutor with %d worker(s) "
            "for %d move(s) (~%d per chunk).",
            n_chunks,
            total,
            chunk_size,
        )
        last_logged_percent = 0
        try:
            with ProcessPoolExecutor(max_workers=n_chunks) as pool:
                futures = [
                    pool.submit(
                        _worker_chunk,
                        worker_index,
                        chunk,
                        depth,
                        num_nodes,
                        progress_shm_name,
                    )
                    for worker_index, chunk in enumerate(chunks)
                ]
                pending = set(futures)
                completed_chunks = 0
                last_heartbeat = time.monotonic()
                while pending:
                    finished, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
                    completed_moves = _sum_shared_fen_progress(progress_buf, n_chunks)
                    if log_progress_percent is not None:
                        last_logged_percent = _advance_fen_progress(
                            completed=completed_moves,
                            total=total,
                            progress_percent=log_progress_percent,
                            last_logged_percent=last_logged_percent,
                            report=_report_progress,
                        )
                    for fut in finished:
                        result = fut.result()
                        done += result["done"]
                        failed += result["failed"]
                        completed_chunks += 1
                        logger.info(
                            "CandidateEvalBackfill: worker chunk finished "
                            "(%d / %d chunks, done=%s failed=%s).",
                            completed_chunks,
                            n_chunks,
                            done,
                            failed,
                        )
                    now = time.monotonic()
                    if now - last_heartbeat >= 60.0:
                        percent = _completed_percent(completed_moves, total)
                        logger.info(
                            "CandidateEvalBackfill: heartbeat %d / %d moves (%d%%), "
                            "%d / %d chunk(s) complete, %d pending.",
                            completed_moves,
                            total,
                            percent,
                            completed_chunks,
                            n_chunks,
                            len(pending),
                        )
                        last_heartbeat = now
        finally:
            progress_shm.close()
            progress_shm.unlink()

    elapsed = time.monotonic() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    logger.info(
        "CandidateEvalBackfill: finished done=%s failed=%s elapsed_s=%.1f rate=%.2f moves/s",
        done,
        failed,
        elapsed,
        rate,
    )
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=CANDIDATE_STOCKFISH_DEPTH)
    parser.add_argument("--nodes", type=int, default=CANDIDATE_STOCKFISH_NODES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=_default_fen_eval_workers())
    parser.add_argument(
        "--log-progress-percent",
        type=int,
        default=_DEFAULT_LOG_PROGRESS_PERCENT,
        help="Log every N%% complete (same idea as StockfishEvaluationTransformation).",
    )
    args = parser.parse_args()

    log_script_runtime_context(logger, script="backfill_candidate_evals")
    progress = args.log_progress_percent
    if progress is not None and progress <= 0:
        progress = None
    return backfill(
        depth=args.depth,
        num_nodes=max(0, args.nodes),
        limit=args.limit,
        workers=max(1, args.workers),
        log_progress_percent=progress,
    )


if __name__ == "__main__":
    run_script_main(main)
