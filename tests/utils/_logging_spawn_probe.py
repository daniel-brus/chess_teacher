"""Standalone spawn/subprocess probe for log shipping (stdlib-only at import time).

Invoke as ``python -m tests.utils._logging_spawn_probe <buffer_dir> <hostname>``
so pytest / Streamlit AppTest cannot poison multiprocessing's ``init_main_from_path``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _ensure_repo_on_sys_path() -> None:
    """Make ``src/`` importable when the child did not inherit pytest's path hacks."""
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "src"
    for path in (str(repo_root), str(src)):
        if path not in sys.path:
            sys.path.insert(0, path)


def run_logging_spawn_probe(buffer_dir: str, hostname: str) -> dict[str, object]:
    """Assert workers skip segment handlers and LogShipper; return a JSON-able result."""
    os.environ["LOG_BUFFER_DIR"] = buffer_dir
    os.environ["HOSTNAME"] = hostname
    os.environ["LOG_SHIP_ENABLED"] = "true"
    os.environ["ENVIRONMENT"] = "test"

    _ensure_repo_on_sys_path()

    import logging as logging_mod

    from chess_teacher.utils.logging import runtime as child_runtime
    from chess_teacher.utils.logging.buffer import SegmentFileHandler as SegmentHandler
    from chess_teacher.utils.logging.config import configure_logging
    from chess_teacher.utils.process_utils import WorkerSafeLogger, is_parent_process

    safe = WorkerSafeLogger("spawn.probe.safe")
    safe.info("worker should not touch buffer/shipper")

    # Even if something calls configure_logging in a spawn child, file+shipper
    # must stay off because is_parent_process() is False.
    configure_logging(log_dir=Path(buffer_dir), force=True)

    return {
        "ok": True,
        "is_parent": is_parent_process(),
        "safe_is_noop": type(safe._get()).__name__ == "_WorkerNoOpLogger",
        "has_segment_handler": any(
            isinstance(handler, SegmentHandler) for handler in logging_mod.getLogger().handlers
        ),
        "shipper_started": child_runtime._shipper is not None,
        "process_name": mp.current_process().name,
    }


def _spawn_worker_target(queue: Any, buffer_dir: str, hostname: str) -> None:
    """Module-level target so spawn can pickle it (must not live under ``__main__``)."""
    try:
        queue.put(run_logging_spawn_probe(buffer_dir, hostname))
    except Exception:
        queue.put({
            "ok": False,
            "error": traceback.format_exc(),
            "process_name": mp.current_process().name,
        })


def _run_in_spawn_child(buffer_dir: str, hostname: str) -> dict[str, object]:
    """Nest a true spawn worker so we exercise SpawnProcess-* naming."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue[dict[str, object]] = ctx.Queue()
    process = ctx.Process(
        target=_spawn_worker_target,
        args=(queue, buffer_dir, hostname),
    )
    process.start()
    process.join(timeout=60)
    if process.exitcode != 0:
        return {
            "ok": False,
            "error": f"nested spawn worker exited with {process.exitcode}",
            "process_name": "unknown",
        }
    return queue.get(timeout=5)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(json.dumps({"ok": False, "error": "usage: -m ... <buffer_dir> <hostname>"}))
        return 2
    try:
        payload = _run_in_spawn_child(args[0], args[1])
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}
    print(json.dumps(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
