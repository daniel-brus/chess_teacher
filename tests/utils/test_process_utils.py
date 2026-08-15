"""Tests for multiprocessing-aware process helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chess_teacher.utils.process_utils import (
    WORKER_NO_OP_LOGGER,
    WorkerSafeLogger,
    is_parent_process,
    run_script_main,
)


def test_is_parent_process_true_in_normal_main() -> None:
    assert is_parent_process() is True


def test_is_parent_process_false_when_parent_process_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "multiprocessing.parent_process",
        lambda: MagicMock(),
    )
    assert is_parent_process() is False


def test_is_parent_process_false_for_spawn_worker_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = MagicMock()
    worker.name = "SpawnProcess-1"
    monkeypatch.setattr("multiprocessing.current_process", lambda: worker)
    monkeypatch.setattr("multiprocessing.parent_process", lambda: None)
    assert is_parent_process() is False


def test_is_parent_process_false_for_spawn_pool_worker_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = MagicMock()
    worker.name = "SpawnPoolWorker-2"
    monkeypatch.setattr("multiprocessing.current_process", lambda: worker)
    monkeypatch.setattr("multiprocessing.parent_process", lambda: None)
    assert is_parent_process() is False


def test_run_script_main_exits_with_return_code() -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_script_main(lambda: 7)
    assert exc_info.value.code == 7


def test_run_script_main_skips_in_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chess_teacher.utils.process_utils.is_parent_process",
        lambda: False,
    )
    called = False

    def main() -> int:
        nonlocal called
        called = True
        return 0

    run_script_main(main)
    assert called is False


def test_worker_safe_logger_uses_no_op_in_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chess_teacher.utils.process_utils.is_parent_process",
        lambda: False,
    )
    logger = WorkerSafeLogger("chess_teacher.tests.worker_safe_logger")
    assert logger._get() is WORKER_NO_OP_LOGGER
