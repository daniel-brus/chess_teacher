"""Tests for multiprocessing-aware process helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chess_teacher.utils.process_utils import (
    WORKER_NO_OP_LOGGER,
    HostPressure,
    WorkerSafeLogger,
    is_parent_process,
    log_heavy_operation,
    log_script_runtime_context,
    run_script_main,
    snapshot_host_pressure,
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


def test_snapshot_host_pressure_reports_process_rss() -> None:
    pressure = snapshot_host_pressure()
    assert isinstance(pressure, HostPressure)
    assert pressure.rss_mb >= 0
    assert pressure.cpu_count >= 1
    fields = pressure.format_fields()
    assert "rss_mb=" in fields
    assert "cpu_count=" in fields


def test_log_script_runtime_context_includes_host_pressure() -> None:
    logger = MagicMock()
    log_script_runtime_context(logger, script="pipeline")
    logger.info.assert_called_once()
    message = logger.info.call_args.args[0]
    formatted = message % logger.info.call_args.args[1:]
    assert formatted.startswith("pipeline runtime context")
    assert "rss_mb=" in formatted
    assert "cpu_count=" in formatted


def test_log_heavy_operation_logs_start_and_finish() -> None:
    logger = MagicMock()
    with log_heavy_operation(logger, "fen eval", unique_fens=12, workers=2):
        pass
    assert logger.info.call_count == 2
    start_msg = logger.info.call_args_list[0].args[0] % logger.info.call_args_list[0].args[1:]
    finish_msg = logger.info.call_args_list[1].args[0] % logger.info.call_args_list[1].args[1:]
    assert start_msg.startswith("fen eval unique_fens=12 workers=2 started")
    assert "rss_mb=" in start_msg
    assert finish_msg.startswith("fen eval unique_fens=12 workers=2 finished")
    assert "duration_s=" in finish_msg
    assert "delta_rss_mb=" in finish_msg


def test_log_heavy_operation_logs_failure() -> None:
    logger = MagicMock()
    with pytest.raises(RuntimeError, match="boom"):
        with log_heavy_operation(logger, "keras fit"):
            raise RuntimeError("boom")
    assert logger.info.call_count == 2
    finish_msg = logger.info.call_args_list[1].args[0] % logger.info.call_args_list[1].args[1:]
    assert "keras fit failed" in finish_msg
    assert "duration_s=" in finish_msg
