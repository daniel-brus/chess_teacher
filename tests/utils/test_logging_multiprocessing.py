"""Log shipping + multiprocessing interaction tests.

Workers must not attach segment handlers or start ``LogShipper`` (Windows spawn
re-imports modules and used to fight the parent for ``app.log`` / shipper threads).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from chess_teacher.utils.logging import runtime as logging_runtime
from chess_teacher.utils.logging.buffer import SegmentFileHandler
from chess_teacher.utils.logging.formatters import ConsoleFormatter, JsonLinesFormatter
from chess_teacher.utils.logging.runtime import (
    attach_handlers,
    reset_log_shipping,
    start_log_shipping,
)
from chess_teacher.utils.process_utils import WORKER_NO_OP_LOGGER, WorkerSafeLogger

pytestmark = pytest.mark.integration


@pytest.fixture
def buffer_dir() -> Path:
    reset_log_shipping()
    path = Path(tempfile.mkdtemp(prefix="log_mp_buffer_"))
    root = logging.getLogger()
    root.handlers.clear()
    yield path
    root.handlers.clear()
    reset_log_shipping()
    shutil.rmtree(path, ignore_errors=True)


def _attach(*, buffer_dir: Path, level: str = "INFO") -> None:
    attach_handlers(
        level=level,
        buffer_dir=buffer_dir,
        console_formatter=ConsoleFormatter(),
        file_formatter=JsonLinesFormatter(),
    )


def test_attach_handlers_skips_segment_and_shipper_in_worker(
    buffer_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chess_teacher.utils.logging.runtime.is_parent_process",
        lambda: False,
    )
    monkeypatch.setenv("LOG_SHIP_ENABLED", "true")
    monkeypatch.setenv("HOSTNAME", "worker-host")

    _attach(buffer_dir=buffer_dir)

    root = logging.getLogger()
    assert not any(isinstance(handler, SegmentFileHandler) for handler in root.handlers)
    assert any(isinstance(handler, logging.StreamHandler) for handler in root.handlers)
    assert logging_runtime._shipper is None


def test_attach_handlers_parent_starts_shipper_when_enabled(
    buffer_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chess_teacher.utils.logging.runtime.is_parent_process",
        lambda: True,
    )
    monkeypatch.setenv("LOG_SHIP_ENABLED", "true")
    monkeypatch.setenv("HOSTNAME", "parent-host")

    _attach(buffer_dir=buffer_dir)

    root = logging.getLogger()
    assert any(isinstance(handler, SegmentFileHandler) for handler in root.handlers)
    assert logging_runtime._shipper is not None
    reset_log_shipping()


def test_attach_handlers_parent_skips_shipper_when_disabled(
    buffer_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chess_teacher.utils.logging.runtime.is_parent_process",
        lambda: True,
    )
    monkeypatch.setenv("LOG_SHIP_ENABLED", "false")
    monkeypatch.setenv("HOSTNAME", "parent-host")

    _attach(buffer_dir=buffer_dir)

    assert any(isinstance(handler, SegmentFileHandler) for handler in logging.getLogger().handlers)
    assert logging_runtime._shipper is None
    assert start_log_shipping(buffer_dir) is None


def test_worker_safe_logger_never_calls_get_logger_in_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chess_teacher.utils.process_utils.is_parent_process",
        lambda: False,
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("get_logger must not run in worker processes")

    monkeypatch.setattr(
        "chess_teacher.utils.logging.get_logger",
        _boom,
        raising=False,
    )
    monkeypatch.setattr(
        "chess_teacher.utils.logging.config.get_logger",
        _boom,
        raising=False,
    )

    logger = WorkerSafeLogger("chess_teacher.tests.mp_safe_logger")
    assert logger._get() is WORKER_NO_OP_LOGGER
    logger.info("must stay a no-op")


def test_spawn_worker_does_not_start_log_shipper_or_segment_handler(
    buffer_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use a subprocess (not mp.Process from pytest) so AppTest cannot poison spawn.

    Streamlit AppTest sets multiprocessing's ``init_main_from_path`` to a page
    script (e.g. ``streamlit_pages/admin.py``). A later ``mp.Process(spawn)`` then
    re-executes that page in the child and crashes on DB auth.
    """
    import json
    import subprocess
    import sys

    monkeypatch.setenv("HOSTNAME", "parent-host")
    monkeypatch.setenv("LOG_SHIP_ENABLED", "true")
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_BUFFER_DIR", str(buffer_dir))

    # Parent owns the primary buffer (the real Streamlit / pipeline process).
    parent_handler = SegmentFileHandler(
        buffer_dir,
        interval_seconds=3600,
        instance_id="parent-host",
        writer_mode="primary",
    )
    assert parent_handler.is_primary_writer

    probe_env = {
        **os.environ,
        "HOSTNAME": "parent-host",
        "LOG_BUFFER_DIR": str(buffer_dir),
        "LOG_SHIP_ENABLED": "true",
        "ENVIRONMENT": "test",
    }
    # Ensure the subprocess can import ``tests`` and ``chess_teacher``.
    repo_root = Path(__file__).resolve().parents[2]
    existing = probe_env.get("PYTHONPATH", "")
    probe_env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(repo_root), str(repo_root / "src"), existing) if p
    )

    completed = subprocess.run(
        [sys.executable, "-m", "tests.utils._logging_spawn_probe", str(buffer_dir), "parent-host"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=probe_env,
    )
    assert completed.returncode == 0, (
        f"probe failed rc={completed.returncode}\n"
        f"stdout={completed.stdout}\nstderr={completed.stderr}"
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result.get("ok") is True, result.get("error", result)
    assert result["is_parent"] is False
    assert result["safe_is_noop"] is True
    assert result["has_segment_handler"] is False
    assert result["shipper_started"] is False
    assert str(result["process_name"]).startswith("Spawn")

    parent_handler.close()
