"""Unit tests for log_shipping."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chess_teacher.utils.exception_utils import ConfigError
from chess_teacher.utils.log_shipping import (
    LogBufferWriterLock,
    LogShipper,
    SegmentFileHandler,
    log_storage_key_for_segment,
    reset_log_shipping,
    shutdown_logging,
)
from chess_teacher.utils.object_storage.filesystem import FilesystemObjectStorage


@pytest.fixture
def buffer_dir():
    reset_log_shipping()
    path = Path(tempfile.mkdtemp(prefix="log_buffer_test_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)
    reset_log_shipping()


@pytest.fixture
def raw_storage():
    root = Path(tempfile.mkdtemp(prefix="log_raw_storage_test_"))
    storage = FilesystemObjectStorage(root)
    yield storage, root
    shutil.rmtree(root, ignore_errors=True)


def test_log_buffer_writer_lock_same_process_second_handler_is_follower(buffer_dir: Path) -> None:
    first = SegmentFileHandler(buffer_dir, interval_seconds=3600, instance_id="host")
    assert first.owns_active_log

    second = SegmentFileHandler(buffer_dir, interval_seconds=3600, instance_id="host")
    assert not second.owns_active_log

    first.close()
    third = SegmentFileHandler(buffer_dir, interval_seconds=3600, instance_id="host")
    assert third.owns_active_log
    third.close()
    second.close()


def test_log_buffer_writer_lock_blocks_other_process(
    buffer_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = buffer_dir / "active" / ".writer.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("424242", encoding="utf-8")
    monkeypatch.setattr(
        "chess_teacher.utils.log_shipping._pid_is_alive",
        lambda pid: pid == 424242,
    )

    with pytest.raises(ConfigError, match="already owned"):
        SegmentFileHandler(buffer_dir, interval_seconds=3600, instance_id="host")


def test_log_buffer_writer_lock_reclaims_stale_pid(
    buffer_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = buffer_dir / "active" / ".writer.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("999999", encoding="utf-8")

    def fake_kill(pid: int, sig: int) -> None:
        raise ProcessLookupError(pid)

    monkeypatch.setattr(os, "kill", fake_kill)

    handler = SegmentFileHandler(buffer_dir, interval_seconds=3600, instance_id="host")
    assert handler.owns_active_log
    assert LogBufferWriterLock(buffer_dir).holder_pid() == os.getpid()
    handler.close()


def test_segment_handler_rotates_on_size(buffer_dir: Path) -> None:
    handler = SegmentFileHandler(
        buffer_dir, max_bytes=200, interval_seconds=3600, instance_id="test-host"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("segment_size_test")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("x" * 100)
    logger.info("y" * 100)

    closed_segments = list((buffer_dir / "closed").rglob("*.ready"))
    assert len(closed_segments) == 1
    assert handler.active_path.exists()
    handler.close()


def test_log_storage_key_for_segment(buffer_dir: Path) -> None:
    ready_path = buffer_dir / "closed" / "2026" / "06" / "07" / "host" / "app-ts.log.ready"
    key = log_storage_key_for_segment(ready_path, buffer_dir)
    assert key == "logs/python/buffer/closed/2026/06/07/host/app-ts.log"


def test_shipper_uploads_and_removes_ready_segment(
    buffer_dir: Path,
    raw_storage: tuple[FilesystemObjectStorage, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = raw_storage
    from chess_teacher.utils.object_storage import factory

    monkeypatch.setattr(factory, "_raw_storage", storage)

    ready_dir = buffer_dir / "closed" / "2026" / "06" / "07" / "test-host"
    ready_dir.mkdir(parents=True)
    ready_path = ready_dir / "app-20260607T120000Z.log.ready"
    payload = json.dumps({"msg": "hello"}) + "\n"
    ready_path.write_bytes(payload.encode("utf-8"))

    shipper = LogShipper(buffer_dir, scan_interval_seconds=3600)
    assert shipper.scan_once() == 1
    assert not ready_path.exists()

    key = log_storage_key_for_segment(
        buffer_dir
        / "closed"
        / "2026"
        / "06"
        / "07"
        / "test-host"
        / "app-20260607T120000Z.log.ready",
        buffer_dir,
    )
    uploaded = storage.read_bytes(key)
    assert uploaded == payload.encode("utf-8")


def test_shipper_keeps_segment_when_upload_fails(
    buffer_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_dir = buffer_dir / "closed" / "2026" / "06" / "07" / "test-host"
    ready_dir.mkdir(parents=True)
    ready_path = ready_dir / "app-20260607T120000Z.log.ready"
    ready_path.write_bytes(b"{}\n")

    failing_storage = MagicMock()
    failing_storage.write_bytes.side_effect = RuntimeError("upload failed")
    from chess_teacher.utils.object_storage import factory

    monkeypatch.setattr(factory, "_raw_storage", failing_storage)

    shipper = LogShipper(buffer_dir, scan_interval_seconds=3600)
    assert shipper.scan_once() == 0
    assert ready_path.exists()


def test_shutdown_rotates_and_drains(
    buffer_dir: Path,
    raw_storage: tuple[FilesystemObjectStorage, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = raw_storage
    from chess_teacher.utils.log_shipping import register_segment_handler, start_log_shipping
    from chess_teacher.utils.object_storage import factory

    monkeypatch.setattr(factory, "_raw_storage", storage)
    monkeypatch.setenv("LOG_SHIP_ENABLED", "true")
    reset_log_shipping()

    handler = SegmentFileHandler(
        buffer_dir,
        max_bytes=10_000,
        interval_seconds=3600,
        instance_id="test-host",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    register_segment_handler(handler)
    start_log_shipping(buffer_dir)

    logger = logging.getLogger("shutdown_test")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("shutdown message")

    shutdown_logging()

    assert list((buffer_dir / "closed").rglob("*.ready")) == []
    assert storage.list_keys("logs/python/buffer/closed", recursive=True)
    handler.close()
