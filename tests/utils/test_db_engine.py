"""Unit tests for db engine singleton."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chess_teacher.utils.db import engine as engine_module
from chess_teacher.utils.db.engine import EnrichedEngine, get_db_engine, reset_db_engine_for_tests


@pytest.fixture(autouse=True)
def _reset_engine_singleton():
    reset_db_engine_for_tests()
    yield
    reset_db_engine_for_tests()


def test_get_db_engine_returns_same_singleton_by_default() -> None:
    fake_engine = MagicMock(spec=EnrichedEngine)
    with patch.object(engine_module, "_create_db_engine", return_value=fake_engine) as create:
        first = get_db_engine()
        second = get_db_engine()

    assert first is second
    create.assert_called_once()


def test_get_db_engine_bypasses_singleton_when_host_override() -> None:
    fake_first = MagicMock(spec=EnrichedEngine)
    fake_second = MagicMock(spec=EnrichedEngine)
    with patch.object(
        engine_module,
        "_create_db_engine",
        side_effect=[fake_first, fake_second],
    ) as create:
        singleton = get_db_engine()
        override = get_db_engine(host="override.example.com")

    assert singleton is fake_first
    assert override is fake_second
    assert singleton is not override
    assert create.call_count == 2


def test_reset_db_engine_for_tests_disposes_engine() -> None:
    fake_engine = MagicMock(spec=EnrichedEngine)
    with patch.object(engine_module, "_create_db_engine", return_value=fake_engine):
        get_db_engine()
        reset_db_engine_for_tests()

    fake_engine.dispose.assert_called_once()


def test_reset_db_client_for_tests_clears_client_singleton() -> None:
    from chess_teacher.utils.db.client import get_db_client, reset_db_client_for_tests

    fake_engine = MagicMock(spec=EnrichedEngine)
    with patch.object(engine_module, "_create_db_engine", return_value=fake_engine):
        first = get_db_client()
        second = get_db_client()
        reset_db_client_for_tests()
        third = get_db_client()

    assert first is second
    assert third is not first
