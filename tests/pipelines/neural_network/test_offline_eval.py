"""Unit tests for offline eval helpers (no Keras / no DB)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chess_teacher.pipelines.neural_network.offline_eval import load_registry_val_datums


def test_load_registry_val_datums_requires_limit_unless_full() -> None:
    with pytest.raises(ValueError, match="limit is required"):
        load_registry_val_datums(MagicMock(), full=False, limit=None)
