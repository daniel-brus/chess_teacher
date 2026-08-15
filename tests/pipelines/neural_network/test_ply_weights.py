"""Unit tests for ply sample weights."""

from __future__ import annotations

import numpy as np
import pytest

from chess_teacher.pipelines.neural_network.ply_weights import (
    ply_sample_weights,
    ply_weight_raw,
)


def test_ply_weight_raw_increases_with_ply() -> None:
    early = float(ply_weight_raw(5))
    late = float(ply_weight_raw(40))
    assert late > early


def test_ply_sample_weights_mean_near_one() -> None:
    w = ply_sample_weights([1, 10, 20, 40, 60])
    assert w.dtype == np.float32
    assert float(np.mean(w)) == pytest.approx(1.0, abs=1e-5)
    assert np.min(w) >= 0.25 - 1e-6
    assert np.max(w) <= 4.0 + 1e-6
