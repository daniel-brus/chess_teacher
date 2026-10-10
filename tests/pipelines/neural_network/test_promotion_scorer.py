"""Hybrid-model compatibility for the legacy promotion scorer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from chess_teacher.pipelines.neural_network import eval_metrics, promotion, train
from chess_teacher.pipelines.neural_network.candidate_eval import MAX_CANDIDATES, MOVE_FEAT_DIM


def test_candidate_style_scorer_uses_model_aware_inputs(monkeypatch: Any) -> None:
    datum = type("Datum", (), {"ply": 1})()
    feats = np.zeros((1, MAX_CANDIDATES, MOVE_FEAT_DIM), dtype=np.float32)
    mask = np.zeros((1, MAX_CANDIDATES), dtype=np.float32)
    mask[0, :2] = 1.0
    labels = np.array([0], dtype=np.int64)
    model = object()
    calls: list[tuple[object, list[object], np.ndarray]] = []

    class _Batch:
        def __init__(self, datums: list[object]) -> None:
            assert datums == [datum]

        def candidate_style_targets(
            self,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
            return feats, mask, labels, [0]

    def predict(
        actual_model: object,
        datums: list[object],
        move_feats: np.ndarray,
    ) -> np.ndarray:
        calls.append((actual_model, datums, move_feats))
        logits = np.full((1, MAX_CANDIDATES), -1.0, dtype=np.float64)
        logits[0, 0] = 1.0
        return logits

    monkeypatch.setattr(promotion, "TrainingBatch", _Batch)
    monkeypatch.setattr(train, "load_candidate_style_from_uri", lambda *a, **k: model)
    monkeypatch.setattr(eval_metrics, "predict_candidate_logits", predict)
    monkeypatch.setattr(
        promotion,
        "candidate_style_sample_weights",
        lambda *a, **k: np.ones((1,), dtype=np.float64),
    )

    score = promotion.CandidateStyleTopKScorer(tracker=MagicMock()).score(
        model_uri="model-uri",
        datums=[datum],  # type: ignore[list-item]
    )

    assert calls == [(model, [datum], feats)]
    assert score.primary == 1.0
