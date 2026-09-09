"""Parent-weight loading for BaselineTrainer (no Keras fit)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from chess_teacher.pipelines.neural_network import train as train_mod
from chess_teacher.pipelines.neural_network.train import BaselineTrainer


class _FakeBatch:
    def __init__(self, datums: list[object]) -> None:
        self.datums = datums

    def candidate_style_targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        n, max_c, feat_dim = 1, 8, 4
        return (
            np.zeros((n, max_c, feat_dim)),
            np.ones((n, max_c)),
            np.array([0]),
            [0],
        )

    def state_matrix(self) -> np.ndarray:
        return np.zeros((1, 16))


def _patch_fit_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_mod, "TrainingBatch", _FakeBatch)
    monkeypatch.setattr(train_mod, "pack_candidate_targets", lambda labels, _mask: labels)
    monkeypatch.setattr(train_mod, "user_not_sf_best_mask", lambda *_a, **_k: np.array([True]))
    monkeypatch.setattr(train_mod, "user_sf_disagree_strength", lambda *_a, **_k: np.array([0.5]))
    monkeypatch.setattr(train_mod, "baseline_disagree_strength", lambda *_a, **_k: np.array([1.0]))
    monkeypatch.setattr(
        train_mod, "user_finetune_sample_weights", lambda *_a, **_k: np.array([1.0])
    )


def _history() -> MagicMock:
    history = MagicMock()
    history.history = {"loss": [0.1]}
    return history


def test_trainer_baseline_disagree_boost_defaults_off() -> None:
    assert BaselineTrainer().baseline_disagree_boost == 1.0


def test_fit_predicts_baseline_before_keras_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = BaselineTrainer(baseline_disagree_boost=4.0, epochs=1)
    call_order: list[str] = []
    model = MagicMock()
    path = Path("same.keras")

    def _predict(*_a: object, **_k: object) -> np.ndarray:
        call_order.append("predict")
        return np.zeros((1, 8), dtype=np.float64)

    def _fit(*_a: object, **_k: object) -> MagicMock:
        call_order.append("fit")
        return _history()

    model.predict.side_effect = _predict
    model.fit.side_effect = _fit
    monkeypatch.setattr(trainer, "load_or_build", lambda **_k: model)
    _patch_fit_stack(monkeypatch)

    datum = MagicMock(game_id="g1", ply=10)
    trainer.fit(
        [datum],
        recency_lambda=None,
        weights_path=path,
        baseline_weights_path=path,
    )
    assert call_order == ["predict", "fit"]


def test_fit_predicts_frozen_baseline_not_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = BaselineTrainer(baseline_disagree_boost=4.0, epochs=1)
    baseline_path = Path("baseline.keras")
    resume_path = Path("resume.keras")
    baseline_model = MagicMock()
    resume_model = MagicMock()
    call_order: list[str] = []

    def _baseline_predict(*_a: object, **_k: object) -> np.ndarray:
        call_order.append("baseline_predict")
        return np.zeros((1, 8), dtype=np.float64)

    def _resume_fit(*_a: object, **_k: object) -> MagicMock:
        call_order.append("resume_fit")
        return _history()

    baseline_model.predict.side_effect = _baseline_predict
    resume_model.fit.side_effect = _resume_fit

    def _load(**kwargs: object) -> MagicMock:
        return baseline_model if kwargs.get("weights_path") == baseline_path else resume_model

    monkeypatch.setattr(trainer, "load_or_build", _load)
    _patch_fit_stack(monkeypatch)

    datum = MagicMock(game_id="g1", ply=10)
    trainer.fit(
        [datum],
        recency_lambda=None,
        weights_path=resume_path,
        baseline_weights_path=baseline_path,
        require_parent_weights=True,
    )
    assert call_order == ["baseline_predict", "resume_fit"]
    baseline_model.fit.assert_not_called()
    resume_model.predict.assert_not_called()


def test_fit_baseline_disagree_requires_baseline_path(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = BaselineTrainer(baseline_disagree_boost=4.0, epochs=1)
    _patch_fit_stack(monkeypatch)
    with pytest.raises(ValueError, match="requires baseline_weights_path"):
        trainer.fit([MagicMock(game_id="g1", ply=10)], recency_lambda=None)


def test_load_or_build_require_parent_raises_when_incompatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = BaselineTrainer()
    path = MagicMock()
    path.is_file.return_value = True
    monkeypatch.setattr(train_mod, "load_candidate_style_keras", lambda *a, **k: MagicMock())
    monkeypatch.setattr(train_mod, "model_is_candidate_style_compatible", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="not candidate_style-compatible"):
        trainer.load_or_build(
            input_dim=8,
            weights_path=path,
            require_compatible_parent=True,
        )


def test_load_or_build_require_parent_raises_when_file_missing() -> None:
    trainer = BaselineTrainer()
    path = MagicMock()
    path.is_file.return_value = False
    with pytest.raises(FileNotFoundError, match="require_compatible_parent"):
        trainer.load_or_build(
            input_dim=8,
            weights_path=path,
            require_compatible_parent=True,
        )
