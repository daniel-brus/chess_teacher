"""Preallocated candidate pack + tf.data batching for low-RAM fit."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from chess_teacher.pipelines.neural_network import train as train_mod
from chess_teacher.pipelines.neural_network.candidate_eval import MAX_CANDIDATES, MOVE_FEAT_DIM
from chess_teacher.pipelines.neural_network.create_training_set import TrainingBatch
from chess_teacher.pipelines.neural_network.train import (
    BaselineTrainer,
    numpy_batches_to_tf_dataset,
)


class _Datum:
    def __init__(self, *, ok: bool, label: int = 0) -> None:
        self._ok = ok
        self._label = label
        self.ply = 12
        self.game_id = "g1"

    def candidate_style_target(self) -> tuple[np.ndarray, np.ndarray, int] | None:
        if not self._ok:
            return None
        feats = np.zeros((MAX_CANDIDATES, MOVE_FEAT_DIM), dtype=np.float32)
        mask = np.ones((MAX_CANDIDATES,), dtype=np.float32)
        return feats, mask, self._label


def test_candidate_style_targets_preallocates_and_skips() -> None:
    batch = TrainingBatch(
        [_Datum(ok=False), _Datum(ok=True, label=2), _Datum(ok=True, label=1)]  # type: ignore[list-item]
    )
    feats, mask, labels, kept = batch.candidate_style_targets()
    assert kept == [1, 2]
    assert feats.shape == (2, MAX_CANDIDATES, MOVE_FEAT_DIM)
    assert mask.shape == (2, MAX_CANDIDATES)
    assert labels.tolist() == [2, 1]


def test_candidate_style_targets_all_kept_no_copy_trim() -> None:
    batch = TrainingBatch([_Datum(ok=True), _Datum(ok=True)])  # type: ignore[list-item]
    feats, _mask, labels, kept = batch.candidate_style_targets()
    assert kept == [0, 1]
    assert feats.shape == (2, MAX_CANDIDATES, MOVE_FEAT_DIM)
    assert labels.shape == (2,)


def test_numpy_batches_to_tf_dataset_shapes() -> None:
    tf = pytest.importorskip("tensorflow")
    n, max_c, feat_dim, state_dim = 5, 8, 4, 16
    x = {
        "state": np.zeros((n, state_dim), dtype=np.float32),
        "move_feats": np.zeros((n, max_c, feat_dim), dtype=np.float32),
    }
    y = np.zeros((n, max_c), dtype=np.float32)
    w = np.ones((n,), dtype=np.float32)
    ds, n_out, bs = numpy_batches_to_tf_dataset(x_inputs=x, y=y, sample_weight=w, batch_size=2)
    assert n_out == n
    assert bs == 2
    batch = next(iter(ds))
    inputs, y_b, w_b = batch
    assert set(inputs.keys()) == {"state", "move_feats"}
    assert int(y_b.shape[0]) == 2
    assert int(w_b.shape[0]) == 2
    assert isinstance(ds, tf.data.Dataset)


def test_fit_uses_tf_dataset_not_raw_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = BaselineTrainer(epochs=1, batch_size=2)
    call: dict[str, object] = {}

    class _FakeBatch:
        def __init__(self, datums: list[object]) -> None:
            self.datums = datums

        def candidate_style_targets(
            self,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
            n = len(self.datums)
            return (
                np.zeros((n, 8, 4), dtype=np.float32),
                np.ones((n, 8), dtype=np.float32),
                np.zeros((n,), dtype=np.int32),
                list(range(n)),
            )

        def state_matrix(self) -> np.ndarray:
            return np.zeros((len(self.datums), 16), dtype=np.float32)

    def _pack(**_k: object) -> np.ndarray:
        return np.zeros((2, 8), dtype=np.float32)

    monkeypatch.setattr(train_mod, "TrainingBatch", _FakeBatch)
    monkeypatch.setattr(train_mod, "pack_candidate_targets_for_loss", _pack)
    monkeypatch.setattr(
        train_mod, "user_not_sf_best_mask", lambda *_a, **_k: np.array([True, False])
    )
    monkeypatch.setattr(
        train_mod, "user_sf_disagree_strength", lambda *_a, **_k: np.array([0.5, 0.1])
    )
    monkeypatch.setattr(train_mod, "candidate_style_sample_weights", lambda *_a, **_k: np.ones(2))

    model = MagicMock()
    history = MagicMock()
    history.history = {"loss": [0.2]}

    def _fit(dataset: object, **kwargs: object) -> MagicMock:
        call["dataset"] = dataset
        call["kwargs"] = kwargs
        # Must be a tf.data dataset (no x= numpy dict).
        assert not isinstance(dataset, dict)
        assert kwargs.get("batch_size") is None
        return history

    model.fit.side_effect = _fit
    monkeypatch.setattr(trainer, "load_or_build", lambda **_k: model)

    datums = [MagicMock(game_id="g1", ply=10), MagicMock(game_id="g2", ply=11)]
    _model, metrics = trainer.fit(datums)
    assert metrics["n_samples"] == 2.0
    assert "dataset" in call
