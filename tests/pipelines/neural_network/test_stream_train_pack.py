"""Preallocated candidate pack + tf.data batching for low-RAM fit."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from chess_teacher.pipelines.neural_network import train as train_mod
from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_KEYS,
    MAX_CANDIDATES,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.candidate_losses import (
    DEFAULT_SOFT_TEMPERATURE_PAWNS,
    pack_candidate_targets_for_loss,
)
from chess_teacher.pipelines.neural_network.create_training_set import TrainingBatch
from chess_teacher.pipelines.neural_network.ply_weights import (
    candidate_style_sample_weights,
    user_finetune_sample_weights,
)
from chess_teacher.pipelines.neural_network.train import (
    BaselineTrainer,
    candidate_batch_dataset,
    global_sample_weights,
    numpy_batches_to_tf_dataset,
    scan_candidate_stream,
)


class _Datum:
    def __init__(self, *, ok: bool, label: int = 0, ply: int = 12, game_id: str = "g1") -> None:
        self._ok = ok
        self._label = label
        self.ply = ply
        self.game_id = game_id

    def state_vector(self) -> np.ndarray:
        return np.asarray([self.ply, self._label, 0, 0, 0], dtype=np.float32)

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
    tf = pytest.importorskip("tensorflow")
    trainer = BaselineTrainer(
        epochs=1,
        batch_size=2,
        style_disagree_boost=1.0,
        max_candidates=MAX_CANDIDATES,
        move_feat_dim=MOVE_FEAT_DIM,
    )
    call: dict[str, object] = {}

    def _forbid_full_tensor(*_a: object, **_k: object) -> None:
        raise AssertionError("fit must not copy a full NumPy tensor into tf.data")

    monkeypatch.setattr(train_mod, "numpy_batches_to_tf_dataset", _forbid_full_tensor)

    model = MagicMock()
    history = MagicMock()
    history.history = {"loss": [0.2]}

    def _fit(dataset: object, **kwargs: object) -> MagicMock:
        call["dataset"] = dataset
        call["kwargs"] = kwargs
        assert not isinstance(dataset, dict)
        assert kwargs.get("batch_size") is None
        assert kwargs.get("steps_per_epoch") == 1
        return history

    model.fit.side_effect = _fit
    monkeypatch.setattr(trainer, "load_or_build", lambda **_k: model)

    datums = [
        _Datum(ok=False),
        _Datum(ok=True, label=0),
        _Datum(ok=True, label=1),
    ]
    _model, metrics = trainer.fit(datums)  # type: ignore[arg-type]
    assert metrics["n_samples"] == 2.0
    assert metrics["n_dropped_missing_candidates"] == 1.0
    assert isinstance(call["dataset"], tf.data.Dataset)


_SMALL_MAX = 4


class _WeightedDatum:
    """Small candidate row with a real feat layout so weight math can run."""

    def __init__(
        self,
        *,
        ply: int,
        label: int,
        delta_pawns: float,
        evals: list[float | None],
        game_id: str,
    ) -> None:
        self.ply = ply
        self.game_id = game_id
        self._label = label
        self._delta_pawns = delta_pawns
        self._evals = evals

    def state_vector(self) -> np.ndarray:
        return np.asarray([self.ply, self._label, 1, 0, 0], dtype=np.float32)

    def candidate_style_target(self) -> tuple[np.ndarray, np.ndarray, int]:
        feats = np.zeros((_SMALL_MAX, MOVE_FEAT_DIM), dtype=np.float32)
        mask = np.zeros((_SMALL_MAX,), dtype=np.float32)
        delta_i = CANDIDATE_MOVE_FEAT_KEYS.index("delta_vs_best")
        eval_i = CANDIDATE_MOVE_FEAT_KEYS.index("evaluation_after_user_pov")
        for i, ev in enumerate(self._evals):
            if ev is None:
                continue
            mask[i] = 1.0
            feats[i, eval_i] = np.float32(np.tanh(float(ev) / 5.0))
        feats[self._label, delta_i] = np.float32(np.tanh(self._delta_pawns / 5.0))
        mask[self._label] = 1.0
        return feats, mask, self._label


def _weighted_datums() -> list[_WeightedDatum]:
    return [
        _WeightedDatum(
            ply=2,
            label=0,
            delta_pawns=0.0,
            evals=[0.2, -0.1, None, None],
            game_id="a",
        ),
        _WeightedDatum(
            ply=30,
            label=1,
            delta_pawns=-1.5,
            evals=[1.0, -0.5, -2.0, None],
            game_id="b",
        ),
        _WeightedDatum(
            ply=55,
            label=0,
            delta_pawns=-4.0,
            evals=[2.0, -3.0, None, None],
            game_id="c",
        ),
    ]


def test_scanned_weights_match_full_tensor_normalize() -> None:
    datums = _weighted_datums()
    scan = scan_candidate_stream(
        datums,  # type: ignore[arg-type]
        style_disagree_boost=3.0,
        style_disagree_scale=2.0,
        forced_scale_pawns=1.5,
    )
    packed = [d.candidate_style_target() for d in datums]
    feats = np.stack([row[0] for row in packed], axis=0)
    mask = np.stack([row[1] for row in packed], axis=0)
    labels = np.asarray([row[2] for row in packed], dtype=np.int32)
    expected = candidate_style_sample_weights(
        [d.ply for d in datums],
        feats,
        labels,
        style_disagree_boost=3.0,
        style_disagree_scale=2.0,
        candidate_mask=mask,
        forced_scale_pawns=1.5,
    )
    got = global_sample_weights(scan.raw_weights, finetune=False)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert scan.n_dropped == 0
    assert not np.allclose(got, np.ones_like(got))


def test_scanned_finetune_weights_match_full_tensor() -> None:
    from datetime import UTC, datetime

    datums = _weighted_datums()
    now = datetime(2026, 10, 7, tzinfo=UTC)
    ends = {
        "a": datetime(2026, 10, 1, tzinfo=UTC),
        "b": datetime(2026, 1, 1, tzinfo=UTC),
        "c": None,
    }
    scan = scan_candidate_stream(
        datums,  # type: ignore[arg-type]
        style_disagree_boost=3.0,
        style_disagree_scale=2.0,
        end_time_by_game_id=ends,  # type: ignore[arg-type]
    )
    packed = [d.candidate_style_target() for d in datums]
    feats = np.stack([row[0] for row in packed], axis=0)
    labels = np.asarray([row[2] for row in packed], dtype=np.int32)
    baseline = np.asarray([1.0, 0.0, 1.0], dtype=np.float64)
    expected = user_finetune_sample_weights(
        [d.ply for d in datums],
        feats,
        labels,
        [ends[d.game_id] for d in datums],
        recency_lambda=0.05,
        recency_boost=4.0,
        style_disagree_boost=3.0,
        style_disagree_scale=2.0,
        baseline_disagree_strength=baseline,
        baseline_disagree_boost=4.0,
        now=now,
    )
    got = global_sample_weights(
        scan.raw_weights,
        finetune=True,
        end_times=scan.end_times,
        recency_lambda=0.05,
        recency_boost=4.0,
        baseline_strength=baseline,
        baseline_disagree_boost=4.0,
        now=now,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert scan.n_missing_end == 1


def test_generated_batches_match_packed_targets(tmp_path: Path) -> None:
    tf = pytest.importorskip("tensorflow")
    rows = _weighted_datums()
    datums = [_Datum(ok=False), *rows]
    scan = scan_candidate_stream(
        datums,  # type: ignore[arg-type]
        style_disagree_boost=1.0,
        style_disagree_scale=2.0,
        batch_dir=str(tmp_path),
        batch_size=2,
    )
    assert scan.n_dropped == 1
    assert len(scan.batch_paths) == 2
    weights = global_sample_weights(scan.raw_weights, finetune=False)
    dataset, n_out, bs = candidate_batch_dataset(
        scan.batch_paths,
        sample_weight=weights,
        batch_size=2,
        loss_kind="sf_mix",
        soft_temperature_pawns=DEFAULT_SOFT_TEMPERATURE_PAWNS,
        state_dim=scan.state_dim,
        max_candidates=scan.max_candidates,
        move_feat_dim=scan.move_feat_dim,
    )
    assert isinstance(dataset, tf.data.Dataset)
    assert n_out == 3
    assert bs == 2
    batches = list(dataset.as_numpy_iterator())
    assert [int(batch[1].shape[0]) for batch in batches] == [2, 1]
    state = np.concatenate([batch[0]["state"] for batch in batches], axis=0)
    feats = np.concatenate([batch[0]["move_feats"] for batch in batches], axis=0)
    y = np.concatenate([batch[1] for batch in batches], axis=0)
    got_w = np.concatenate([batch[2] for batch in batches], axis=0)
    packed = [d.candidate_style_target() for d in scan.datums]
    expect_feats = np.stack([row[0] for row in packed], axis=0)
    expect_mask = np.stack([row[1] for row in packed], axis=0)
    expect_labels = np.asarray([row[2] for row in packed], dtype=np.int32)
    expect_state = np.stack([d.state_vector() for d in scan.datums], axis=0)
    expect_y = pack_candidate_targets_for_loss(
        loss_kind="sf_mix",
        labels=expect_labels,
        mask=expect_mask,
        move_feats=expect_feats,
    )
    np.testing.assert_allclose(feats, expect_feats)
    np.testing.assert_allclose(state, expect_state)
    np.testing.assert_allclose(y, expect_y)
    np.testing.assert_allclose(got_w, weights)


def test_second_epoch_does_not_repack(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tensorflow")
    calls = {"n": 0}

    class _CountingDatum(_WeightedDatum):
        def candidate_style_target(self) -> tuple[np.ndarray, np.ndarray, int]:
            calls["n"] += 1
            return super().candidate_style_target()

    datums = [
        _CountingDatum(
            ply=4,
            label=0,
            delta_pawns=0.0,
            evals=[0.0, -0.2, None, None],
            game_id="a",
        ),
        _CountingDatum(
            ply=12,
            label=1,
            delta_pawns=-0.4,
            evals=[0.3, 0.0, None, None],
            game_id="b",
        ),
    ]
    trainer = BaselineTrainer(
        epochs=2,
        batch_size=2,
        hidden=4,
        score_hidden=4,
        max_candidates=_SMALL_MAX,
        move_feat_dim=MOVE_FEAT_DIM,
        style_disagree_boost=1.0,
    )
    recorded: dict[str, int] = {}

    def _load(**kwargs: object) -> object:
        model = trainer.build(int(kwargs["input_dim"]))  # type: ignore[arg-type]
        real_fit = model.fit

        def _wrapped(*args: object, **fit_kwargs: object) -> object:
            recorded["before"] = calls["n"]
            out = real_fit(*args, **fit_kwargs)
            recorded["after"] = calls["n"]
            return out

        model.fit = _wrapped  # type: ignore[method-assign]
        return model

    monkeypatch.setattr(trainer, "load_or_build", _load)
    _model, metrics = trainer.fit(datums)  # type: ignore[arg-type]
    assert metrics["epochs"] == 2.0
    # Scan writes batch files. Both epochs read those files and do not repack.
    assert recorded["before"] == 2
    assert recorded["after"] == 2
