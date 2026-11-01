"""Smoke tests for hybrid board+state encoder build + predict path."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from chess_teacher.pipelines.neural_network.board_encoder import (
    HybridBoardTrainer,
    model_is_hybrid_board_compatible,
)
from chess_teacher.pipelines.neural_network.board_tensor import (
    BOARD_TENSOR_CHANNELS,
    BOARD_TENSOR_SHAPE,
)
from chess_teacher.pipelines.neural_network.candidate_eval import MAX_CANDIDATES, MOVE_FEAT_DIM
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.pipelines.neural_network.train import CandidateStreamScan
from chess_teacher.pipelines.neural_network.training_slot import FitMemoryError
from chess_teacher.utils.chess_utils import Color


def test_tensorflow_thread_caps_are_set_before_import(monkeypatch) -> None:
    import chess_teacher.pipelines.neural_network.tf_runtime as tf_runtime

    monkeypatch.setattr(tf_runtime, "_env_configured", False)
    for name in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "TF_ENABLE_ONEDNN_OPTS"):
        monkeypatch.delenv(name, raising=False)
    tf_runtime.ensure_tensorflow_logging()
    assert tf_runtime.os.environ["TF_NUM_INTRAOP_THREADS"] == "1"
    assert tf_runtime.os.environ["TF_NUM_INTEROP_THREADS"] == "1"
    assert tf_runtime.os.environ["TF_ENABLE_ONEDNN_OPTS"] == "0"


def test_hybrid_build_shapes_and_compat() -> None:
    ensure_tensorflow_logging()
    trainer = HybridBoardTrainer(
        epochs=1, batch_size=4, hidden=32, score_hidden=16, conv_filters=16
    )
    state_dim = 20
    model = trainer.build(state_dim=state_dim)
    assert model_is_hybrid_board_compatible(
        model,
        max_candidates=MAX_CANDIDATES,
        move_feat_dim=MOVE_FEAT_DIM,
        board_channels=BOARD_TENSOR_CHANNELS,
    )
    n = 3
    board = np.zeros((n, 8, 8, BOARD_TENSOR_CHANNELS), dtype=np.float32)
    state = np.zeros((n, state_dim), dtype=np.float32)
    feats = np.zeros((n, MAX_CANDIDATES, MOVE_FEAT_DIM), dtype=np.float32)
    out = model.predict({"board": board, "state": state, "move_feats": feats}, verbose=0)
    assert out.shape == (n, MAX_CANDIDATES)


def test_default_conv_filters_bumped() -> None:
    assert HybridBoardTrainer.DEFAULT_CONV_FILTERS == 64


def test_hybrid_weight_knobs_match_baseline_surface() -> None:
    trainer = HybridBoardTrainer(
        forced_scale_pawns=1.5,
        baseline_disagree_boost=1.25,
        recency_boost=1.5,
    )
    assert trainer.forced_scale_pawns == 1.5
    assert trainer.baseline_disagree_boost == 1.25
    assert trainer.recency_boost == 1.5
    assert trainer.forced_scale_pawns is not None


def test_hybrid_fit_streams_spilled_board_batches() -> None:
    ensure_tensorflow_logging()
    trainer = HybridBoardTrainer(
        epochs=1,
        batch_size=2,
        hidden=8,
        score_hidden=4,
        conv_filters=2,
        style_disagree_boost=1.0,
    )

    class _Datum:
        ply = 12
        game_id = "game"
        fen_before = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        color = Color.WHITE

        def state_vector(self) -> np.ndarray:
            return np.zeros((20,), dtype=np.float32)

        def candidate_style_target(
            self,
        ) -> tuple[np.ndarray, np.ndarray, int]:
            return (
                np.zeros((MAX_CANDIDATES, MOVE_FEAT_DIM), dtype=np.float32),
                np.ones((MAX_CANDIDATES,), dtype=np.float32),
                0,
            )

    model, metrics = trainer.fit([_Datum(), _Datum(), _Datum()])  # type: ignore[arg-type]
    assert metrics["n_samples"] == 3.0
    assert metrics["epochs"] == 1.0
    assert model.input_shape[0][1:] == BOARD_TENSOR_SHAPE


def test_default_epochs_are_20_cold_and_10_warm() -> None:
    trainer = HybridBoardTrainer()
    assert trainer.epochs is None
    assert trainer.resolve_epochs(cold_start=True) == 20
    assert trainer.resolve_epochs(cold_start=False) == 10
    assert trainer.describe_epochs() == "10 warm-start / 20 cold-start"

    explicit = HybridBoardTrainer(epochs=7)
    assert explicit.resolve_epochs(cold_start=True) == 7
    assert explicit.resolve_epochs(cold_start=False) == 7
    assert explicit.describe_epochs() == "7"


def test_fit_uses_warm_start_epochs_when_parent_weights_load(monkeypatch) -> None:
    trainer = HybridBoardTrainer(style_disagree_boost=1.0, batch_size=2)
    model = MagicMock()
    model.fit.return_value = MagicMock(history={"loss": [0.2]})
    dataset = MagicMock()
    dataset.repeat.return_value = dataset
    dataset.prefetch.return_value = dataset
    scan = CandidateStreamScan(
        datums=[object(), object()],  # type: ignore[list-item]
        raw_weights=np.ones((2,), dtype=np.float64),
        end_times=[None, None],
        n_input=2,
        n_dropped=0,
        n_missing_end=0,
        disagree_frac=0.0,
        mean_strength=0.0,
        state_dim=4,
        max_candidates=trainer.max_candidates,
        move_feat_dim=trainer.move_feat_dim,
        batch_paths=["batch-0.npz"],
    )

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.board_encoder.scan_candidate_stream",
        lambda *args, **kwargs: scan,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.board_encoder.candidate_batch_dataset",
        lambda *args, **kwargs: (dataset, 2, 2),
    )
    monkeypatch.setattr(trainer, "load_or_build", lambda **kwargs: (model, False))
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.board_encoder.fit_host_has_memory",
        lambda: True,
    )

    _fitted, metrics = trainer.fit([object(), object()])  # type: ignore[list-item]

    assert metrics["epochs"] == 10.0
    assert model.fit.call_count == 10
    assert model.fit.call_args.kwargs["epochs"] == 1
    dataset.repeat.assert_not_called()


def test_fit_keeps_20_epochs_on_cold_start(monkeypatch) -> None:
    trainer = HybridBoardTrainer(style_disagree_boost=1.0, batch_size=2)
    model = MagicMock()
    model.fit.return_value = MagicMock(history={"loss": [0.2]})
    dataset = MagicMock()
    dataset.repeat.return_value = dataset
    dataset.prefetch.return_value = dataset
    scan = CandidateStreamScan(
        datums=[object()],  # type: ignore[list-item]
        raw_weights=np.ones((1,), dtype=np.float64),
        end_times=[None],
        n_input=1,
        n_dropped=0,
        n_missing_end=0,
        disagree_frac=0.0,
        mean_strength=0.0,
        state_dim=4,
        max_candidates=trainer.max_candidates,
        move_feat_dim=trainer.move_feat_dim,
        batch_paths=["batch-0.npz"],
    )

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.board_encoder.scan_candidate_stream",
        lambda *args, **kwargs: scan,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.board_encoder.candidate_batch_dataset",
        lambda *args, **kwargs: (dataset, 1, 1),
    )
    monkeypatch.setattr(trainer, "load_or_build", lambda **kwargs: (model, True))
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.board_encoder.fit_host_has_memory",
        lambda: True,
    )

    _fitted, metrics = trainer.fit([object()])  # type: ignore[list-item]

    assert metrics["epochs"] == 20.0
    assert model.fit.call_count == 20
    assert model.fit.call_args.kwargs["epochs"] == 1


def _patched_fit(monkeypatch, trainer: HybridBoardTrainer, *, memory: list[bool]):
    model = MagicMock()
    model.fit.return_value = MagicMock(history={"loss": [0.2]})
    dataset = MagicMock()
    dataset.prefetch.return_value = dataset
    scan = CandidateStreamScan(
        datums=[object()],  # type: ignore[list-item]
        raw_weights=np.ones((1,), dtype=np.float64),
        end_times=[None],
        n_input=1,
        n_dropped=0,
        n_missing_end=0,
        disagree_frac=0.0,
        mean_strength=0.0,
        state_dim=4,
        max_candidates=trainer.max_candidates,
        move_feat_dim=trainer.move_feat_dim,
        batch_paths=["batch-0.npz"],
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.board_encoder.scan_candidate_stream",
        lambda *args, **kwargs: scan,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.board_encoder.candidate_batch_dataset",
        lambda *args, **kwargs: (dataset, 1, 1),
    )
    monkeypatch.setattr(trainer, "load_or_build", lambda **kwargs: (model, True))
    answers = iter(memory)

    def _memory() -> bool:
        return next(answers)

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.board_encoder.fit_host_has_memory",
        _memory,
    )
    return model


def test_fit_refuses_before_keras_when_memory_is_tight(monkeypatch) -> None:
    trainer = HybridBoardTrainer(epochs=3, style_disagree_boost=1.0, batch_size=2)
    model = _patched_fit(monkeypatch, trainer, memory=[False])
    with pytest.raises(FitMemoryError):
        trainer.fit([object()])  # type: ignore[list-item]
    model.fit.assert_not_called()


def test_fit_stops_before_a_later_epoch_when_memory_drops(monkeypatch) -> None:
    trainer = HybridBoardTrainer(epochs=4, style_disagree_boost=1.0, batch_size=2)
    model = _patched_fit(monkeypatch, trainer, memory=[True, True, False])
    _fitted, metrics = trainer.fit([object()])  # type: ignore[list-item]
    assert model.fit.call_count == 2
    assert metrics["epochs"] == 2.0
