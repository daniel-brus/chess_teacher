"""Smoke tests for hybrid board+state encoder build + predict path."""

from __future__ import annotations

import numpy as np

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
from chess_teacher.utils.chess_utils import Color


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
    assert model.input_shape[0][1:] == BOARD_TENSOR_SHAPE
