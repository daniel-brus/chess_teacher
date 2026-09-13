"""Smoke tests for hybrid board+state encoder build + predict path."""

from __future__ import annotations

import numpy as np

from chess_teacher.pipelines.neural_network.board_encoder import (
    HybridBoardTrainer,
    model_is_hybrid_board_compatible,
)
from chess_teacher.pipelines.neural_network.board_tensor import BOARD_TENSOR_CHANNELS
from chess_teacher.pipelines.neural_network.candidate_eval import MAX_CANDIDATES, MOVE_FEAT_DIM
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging


def test_hybrid_build_shapes_and_compat() -> None:
    ensure_tensorflow_logging()
    trainer = HybridBoardTrainer(epochs=1, batch_size=4, hidden=32, score_hidden=16, conv_filters=16)
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
