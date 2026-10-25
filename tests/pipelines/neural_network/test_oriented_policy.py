"""Orientation of the 8x8x73 policy matches the flipped board tensor."""

from __future__ import annotations

import chess
import numpy as np

from chess_teacher.pipelines.neural_network.board_tensor import oriented_rc
from chess_teacher.pipelines.neural_network.candidate_eval import (
    MAX_CANDIDATES,
    MOVE_FEAT_DIM,
    capped_candidate_rows,
    pack_candidate_tensors,
)
from chess_teacher.pipelines.neural_network.move_encoding import (
    NUM_MOVE_PLANES,
    MoveEncoder,
)
from chess_teacher.pipelines.neural_network.oriented_policy import (
    oriented_policy_index,
)
from chess_teacher.pipelines.neural_network.spatial_policy import (
    SpatialPolicyTrainer,
    gather_candidate_logits,
)
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.chess_utils import Color

_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_white_e2e4_matches_absolute_encoder() -> None:
    assert oriented_policy_index("e2e4", flip=False) == MoveEncoder.encode("e2e4")
    assert oriented_policy_index("e1g1", flip=False) == MoveEncoder.encode("e1g1")


def test_white_index_matches_flattened_plane() -> None:
    move = chess.Move.from_uci("e2e4")
    row, col = oriented_rc(move.from_square, flip=False)
    index = oriented_policy_index(move, flip=False)
    plane = index % NUM_MOVE_PLANES
    assert index == (row * 8 + col) * NUM_MOVE_PLANES + plane


def test_black_pawn_push_is_plus_rank_in_oriented_frame() -> None:
    index = oriented_policy_index("e7e5", flip=True)
    decoded = MoveEncoder.decode(index)
    assert chess.square_file(decoded.to_square) == chess.square_file(decoded.from_square)
    assert chess.square_rank(decoded.to_square) - chess.square_rank(decoded.from_square) == 2


def test_gathered_logits_use_pack_slot_order() -> None:
    board = chess.Board(_START)
    legal = tuple(sorted(m.uci() for m in board.legal_moves))
    evals = {uci: (0.4 if uci == "e2e4" else 0.1) for uci in legal}

    class _Datum:
        def __init__(self) -> None:
            self.candidate_evaluations = {"evals_white_pov": evals}
            self.color = Color.WHITE
            self.move_uci = "e2e4"
            self.legal_move_ucis = legal
            self.fen_before = _START

    packed = pack_candidate_tensors(
        evals,
        fen_before=_START,
        color_is_white=True,
        user_move_uci="e2e4",
        legal_ucis=legal,
    )
    assert packed is not None
    _feats, _mask, label = packed
    rows = capped_candidate_rows(
        evals,
        color_is_white=True,
        user_move_uci="e2e4",
        legal_ucis=legal,
    )
    assert rows is not None
    assert rows[label][0] == "e2e4"

    policy = np.zeros((1, MoveEncoder.vocab_size), dtype=np.float64)
    played = oriented_policy_index("e2e4", flip=False)
    other = oriented_policy_index("g1f3", flip=False)
    policy[0, played] = 3.5
    policy[0, other] = -2.0
    gathered = gather_candidate_logits(policy, [_Datum()])  # type: ignore[list-item]
    other_slot = next(i for i, row in enumerate(rows) if row[0] == "g1f3")
    assert gathered.shape == (1, MAX_CANDIDATES)
    assert gathered[0, label] == 3.5
    assert gathered[0, other_slot] == -2.0


def test_spatial_build_output_is_policy_vocab() -> None:
    ensure_tensorflow_logging()
    trainer = SpatialPolicyTrainer(epochs=1, batch_size=2, filters=4, blocks=1)
    model = trainer.build()
    n = 2
    board = np.zeros((n, 8, 8, 17), dtype=np.float32)
    mask = np.zeros((n, MoveEncoder.vocab_size), dtype=np.float32)
    mask[:, MoveEncoder.encode("e2e4")] = 1.0
    out = model.predict({"board": board, "legal_mask": mask}, verbose=0)
    assert out.shape == (n, MoveEncoder.vocab_size)
    assert int(np.argmax(out[0])) == MoveEncoder.encode("e2e4")


def test_spatial_fit_one_step_on_stub_datums() -> None:
    ensure_tensorflow_logging()
    board = chess.Board(_START)
    legal = tuple(m.uci() for m in board.legal_moves)

    class _Datum:
        ply = 8
        game_id = "game"
        fen_before = _START
        color = Color.WHITE
        move_uci = "e2e4"
        legal_move_ucis = legal

        def state_vector(self) -> np.ndarray:
            return np.zeros((20,), dtype=np.float32)

        def candidate_style_target(self) -> tuple[np.ndarray, np.ndarray, int]:
            return (
                np.zeros((MAX_CANDIDATES, MOVE_FEAT_DIM), dtype=np.float32),
                np.ones((MAX_CANDIDATES,), dtype=np.float32),
                0,
            )

    trainer = SpatialPolicyTrainer(
        epochs=2,
        batch_size=2,
        filters=4,
        blocks=1,
        style_disagree_boost=1.0,
    )
    model, metrics = trainer.fit([_Datum(), _Datum(), _Datum()])  # type: ignore[list-item]
    assert metrics["n_samples"] == 3.0
    assert metrics["epochs"] == 2.0
    assert metrics["param_count"] > 0
    assert len(metrics["epoch_losses"]) == 2
    assert all(loss == loss for loss in metrics["epoch_losses"])
    assert model.output_shape[-1] == MoveEncoder.vocab_size
