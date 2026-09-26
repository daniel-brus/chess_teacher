"""Score path for the user promotion sibling, without Postgres or local artifacts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import pytest

from chess_teacher.pipelines.neural_network.board_encoder import HybridBoardTrainer
from chess_teacher.pipelines.neural_network.candidate_eval import build_candidate_payload
from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingBatch,
    TrainingDatumBuilder,
)
from chess_teacher.pipelines.preprocessing.moves import Move, MoveCharacteristics
from chess_teacher.utils.chess_utils import Color
from scripts.ops.offline_user_promotion import run_offline_user_promotion

_FEN = chess.STARTING_FEN


def _one_datum() -> object:
    board = chess.Board(_FEN)
    board.push(chess.Move.from_uci("e2e4"))
    move = Move(
        move_id="m1",
        game_id="g1",
        account_id="acct-1",
        move_nr=1,
        ply=1,
        move_san="e4",
        move_uci="e2e4",
        fen_before=_FEN,
        fen_after=board.fen(en_passant="fen"),
    )
    evals = {m.uci(): 0.1 for m in chess.Board(_FEN).legal_moves}
    evals["e2e4"] = 0.4
    chars = MoveCharacteristics(
        move_id="m1",
        game_id="g1",
        account_id="acct-1",
        evaluation_before=0.2,
        candidate_evaluations=build_candidate_payload(evals),
    )
    return TrainingDatumBuilder.from_db_rows(move, chars, color=Color.WHITE, game_id="g1")


def _save_tiny_hybrid(path: Path, *, seed_bias: float) -> None:
    datum = _one_datum()
    state_dim = int(TrainingBatch([datum]).state_matrix().shape[1])
    trainer = HybridBoardTrainer(
        epochs=1,
        batch_size=2,
        hidden=8,
        score_hidden=4,
        conv_filters=4,
    )
    model = trainer.build(state_dim=state_dim)
    # Nudge one kernel so the two files are not byte-identical.
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            weights[0] = weights[0] + seed_bias
            layer.set_weights(weights)
            break
    HybridBoardTrainer.save(model, path)


def test_user_promotion_scores_two_keras_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    user_path = tmp_path / "user.keras"
    baseline_path = tmp_path / "baseline.keras"
    _save_tiny_hybrid(user_path, seed_bias=0.0)
    _save_tiny_hybrid(baseline_path, seed_bias=0.25)
    datum = _one_datum()

    with (
        patch("scripts.ops.offline_user_promotion.get_db_client", return_value=MagicMock()),
        patch(
            "scripts.ops.offline_user_promotion.load_account_registry_bucket_datums",
            return_value=[datum],
        ) as load_val,
    ):
        code = run_offline_user_promotion(
            account_id="acct-1",
            split_version="baseline-v1",
            user_weights=str(user_path),
            baseline_weights=str(baseline_path),
            val_limit=None,
        )

    assert code == 0
    assert load_val.call_args.kwargs["bucket"].value == "val"
    assert load_val.call_args.args[0] == "acct-1"
    out = capsys.readouterr().out
    assert "offline user promotion compare" in out
    assert "disagree_t1=" in out
    assert "no_promote=true" in out
    assert "primary=top1_sf_disagree" in out
