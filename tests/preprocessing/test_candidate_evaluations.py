from __future__ import annotations

import chess
import polars as pl
import pytest

from chess_teacher.pipelines.fen_eval_cache.service import (
    PositionEvalService,
    reset_position_eval_service_for_tests,
)
from chess_teacher.pipelines.fen_eval_cache.store import MemoryEvalStore
from chess_teacher.pipelines.neural_network.candidate_eval import PAYLOAD_KEY_EVALS
from chess_teacher.pipelines.preprocessing.move_characteristics.candidate_evaluations import (
    CandidateEvaluationsTransformation,
)

_START = chess.STARTING_FEN
_AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def _sample_moves_df() -> pl.DataFrame:
    return pl.DataFrame({
        "move_id": ["m1", "m2", "m3"],
        "game_id": ["g1", "g1", "g2"],
        "account_id": ["a1", "a1", "a1"],
        "fen_before": [_START, _START, _AFTER_E4],
        "ply": [1, 1, 2],
    })


def test_candidate_evaluations_dedupes_shared_fens(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluate_calls: list[str] = []

    def compute_candidates(fen: str, num_nodes: int) -> dict[str, float]:
        del num_nodes
        evaluate_calls.append(fen)
        return {"e2e4": 0.2} if fen == _START else {"e7e5": 0.1}

    service = PositionEvalService(
        store=MemoryEvalStore(),
        compute_scalar=lambda _fen: 0.0,
        compute_candidates=compute_candidates,
        max_ply=64,
        n_workers=1,
    )
    reset_position_eval_service_for_tests(service)
    monkeypatch.setattr(
        "chess_teacher.pipelines.preprocessing.move_characteristics.candidate_evaluations.get_position_eval_service",
        lambda: service,
    )

    result = CandidateEvaluationsTransformation(
        log_progress_percent=None,
        checkpoint_percent=0,
        n_workers=1,
    ).transform(_sample_moves_df())

    assert set(evaluate_calls) == {_START, _AFTER_E4}
    assert result.height == 3
    payload_m1 = result["candidate_evaluations"][0]
    payload_m2 = result["candidate_evaluations"][1]
    assert payload_m1 == payload_m2
    assert payload_m1[PAYLOAD_KEY_EVALS]["e2e4"] == pytest.approx(0.2)
    assert result["candidate_evaluations"][2][PAYLOAD_KEY_EVALS]["e7e5"] == pytest.approx(0.1)
    reset_position_eval_service_for_tests()
