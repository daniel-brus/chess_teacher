from __future__ import annotations

import chess
import polars as pl
import pytest

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
    })


def test_candidate_evaluations_dedupes_shared_fens(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluate_calls: list[str] = []

    def _fake_evaluate_all_legal_after(
        _engine: object,
        fen_before: str,
        *,
        num_nodes: int | None = None,
    ) -> dict[str, float]:
        del num_nodes
        evaluate_calls.append(fen_before)
        return {"e2e4": 0.2} if fen_before == _START else {"e7e5": 0.1}

    monkeypatch.setattr(
        "chess_teacher.pipelines.preprocessing.move_characteristics.candidate_evaluations.evaluate_all_legal_after",
        _fake_evaluate_all_legal_after,
    )
    class _FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def __enter__(self) -> _FakeEngine:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "chess_teacher.pipelines.preprocessing.move_characteristics.candidate_evaluations.StockfishEngine",
        _FakeEngine,
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
