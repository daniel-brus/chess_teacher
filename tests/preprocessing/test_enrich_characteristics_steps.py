"""Tests for MoveCharacteristics expensive-column completeness helpers and enrich steps."""

from __future__ import annotations

from unittest.mock import MagicMock

from chess_teacher.pipelines.modes import PipelineMode
from chess_teacher.pipelines.preprocessing.moves import MoveCharacteristics
from chess_teacher.pipelines.preprocessing.pipeline_steps import (
    EnrichCheapMoveCharacteristicsStep,
    EnrichExpensiveMoveCharacteristicsStep,
)
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext


def test_sql_expensive_complete_and_incomplete() -> None:
    complete = MoveCharacteristics.sql_expensive_complete("mc")
    incomplete = MoveCharacteristics.sql_expensive_incomplete("mc")
    assert "mc.evaluation_after IS NOT NULL" in complete
    assert "mc.candidate_evaluations IS NOT NULL" in complete
    assert " AND " in complete
    assert "mc.evaluation_after IS NULL" in incomplete
    assert "mc.candidate_evaluations IS NULL" in incomplete
    assert incomplete.startswith("(") and incomplete.endswith(")")


def test_is_expensive_complete() -> None:
    incomplete = MoveCharacteristics(move_id="m1", game_id="g1", account_id="a1")
    assert incomplete.is_expensive_complete() is False
    partial = MoveCharacteristics(
        move_id="m1",
        game_id="g1",
        account_id="a1",
        evaluation_after=0.5,
    )
    assert partial.is_expensive_complete() is False
    complete = MoveCharacteristics(
        move_id="m1",
        game_id="g1",
        account_id="a1",
        evaluation_after=0.5,
        candidate_evaluations={"version": 1, "evals": {}},
    )
    assert complete.is_expensive_complete() is True


def test_cheap_step_excludes_stockfish_transforms() -> None:
    step = EnrichCheapMoveCharacteristicsStep()
    names = [type(t).__name__ for t in step.transformations]
    assert "StockfishEvaluationTransformation" not in names
    assert "CandidateEvaluationsTransformation" not in names
    assert "MaterialBalanceTransformation" in names
    assert "MoveFlagsTransformation" in names


def test_expensive_step_only_stockfish_transforms() -> None:
    step = EnrichExpensiveMoveCharacteristicsStep()
    names = [type(t).__name__ for t in step.transformations]
    assert names.count("StockfishEvaluationTransformation") == 1
    assert names.count("CandidateEvaluationsTransformation") == 1
    assert "MaterialBalanceTransformation" not in names
    assert step.on is None


def test_expensive_load_filters_incomplete_in_incremental() -> None:
    step = EnrichExpensiveMoveCharacteristicsStep(mode=PipelineMode.INCREMENTAL)
    db_client = MagicMock()
    db_client.table_exists.return_value = True
    captured_sql: list[str] = []

    def capture_query(sql: str, _params: dict) -> list[dict]:
        captured_sql.append(sql)
        return [
            {
                "move_id": "m1",
                "game_id": "g1",
                "account_id": "acct-1",
                "fen_before": "fen-a",
                "fen_after": "fen-b",
                "move_uci": "e2e4",
            }
        ]

    db_client.engine.execute_parameterized_query.side_effect = capture_query
    context = PipelineContext(user_id="u1", account_id="acct-1")
    df = step._load_records(db_client, context)
    assert df.height == 1
    assert len(captured_sql) == 1
    assert "evaluation_after IS NULL" in captured_sql[0]
    assert "candidate_evaluations IS NULL" in captured_sql[0]
    assert "acct-1" in captured_sql[0]


def test_expensive_load_skips_incomplete_filter_on_reprocess() -> None:
    step = EnrichExpensiveMoveCharacteristicsStep(mode=PipelineMode.REPROCESS)
    db_client = MagicMock()
    db_client.table_exists.return_value = True
    captured_sql: list[str] = []

    def capture_query(sql: str, _params: dict) -> list[dict]:
        captured_sql.append(sql)
        return []

    db_client.engine.execute_parameterized_query.side_effect = capture_query
    context = PipelineContext(user_id="u1", account_id="acct-1")
    df = step._load_records(db_client, context)
    assert df.height == 0
    assert "evaluation_after IS NULL" not in captured_sql[0]
    assert 'm."account_id"' in captured_sql[0] or 'm."account_id" =' in captured_sql[0]


def test_expensive_step_save_path_uses_upsert() -> None:
    """Smoke: empty load short-circuits without error."""
    step = EnrichExpensiveMoveCharacteristicsStep()
    db_client = MagicMock()
    db_client.ensure_metadata.return_value = None
    db_client.table_exists.return_value = True
    db_client.engine.execute_parameterized_query.return_value = []

    context = PipelineContext(user_id="u1", account_id="acct-1")
    step.run(db_client, context)
    db_client.merge.assert_not_called()
