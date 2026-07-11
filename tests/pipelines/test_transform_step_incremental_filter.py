"""TransformStep integration tests for incremental filtering against a mock target table."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import polars as pl
import pytest

from chess_teacher.pipelines.ingestion.raw_games import RawGame
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.utils.db.client import WriteResult, WriteStrategy
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext
from chess_teacher.utils.pipeline_utils.pipeline_steps import LoadingStrategy, TransformStep

_ACCOUNT_ID = "acct-1"
_INGESTED_AT = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


def _source_table_df() -> pl.DataFrame:
    """Mock rows read from the source table (games.raw_games)."""
    return pl.DataFrame({
        "game_id": ["game-1", "game-2", "game-3"],
        "platform_game_id": ["platform-1", "platform-2", "platform-3"],
        "account_id": [_ACCOUNT_ID, _ACCOUNT_ID, _ACCOUNT_ID],
        "raw_response": ["{}", "{}", "{}"],
        "source_file": [
            "ingested/acct-1/2024/01/01/a.jsonl",
            "ingested/acct-1/2024/01/01/b.jsonl",
            "ingested/acct-1/2024/01/01/c.jsonl",
        ],
        "ingested_at": [_INGESTED_AT, _INGESTED_AT, _INGESTED_AT],
    })


def _mock_db_client(*, target_game_ids: list[str]) -> MagicMock:
    """Mock DB client whose target lookup returns ``target_game_ids``."""
    db_client = MagicMock()
    db_client.ensure_metadata.return_value = None
    db_client.table_exists.return_value = True
    db_client.engine.execute_parameterized_query.return_value = [
        {"game_id": game_id} for game_id in target_game_ids
    ]
    return db_client


def _run_transform_step_and_capture_saved(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on: str | None,
    target_game_ids: list[str],
) -> tuple[pl.DataFrame, MagicMock]:
    """Run a TransformStep with mocked source/target tables and return the saved frame."""
    source_df = _source_table_df()
    db_client = _mock_db_client(target_game_ids=target_game_ids)

    step = TransformStep(
        name="TestRawGamesToGames",
        source_data_class=RawGame,
        target_data_class=Game,
        on=on,
        transformations=[],
        loading_strategy=LoadingStrategy.MERGE,
    )
    step._incremental_filter.db_client = db_client
    step.transformations = [step._incremental_filter]

    saved_frames: list[pl.DataFrame] = []

    def capture_save(
        _db_client: MagicMock,
        _table_metadata: object,
        data: pl.DataFrame,
    ) -> WriteResult:
        saved_frames.append(data)
        return WriteResult(strategy=WriteStrategy.MERGE, rows_inserted=data.height)

    monkeypatch.setattr(step, "_load_records", lambda _db, _ctx: source_df)
    monkeypatch.setattr(step, "_save_records", capture_save)

    step.run(db_client, PipelineContext(account_id=_ACCOUNT_ID))

    assert len(saved_frames) == 1
    return saved_frames[0], db_client


def test_transform_step_incremental_filter_on_skips_rows_in_target_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_df, db_client = _run_transform_step_and_capture_saved(
        monkeypatch,
        on="game_id",
        target_game_ids=["game-1"],
    )

    assert saved_df.height == 2
    assert saved_df["game_id"].to_list() == ["game-2", "game-3"]

    sql = db_client.engine.execute_parameterized_query.call_args[0][0]
    assert 'FROM "games"."games"' in sql
    assert "\"account_id\" = 'acct-1'" in sql


def test_transform_step_incremental_filter_off_keeps_all_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_df, db_client = _run_transform_step_and_capture_saved(
        monkeypatch,
        on=None,
        target_game_ids=["game-1", "game-2"],
    )

    assert saved_df.height == 3
    assert saved_df["game_id"].to_list() == ["game-1", "game-2", "game-3"]
    db_client.engine.execute_parameterized_query.assert_not_called()


def test_transform_step_skips_save_when_incremental_filter_removes_all_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty frame after incremental filter must not run later transforms (e.g. PGN filter)."""
    from chess_teacher.pipelines.preprocessing.pipeline_steps import RawGamesToGamesStep

    monkeypatch.setattr(
        "chess_teacher.utils.pipeline_utils.transformations.get_db_client",
        lambda: MagicMock(),
    )

    source_df = _source_table_df()
    db_client = _mock_db_client(target_game_ids=["game-1", "game-2", "game-3"])

    step = RawGamesToGamesStep()
    step._incremental_filter.db_client = db_client

    saved_frames: list[pl.DataFrame] = []

    def capture_save(
        _db_client: MagicMock,
        _table_metadata: object,
        data: pl.DataFrame,
    ) -> WriteResult:
        saved_frames.append(data)
        return WriteResult(strategy=WriteStrategy.MERGE, rows_inserted=data.height)

    monkeypatch.setattr(step, "_load_records", lambda _db, _ctx: source_df)
    monkeypatch.setattr(step, "_save_records", capture_save)

    step.run(db_client, PipelineContext(account_id=_ACCOUNT_ID))

    assert saved_frames == []
