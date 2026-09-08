"""TransformStep batched load: keyset pages + save per page."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import polars as pl
import pytest

from chess_teacher.pipelines.ingestion.raw_games import RawGame
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.pipeline_steps import RawGamesToGamesStep
from chess_teacher.utils.db.client import WriteResult, WriteStrategy
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext
from chess_teacher.utils.pipeline_utils.pipeline_steps import LoadingStrategy, TransformStep

_ACCOUNT_ID = "acct-1"
_INGESTED_AT = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


def _raw_row(game_id: str) -> dict[str, object]:
    return {
        "game_id": game_id,
        "platform_game_id": f"platform-{game_id}",
        "account_id": _ACCOUNT_ID,
        "raw_response": "{}",
        "source_file": f"ingested/{_ACCOUNT_ID}/{game_id}.jsonl",
        "ingested_at": _INGESTED_AT,
    }


def test_raw_games_to_games_step_enables_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chess_teacher.utils.pipeline_utils.transformations.get_db_client",
        lambda: MagicMock(),
    )
    step = RawGamesToGamesStep()
    assert step.batch_size == 500


def test_transform_step_batched_run_saves_each_page(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = {
        None: pl.DataFrame([_raw_row("game-1"), _raw_row("game-2")]),
        "game-2": pl.DataFrame([_raw_row("game-3")]),
        "game-3": pl.DataFrame(),
    }
    load_calls: list[str | None] = []

    def load_page(
        _db: MagicMock, _ctx: PipelineContext, *, after_key: str | None = None
    ) -> pl.DataFrame:
        load_calls.append(after_key)
        return pages[after_key]

    db_client = MagicMock()
    db_client.ensure_metadata.return_value = None
    db_client.table_exists.return_value = True

    step = TransformStep(
        name="TestBatchedRawGames",
        source_data_class=RawGame,
        target_data_class=Game,
        on="game_id",
        transformations=[],
        loading_strategy=LoadingStrategy.MERGE,
        batch_size=2,
    )
    step._incremental_filter.db_client = db_client
    step.transformations = [step._incremental_filter]

    saved_heights: list[int] = []

    def capture_save(
        _db_client: MagicMock,
        _table_metadata: object,
        data: pl.DataFrame,
    ) -> WriteResult:
        saved_heights.append(data.height)
        return WriteResult(strategy=WriteStrategy.MERGE, rows_inserted=data.height)

    monkeypatch.setattr(step, "_load_records", load_page)
    monkeypatch.setattr(step, "_save_records", capture_save)

    step.run(db_client, PipelineContext(account_id=_ACCOUNT_ID))

    assert load_calls == [None, "game-2", "game-3"]
    assert saved_heights == [2, 1]


def test_transform_step_load_records_batch_adds_order_limit_and_keyset() -> None:
    db_client = MagicMock()
    db_client.table_exists.return_value = True
    db_client.read.return_value = pl.DataFrame()

    step = TransformStep(
        name="TestBatchedRawGames",
        source_data_class=RawGame,
        target_data_class=Game,
        on="game_id",
        transformations=[],
        loading_strategy=LoadingStrategy.MERGE,
        batch_size=500,
    )
    step._incremental_filter.set_scope_where("\"account_id\" = 'acct-1'")
    step._load_records(db_client, PipelineContext(account_id=_ACCOUNT_ID), after_key="game-100")

    kwargs = db_client.read.call_args.kwargs
    assert kwargs["limit"] == 500
    assert kwargs["order_by"] == '"game_id"'
    assert "\"game_id\" > 'game-100'" in kwargs["where"]
    assert "NOT EXISTS" in kwargs["where"]
