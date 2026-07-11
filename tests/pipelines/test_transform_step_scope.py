from unittest.mock import MagicMock

import polars as pl
import pytest

from chess_teacher.pipelines.ingestion.raw_games import RawGame
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.utils.exception_utils import PipelineError
from chess_teacher.utils.metadata_utils import ColumnMetadata, TableMetadata
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext
from chess_teacher.utils.pipeline_utils.pipeline_steps import (
    LoadingStrategy,
    LoadToDatabaseStep,
    TransformStep,
)


def test_context_where_clause_filters_by_account_id() -> None:
    where = TransformStep._context_where_clause(
        Game.get_metadata(),
        PipelineContext(account_id="acct-1"),
    )
    assert where == "\"account_id\" = 'acct-1'"


def test_context_where_clause_filters_by_user_id_when_no_account_id() -> None:
    table = TableMetadata(
        schema_name="platform",
        table_name="users",
        columns=(ColumnMetadata(name="user_id", data_type="text", nullable=False),),
        primary_key=("user_id",),
    )
    where = TransformStep._context_where_clause(
        table,
        PipelineContext(user_id="user-1"),
    )
    assert where == "\"user_id\" = 'user-1'"


def test_context_where_clause_prefers_account_id_over_user_id() -> None:
    table = TableMetadata(
        schema_name="demo",
        table_name="scoped",
        columns=(
            ColumnMetadata(name="account_id", data_type="text", nullable=False),
            ColumnMetadata(name="user_id", data_type="text", nullable=False),
        ),
        primary_key=("account_id",),
    )
    where = TransformStep._context_where_clause(
        table,
        PipelineContext(user_id="user-1", account_id="acct-1"),
    )
    assert where == "\"account_id\" = 'acct-1'"


def test_context_where_clause_returns_none_without_scope() -> None:
    where = TransformStep._context_where_clause(
        Game.get_metadata(),
        PipelineContext(),
    )
    assert where is None


def test_context_where_clause_raises_when_account_column_missing() -> None:
    table = TableMetadata(
        schema_name="other",
        table_name="reference",
        columns=(ColumnMetadata(name="code", data_type="text", nullable=False),),
        primary_key=("code",),
    )
    with pytest.raises(PipelineError, match="account_id column"):
        TransformStep._context_where_clause(
            table,
            PipelineContext(account_id="acct-1"),
        )


def test_context_where_clause_raises_when_user_column_missing() -> None:
    table = TableMetadata(
        schema_name="other",
        table_name="reference",
        columns=(ColumnMetadata(name="code", data_type="text", nullable=False),),
        primary_key=("code",),
    )
    with pytest.raises(PipelineError, match="user_id column"):
        TransformStep._context_where_clause(
            table,
            PipelineContext(user_id="user-1"),
        )


def test_incremental_filter_is_same_instance_as_first_transformation() -> None:
    step = TransformStep(
        name="TestStep",
        source_data_class=RawGame,
        target_data_class=Game,
        on="game_id",
        loading_strategy=LoadingStrategy.MERGE,
    )
    assert step.transformations[0] is step._incremental_filter


def test_transform_step_run_sets_scope_where_before_parent_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = TransformStep(
        name="TestStep",
        source_data_class=RawGame,
        target_data_class=Game,
        on="game_id",
        loading_strategy=LoadingStrategy.MERGE,
    )
    order: list[str] = []

    def track_set_scope(where: str | None) -> None:
        order.append("set_scope")
        step._incremental_filter.scope_where = where

    monkeypatch.setattr(step._incremental_filter, "set_scope_where", track_set_scope)

    def parent_run(
        self: LoadToDatabaseStep,
        db_client: MagicMock,
        context: PipelineContext,
    ) -> None:
        order.append("parent_run")
        assert step._incremental_filter.scope_where == "\"account_id\" = 'acct-1'"

    monkeypatch.setattr(LoadToDatabaseStep, "run", parent_run)

    step.run(MagicMock(), PipelineContext(account_id="acct-1"))

    assert order == ["set_scope", "parent_run"]


def test_transform_step_run_applies_scope_during_transform_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = TransformStep(
        name="TestStep",
        source_data_class=RawGame,
        target_data_class=Game,
        on="game_id",
        loading_strategy=LoadingStrategy.MERGE,
    )
    scope_during_transform: dict[str, str | None] = {}

    class StopAfterIncrementalFilterError(Exception):
        pass

    def capture_scope(df: pl.DataFrame) -> pl.DataFrame:
        scope_during_transform["value"] = step._incremental_filter.scope_where
        raise StopAfterIncrementalFilterError

    monkeypatch.setattr(step._incremental_filter, "transform", capture_scope)
    monkeypatch.setattr(
        step,
        "_load_records",
        lambda db_client, context: pl.DataFrame({
            "game_id": ["game-1"],
            "account_id": ["acct-1"],
        }),
    )

    db_client = MagicMock()
    db_client.ensure_metadata.return_value = None

    with pytest.raises(StopAfterIncrementalFilterError):
        step.run(db_client, PipelineContext(account_id="acct-1"))

    assert scope_during_transform["value"] == "\"account_id\" = 'acct-1'"
