import pytest

from chess_teacher.ingestion.raw_games import RawGame
from chess_teacher.pipelines.pipeline_base import PipelineContext
from chess_teacher.pipelines.pipeline_steps import TransformStep
from chess_teacher.utils.exception_utils import PipelineError
from chess_teacher.utils.metadata_utils import ColumnMetadata, TableMetadata


def test_context_where_clause_filters_by_account_id() -> None:
    where = TransformStep._context_where_clause(
        RawGame.get_metadata(),
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
        RawGame.get_metadata(),
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
