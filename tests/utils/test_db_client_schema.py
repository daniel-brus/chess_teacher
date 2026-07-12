import polars as pl
import pytest

from chess_teacher.platform.user import User
from chess_teacher.utils.db.client import (
    _alter_column_type_sql,
    _column_defaults_equivalent,
    _normalize_default_literal,
    _normalize_pg_data_type_for_compare,
    _pg_data_types_equivalent,
    _require_pg_data_type,
    _require_using_expression,
    _rows_to_polars_dataframe,
)
from chess_teacher.utils.metadata_utils import ColumnMetadata, TableMetadata


def test_require_pg_data_type_accepts_common_types() -> None:
    assert _require_pg_data_type("text") == "text"
    assert _require_pg_data_type("TIMESTAMPTZ") == "timestamptz"


def test_require_pg_data_type_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid PostgreSQL data type"):
        _require_pg_data_type("text; drop table")


def test_alter_column_type_sql_default_using() -> None:
    sql = _alter_column_type_sql('"games"."raw_games"', "eco_code", "integer", using=None)
    assert sql == (
        'ALTER TABLE "games"."raw_games" '
        'ALTER COLUMN "eco_code" TYPE integer '
        'USING "eco_code"::integer;'
    )


def test_alter_column_type_sql_custom_using() -> None:
    sql = _alter_column_type_sql(
        '"games"."raw_games"',
        "rating",
        "integer",
        using="NULLIF(trim(rating), '')::integer",
    )
    assert "USING NULLIF(trim(rating), '')::integer;" in sql


def test_require_using_expression_rejects_semicolon() -> None:
    with pytest.raises(ValueError, match="semicolons"):
        _require_using_expression("1; DROP TABLE x")


def test_normalize_pg_data_type_timestamptz_alias() -> None:
    assert _normalize_pg_data_type_for_compare(
        "timestamptz"
    ) == _normalize_pg_data_type_for_compare("timestamp with time zone")


def test_normalize_pg_data_type_time_alias() -> None:
    assert _normalize_pg_data_type_for_compare("time") == _normalize_pg_data_type_for_compare(
        "time without time zone"
    )


def test_pg_data_types_equivalent_accepts_matching_text() -> None:
    assert _pg_data_types_equivalent("text", "text")


def test_pg_data_types_equivalent_rejects_real_mismatch() -> None:
    assert not _pg_data_types_equivalent("text", "integer")


def test_normalize_default_literal_boolean() -> None:
    assert _normalize_default_literal("FALSE") == "false"
    assert _normalize_default_literal("false") == "false"


def test_normalize_default_literal_strips_cast() -> None:
    assert _normalize_default_literal("'Free'::text") == "Free"
    assert _normalize_default_literal("'03:00:00'::time without time zone") == "03:00:00"


def test_old_default_comparison_was_false_positive_for_boolean() -> None:
    col = ColumnMetadata(
        name="email_verified",
        data_type="boolean",
        nullable=False,
        default=False,
    )
    assert str(col.default) != "false"
    assert _column_defaults_equivalent(col, "false")


def test_users_table_login_false_positive_columns() -> None:
    """Regression: information_schema shapes that triggered ensure_metadata every login."""
    cols = User.get_metadata().columns_by_name()

    assert _pg_data_types_equivalent(
        cols["latest_login"].data_type,
        "timestamp with time zone",
    )
    assert _pg_data_types_equivalent(cols["latest_pipeline_run"].data_type, "text")
    assert _pg_data_types_equivalent(cols["cron_time"].data_type, "time without time zone")

    assert _column_defaults_equivalent(cols["email_verified"], "false")
    assert _column_defaults_equivalent(cols["tier"], "'Free'::text")
    assert _column_defaults_equivalent(
        cols["cron_time"],
        "'03:00:00'::time without time zone",
    )
    assert _column_defaults_equivalent(cols["timezone"], "'Europe/Amsterdam'::text")


def test_create_indexes_sql_single_column() -> None:
    table = TableMetadata._from_dict_raw({
        "schema": "games",
        "table": "moves",
        "primary_key": ["move_id"],
        "indexes": [{"name": "idx_moves_game_id", "columns": ["game_id"]}],
        "columns": [
            {"name": "move_id", "data_type": "text", "nullable": False},
            {"name": "game_id", "data_type": "text", "nullable": False},
        ],
    })
    assert table.indexes[0].name == "idx_moves_game_id"
    assert table.indexes[0].columns == ("game_id",)
    sql = table.create_indexes_sql()
    assert len(sql) == 1
    assert (
        sql[0] == 'CREATE INDEX IF NOT EXISTS "idx_moves_game_id" ON "games"."moves" ("game_id");'
    )


def test_create_indexes_sql_composite_column() -> None:
    table = TableMetadata._from_dict_raw({
        "schema": "games",
        "table": "moves",
        "primary_key": ["move_id"],
        "indexes": [{"name": "idx_moves_game_move", "columns": ["game_id", "move_nr"]}],
        "columns": [
            {"name": "move_id", "data_type": "text", "nullable": False},
            {"name": "game_id", "data_type": "text", "nullable": False},
            {"name": "move_nr", "data_type": "integer", "nullable": False},
        ],
    })
    sql = table.create_indexes_sql()
    assert (
        sql[0] == 'CREATE INDEX IF NOT EXISTS "idx_moves_game_move" ON "games"."moves" '
        '("game_id", "move_nr");'
    )


def test_parse_indexes_skips_primary_key_duplicate() -> None:
    table = TableMetadata._from_dict_raw({
        "schema": "games",
        "table": "moves",
        "primary_key": ["move_id"],
        "indexes": [{"name": "idx_moves_pk", "columns": ["move_id"]}],
        "columns": [
            {"name": "move_id", "data_type": "text", "nullable": False},
        ],
    })
    assert table.indexes == ()


def test_rows_to_polars_dataframe_uses_metadata_for_nullable_text() -> None:
    table = TableMetadata._from_dict_raw({
        "schema": "games",
        "table": "moves",
        "primary_key": ["move_id"],
        "columns": [
            {"name": "move_id", "data_type": "text", "nullable": False},
            {
                "name": "previous_opponent_move_san",
                "data_type": "text",
                "nullable": True,
            },
        ],
    })
    rows = [{"move_id": f"id-{index}", "previous_opponent_move_san": None} for index in range(120)]
    rows.append({"move_id": "id-120", "previous_opponent_move_san": "e4"})

    with pytest.raises(pl.exceptions.ComputeError, match='could not append value: "e4"'):
        pl.DataFrame(rows)

    df = _rows_to_polars_dataframe(rows, table)
    assert df.schema["previous_opponent_move_san"] == pl.String
    assert df["previous_opponent_move_san"].item(-1) == "e4"
