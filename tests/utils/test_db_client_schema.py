import pytest

from chess_teacher.utils.db_client import (
    _alter_column_type_sql,
    _require_pg_data_type,
    _require_using_expression,
)


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
