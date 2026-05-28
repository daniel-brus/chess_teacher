from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import StrEnum
from typing import Any, Literal, overload
from uuid import uuid4

import polars as pl
from sqlalchemy import text

from chess_teacher.utils.db_engine import EnrichedEngine, get_db_engine
from chess_teacher.utils.exception_utils import DatabaseError, MetadataError
from chess_teacher.utils.general_utils import generate_ident_is_literal, quote_ident, quote_literal
from chess_teacher.utils.logging_utils import get_logger
from chess_teacher.utils.metadata_utils import TableMetadata

MERGE_COPY_THRESHOLD = 1000

# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


WhenMatched = Literal["update", "delete", "ignore"]
WhenNotMatchedByTarget = Literal["insert", "ignore"]
WhenNotMatchedBySource = Literal["delete", "ignore"]


class WriteStrategy(StrEnum):
    APPEND = "append"
    INSERT_IGNORE = "insert_ignore"
    OVERWRITE = "overwrite"
    MERGE = "merge"


@dataclass(frozen=True, slots=True)
class MergeStrategy:
    """Postgres MERGE behaviour for matched / unmatched source and target rows."""

    when_matched: WhenMatched = "update"
    when_not_matched_by_target: WhenNotMatchedByTarget = "insert"
    when_not_matched_by_source: WhenNotMatchedBySource = "ignore"

    def __post_init__(self) -> None:
        if (
            self.when_matched == "ignore"
            and self.when_not_matched_by_target == "ignore"
            and self.when_not_matched_by_source == "ignore"
        ):
            raise ValueError(
                "Invalid MergeStrategy: all when_* actions are 'ignore' — "
                "merge would produce no SQL clauses."
            )

    @classmethod
    def upsert(cls) -> MergeStrategy:
        return cls()

    @classmethod
    def full_sync(cls) -> MergeStrategy:
        return cls(when_not_matched_by_source="delete")

    @classmethod
    def insert_new(cls) -> MergeStrategy:
        return cls(when_matched="ignore")


@dataclass
class WriteResult:
    strategy: WriteStrategy
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_deleted: int = 0

    @property
    def total_affected(self) -> int:
        return self.rows_inserted + self.rows_updated + self.rows_deleted


@dataclass
class SchemaDiff:
    """Result of schema_diff() — describes divergence between TableMetadata and live DB."""

    missing_columns: list[str] = field(default_factory=list)  # in metadata, not in DB
    extra_columns: list[str] = field(default_factory=list)  # in DB, not in metadata
    type_mismatches: dict[str, tuple[str, str]] = field(
        default_factory=dict
    )  # col -> (expected, actual)
    nullable_mismatches: dict[str, tuple[bool, bool]] = field(
        default_factory=dict
    )  # col -> (expected, actual)
    default_mismatches: dict[str, tuple[Any, Any]] = field(
        default_factory=dict
    )  # col -> (expected, actual)
    comment_mismatches: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)

    @property
    def is_match(self) -> bool:
        return not (
            self.missing_columns
            or self.extra_columns
            or self.type_mismatches
            or self.nullable_mismatches
            or self.default_mismatches
            or self.comment_mismatches
        )

    @property
    def is_destructive(self) -> bool:
        """True if resolving this diff would require dropping columns or data."""
        return bool(self.extra_columns or self.type_mismatches)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_records(data: list[dict] | pl.DataFrame) -> list[dict]:
    """Normalise input to list[dict]."""
    if isinstance(data, pl.DataFrame):
        return data.to_dicts()
    if isinstance(data, list):
        return data
    raise TypeError(f"Expected list[dict] or pl.DataFrame, got {type(data)}")


def _require_where(where: str | None, operation: str) -> str:
    """Guard against accidental full-table mutations."""
    if not where or not where.strip():
        raise ValueError(
            f"'{operation}' requires an explicit WHERE clause. "
            "Use truncate_table() if you intend to affect all rows."
        )
    return where.strip()


def _build_insert_sql(
    records: list[dict],
    table: TableMetadata,
    *,
    on_conflict: Literal["error", "nothing"] = "error",
) -> tuple[str, list[dict]]:
    """Build parameterised INSERT statement.

    Returns (sql_template, records) where sql_template uses
    SQLAlchemy :col_name bindparam syntax.
    """
    if not records:
        raise ValueError("Cannot insert empty dataset.")

    col_names = list(records[0].keys())
    quoted_cols = ", ".join(quote_ident(c) for c in col_names)
    placeholders = ", ".join(f":{c}" for c in col_names)
    base = f"INSERT INTO {table.qualified_name_sql()} ({quoted_cols})\nVALUES ({placeholders})"
    if on_conflict == "nothing":
        base += "\nON CONFLICT DO NOTHING"
    return base, records


def _value_to_typed_sql(value: Any, data_type: str) -> str:
    if value is None:
        return f"NULL::{data_type}"
    if isinstance(value, bool):
        literal = "TRUE" if value else "FALSE"
    elif isinstance(value, int | float):
        literal = str(value)
    elif isinstance(value, date | datetime | time):
        literal = quote_literal(value.isoformat())
    else:
        literal = quote_literal(str(value))
    return f"{literal}::{data_type}"


def _build_source_cte(records: list[dict], table: TableMetadata) -> str:
    col_names = list(records[0].keys())
    columns_by_name = table.columns_by_name()

    def row_to_sql(row: dict) -> str:
        return (
            "("
            + ", ".join(
                _value_to_typed_sql(row.get(c), columns_by_name[c].data_type) for c in col_names
            )
            + ")"
        )

    values_rows = ",\n    ".join(row_to_sql(r) for r in records)
    quoted_cols_csv = ", ".join(quote_ident(c) for c in col_names)
    return f"WITH _source({quoted_cols_csv}) AS (\n  VALUES\n    {values_rows}\n)"


def _staging_table_name() -> str:
    return f"_merge_staging_{uuid4().hex}"


def _build_create_staging_sql(staging_name: str, col_names: list[str], table: TableMetadata) -> str:
    columns_by_name = table.columns_by_name()
    col_defs = ", ".join(f"{quote_ident(c)} {columns_by_name[c].data_type}" for c in col_names)
    return f"CREATE TEMP TABLE {quote_ident(staging_name)} ({col_defs}) ON COMMIT DROP"


def _build_join_condition(match_keys: list[str], match_condition: str | None) -> str:
    join_condition = " AND ".join(
        f"_target.{quote_ident(k)} = _source.{quote_ident(k)}" for k in match_keys
    )
    if match_condition:
        join_condition = f"({join_condition}) AND ({match_condition})"
    return join_condition


def _build_count_matched_sql(
    table: TableMetadata,
    join_condition: str,
    *,
    source_prefix: str = "",
    source_from: str = "_source",
) -> str:
    return (
        f"{source_prefix}"
        f"SELECT COUNT(*) AS matched_count\n"
        f"FROM {source_from}\n"
        f"WHERE EXISTS (\n"
        f"  SELECT 1 FROM {table.qualified_name_sql()} _target\n"
        f"  WHERE {join_condition}\n"
        f")"
    )


def _build_count_deletes_sql(
    table: TableMetadata,
    join_condition: str,
    *,
    source_prefix: str = "",
    source_from: str = "_source",
) -> str:
    return (
        f"{source_prefix}"
        f"SELECT COUNT(*) AS delete_count\n"
        f"FROM {table.qualified_name_sql()} _target\n"
        f"WHERE NOT EXISTS (\n"
        f"  SELECT 1 FROM {source_from}\n"
        f"  WHERE {join_condition}\n"
        f")"
    )


def _build_merge_sql(
    table: TableMetadata,
    col_names: list[str],
    *,
    match_keys: list[str],
    strategy: MergeStrategy,
    match_condition: str | None,
    source_prefix: str = "",
    source_using: str = "_source",
) -> str:
    """Build a Postgres 16 MERGE statement from an inline CTE or staging table source."""
    if not col_names:
        raise ValueError("Cannot merge empty dataset.")
    if not match_keys:
        raise ValueError("merge() requires at least one match_key.")

    non_match_cols = [c for c in col_names if c not in match_keys]
    quoted_cols_csv = ", ".join(quote_ident(c) for c in col_names)
    join_condition = _build_join_condition(match_keys, match_condition)

    merge_head = (
        f"{source_prefix}"
        f"MERGE INTO {table.qualified_name_sql()} AS _target\n"
        f"USING {source_using}\n"
        f"ON {join_condition}"
    )

    clauses: list[str] = []

    # WHEN MATCHED
    if strategy.when_matched == "update" and non_match_cols:
        set_clause = ", ".join(
            f"{quote_ident(c)} = _source.{quote_ident(c)}" for c in non_match_cols
        )
        clauses.append(f"WHEN MATCHED THEN\n  UPDATE SET {set_clause}")
    elif strategy.when_matched == "delete":
        clauses.append("WHEN MATCHED THEN\n  DELETE")
    # "ignore" → no WHEN MATCHED clause

    # WHEN NOT MATCHED BY TARGET
    if strategy.when_not_matched_by_target == "insert":
        clauses.append(
            f"WHEN NOT MATCHED THEN\n"
            f"  INSERT ({quoted_cols_csv})\n"
            f"  VALUES ({', '.join(f'_source.{quote_ident(c)}' for c in col_names)})"
        )

    # WHEN NOT MATCHED BY SOURCE (Postgres 16+)
    if strategy.when_not_matched_by_source == "delete":
        clauses.append("WHEN NOT MATCHED BY SOURCE THEN\n  DELETE")

    if not clauses:
        raise ValueError("merge() produced no action clauses — check MergeStrategy.")

    return merge_head + "\n" + "\n".join(clauses) + ";"


# ---------------------------------------------------------------------------
# DatabaseClient
# ---------------------------------------------------------------------------


class DatabaseClient:
    """Higher-level database client for reads, writes, and targeted mutations.

    Args:
        engine: Optional pre-built EnrichedEngine. If omitted, one is created
                from environment variables via get_db_engine().
    """

    def __init__(self, engine: EnrichedEngine | None = None) -> None:
        self.engine = engine or get_db_engine()
        self.logger = get_logger()

    # ------------------------------------------------------------------
    # Write strategies
    # ------------------------------------------------------------------

    def insert(
        self,
        data: list[dict] | pl.DataFrame,
        table: TableMetadata,
        *,
        on_conflict: Literal["error", "nothing"] = "error",
    ) -> WriteResult:
        """Insert records with configurable conflict handling.

        Args:
            data: Records to insert (list of dicts or Polars DataFrame)
            table: Target table metadata
            on_conflict: "error" raises on PK conflict, "nothing" silently skips conflicting rows

        Returns:
            WriteResult with rows_inserted count
        """
        records = _to_records(data)
        if not records:
            strategy = (
                WriteStrategy.APPEND if on_conflict == "error" else WriteStrategy.INSERT_IGNORE
            )
            self.logger.debug("insert → %s: no records to insert", table.qualified_name_sql())
            return WriteResult(strategy=strategy, rows_inserted=0)

        try:
            sql, records = _build_insert_sql(records, table, on_conflict=on_conflict)
            inserted = self.engine.execute_write(sql, records) if records else 0

            if on_conflict == "error":
                self.logger.debug(
                    "insert → %s: %d rows inserted", table.qualified_name_sql(), inserted
                )
            else:
                self.logger.debug(
                    "insert → %s: %d/%d rows inserted (skipped %d conflicts)",
                    table.qualified_name_sql(),
                    inserted,
                    len(records),
                    len(records) - inserted,
                )
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"Error occurred while inserting data to {table.qualified_name_sql()}: {e}"
                )
            )

        strategy = WriteStrategy.APPEND if on_conflict == "error" else WriteStrategy.INSERT_IGNORE
        return WriteResult(strategy=strategy, rows_inserted=inserted)

    def overwrite(
        self,
        data: list[dict] | pl.DataFrame,
        table: TableMetadata,
        *,
        cascade: bool = False,
    ) -> WriteResult:
        """TRUNCATE then INSERT. Full table replacement.

        Args:
            cascade: Pass TRUNCATE ... CASCADE to handle foreign-key dependents.
        """
        records = _to_records(data)
        self.truncate_table(table, cascade=cascade)

        if records:
            sql_insert, records = _build_insert_sql(records, table, on_conflict="error")
            self.engine.execute_parameterized_query(sql_insert, records)

        self.logger.info(
            "overwrite → %s: table truncated, %d rows inserted",
            table.qualified_name_sql(),
            len(records),
        )
        return WriteResult(strategy=WriteStrategy.OVERWRITE, rows_inserted=len(records))

    def merge(
        self,
        data: list[dict] | pl.DataFrame,
        table: TableMetadata,
        *,
        strategy: MergeStrategy = MergeStrategy.upsert(),
        match_keys: list[str] | None = None,
        match_condition: str | None = None,
        use_copy: bool = False,
    ) -> WriteResult:
        """Postgres MERGE with row count tracking.

        Large loads (more than ``MERGE_COPY_THRESHOLD`` rows, or ``use_copy=True``) stage
        source rows in a temp table via psycopg3 COPY, then MERGE from that table.

        Args:
            strategy: Merge behaviour (defaults to upsert via MergeStrategy.upsert()).
            match_keys: Columns to join on. Defaults to table.primary_key.
            match_condition: Optional extra SQL condition appended to the ON clause.
            use_copy: When True, always use COPY staging. When False, staging is still used
                if the record count exceeds ``MERGE_COPY_THRESHOLD``.
        """
        records = _to_records(data)
        if not records:
            self.logger.debug("merge → %s: no records to merge", table.qualified_name_sql())
            return WriteResult(strategy=WriteStrategy.MERGE)

        resolved_keys = match_keys or list(table.primary_key)
        if not resolved_keys:
            self.logger.log_and_raise(
                ValueError("merge() requires match_keys or a primary_key defined on TableMetadata.")
            )

        use_staging = use_copy or len(records) > MERGE_COPY_THRESHOLD
        source_mode = "copy" if use_staging else "inline"

        try:
            if use_staging:
                matched_count, deleted_count = self._merge_via_copy(
                    records,
                    table,
                    resolved_keys=resolved_keys,
                    strategy=strategy,
                    match_condition=match_condition,
                )
            else:
                matched_count = self._count_matches(records, table, resolved_keys, match_condition)
                deleted_count = 0
                if strategy.when_not_matched_by_source == "delete":
                    deleted_count = self._count_deletes(
                        records, table, resolved_keys, match_condition
                    )
                col_names = list(records[0].keys())
                source_prefix = _build_source_cte(records, table) + "\n"
                sql = _build_merge_sql(
                    table,
                    col_names,
                    match_keys=resolved_keys,
                    strategy=strategy,
                    match_condition=match_condition,
                    source_prefix=source_prefix,
                    source_using="_source",
                )
                self.engine.execute_statements([sql])
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"Error occurred while merging data into {table.qualified_name_sql()}: {e}"
                )
            )

        non_matched_count = len(records) - matched_count
        rows_inserted = non_matched_count if strategy.when_not_matched_by_target == "insert" else 0
        rows_updated = matched_count if strategy.when_matched == "update" else 0

        self.logger.debug(
            "merge → %s: inserted=%d, updated=%d, deleted=%d (source=%d records, mode=%s)",
            table.qualified_name_sql(),
            rows_inserted,
            rows_updated,
            deleted_count,
            len(records),
            source_mode,
        )
        return WriteResult(
            strategy=WriteStrategy.MERGE,
            rows_inserted=rows_inserted,
            rows_updated=rows_updated,
            rows_deleted=deleted_count,
        )

    # ------------------------------------------------------------------
    # Targeted mutations
    # ------------------------------------------------------------------

    def update_where(
        self,
        table: TableMetadata,
        values: dict[str, Any],
        where: str,
    ) -> int:
        """UPDATE specific columns for rows matching WHERE clause.

        Args:
            values: Column → new value mapping.
            where:  SQL WHERE clause (required — no full-table updates).

        Returns:
            Number of affected rows.
        """
        try:
            _require_where(where, "update_where")
            if not values:
                self.logger.log_and_raise(
                    ValueError("update_where() requires at least one column to update.")
                )

            set_clause = ", ".join(
                generate_ident_is_literal(col, val) for col, val in values.items()
            )
            sql = f"UPDATE {table.qualified_name_sql()}\nSET {set_clause}\nWHERE {where};"
            affected = self.engine.execute_write(sql, {}) if values else 0
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"Error occurred while updating data in {table.qualified_name_sql()}: {e}"
                )
            )
        self.logger.debug(
            "update_where → %s: %d rows updated", table.qualified_name_sql(), affected
        )
        return affected

    def delete_where(
        self,
        table: TableMetadata,
        where: str,
    ) -> int:
        """DELETE rows matching WHERE clause.

        Args:
            where: SQL WHERE clause (required — no full-table deletes).

        Returns:
            Number of deleted rows.
        """
        try:
            _require_where(where, "delete_where")
            sql = f"DELETE FROM {table.qualified_name_sql()}\nWHERE {where};"
            affected = self.engine.execute_write(sql, {})
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"Error occurred while deleting data from {table.qualified_name_sql()}: {e}"
                )
            )
        self.logger.debug(
            "delete_where → %s: %d rows deleted", table.qualified_name_sql(), affected
        )
        return affected

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def schema_exists(self, table: TableMetadata) -> bool:
        """Return True if the schema exists in the database."""
        sql = """
            SELECT 1 FROM information_schema.schemata
            WHERE schema_name = :schema
        """
        try:
            result = self.engine.execute_parameterized_query(sql, {"schema": table.schema_name})
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"Error occurred while checking schema existence for {table.schema_name}: {e}"
                )
            )
        return len(result) > 0 if result else False

    def table_exists(self, table: TableMetadata) -> bool:
        """Return True if the table exists in the database."""
        sql = """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = :schema
              AND table_name   = :table
        """
        try:
            result = self.engine.execute_parameterized_query(
                sql, {"schema": table.schema_name, "table": table.table_name}
            )
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"Error occurred while checking table existence for {table.table_name}: {e}"
                )
            )
        return len(result) > 0 if result else False

    def schema_diff(self, table: TableMetadata) -> SchemaDiff:
        """Compare TableMetadata against the live table definition.

        Checks: column presence, data types, nullability, defaults, and comments.
        Does NOT check primary key constraints (out of scope for now).

        Returns:
            SchemaDiff dataclass. Call .is_match to check for clean state.
        """
        # --- fetch live column info ---
        col_sql = """
            SELECT
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                pgd.description AS comment
            FROM information_schema.columns c
            LEFT JOIN pg_catalog.pg_statio_all_tables st
                ON st.schemaname = c.table_schema
               AND st.relname    = c.table_name
            LEFT JOIN pg_catalog.pg_description pgd
                ON pgd.objoid    = st.relid
               AND pgd.objsubid  = c.ordinal_position
            WHERE c.table_schema = :schema
              AND c.table_name   = :table
        """
        try:
            rows = self.engine.execute_parameterized_query(
                col_sql, {"schema": table.schema_name, "table": table.table_name}
            )

            if not rows:
                # Table doesn't exist — treat all metadata columns as missing
                return SchemaDiff(missing_columns=[c.name for c in table.columns])

            live: dict[str, dict[str, Any]] = {
                r["column_name"]: {
                    "data_type": r["data_type"].lower(),
                    "nullable": r["is_nullable"] == "YES",
                    "default": r["column_default"],
                    "comment": r["comment"],
                }
                for r in rows
            }
            meta: dict[str, Any] = {c.name: c for c in table.columns}
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"Error occurred while fetching column info for {table.qualified_name_sql()}: {e}"
                )
            )

        try:
            diff = SchemaDiff()

            diff.missing_columns = [name for name in meta if name not in live]
            diff.extra_columns = [name for name in live if name not in meta]

            for name, col in meta.items():
                if name not in live:
                    continue  # already captured in missing_columns
                live_col = live[name]

                if col.data_type != live_col["data_type"]:
                    diff.type_mismatches[name] = (col.data_type, live_col["data_type"])

                if col.nullable != live_col["nullable"]:
                    diff.nullable_mismatches[name] = (col.nullable, live_col["nullable"])

                # Normalize default: metadata stores raw value, DB stores SQL expression
                meta_default = str(col.default) if col.default is not None else None
                live_default = live_col["default"]
                if meta_default != live_default:
                    diff.default_mismatches[name] = (meta_default, live_default)

                meta_comment = col.comment
                live_comment = live_col["comment"]
                if meta_comment != live_comment:
                    diff.comment_mismatches[name] = (meta_comment, live_comment)
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"Error occurred while comparing schema for {table.qualified_name_sql()}: {e}"
                )
            )
        return diff

    def get_row_count(self, table: TableMetadata, *, where: str | None = None) -> int:
        """Return the number of rows in the table, optionally filtered.

        Args:
            where: Optional WHERE clause. Required to pass _require_where
                   check when provided — avoids silent full-table scans
                   being mistaken for filtered counts.
        """
        where_clause = ""
        try:
            if where is not None:
                where_clause = f"WHERE {_require_where(where, 'get_row_count')}"

            sql = f"SELECT COUNT(*) FROM {table.qualified_name_sql()} {where_clause};"
            result = self.engine.execute_parameterized_query(sql, {})
            count = result[0]["count"] if result else 0
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"Error occurred while fetching row count for {table.qualified_name_sql()}: {e}"
                )
            )
        return count

    def exists(self, table: TableMetadata, where: str) -> bool:
        """Return True if at least one row matches the WHERE clause.

        Args:
            where: SQL WHERE clause (required).
        """
        try:
            _require_where(where, "exists")
            sql = f"SELECT EXISTS (SELECT 1 FROM {table.qualified_name_sql()} WHERE {where});"
            rows = self.engine.execute_parameterized_query(sql, {})
            result = bool(rows[0]["exists"]) if rows else False
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"Error occurred while checking existence for {table.qualified_name_sql()}: {e}"
                )
            )
        return result

    # ------------------------------------------------------------------
    # Setup & teardown
    # ------------------------------------------------------------------

    def ensure_schema(self, table: TableMetadata) -> None:
        """Create schema if it does not exist. No-op if already present."""
        try:
            self.engine.execute_statements([table.create_schema_sql(if_not_exists=True)])
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(f"Error occurred while ensuring schema for {table.schema_name}: {e}")
            )
        self.logger.debug("ensure_schema → %s: ok", table.schema_name)

    def ensure_table(self, table: TableMetadata) -> None:
        """Create table if it does not exist. No-op if already present.

        Does not validate or alter an existing table — use ensure_metadata() for that.
        """
        try:
            self.engine.execute_statements([
                table.create_schema_sql(if_not_exists=True),
                table.create_table_sql(if_not_exists=True),
            ])
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"Error occurred while ensuring table for {table.qualified_name_sql()}: {e}"
                )
            )
        self.logger.debug("ensure_table → %s: ok", table.qualified_name_sql())

    def ensure_metadata(self, table: TableMetadata) -> None:
        """Reconcile the live table definition against TableMetadata.

        Safe operations (performed automatically):
        - CREATE SCHEMA / CREATE TABLE if missing
        - ALTER TABLE ADD COLUMN for missing columns
        - ALTER COLUMN TYPE via USING cast for type mismatches
        - SET/DROP NOT NULL for nullable mismatches
        - SET/DROP DEFAULT for default mismatches
        - COMMENT ON COLUMN / TABLE for comment mismatches

        Destructive operations (always raise):
        - Extra columns in DB not present in metadata → raises SchemaDiffError
        - Type cast failure → raises SchemaDiffError

        Raises:
            SchemaDiffError: if diff contains destructive changes or cast fails.
        """
        self.ensure_schema(table)
        self.ensure_table(table)

        diff = self.schema_diff(table)

        if diff.is_match:
            self.logger.debug(
                "ensure_metadata → %s: schema matches, nothing to do", table.qualified_name_sql()
            )
            return

        # Fail fast on destructive changes
        if diff.extra_columns:
            raise MetadataError(
                f"ensure_metadata aborted — live table has extra columns not in metadata: "
                f"{diff.extra_columns}. Remove them manually or update TableMetadata."
            )

        statements: list[str] = []
        qname = table.qualified_name_sql()
        col_by_name = {c.name: c for c in table.columns}

        # ADD missing columns
        for col_name in diff.missing_columns:
            col = col_by_name[col_name]
            statements.append(f"ALTER TABLE {qname} ADD COLUMN {col.column_def_sql()};")
            self.logger.info("ensure_metadata → ADD COLUMN %s.%s", qname, col_name)

        # ALTER TYPE (try USING cast — Postgres will error if cast is invalid)
        for col_name, (expected, _actual) in diff.type_mismatches.items():
            statements.append(
                f"ALTER TABLE {qname} "
                f"ALTER COLUMN {quote_ident(col_name)} "
                f"TYPE {expected} "
                f"USING {quote_ident(col_name)}::{expected};"
            )
            self.logger.info("ensure_metadata → ALTER TYPE %s.%s to %s", qname, col_name, expected)

        # SET/DROP NOT NULL
        for col_name, (expected_nullable, _) in diff.nullable_mismatches.items():
            action = "DROP NOT NULL" if expected_nullable else "SET NOT NULL"
            statements.append(f"ALTER TABLE {qname} ALTER COLUMN {quote_ident(col_name)} {action};")
            self.logger.info("ensure_metadata → %s on %s.%s", action, qname, col_name)

        # SET/DROP DEFAULT
        for col_name, (expected_default, _) in diff.default_mismatches.items():
            col = col_by_name[col_name]
            if expected_default is None:
                action = "DROP DEFAULT"
            else:
                action = f"SET DEFAULT {col._format_default_value()}"
            statements.append(f"ALTER TABLE {qname} ALTER COLUMN {quote_ident(col_name)} {action};")
            self.logger.info("ensure_metadata → %s on %s.%s", action, qname, col_name)

        # Execute all ALTER statements in one transaction
        try:
            self.engine.execute_statements(statements)
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"ensure_metadata failed while applying schema changes to {qname}: {e}"
                )
            )

        # Sync comments separately (COMMENT ON is not transactional in PG)
        comment_stmts = table.comment_sql()
        if comment_stmts:
            self.engine.execute_statements(comment_stmts)

        self.logger.info("ensure_metadata → %s: schema reconciled", qname)

    def truncate_table(self, table: TableMetadata, *, cascade: bool = False) -> None:
        """TRUNCATE the table. Explicit, destructive — use intentionally.

        Args:
            cascade: Also truncate tables with foreign key references.
        """
        cascade_sql = " CASCADE" if cascade else ""
        sql = f"TRUNCATE TABLE {table.qualified_name_sql()}{cascade_sql};"
        self.engine.execute_statements([sql])
        self.logger.info("truncate_table → %s", table.qualified_name_sql())

    def drop_table(self, table: TableMetadata, *, cascade: bool = False) -> None:
        """DROP the table. Irreversible — use intentionally.

        Args:
            cascade: Also drop dependent objects (views, FKs).
        """
        cascade_sql = " CASCADE" if cascade else ""
        sql = f"DROP TABLE IF EXISTS {table.qualified_name_sql()}{cascade_sql};"
        self.engine.execute_statements([sql])
        self.logger.info("drop_table → %s", table.qualified_name_sql())

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @overload
    def read(
        self,
        table: TableMetadata,
        *,
        columns: list[str] | None = None,
        where: str | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        as_polars: Literal[False] = False,
    ) -> list[dict[str, Any]]: ...

    @overload
    def read(
        self,
        table: TableMetadata,
        *,
        columns: list[str] | None = None,
        where: str | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        as_polars: Literal[True],
    ) -> pl.DataFrame: ...

    def read(
        self,
        table: TableMetadata,
        *,
        columns: list[str] | None = None,
        where: str | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        as_polars: bool = False,
    ) -> list[dict[str, Any]] | pl.DataFrame:
        """Read rows from a table with optional filtering and projection.

        Args:
            columns:   Columns to SELECT. Defaults to all (*).
            where:     Optional WHERE clause.
            order_by:  Optional ORDER BY clause.
            limit:     Optional row limit.
            as_polars: Return pl.DataFrame instead of list[dict].

        Returns:
            list[dict] by default, or pl.DataFrame if as_polars=True.
        """
        col_clause = ", ".join(quote_ident(c) for c in columns) if columns else "*"
        sql = f"SELECT {col_clause} FROM {table.qualified_name_sql()}"

        if where:
            sql += f"\nWHERE {where}"
        if order_by:
            sql += f"\nORDER BY {order_by}"
        if limit is not None:
            sql += f"\nLIMIT {limit}"
        sql += ";"

        rows = self.engine.execute_parameterized_query(sql, {})
        self.logger.debug("read → %s: %d rows returned", table.qualified_name_sql(), len(rows))

        if as_polars:
            return pl.DataFrame(rows)
        return rows

    # ------------------------------------------------------------------
    # Helper methods for merge
    # ------------------------------------------------------------------

    def _count_matches(
        self,
        records: list[dict],
        table: TableMetadata,
        resolved_keys: list[str],
        match_condition: str | None,
    ) -> int:
        """Count how many source records match existing target rows."""
        source_cte = _build_source_cte(records, table)

        # Build join condition
        join_condition = " AND ".join(
            f"_target.{quote_ident(k)} = _source.{quote_ident(k)}" for k in resolved_keys
        )
        if match_condition:
            join_condition = f"({join_condition}) AND ({match_condition})"

        # Count matched rows (identifiers quoted; row values in VALUES CTE)
        qname = table.qualified_name_sql()
        count_matched_sql = (
            f"{source_cte}\n"
            f"SELECT COUNT(*) AS matched_count\n"
            f"FROM _source\n"
            f"WHERE EXISTS (\n"
            f"  SELECT 1 FROM {qname} _target\n"
            f"  WHERE {join_condition}\n"
            f")"
        )

        join_condition = _build_join_condition(resolved_keys, match_condition)
        source_prefix = _build_source_cte(records, table) + "\n"
        count_matched_sql = _build_count_matched_sql(
            table,
            join_condition,
            source_prefix=source_prefix,
            source_from="_source",
        )
        result = self.engine.execute_parameterized_query(count_matched_sql, {})
        return result[0]["matched_count"] if result else 0

    def _count_deletes(
        self,
        records: list[dict],
        table: TableMetadata,
        resolved_keys: list[str],
        match_condition: str | None,
    ) -> int:
        """Count how many target rows have no match in source (for WHEN NOT MATCHED BY SOURCE)."""
        source_cte = _build_source_cte(records, table)

        # Build join condition
        join_condition = " AND ".join(
            f"_target.{quote_ident(k)} = _source.{quote_ident(k)}" for k in resolved_keys
        )
        if match_condition:
            join_condition = f"({join_condition}) AND ({match_condition})"

        # Count rows to delete (identifiers quoted; row values in VALUES CTE)
        qname = table.qualified_name_sql()
        count_unmatched_target_sql = (
            f"{source_cte}\n"
            f"SELECT COUNT(*) AS delete_count\n"
            f"FROM {qname} _target\n"
            f"WHERE NOT EXISTS (\n"
            f"  SELECT 1 FROM _source\n"
            f"  WHERE {join_condition}\n"
            f")"
        )

        result = self.engine.execute_parameterized_query(count_unmatched_target_sql, {})
        return result[0]["delete_count"] if result else 0
        join_condition = _build_join_condition(resolved_keys, match_condition)
        source_prefix = _build_source_cte(records, table) + "\n"
        count_unmatched_target_sql = _build_count_deletes_sql(
            table,
            join_condition,
            source_prefix=source_prefix,
            source_from="_source",
        )
        result = self.engine.execute_parameterized_query(count_unmatched_target_sql, {})
        return result[0]["delete_count"] if result else 0

    def _merge_via_copy(
        self,
        records: list[dict],
        table: TableMetadata,
        *,
        resolved_keys: list[str],
        strategy: MergeStrategy,
        match_condition: str | None,
    ) -> tuple[int, int]:
        """Stage source rows with COPY, then count and MERGE in one transaction."""
        staging_name = _staging_table_name()
        col_names = list(records[0].keys())
        join_condition = _build_join_condition(resolved_keys, match_condition)
        staging_from = f"{quote_ident(staging_name)} AS _source"
        staging_using = staging_from

        create_sql = _build_create_staging_sql(staging_name, col_names, table)
        count_matched_sql = _build_count_matched_sql(
            table, join_condition, source_from=staging_from
        )
        count_deletes_sql = _build_count_deletes_sql(
            table, join_condition, source_from=staging_from
        )
        merge_sql = _build_merge_sql(
            table,
            col_names,
            match_keys=resolved_keys,
            strategy=strategy,
            match_condition=match_condition,
            source_using=staging_using,
        )

        try:
            with self.engine.begin() as conn:
                conn.execute(text(create_sql))
                self.engine.copy_records(conn, staging_name, col_names, records)

                matched_row = conn.execute(text(count_matched_sql)).mappings().first()
                matched_count = int(matched_row["matched_count"]) if matched_row else 0

                deleted_count = 0
                if strategy.when_not_matched_by_source == "delete":
                    delete_row = conn.execute(text(count_deletes_sql)).mappings().first()
                    deleted_count = int(delete_row["delete_count"]) if delete_row else 0

                conn.execute(text(merge_sql))
        except Exception as e:
            self.logger.log_and_raise(
                DatabaseError(
                    f"Error occurred while COPY merge into {table.qualified_name_sql()} "
                    f"(staging={staging_name}): {e}"
                )
            )

        return matched_count, deleted_count


def get_db_client(engine: EnrichedEngine | None = None) -> DatabaseClient:
    """Factory function to create a DatabaseClient with optional custom engine."""
    return DatabaseClient(engine=engine)
