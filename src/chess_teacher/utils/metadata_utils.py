from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from chess_teacher.utils.exception_utils import MetadataError
from chess_teacher.utils.general_utils import load_yaml, quote_ident, quote_literal, require_ident
from chess_teacher.utils.logging_utils import get_logger

logger = get_logger()

# TODO: add other schema functionality (https://docs.sqlalchemy.org/en/21/core/metadata.html#sqlalchemy.schema.Column)
# e.g. foreignkey, constraint (constraint: maybe a nice class to define? )


@dataclass(frozen=True)
class ColumnMetadata:
    name: str
    data_type: str
    comment: str | None = None
    nullable: bool = True
    default: Any | None = None

    def __post_init__(self) -> None:
        """Normalize entries"""
        name = require_ident(self.name.strip().lower(), what="column name")
        object.__setattr__(self, "name", name)
        # TODO: moet niet ook require_indent?
        object.__setattr__(self, "data_type", self.data_type.strip().lower())
        if self.comment is not None:
            object.__setattr__(self, "comment", self.comment.strip() or None)

        # double check that name & data_type are not empty string
        if (not self.name) or (not self.data_type):
            logger.log_and_raise(
                MetadataError(
                    f"Column name and data_type cannot be empty. "
                    f"Got name='{self.name}', data_type='{self.data_type}'"
                )
            )

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> ColumnMetadata:
        try:
            name = raw.get("name", "")
            data_type = raw.get("data_type", "")
            comment = raw.get("comment", None)
            nullable = raw.get("nullable", True)
            default = raw.get("default", None)
            col = ColumnMetadata(
                name=name,
                data_type=data_type,
                comment=comment,
                nullable=bool(nullable),
                default=default,
            )
        except Exception as e:
            logger.log_and_raise(MetadataError(f"Error parsing column metadata from dict: {e}"))
        return col

    def column_def_sql(self) -> str:
        parts: list[str] = [quote_ident(self.name), self.data_type]
        if not self.nullable:
            parts.append("NOT NULL")
        if self.default is not None:
            parts.append(f"DEFAULT {self._format_default_value()}")
        return " ".join(parts)

    def _format_default_value(self) -> str:
        """Format default value as valid SQL literal.

        Handles: str, int, float, bool, (date)(time), None
        """
        if self.default is None:
            return "NULL"
        if isinstance(self.default, bool):
            return "TRUE" if self.default else "FALSE"
        if isinstance(self.default, int | float):
            return str(self.default)
        if isinstance(self.default, str):
            return quote_literal(self.default)
        if isinstance(
            self.default,
            (date | datetime | time),
        ):
            return quote_literal(self.default.isoformat())
        # Fallback: convert to string and quote
        return quote_literal(str(self.default))


@dataclass(frozen=True, init=False)
class TableMetadata:
    schema_name: str
    table_name: str
    columns: tuple[ColumnMetadata, ...]
    comment: str | None = None
    primary_key: tuple[str, ...] = field(default_factory=tuple)

    def __init__(
        self,
        key: str | None = None,
        *,
        schema_name: str | None = None,
        table_name: str | None = None,
        columns: tuple[ColumnMetadata, ...] | None = None,
        comment: str | None = None,
        primary_key: tuple[str, ...] | None = None,
        yaml_path: str | Path | None = None,
    ):
        """Initialize TableMetadata.

        Two modes:
        1. Load from YAML by key: TableMetadata(key="mykey", yaml_path="path/to/metadata.yml")
        2. Direct: TableMetadata(schema_name="schema", table_name="table", columns=[...])
        """
        # Mode 1: Load from YAML by key
        if key is not None and (schema_name is None and table_name is None and columns is None):
            if yaml_path is None:  # find yml from folder where TableMetadata is initialized
                caller_frame = inspect.stack()[1]
                caller_file = Path(caller_frame.filename)
                yaml_path = caller_file.parent / "metadata.yml"

            loaded = self._load_from_yaml_by_key(key, yaml_path)
            object.__setattr__(self, "schema_name", loaded.schema_name)
            object.__setattr__(self, "table_name", loaded.table_name)
            object.__setattr__(self, "columns", loaded.columns)
            object.__setattr__(self, "comment", loaded.comment)
            object.__setattr__(self, "primary_key", loaded.primary_key)
        # Mode 2: Direct initialization
        else:
            if schema_name is None or table_name is None or columns is None:
                logger.log_and_raise(
                    MetadataError(
                        "Either provide 'key' for YAML loading, or provide schema_name, table_name, and columns"
                    )
                )
            object.__setattr__(self, "schema_name", schema_name)
            object.__setattr__(self, "table_name", table_name)
            object.__setattr__(self, "columns", columns)
            object.__setattr__(self, "comment", comment)
            object.__setattr__(self, "primary_key", primary_key or ())

    @staticmethod
    def _load_from_yaml_by_key(key: str, path: str | Path) -> TableMetadata:
        """Load a table metadata from YAML by key.

        Expects YAML structure with top-level keys:
            tables:
              mykey:
                schema: schema_name
                table: table_name
                columns: [...]
              otherkey:
                ...
        """
        try:
            data = load_yaml(path)

            tables = data.get("tables")
            if not isinstance(tables, dict):
                raise MetadataError("metadata.yml must contain a 'tables' mapping at the top level")

            if key not in tables:
                available_keys = list(tables.keys())
                raise MetadataError(
                    f"Key '{key}' not found in tables. Available keys: {available_keys}"
                )

            table_data = tables[key]
            if not isinstance(table_data, dict):
                raise MetadataError(f"Table config for key '{key}' must be a mapping/object")

            # Use from_dict to parse the table config
            result = TableMetadata._from_dict_raw(table_data)
        except Exception as exc:
            logger.log_and_raise(
                exc, "Error loading TableMetadata from YAML for key '{key}' at path '{path}': {exc}"
            )
        return result

    @staticmethod
    def _from_dict_raw(raw: dict[str, Any]) -> TableMetadata:
        """Internal method to create TableMetadata from dict without going through __init__.

        Expects table configuration with top-level keys:
            schema: schema_name
            table: table_name
            columns: [...]
        """
        schema_name = require_ident(
            raw.get("schema", ""),
            what="schema name",
        ).lower()
        table_name = require_ident(
            raw.get("table", ""),
            what="table name",
        ).lower()

        columns_raw = raw.get("columns")
        if not isinstance(columns_raw, list) or not columns_raw:
            raise MetadataError("metadata must contain a non-empty 'columns' list")
        parsed_columns = [ColumnMetadata.from_dict(c) for c in columns_raw]

        comment = raw.get("comment")
        comment = comment.strip() if isinstance(comment, str) and comment.strip() else None

        pk_raw = raw.get("primary_key") or raw.get("primaryKey") or []
        if isinstance(pk_raw, str):
            pk_list: list[str] = [pk_raw]
        elif isinstance(pk_raw, list):
            pk_list = [str(x) for x in pk_raw]
        else:
            pk_list = []
        primary_key = tuple(require_ident(x, what="primary key column").lower() for x in pk_list)

        if primary_key:
            column_names = {c.name for c in parsed_columns}
            missing = [c for c in primary_key if c not in column_names]
            if missing:
                raise MetadataError(f"primary_key columns not present in columns: {missing}")
            columns_by_name = {c.name: c for c in parsed_columns}
            pk_cols = [columns_by_name[name] for name in primary_key]
            pk_set = set(primary_key)
            non_pk_cols = [c for c in parsed_columns if c.name not in pk_set]
            columns = tuple([*pk_cols, *non_pk_cols])
        else:
            columns = tuple(parsed_columns)

        # Create without using __init__
        obj = object.__new__(TableMetadata)
        object.__setattr__(obj, "schema_name", schema_name)
        object.__setattr__(obj, "table_name", table_name)
        object.__setattr__(obj, "columns", columns)
        object.__setattr__(obj, "comment", comment)
        object.__setattr__(obj, "primary_key", primary_key)
        return obj

    def qualified_name_sql(self) -> str:
        return f"{quote_ident(self.schema_name)}.{quote_ident(self.table_name)}"

    def create_table_sql(self, *, if_not_exists: bool = True) -> str:
        cols_sql = ",\n  ".join(c.column_def_sql() for c in self.columns)
        constraints: list[str] = []
        if self.primary_key:
            pk_cols = ", ".join(quote_ident(c) for c in self.primary_key)
            constraints.append(f"PRIMARY KEY ({pk_cols})")
        if constraints:
            cols_sql = cols_sql + ",\n  " + ",\n  ".join(constraints)

        ine = "IF NOT EXISTS " if if_not_exists else ""
        return f"CREATE TABLE {ine}{self.qualified_name_sql()} (\n  {cols_sql}\n);"

    def create_schema_sql(self, *, if_not_exists: bool = True) -> str:
        ine = "IF NOT EXISTS " if if_not_exists else ""
        return f"CREATE SCHEMA {ine}{quote_ident(self.schema_name)};"

    def comment_sql(self) -> list[str]:
        stmts: list[str] = []
        qname = self.qualified_name_sql()
        if self.comment:
            stmts.append(f"COMMENT ON TABLE {qname} IS {quote_literal(self.comment)};")
        for col in self.columns:
            if col.comment:
                stmts.append(
                    f"COMMENT ON COLUMN {qname}.{quote_ident(col.name)} IS {quote_literal(col.comment)};"
                )
        return stmts

    def ddl(self) -> list[str]:
        return [self.create_schema_sql(), self.create_table_sql(), *self.comment_sql()]
