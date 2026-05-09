from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from chess_teacher.utils.logging_utils import get_logger

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

logger = get_logger()


def _require_ident(value: str, *, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    if not _IDENT_RE.match(value):
        raise ValueError(
            f"Invalid {what} '{value}'. Use letters/numbers/underscore, start with letter/_"
        )
    return value


def _quote_ident(value: str) -> str:
    _require_ident(value, what="identifier")
    return f'"{value}"'


def _quote_literal(value: str) -> str:
    if value is None:
        return "NULL"
    if not isinstance(value, str):
        value = str(value)
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class ColumnMetadata:
    name: str
    data_type: str
    comment: str | None = None
    nullable: bool = True
    default: str | None = None

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> ColumnMetadata:
        name = _require_ident(raw.get("name", ""), what="column name")
        data_type = raw.get("data_type")
        if not isinstance(data_type, str) or not data_type.strip():
            raise ValueError(f"Column '{name}' missing 'type'")
        comment = raw.get("comment")
        nullable = raw.get("nullable", True)
        default = raw.get("default")
        return ColumnMetadata(
            name=name,
            data_type=data_type.strip(),
            comment=comment.strip() if isinstance(comment, str) and comment.strip() else None,
            nullable=bool(nullable),
            default=default.strip() if isinstance(default, str) and default.strip() else None,
        )

    def column_def_sql(self) -> str:
        parts: list[str] = [_quote_ident(self.name), self.data_type]
        if not self.nullable:
            parts.append("NOT NULL")
        if self.default is not None:
            parts.append(f"DEFAULT {self.default}")
        return " ".join(parts)


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
            if yaml_path is None:
                yaml_path = Path("metadata.yml")  # default path
            loaded = self._load_from_yaml_by_key(key, yaml_path)
            object.__setattr__(self, "schema_name", loaded.schema_name)
            object.__setattr__(self, "table_name", loaded.table_name)
            object.__setattr__(self, "columns", loaded.columns)
            object.__setattr__(self, "comment", loaded.comment)
            object.__setattr__(self, "primary_key", loaded.primary_key)
        # Mode 2: Direct initialization
        else:
            if schema_name is None or table_name is None or columns is None:
                raise ValueError(
                    "Either provide 'key' for YAML loading, or provide schema_name, table_name, and columns"
                )
            object.__setattr__(self, "schema_name", schema_name)
            object.__setattr__(self, "table_name", table_name)
            object.__setattr__(self, "columns", columns)
            object.__setattr__(self, "comment", comment)
            object.__setattr__(self, "primary_key", primary_key or ())

    @staticmethod
    def _load_from_yaml_by_key(key: str, path: str | Path) -> TableMetadata:
        """Load a table metadata from YAML by key.

        Expects YAML structure:
            tables:
              mykey:
                schema: schema_name
                table: table_name
                columns: [...]
              otherkey:
                ...
        """
        try:
            p = Path(path)
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("metadata.yml must contain a YAML mapping/object at the top level")

            tables = data.get("tables")
            if not isinstance(tables, dict):
                raise ValueError("metadata.yml must contain a 'tables' mapping at the top level")

            if key not in tables:
                available_keys = list(tables.keys())
                raise ValueError(
                    f"Key '{key}' not found in tables. Available keys: {available_keys}"
                )

            table_data = tables[key]
            if not isinstance(table_data, dict):
                raise ValueError(f"Table config for key '{key}' must be a mapping/object")

            # Use from_dict to parse the table config
            result = TableMetadata._from_dict_raw(table_data)
        except Exception as exc:
            logger.error("Error loading metadata from %s with key '%s': %s", path, key, exc)
            raise ValueError(
                f"Error occurred while loading metadata from {path} with key '{key}': {exc}"
            )
        return result

    @staticmethod
    def _from_dict_raw(raw: dict[str, Any]) -> TableMetadata:
        """Internal method to create TableMetadata from dict without going through __init__."""
        # Accept either top-level keys or a nested "table" object.
        table_raw = raw.get("table")
        if isinstance(table_raw, dict):
            raw = {**raw, **table_raw}

        schema_name = _require_ident(
            raw.get("schema", ""),
            what="schema name",
        )
        table_name = _require_ident(
            raw.get("table", ""),
            what="table name",
        )

        columns_raw = raw.get("columns")
        if not isinstance(columns_raw, list) or not columns_raw:
            raise ValueError("metadata must contain a non-empty 'columns' list")
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
        primary_key = tuple(_require_ident(x, what="primary key column") for x in pk_list)

        if primary_key:
            column_names = {c.name for c in parsed_columns}
            missing = [c for c in primary_key if c not in column_names]
            if missing:
                raise ValueError(f"primary_key columns not present in columns: {missing}")
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
        return f"{_quote_ident(self.schema_name)}.{_quote_ident(self.table_name)}"

    def create_table_sql(self, *, if_not_exists: bool = True) -> str:
        cols_sql = ",\n  ".join(c.column_def_sql() for c in self.columns)
        constraints: list[str] = []
        if self.primary_key:
            pk_cols = ", ".join(_quote_ident(c) for c in self.primary_key)
            constraints.append(f"PRIMARY KEY ({pk_cols})")
        if constraints:
            cols_sql = cols_sql + ",\n  " + ",\n  ".join(constraints)

        ine = "IF NOT EXISTS " if if_not_exists else ""
        return f"CREATE TABLE {ine}{self.qualified_name_sql()} (\n  {cols_sql}\n);"

    def create_schema_sql(self, *, if_not_exists: bool = True) -> str:
        ine = "IF NOT EXISTS " if if_not_exists else ""
        return f"CREATE SCHEMA {ine}{_quote_ident(self.schema_name)};"

    def comment_sql(self) -> list[str]:
        stmts: list[str] = []
        qname = self.qualified_name_sql()
        if self.comment:
            stmts.append(f"COMMENT ON TABLE {qname} IS {_quote_literal(self.comment)};")
        for col in self.columns:
            if col.comment:
                stmts.append(
                    f"COMMENT ON COLUMN {qname}.{_quote_ident(col.name)} IS {_quote_literal(col.comment)};"
                )
        return stmts

    def ddl(self) -> list[str]:
        return [self.create_schema_sql(), self.create_table_sql(), *self.comment_sql()]


# TODO: stop het onderste in een andere class? Of helemaal eruit gooien?


def ddls_from_metadata_files(
    paths: Iterable[str | Path],
    table_key: str = "default",
    log_warnings: bool = True,
) -> list[str]:
    """Generate DDL statements from metadata YAML files.

    Args:
        paths: Paths to metadata.yml files
        table_key: The key under 'tables' in the YAML to load
        log_warnings: Whether to log warnings for failed files

    Returns:
        List of DDL statements
    """
    ddls: list[str] = []
    for path in paths:
        try:
            ddls.extend(TableMetadata(key=table_key, yaml_path=path).ddl())
        except Exception as exc:
            # Best-effort: continue generating DDL for other files.
            if log_warnings:
                logger.warning("Skipping metadata file %s: %s", path, exc)
    return ddls


def ddls_from_current_directory_metadata(
    cwd: str | Path = ".",
    table_key: str = "default",
) -> list[str]:
    """Loads `metadata.yml` from the given directory and returns its DDL statements.

    Only looks for a single file named exactly `metadata.yml` in that directory.
    Returns an empty list if the file does not exist.

    Args:
        cwd: Directory to search for metadata.yml
        table_key: The key under 'tables' in the YAML to load

    Returns:
        List of DDL statements
    """

    directory = Path(cwd)
    metadata_path = directory / "metadata.yml"
    if not metadata_path.is_file():
        return []
    return ddls_from_metadata_files([metadata_path], table_key=table_key, log_warnings=False)
