from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils.logging_utils import get_logger

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


@dataclass(frozen=True)
class TableMetadata:
    schema_name: str
    table_name: str
    columns: tuple[ColumnMetadata, ...]
    comment: str | None = None
    primary_key: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> TableMetadata:
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

        return TableMetadata(
            schema_name=schema_name,
            table_name=table_name,
            columns=columns,
            comment=comment,
            primary_key=primary_key,
        )

    @staticmethod
    def from_yaml(path: str | Path) -> TableMetadata:
        try:
            p = Path(path)
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("metadata.yml must contain a YAML mapping/object at the top level")
            result = TableMetadata.from_dict(data)
        except Exception as exc:
            logger.error("Error loading metadata from %s: %s", path, exc)
            raise ValueError(f"Error occurred while loading metadata from {path}: {exc}")
        return result

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


def ddls_from_metadata_files(paths: Iterable[str | Path], log_warnings: bool = True) -> list[str]:
    ddls: list[str] = []
    for path in paths:
        try:
            ddls.extend(TableMetadata.from_yaml(path).ddl())
        except Exception as exc:
            # Best-effort: continue generating DDL for other files.
            if log_warnings:
                logger.warning("Skipping metadata file %s: %s", path, exc)
    return ddls


def ddls_from_current_directory_metadata(cwd: str | Path = ".") -> list[str]:
    """
    Loads `metadata.yml` from the given directory and returns its DDL statements.

    Only looks for a single file named exactly `metadata.yml` in that directory.
    Returns an empty list if the file does not exist.
    """

    directory = Path(cwd)
    metadata_path = directory / "metadata.yml"
    if not metadata_path.is_file():
        return []
    return ddls_from_metadata_files([metadata_path], log_warnings=False)
