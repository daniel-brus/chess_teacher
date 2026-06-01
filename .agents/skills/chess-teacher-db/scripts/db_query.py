#!/usr/bin/env python3
"""Read-only database inspection for chess_teacher (DatabaseClient only)."""

from __future__ import annotations

# Agent runs: tag logs separately from LOCAL/DEV (load_dotenv will not override).
import os

os.environ["ENVIRONMENT"] = "AGENT"

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from chess_teacher.utils.db_client import get_db_client
from chess_teacher.utils.general_utils import quote_ident
from chess_teacher.utils.metadata_utils import TableMetadata

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke|merge|copy)\b",
    re.IGNORECASE,
)


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise SystemExit("Could not find repository root (pyproject.toml).")


def package_root() -> Path:
    import chess_teacher

    return Path(chess_teacher.__file__).resolve().parent


def _has_tables_section(path: Path) -> bool:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(raw, dict) and isinstance(raw.get("tables"), dict)


@lru_cache(maxsize=1)
def discover_domains() -> dict[str, Path]:
    """Map domain id -> metadata.yml path.

      Domain id is the folder path under the chess_teacher package that contains
      metadata.yml (e.g. ``ingestion``, ``etl_stockfish``). Nested folders use
    slashes (e.g. ``foo/bar``).
    """
    root = package_root()
    domains: dict[str, Path] = {}
    for yml in sorted(root.rglob("metadata.yml")):
        if not _has_tables_section(yml):
            continue
        rel_parent = yml.parent.relative_to(root)
        domain_id = rel_parent.as_posix()
        if domain_id in domains:
            raise SystemExit(
                f"Duplicate domain id '{domain_id}' for metadata files:\n"
                f"  {domains[domain_id]}\n"
                f"  {yml}"
            )
        domains[domain_id] = yml
    if not domains:
        raise SystemExit(f"No metadata.yml with a tables section found under {root}")
    return domains


def resolve_domain(domain: str) -> Path:
    domains = discover_domains()
    if domain not in domains:
        known = ", ".join(sorted(domains))
        raise SystemExit(f"Unknown domain '{domain}'. Known domains: {known}")
    return domains[domain]


def load_table(domain: str, table_key: str) -> TableMetadata:
    return TableMetadata(key=table_key, yaml_path=resolve_domain(domain))


def list_table_keys(domain: str) -> list[str]:
    path = resolve_domain(domain)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tables = raw.get("tables") if isinstance(raw, dict) else None
    if not isinstance(tables, dict):
        raise SystemExit(f"No tables section in {path}")
    return sorted(tables.keys())


def validate_sql_fragment(fragment: str, *, label: str) -> str:
    """Reject obvious injection / multi-statement in user-supplied SQL fragments."""
    text = fragment.strip()
    if not text:
        raise SystemExit(f"{label} must not be empty.")
    if ";" in text:
        raise SystemExit(f"{label} must not contain ';'.")
    if _WRITE_KEYWORDS.search(text):
        raise SystemExit(f"{label} contains disallowed SQL keyword.")
    return text


def validate_select_sql(sql: str) -> str:
    text = sql.strip().rstrip(";")
    if not text:
        raise SystemExit("SQL must not be empty.")
    if ";" in text:
        raise SystemExit("Only a single SELECT statement is allowed.")
    first = text.lstrip().split(None, 1)[0].upper() if text.lstrip() else ""
    if first not in {"SELECT", "WITH"}:
        raise SystemExit("Only SELECT (or WITH ... SELECT) queries are allowed.")
    if _WRITE_KEYWORDS.search(text):
        raise SystemExit("Query contains disallowed SQL keyword.")
    return text


def validate_columns(table: TableMetadata, columns: list[str]) -> None:
    known = {c.name for c in table.columns}
    unknown = [c for c in columns if c not in known]
    if unknown:
        raise SystemExit(f"Unknown column(s) for table '{table.table_name}': {unknown}")


def emit(data: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, default=str))
    else:
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{key}: {value}")
        elif isinstance(data, list):
            for row in data:
                print(row)
        else:
            print(data)


def _path_for_display(path: Path) -> str:
    root = repo_root()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def cmd_list_domains(args: argparse.Namespace) -> None:
    items = [
        {
            "domain": domain_id,
            "folder": yml.parent.name,
            "metadata_path": _path_for_display(yml),
        }
        for domain_id, yml in sorted(discover_domains().items())
    ]
    emit(
        {
            "package_root": _path_for_display(package_root()),
            "environment": os.environ.get("ENVIRONMENT"),
            "domains": items,
        },
        as_json=args.json,
    )


def cmd_list_tables(args: argparse.Namespace) -> None:
    keys = list_table_keys(args.domain)
    emit(
        {
            "domain": args.domain,
            "metadata_path": _path_for_display(resolve_domain(args.domain)),
            "tables": keys,
        },
        as_json=args.json,
    )


def cmd_read(args: argparse.Namespace) -> None:
    table = load_table(args.domain, args.table)
    where = validate_sql_fragment(args.where, label="WHERE") if args.where else None
    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
    if columns:
        validate_columns(table, columns)

    client = get_db_client()
    rows = client.read(
        table,
        columns=columns,
        where=where,
        order_by=args.order_by,
        limit=args.limit,
    )
    emit(
        {
            "domain": args.domain,
            "table": args.table,
            "row_count": len(rows),
            "rows": rows,
        },
        as_json=args.json,
    )


def cmd_count(args: argparse.Namespace) -> None:
    table = load_table(args.domain, args.table)
    where = validate_sql_fragment(args.where, label="WHERE") if args.where else None
    client = get_db_client()
    count = client.get_row_count(table, where=where)
    emit(
        {
            "domain": args.domain,
            "table": args.table,
            "where": where,
            "count": count,
        },
        as_json=args.json,
    )


def cmd_exists(args: argparse.Namespace) -> None:
    table = load_table(args.domain, args.table)
    where = validate_sql_fragment(args.where, label="WHERE")
    client = get_db_client()
    result = client.exists(table, where)
    emit(
        {
            "domain": args.domain,
            "table": args.table,
            "where": where,
            "exists": result,
        },
        as_json=args.json,
    )


def cmd_schema(args: argparse.Namespace) -> None:
    table = load_table(args.domain, args.table)
    client = get_db_client()
    diff = client.schema_diff(table)
    emit(
        {
            "domain": args.domain,
            "table": args.table,
            "schema_exists": client.schema_exists(table),
            "table_exists": client.table_exists(table),
            "schema_match": diff.is_match,
            "missing_columns": diff.missing_columns,
            "extra_columns": diff.extra_columns,
            "type_mismatches": diff.type_mismatches,
            "nullable_mismatches": diff.nullable_mismatches,
        },
        as_json=args.json,
    )


def cmd_all_match(args: argparse.Namespace) -> None:
    table = load_table(args.domain, args.table)
    condition = validate_sql_fragment(args.condition, label="CONDITION")
    violation_where = f"NOT ({condition})"

    client = get_db_client()
    total = client.get_row_count(table)
    violations = client.get_row_count(table, where=violation_where)
    sample: list[dict[str, Any]] = []
    if violations and args.sample_limit > 0:
        sample = client.read(table, where=violation_where, limit=args.sample_limit)

    emit(
        {
            "domain": args.domain,
            "table": args.table,
            "condition": condition,
            "total_rows": total,
            "violations": violations,
            "all_match": violations == 0,
            "sample_violations": sample,
        },
        as_json=args.json,
    )


def cmd_unique(args: argparse.Namespace) -> None:
    table = load_table(args.domain, args.table)
    columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    if not columns:
        raise SystemExit("--columns requires at least one column name.")
    validate_columns(table, columns)

    quoted = ", ".join(quote_ident(c) for c in columns)
    qname = table.qualified_name_sql()
    groups_sql = validate_select_sql(
        f"SELECT {quoted}, COUNT(*) AS duplicate_count\n"
        f"FROM {qname}\n"
        f"GROUP BY {quoted}\n"
        f"HAVING COUNT(*) > 1\n"
        f"ORDER BY duplicate_count DESC\n"
        f"LIMIT {args.limit}"
    )
    summary_sql = validate_select_sql(
        f"SELECT COUNT(*) AS duplicate_group_count\n"
        f"FROM (\n"
        f"  SELECT {quoted}\n"
        f"  FROM {qname}\n"
        f"  GROUP BY {quoted}\n"
        f"  HAVING COUNT(*) > 1\n"
        f") AS _dup_groups"
    )

    client = get_db_client()
    summary_rows = client.engine.execute_parameterized_query(summary_sql, {})
    groups = client.engine.execute_parameterized_query(groups_sql, {})
    duplicate_group_count = int(summary_rows[0]["duplicate_group_count"]) if summary_rows else 0

    emit(
        {
            "domain": args.domain,
            "table": args.table,
            "columns": columns,
            "is_unique": duplicate_group_count == 0,
            "duplicate_group_count": duplicate_group_count,
            "sample_duplicate_groups": groups,
        },
        as_json=args.json,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only chess_teacher database queries via DatabaseClient.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (recommended for agents).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-domains", help="Discover domains (folders with metadata.yml).")
    p.set_defaults(func=cmd_list_domains)

    p = sub.add_parser("list-tables", help="List table keys in a domain.")
    p.add_argument("domain", help="Domain id from list-domains (package folder path).")
    p.set_defaults(func=cmd_list_tables)

    p = sub.add_parser("read", help="SELECT rows (optional filter/limit).")
    p.add_argument("domain")
    p.add_argument("table", help="Table key from metadata.yml.")
    p.add_argument("--where", help="SQL expression (no WHERE keyword).")
    p.add_argument("--columns", help="Comma-separated column names.")
    p.add_argument("--order-by", help="ORDER BY clause (no ORDER BY keyword).")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("count", help="COUNT rows.")
    p.add_argument("domain")
    p.add_argument("table")
    p.add_argument("--where", help="SQL expression (no WHERE keyword).")
    p.set_defaults(func=cmd_count)

    p = sub.add_parser("exists", help="EXISTS check for rows matching WHERE.")
    p.add_argument("domain")
    p.add_argument("table")
    p.add_argument("--where", required=True, help="SQL expression (no WHERE keyword).")
    p.set_defaults(func=cmd_exists)

    p = sub.add_parser("schema", help="Table/schema introspection (read-only).")
    p.add_argument("domain")
    p.add_argument("table")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser(
        "all-match",
        help="True when every row satisfies CONDITION (SQL expression).",
    )
    p.add_argument("domain")
    p.add_argument("table")
    p.add_argument(
        "--condition",
        required=True,
        help='SQL boolean expression, e.g. "rating IS NOT NULL" or "length(pgn) > 0".',
    )
    p.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Max violating rows to return when not all match (0 to skip).",
    )
    p.set_defaults(func=cmd_all_match)

    p = sub.add_parser(
        "unique",
        help="Check uniqueness of one or more columns (GROUP BY / HAVING).",
    )
    p.add_argument("domain")
    p.add_argument("table")
    p.add_argument(
        "--columns",
        required=True,
        help="Comma-separated columns that should be unique together.",
    )
    p.add_argument("--limit", type=int, default=20, help="Max duplicate groups to sample.")
    p.set_defaults(func=cmd_unique)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
