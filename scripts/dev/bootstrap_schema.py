"""Create all tables from metadata.yml in the connected database (ensure_metadata)."""

from __future__ import annotations

from pathlib import Path

import yaml

from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.metadata_utils import TableMetadata

_ROOT = Path(__file__).resolve().parents[2]


def _metadata_files() -> list[Path]:
    package_root = _ROOT / "src" / "chess_teacher"
    files: list[Path] = []
    for yml in sorted(package_root.rglob("metadata.yml")):
        raw = yaml.safe_load(yml.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("tables"), dict):
            files.append(yml)
    if not files:
        raise SystemExit(f"No metadata.yml with tables found under {package_root}")
    return files


def main() -> int:
    db_client = get_db_client()
    count = 0
    for yml in _metadata_files():
        raw = yaml.safe_load(yml.read_text(encoding="utf-8"))
        for table_key in raw["tables"]:
            table = TableMetadata(key=table_key, yaml_path=yml)
            db_client.ensure_metadata(table)
            count += 1
            print(f"ensure_metadata: {table.qualified_name_sql()}")
    print(f"Done ({count} tables).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
