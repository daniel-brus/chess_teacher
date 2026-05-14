import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml


def generate_hash(input: str | list[str]) -> str:
    """Generate a sha256hash for the given input string."""
    if isinstance(input, list):
        input_string = ",".join(input)
    else:
        input_string = input

    return hashlib.sha256(input_string.encode()).hexdigest()


def build_daily_path(base_dir: Path, file_name: str) -> Path:
    """
    Example:
        base_dir/2026/05/08/app.log
    """

    daily_dir = base_dir / datetime.now(UTC).strftime("%Y/%m/%d")

    daily_dir.mkdir(parents=True, exist_ok=True)

    return daily_dir / file_name


def build_day_hour_minute_path(base_dir: Path, file_name: str) -> Path:
    """
    Example:
        base_dir/2026/05/08/1530/app.log
    """

    daily_dir = base_dir / datetime.now(UTC).strftime("%Y/%m/%d/%H%M")

    daily_dir.mkdir(parents=True, exist_ok=True)

    return daily_dir / file_name


def load_yaml(path: Path | str) -> dict:
    p = Path(path)
    filename = p.name
    if not filename.endswith((".yml", ".yaml")):
        raise ValueError(f"Invalid file type for {filename}. Expected .yml or .yaml")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"File {filename} must contain a YAML mapping/object at the top level")
    return data


### SQL-specific helpers ###


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def require_ident(value: str, *, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string")
    if not _IDENT_RE.match(value):
        raise ValueError(
            f"Invalid {what} '{value}'. Use letters/numbers/underscore, start with letter/_"
        )
    return value


def quote_ident(value: str) -> str:
    require_ident(value, what="identifier")
    return f'"{value}"'


def quote_literal(value: str) -> str:
    if value is None:
        return "NULL"
    if not isinstance(value, str):
        value = str(value)
    return "'" + value.replace("'", "''") + "'"
