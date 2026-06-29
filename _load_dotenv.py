"""Emit Windows ``cmd`` ``set`` commands for every key in project ``.env``.

Used by the makefile to load env vars into the shell without importing
``chess_teacher.utils`` (and its logging setup).

Exits non-zero if required variables (e.g. ``APP_PORT``) are missing.
"""

import sys
from pathlib import Path

from dotenv import dotenv_values

_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _ROOT / ".env"
_REQUIRED = ("APP_PORT",)


def _cmd_set(key: str, value: str) -> str:
    escaped = value.replace("%", "%%").replace('"', '""')
    return f'set "{key}={escaped}"'


def main() -> int:
    if not _ENV_FILE.is_file():
        print(f"Missing .env at {_ENV_FILE}", file=sys.stderr)
        return 1

    values = dotenv_values(_ENV_FILE)
    missing = [key for key in _REQUIRED if not values.get(key) or not str(values[key]).strip()]
    if missing:
        print(f"Missing required .env variable(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    for key, value in values.items():
        if value is None:
            continue
        print(_cmd_set(key, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
