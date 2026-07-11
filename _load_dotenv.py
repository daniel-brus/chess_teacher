"""Load project ``.env`` and run Streamlit locally without importing ``chess_teacher``.

Used by the makefile so env vars (particularly APP_PORT) are available.

Exits non-zero if required variables (e.g. ``APP_PORT``) are missing.
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _ROOT / ".env"
_REQUIRED = ("APP_PORT",)


def main() -> int:
    if not _ENV_FILE.is_file():
        print(f"Missing .env at {_ENV_FILE}", file=sys.stderr)
        return 1

    load_dotenv(_ENV_FILE, override=False)

    missing = [key for key in _REQUIRED if not os.environ.get(key, "").strip()]
    if missing:
        print(f"Missing required .env variable(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    port = os.environ["APP_PORT"]
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            f"--server.port={port}",
        ],
        cwd=_ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
