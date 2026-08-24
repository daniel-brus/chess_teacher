"""Run Streamlit locally without importing ``chess_teacher``.

Used by the makefile. Expects ``doppler run`` to inject secrets and config;
falls back to project ``.env`` when present (legacy local setup).

Exits non-zero if required variables (e.g. ``APP_PORT``) are missing.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _ROOT / ".env"
_REQUIRED = ("APP_PORT",)


def main() -> int:
    if _ENV_FILE.is_file():
        load_dotenv(_ENV_FILE, override=False)

    missing = [key for key in _REQUIRED if not os.environ.get(key, "").strip()]
    if missing:
        print(f"Missing required environment variable(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    port = os.environ["APP_PORT"]
    os.chdir(_ROOT)
    sys.argv = [
        "streamlit",
        "run",
        "streamlit_app.py",
        f"--server.port={port}",
    ]
    from streamlit.web import cli as stcli

    stcli.main()
    return 0


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
