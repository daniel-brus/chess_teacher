#!/usr/bin/env python3
"""Launcher for ``scripts/tools/agent_db_query.py`` (keeps skill path stable)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise SystemExit("Could not find repository root (pyproject.toml).")


if __name__ == "__main__":
    target = _repo_root() / "scripts" / "tools" / "agent_db_query.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
