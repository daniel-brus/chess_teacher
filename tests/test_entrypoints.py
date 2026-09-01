"""Ensure executable scripts use run_script_main for Windows spawn safety."""

from __future__ import annotations

import ast
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _has_main_guard(module: ast.Module) -> bool:
    for node in module.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        if not isinstance(test.ops[0], ast.Eq):
            continue
        left = test.left
        comparators = test.comparators
        if (
            isinstance(left, ast.Name)
            and left.id == "__name__"
            and len(comparators) == 1
            and isinstance(comparators[0], ast.Constant)
            and comparators[0].value == "__main__"
        ):
            return True
    return False


def test_script_entrypoints_use_run_script_main() -> None:
    missing: list[str] = []
    for path in sorted(_SCRIPTS_DIR.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        if "dev" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(path))
        if not _has_main_guard(module):
            continue
        if "run_script_main" not in source:
            missing.append(str(path.relative_to(_SCRIPTS_DIR)))
    assert not missing, f"Scripts missing run_script_main: {', '.join(missing)}"
