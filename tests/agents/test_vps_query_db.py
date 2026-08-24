"""Unit tests for VPS skill remote DB argv building (no SSH)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not find repository root")


_SCRIPT = _repo_root() / ".agents" / "skills" / "chess-teacher-vps" / "scripts" / "vps_query.py"


def _load_vps_query():
    spec = importlib.util.spec_from_file_location("vps_query_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vps_query():
    return _load_vps_query()


def test_remote_db_argv_is_quoted(vps_query) -> None:
    remote = vps_query._remote_db_argv(
        "count",
        "pipelines/preprocessing",
        "move_characteristics",
        "--where",
        "candidate_evaluations IS NOT NULL",
    )
    assert "kubectl exec" in remote
    assert "deploy/streamlit" in remote
    assert "scripts/tools/agent_db_query.py" in remote
    assert "'candidate_evaluations IS NOT NULL'" in remote or (
        '"candidate_evaluations IS NOT NULL"' in remote
    )


def test_rejects_write_sql_fragment(vps_query) -> None:
    with pytest.raises(SystemExit):
        vps_query._validate_sql_fragment("1=1; DROP TABLE x", label="WHERE")


def test_rejects_bad_domain(vps_query) -> None:
    with pytest.raises(SystemExit):
        vps_query._validate_domain_id("../etc")
