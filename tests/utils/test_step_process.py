"""The child interpreter starts on demand and answers one pickle call at a time."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chess_teacher.utils.pipeline_utils.step_process import StepProcess, StepProcessError


def test_step_process_roundtrip_failure_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[2]
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(p for p in (str(repo), str(repo / "src"), existing) if p),
    )
    process = StepProcess("tests.utils._echo_step_worker")
    process.start()
    try:
        assert process.running is True
        assert process.call("echo", value="hi") == "hi"
        with pytest.raises(StepProcessError, match="nope"):
            process.call("fail")
        assert process.running is True
    finally:
        process.close()
    assert process.running is False
    process.close()


def test_call_before_start_raises() -> None:
    process = StepProcess("tests.utils._echo_step_worker")
    with pytest.raises(StepProcessError, match="not running"):
        process.call("echo", value="hi")


def test_module_name_rejects_whitespace() -> None:
    with pytest.raises(ValueError):
        StepProcess("not a module")
