"""The Streamlit pipeline supervisor stays up when the page watcher stops."""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

import pytest

from streamlit_utils.pipeline_subprocess import (
    find_pipeline_run,
    follow_pipeline,
    pipeline_command,
    pipeline_is_running,
    start_detached_pipeline,
)


class _Progress:
    def __init__(self, *, stop_on: str | None = None) -> None:
        self.events: list[tuple[str, object]] = []
        self._stop_on = stop_on
        self._final_state: str | None = None

    def next(self, message: str) -> None:
        self.events.append(("next", message))
        if self._stop_on == "next":
            raise RuntimeError("page refresh")

    def update(self, message: str) -> None:
        self.events.append(("update", message))

    def pop(self, amount: int = 1) -> None:
        self.events.append(("pop", amount))

    def success(self, message: str) -> None:
        self._final_state = "complete"
        self.events.append(("success", message))

    def warning(self, message: str) -> None:
        self.events.append(("warning", message))

    def error(self, message: str) -> None:
        self._final_state = "error"
        self.events.append(("error", message))

    def clear(self) -> None:
        self.events.append(("clear", None))


def _child_pids(pid: int) -> list[int]:
    try:
        text = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8")
    except OSError:
        return []
    return [int(part) for part in text.split()]


def _stop(pid: int) -> None:
    for child in _child_pids(pid):
        try:
            os.kill(child, signal.SIGKILL)
        except OSError:
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        return


def _python(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def _use_run_dir(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("STREAMLIT_PIPELINE_RUN_DIR", str(path))


def test_pipeline_command_emits_progress_on_stdout() -> None:
    command = pipeline_command("abc")
    assert command[-3:] == ["--user-id", "abc", "--progress-stdout"]
    assert command[1].endswith("scripts/entrypoints/pipeline.py")


def test_follow_to_exit_replays_the_progress_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_run_dir(monkeypatch, tmp_path)
    user_id = "follow-to-exit"
    line = json.dumps({"op": "success", "message": "done"})
    run = start_detached_pipeline(
        user_id,
        command=_python(f"import sys\nsys.stdout.write({line!r} + '\\n')\nsys.stdout.flush()\n"),
    )
    try:
        progress = _Progress()
        exit_code = follow_pipeline(run, progress)
        assert exit_code == 0
        assert progress.events == [("success", "done")]
        assert pipeline_is_running(run) is False
    finally:
        _stop(run.pid)


def test_stopped_watcher_leaves_the_pipeline_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _use_run_dir(monkeypatch, tmp_path)
    user_id = "refresh-user"
    line = json.dumps({"op": "next", "message": "still going"})
    run = start_detached_pipeline(
        user_id,
        command=_python(
            "import sys, time\n"
            f"sys.stdout.write({line!r} + '\\n')\n"
            "sys.stdout.flush()\n"
            "time.sleep(30)\n"
        ),
    )
    try:
        assert os.getpgid(run.pid) == run.pid
        assert os.getpgid(run.pid) != os.getpgrp()
        with pytest.raises(RuntimeError, match="page refresh"):
            follow_pipeline(run, _Progress(stop_on="next"))
        assert pipeline_is_running(run) is True
        os.kill(run.pid, 0)
        assert _child_pids(run.pid)
        again = start_detached_pipeline(user_id, command=_python("import time; time.sleep(30)"))
        assert again.pid == run.pid
        assert find_pipeline_run(user_id) is not None
    finally:
        _stop(run.pid)


def test_nonzero_exit_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_run_dir(monkeypatch, tmp_path)
    run = start_detached_pipeline("fails", command=_python("import sys; sys.exit(3)"))
    try:
        progress = _Progress()
        assert follow_pipeline(run, progress) == 3
        assert progress.events == [("error", "Pipeline subprocess exited with code 3.")]
    finally:
        _stop(run.pid)
