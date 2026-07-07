"""Tests for JSON-lines progress streaming."""

from __future__ import annotations

import io
import json

import pytest

from chess_teacher.utils.pipeline_utils.json_lines_progress import (
    JsonLinesProgressWindow,
    apply_progress_event,
)
from chess_teacher.utils.pipeline_utils.pipeline_helpers import ProgressWindow


class _RecordingProgressWindow:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def next(self, message: str) -> None:
        self.events.append(("next", message))

    def update(self, message: str) -> None:
        self.events.append(("update", message))

    def pop(self, amount: int = 1) -> None:
        self.events.append(("pop", amount))

    def success(self, message: str) -> None:
        self.events.append(("success", message))

    def warning(self, message: str) -> None:
        self.events.append(("warning", message))

    def error(self, message: str) -> None:
        self.events.append(("error", message))

    def clear(self) -> None:
        self.events.append(("clear", None))


def test_json_lines_progress_window_writes_one_event_per_line() -> None:
    buffer = io.StringIO()
    window = JsonLinesProgressWindow(buffer)

    window.next("hello")
    window.update("world")
    window.pop(2)
    window.success("done")

    lines = buffer.getvalue().splitlines()
    assert lines == [
        json.dumps({"op": "next", "message": "hello"}),
        json.dumps({"op": "update", "message": "world"}),
        json.dumps({"op": "pop", "amount": 2}),
        json.dumps({"op": "success", "message": "done"}),
    ]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"op": "next", "message": "a"}, ("next", "a")),
        ({"op": "update", "message": "b"}, ("update", "b")),
        ({"op": "pop", "amount": 3}, ("pop", 3)),
        ({"op": "pop"}, ("pop", 1)),
        ({"op": "warning", "message": "w"}, ("warning", "w")),
        ({"op": "error", "message": "e"}, ("error", "e")),
        ({"op": "clear"}, ("clear", None)),
    ],
)
def test_apply_progress_event(payload: dict[str, object], expected: tuple[str, object]) -> None:
    progress = _RecordingProgressWindow()
    apply_progress_event(progress, payload)
    assert progress.events == [expected]


def test_apply_progress_event_rejects_unknown_op() -> None:
    progress: ProgressWindow = _RecordingProgressWindow()
    with pytest.raises(ValueError, match="Unknown progress event op"):
        apply_progress_event(progress, {"op": "nope"})
