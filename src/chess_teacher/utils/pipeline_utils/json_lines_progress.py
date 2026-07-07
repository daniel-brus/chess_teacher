"""JSON-lines ``ProgressWindow`` for cross-process progress reporting."""

from __future__ import annotations

import json
from typing import IO, Any, Literal, TypedDict, cast

from chess_teacher.utils.pipeline_utils.pipeline_helpers import ProgressWindow

ProgressOp = Literal["next", "update", "pop", "success", "warning", "error", "clear"]


class _ProgressEventBase(TypedDict):
    op: ProgressOp


class _MessageEvent(_ProgressEventBase):
    message: str


class _PopEvent(_ProgressEventBase):
    amount: int


class JsonLinesProgressWindow:
    """Write ``ProgressWindow`` events as one JSON object per line."""

    def __init__(self, out: IO[str]) -> None:
        self._out = out

    def _emit(self, payload: dict[str, Any]) -> None:
        self._out.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._out.flush()

    def next(self, message: str) -> None:
        self._emit({"op": "next", "message": message})

    def update(self, message: str) -> None:
        self._emit({"op": "update", "message": message})

    def pop(self, amount: int = 1) -> None:
        self._emit({"op": "pop", "amount": amount})

    def success(self, message: str) -> None:
        self._emit({"op": "success", "message": message})

    def warning(self, message: str) -> None:
        self._emit({"op": "warning", "message": message})

    def error(self, message: str) -> None:
        self._emit({"op": "error", "message": message})

    def clear(self) -> None:
        self._emit({"op": "clear"})


def apply_progress_event(progress: ProgressWindow, payload: dict[str, Any]) -> None:
    """Apply one decoded JSON-lines progress event to a ``ProgressWindow``."""
    op = payload.get("op")
    if op == "next":
        progress.next(cast(_MessageEvent, payload)["message"])
    elif op == "update":
        progress.update(cast(_MessageEvent, payload)["message"])
    elif op == "pop":
        event = cast(_PopEvent, payload)
        progress.pop(event.get("amount", 1))
    elif op == "success":
        progress.success(cast(_MessageEvent, payload)["message"])
    elif op == "warning":
        progress.warning(cast(_MessageEvent, payload)["message"])
    elif op == "error":
        progress.error(cast(_MessageEvent, payload)["message"])
    elif op == "clear":
        progress.clear()
    else:
        raise ValueError(f"Unknown progress event op: {op!r}")
