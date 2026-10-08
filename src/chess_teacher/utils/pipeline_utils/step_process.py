"""One child interpreter shared by the pipeline steps that hold the same object.

The runner starts it on the first of those steps and exits it when no later
step still carries it. The child speaks pickle on stdout. Its logs stay on stderr.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping
from typing import Any, BinaryIO

from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import snapshot_host_pressure

logger = get_logger()

_SHUTDOWN = "shutdown"


class StepProcessError(RuntimeError):
    """The child failed an operation or exited before answering."""


class StepProcess:
    """A fresh interpreter. ``call`` runs one operation and returns its result."""

    def __init__(self, module: str) -> None:
        if not module or any(char.isspace() for char in module):
            raise ValueError(f"step process module must be a dotted name, got {module!r}")
        self._module = module
        self._proc: subprocess.Popen[bytes] | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Launch the child if it is not already running."""
        if self.running:
            return
        self._proc = subprocess.Popen(
            [sys.executable, "-m", self._module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        logger.info("Step process started module=%s", self._module)

    def call(self, op: str, **payload: Any) -> Any:
        """Run ``op`` in the child. The child must already be running."""
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None or proc.stdout is None:
            raise StepProcessError(f"step process is not running for {op}")
        try:
            pickle.dump({"op": op, "payload": payload}, proc.stdin, protocol=4)
            proc.stdin.flush()
            # The child is one we spawned. Its stdout is not an untrusted stream.
            response = pickle.load(proc.stdout)  # nosec B301
        except (EOFError, pickle.UnpicklingError, OSError) as exc:
            raise StepProcessError(f"step process exited during {op}: {exc}") from exc
        if not isinstance(response, dict) or "ok" not in response:
            raise StepProcessError(f"step process returned an invalid response for {op}")
        if not response["ok"]:
            detail = str(response.get("traceback") or response.get("error") or op)
            raise StepProcessError(detail)
        return response.get("result")

    def close(self) -> None:
        """Ask the child to exit. Kill it only if it ignores that request."""
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if proc.poll() is None:
            _request_shutdown(proc)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Step process ignored shutdown; killing module=%s", self._module)
                proc.kill()
                proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()
        logger.info(
            "Step process exited module=%s %s",
            self._module,
            snapshot_host_pressure().format_fields(),
        )


def serve(ops: Mapping[str, Callable[..., Any]], protocol_out: BinaryIO) -> None:
    """Read operations from stdin until shutdown. ``protocol_out`` carries the replies."""
    while True:
        try:
            # Requests come from the parent that spawned this process.
            message = pickle.load(sys.stdin.buffer)  # nosec B301
        except EOFError:
            return
        if not isinstance(message, dict):
            _reply(protocol_out, ok=False, error="step process request must be a dict")
            continue
        op = message.get("op")
        if op == _SHUTDOWN:
            return
        handler = ops.get(op) if isinstance(op, str) else None
        if handler is None:
            _reply(protocol_out, ok=False, error=f"unknown step process op: {op}")
            continue
        payload = message.get("payload") or {}
        if not isinstance(payload, dict):
            _reply(protocol_out, ok=False, error=f"payload for {op} must be a dict")
            continue
        try:
            result = handler(**payload)
        except Exception as exc:
            _reply(
                protocol_out,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            continue
        _reply(protocol_out, ok=True, result=result)


def _reply(
    protocol_out: BinaryIO,
    *,
    ok: bool,
    result: Any = None,
    error: str | None = None,
    traceback: str | None = None,
) -> None:
    body: dict[str, Any] = {"ok": ok}
    if ok:
        body["result"] = result
    else:
        body["error"] = error
        body["traceback"] = traceback
    pickle.dump(body, protocol_out, protocol=4)
    protocol_out.flush()


def _request_shutdown(proc: subprocess.Popen[bytes]) -> None:
    if proc.stdin is None:
        return
    try:
        pickle.dump({"op": _SHUTDOWN, "payload": {}}, proc.stdin, protocol=4)
        proc.stdin.flush()
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        return
