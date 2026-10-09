"""Start a pipeline outside Streamlit's process group and follow its log.

A page refresh used to close the progress pipe and kill the run. The page now
starts a supervisor in its own session, and that supervisor owns the pipeline.
Progress is a file, so the watcher can stop and a later page load can attach
again. The watcher never signals the supervisor.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# ``python -m`` only puts the working directory on the path. CI installs
# dependencies without an editable install, so the supervisor has to add
# ``src`` itself before importing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
for _import_root in (str(_REPO_ROOT), str(_SRC_ROOT)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from chess_teacher.utils.env_utils import get_optional_env_variable  # noqa: E402
from chess_teacher.utils.logging import get_logger  # noqa: E402
from chess_teacher.utils.pipeline_utils.json_lines_progress import (  # noqa: E402
    apply_progress_event,
)
from chess_teacher.utils.pipeline_utils.pipeline_helpers import ProgressWindow  # noqa: E402

logger = get_logger()

_RUN_DIR_ENV = "STREAMLIT_PIPELINE_RUN_DIR"
_DEFAULT_RUN_DIR = str(Path(tempfile.gettempdir()) / "chess-teacher-streamlit-pipelines")
_POLL_SECONDS = 0.25
_STARTUP_GRACE_SECONDS = 5.0
_RUNNING_STATES = frozenset({"R", "S", "D", "T", "t"})

_PIPELINE_SCRIPT = _REPO_ROOT / "scripts" / "entrypoints" / "pipeline.py"


@dataclass(frozen=True)
class DetachedPipeline:
    """One Streamlit-started pipeline, addressed by files on disk."""

    user_id: str
    pid: int
    log_path: Path
    exit_path: Path
    meta_path: Path
    started_at: float


def pipeline_command(user_id: str) -> list[str]:
    """Argv for the pipeline CLI the supervisor runs."""
    return [
        sys.executable,
        str(_PIPELINE_SCRIPT),
        "--user-id",
        user_id,
        "--progress-stdout",
    ]


def find_pipeline_run(user_id: str) -> DetachedPipeline | None:
    """Return a live run, or a finished one whose log has not been shown yet."""
    run = _load_run(user_id)
    if run is None:
        return None
    if pipeline_is_running(run):
        return run
    _reap(run.pid)
    if run.exit_path.is_file() or _log_has_bytes(run.log_path):
        return run
    clear_pipeline_run(user_id)
    return None


def pipeline_is_running(run: DetachedPipeline) -> bool:
    """True while the supervisor that owns this run is still alive."""
    if _argv_is_supervisor(_argv(run.pid), run.user_id) and _is_alive(run.pid):
        return True
    # posix_spawn has usually finished before we record the pid. Keep a short
    # grace so a cmdline that is not visible yet is not treated as pid reuse.
    age = time.time() - run.started_at
    return _is_alive(run.pid) and age < _STARTUP_GRACE_SECONDS


def start_detached_pipeline(
    user_id: str,
    *,
    command: list[str] | None = None,
) -> DetachedPipeline:
    """Start a supervisor unless this user already has one running.

    ``command`` replaces the pipeline CLI. Tests use it to spawn a sleeper.
    """
    existing = find_pipeline_run(user_id)
    if existing is not None and pipeline_is_running(existing):
        return existing
    clear_pipeline_run(user_id)

    meta_path, log_path, exit_path = _paths(user_id)
    argv = [
        sys.executable,
        "-m",
        "streamlit_utils.pipeline_subprocess",
        "supervise",
        user_id,
        str(log_path),
        str(exit_path),
    ]
    if command is not None:
        argv.extend(["--", *command])

    process = subprocess.Popen(
        argv,
        cwd=_REPO_ROOT,
        env=_child_env(),
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=None,
    )
    started_at = time.time()
    payload = {
        "pid": process.pid,
        "user_id": user_id,
        "log_path": str(log_path),
        "exit_path": str(exit_path),
        "started_at": started_at,
    }
    _write_meta(meta_path, payload)
    return DetachedPipeline(
        user_id=user_id,
        pid=process.pid,
        log_path=log_path,
        exit_path=exit_path,
        meta_path=meta_path,
        started_at=started_at,
    )


def follow_pipeline(run: DetachedPipeline, progress: ProgressWindow) -> int:
    """Apply progress events until the supervisor exits. Never signals it."""
    set_pacing = getattr(progress, "set_pacing", None)
    if callable(set_pacing):
        set_pacing(False)

    offset = 0
    pending = ""

    def drain() -> None:
        nonlocal offset, pending
        if not run.log_path.is_file():
            return
        with run.log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offset = handle.tell()
        if not chunk:
            return
        pending += chunk
        parts = pending.split("\n")
        pending = parts.pop()
        for line in parts:
            _apply_line(progress, line)

    drain()
    if callable(set_pacing):
        set_pacing(True)

    while pipeline_is_running(run):
        drain()
        checkpoint = getattr(progress, "checkpoint", None)
        if callable(checkpoint):
            checkpoint()
        time.sleep(_POLL_SECONDS)

    _reap(run.pid)
    drain()
    if pending.strip():
        _apply_line(progress, pending)

    code = _read_exit_code(run.exit_path)
    if code != 0 and getattr(progress, "_final_state", None) is None:
        progress.error(f"Pipeline subprocess exited with code {code}.")
    return code


def clear_pipeline_run(user_id: str) -> None:
    """Remove the bookkeeping files for this user. Does not signal a process."""
    for path in _paths(user_id):
        path.unlink(missing_ok=True)


def supervise(
    user_id: str,
    log_path: Path,
    exit_path: Path,
    command: list[str] | None = None,
) -> int:
    """Run the pipeline and record its exit code. The log is the progress file."""
    if command is None:
        command = pipeline_command(user_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=_REPO_ROOT,
            env=_child_env(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=None,
        )
        code = process.wait()
    exit_path.write_text(f"{code}\n", encoding="utf-8")
    return code


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 4 or args[0] != "supervise":
        print(
            "usage: python -m streamlit_utils.pipeline_subprocess "
            "supervise USER_ID LOG EXIT [-- COMMAND ...]",
            file=sys.stderr,
        )
        return 2
    command = None
    if len(args) > 4:
        if args[4] != "--" or len(args) < 6:
            print("supervise expected '-- COMMAND ...'", file=sys.stderr)
            return 2
        command = args[5:]
    return supervise(args[1], Path(args[2]), Path(args[3]), command)


def _child_env() -> dict[str, str]:
    """Import roots for a fresh interpreter that did not inherit pytest's path."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(_REPO_ROOT), str(_SRC_ROOT)]
    if existing:
        parts.extend(existing.split(os.pathsep))
    unique: list[str] = []
    for part in parts:
        if part and part not in unique:
            unique.append(part)
    env["PYTHONPATH"] = os.pathsep.join(unique)
    return env


def _run_dir() -> Path:
    path = Path(get_optional_env_variable(_RUN_DIR_ENV, _DEFAULT_RUN_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _token(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def _paths(user_id: str) -> tuple[Path, Path, Path]:
    root = _run_dir()
    name = _token(user_id)
    return root / f"{name}.json", root / f"{name}.log", root / f"{name}.exit"


def _write_meta(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _load_run(user_id: str) -> DetachedPipeline | None:
    meta_path, log_path, exit_path = _paths(user_id)
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("user_id") != user_id:
        return None
    try:
        pid = int(payload["pid"])
        started_at = float(payload["started_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return DetachedPipeline(
        user_id=user_id,
        pid=pid,
        log_path=Path(str(payload.get("log_path", log_path))),
        exit_path=Path(str(payload.get("exit_path", exit_path))),
        meta_path=meta_path,
        started_at=started_at,
    )


def _argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part]


def _argv_is_supervisor(argv: list[str], user_id: str) -> bool:
    return (
        "supervise" in argv
        and user_id in argv
        and any("pipeline_subprocess" in part for part in argv)
    )


def _proc_state(pid: int) -> str:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return ""
    end = stat.rfind(")")
    if end < 0 or end + 2 >= len(stat):
        return ""
    return stat[end + 2 : end + 3]


def _is_alive(pid: int) -> bool:
    return pid > 0 and _proc_state(pid) in _RUNNING_STATES


def _log_has_bytes(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _apply_line(progress: ProgressWindow, line: str) -> None:
    text = line.strip()
    if not text:
        return
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Skipping a pipeline progress line that is not JSON.")
        return
    if not isinstance(payload, dict):
        logger.warning("Skipping a pipeline progress line that is not an object.")
        return
    try:
        apply_progress_event(progress, payload)
    except ValueError:
        logger.warning("Skipping a pipeline progress event with an unknown op.")


def _read_exit_code(path: Path) -> int:
    for _ in range(25):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            time.sleep(0.05)
            continue
        if not text:
            time.sleep(0.05)
            continue
        try:
            return int(text)
        except ValueError:
            return 1
    return 1


def _reap(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return


if __name__ == "__main__":
    raise SystemExit(main())
