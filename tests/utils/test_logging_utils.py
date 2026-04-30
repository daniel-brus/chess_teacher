import logging
import os
import tempfile
from pathlib import Path

from src.utils.logging_utils import configure_logging


def _pick_writable_temp_root() -> str:
    candidates: list[str] = []

    windows_tmp = r"C:\tmp"
    if os.name == "nt" and os.path.isdir(windows_tmp):
        candidates.append(windows_tmp)

    project_tmp = Path(__file__).resolve().parents[1] / "data" / "tmp"
    project_tmp.mkdir(parents=True, exist_ok=True)
    candidates.append(str(project_tmp))

    env_candidates = [
        os.getenv("RUNNER_TEMP"),
        os.getenv("TMPDIR"),
        os.getenv("TEMP"),
        os.getenv("TMP"),
    ]
    candidates.extend([c for c in env_candidates if c])

    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        try:
            with tempfile.TemporaryDirectory(dir=candidate) as probe_dir:
                probe_file = Path(probe_dir) / "probe.txt"
                probe_file.write_text("ok", encoding="utf-8")
            return candidate
        except OSError:
            continue

    return "."


def test_configure_logging_is_idempotent():
    root = logging.getLogger()
    root.handlers.clear()
    if hasattr(root, "_chess_teacher_logging_configured"):
        delattr(root, "_chess_teacher_logging_configured")

    with tempfile.TemporaryDirectory(dir=_pick_writable_temp_root()) as temp_dir:
        log_file = Path(temp_dir) / "app.log"
        configure_logging(force=True, level="INFO", log_file=str(log_file))
        handler_count = len(root.handlers)

        configure_logging(level="INFO", log_file=str(log_file))
        assert len(root.handlers) == handler_count
