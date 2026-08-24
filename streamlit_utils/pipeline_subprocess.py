"""Run the pipeline CLI in a subprocess and stream JSON-lines progress events."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import IO

from chess_teacher.utils.pipeline_utils.json_lines_progress import apply_progress_event
from chess_teacher.utils.pipeline_utils.pipeline_helpers import ProgressWindow

_PIPELINE_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "entrypoints" / "pipeline.py"
)


def pipeline_script_path() -> Path:
    return _PIPELINE_SCRIPT


def _drain_progress_stdout(stdout: IO[str], progress: ProgressWindow) -> None:
    for raw_line in stdout:
        line = raw_line.strip()
        if not line:
            continue
        apply_progress_event(progress, json.loads(line))


def run_pipeline_subprocess(
    *,
    user_id: str,
    progress: ProgressWindow,
) -> int:
    """Run ``scripts/entrypoints/pipeline.py`` and mirror its stdout progress stream."""
    if not _PIPELINE_SCRIPT.is_file():
        progress.error(f"Pipeline script not found: {_PIPELINE_SCRIPT}")
        return 1

    process = subprocess.Popen(
        [
            sys.executable,
            str(_PIPELINE_SCRIPT),
            "--user-id",
            user_id,
            "--progress-stdout",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        progress.error("Pipeline subprocess stdout pipe was not available.")
        return 1

    # Read progress on the main thread: Streamlit widgets require ScriptRunContext.
    stdout = process.stdout
    try:
        _drain_progress_stdout(stdout, progress)
        exit_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    finally:
        stdout.close()

    if exit_code != 0 and getattr(progress, "_final_state", None) is None:
        progress.error(f"Pipeline subprocess exited with code {exit_code}.")
    return exit_code
