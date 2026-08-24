"""Tests for scripts/utils/run_script_job.py (validation + manifest render)."""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from scripts.utils.run_script_job import (
    ALLOWED_SCRIPTS,
    render_script_job_manifest,
    resolve_script_relpath,
    script_job_name,
    validate_script_args,
)
from scripts.utils.run_script_job import main as run_script_job_main


def test_resolve_script_relpath_short_name() -> None:
    assert resolve_script_relpath("baseline_training") == "entrypoints/baseline_training.py"
    assert (
        resolve_script_relpath("ops/backfill_candidate_evals.py")
        == "ops/backfill_candidate_evals.py"
    )


def test_validate_script_args_rejects_shell_metacharacters() -> None:
    with pytest.raises(SystemExit, match="Disallowed characters"):
        validate_script_args(["--workers", "4; rm -rf /"])


def test_script_job_name_is_dns_safe_and_bounded() -> None:
    fixed = datetime(2026, 8, 23, 19, 15, 0, tzinfo=UTC)
    name = script_job_name("entrypoints/baseline_training.py", now=fixed)
    assert name == "script-baseline-training-20260823191500"
    assert len(name) <= 63


def test_render_script_job_manifest_smoke() -> None:
    manifest = render_script_job_manifest(
        job_name="script-baseline-training-20260823191500",
        script_relpath="entrypoints/baseline_training.py",
        script_args=["--help"],
        image="registry.example/chess-teacher:abc123",
        image_pull_policy="Always",
        namespace="chess-teacher",
    )

    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["labels"]["chess-teacher.io/script"] == "baseline_training.py"
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["python", "scripts/entrypoints/baseline_training.py"]
    assert container["args"] == ["--help"]
    assert container["image"] == "registry.example/chess-teacher:abc123"


def test_dry_run_output_is_valid_yaml(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PIPELINE_JOB_IMAGE", "registry.example/chess-teacher:test")
    monkeypatch.setenv("IMAGE_PULL_POLICY", "Always")

    exit_code = run_script_job_main(["baseline_training", "--dry-run"])
    assert exit_code == 0

    parsed = yaml.safe_load(capsys.readouterr().out)
    container = parsed["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["python", "scripts/entrypoints/baseline_training.py"]
    assert container["image"] == "registry.example/chess-teacher:test"


def test_non_whitelisted_script_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_JOB_IMAGE", "registry.example/chess-teacher:test")
    monkeypatch.setenv("IMAGE_PULL_POLICY", "Always")

    with pytest.raises(SystemExit, match="not whitelisted"):
        run_script_job_main(["dispatcher", "--dry-run"])


def test_allowed_scripts_exist_on_disk() -> None:
    scripts_root = Path(__file__).resolve().parent.parent.parent / "scripts"
    for relpath in ALLOWED_SCRIPTS:
        assert (scripts_root / relpath).is_file(), f"Missing whitelisted script: {relpath}"


def _bash_wrapper_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "orchestration"
        / "k8s"
        / "run-script-job.sh"
    )


def test_bash_wrapper_whitelist_matches_python() -> None:
    text = _bash_wrapper_path().read_text(encoding="utf-8")
    for relpath in ALLOWED_SCRIPTS:
        assert relpath in text, f"{relpath} missing from run-script-job.sh"


def test_bash_wrapper_has_valid_syntax() -> None:
    """Reject Python-isms (!r) and other bash parse errors (normalize CRLF for Windows)."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available")

    source = _bash_wrapper_path().read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    assert b"!r}" not in source, "Python !r repr leaked into bash parameter expansion"
    # Bytes stdin avoids Windows pipe CRLF translation that breaks bash -n.
    result = subprocess.run(
        ["bash", "-n"],
        input=source,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
