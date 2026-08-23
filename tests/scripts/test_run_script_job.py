"""Tests for scripts/run_script_job.py (validation + manifest render)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import yaml

from scripts.run_script_job import (
    ALLOWED_SCRIPTS,
    normalize_script_basename,
    render_script_job_manifest,
    script_job_name,
    validate_script_args,
)
from scripts.run_script_job import main as run_script_job_main


def test_normalize_script_basename_adds_py_suffix() -> None:
    assert normalize_script_basename("backfill_candidate_evals") == "backfill_candidate_evals.py"
    assert normalize_script_basename("backfill_candidate_evals.py") == "backfill_candidate_evals.py"


def test_validate_script_args_rejects_shell_metacharacters() -> None:
    with pytest.raises(SystemExit, match="Disallowed characters"):
        validate_script_args(["--workers", "4; rm -rf /"])


def test_validate_script_args_rejects_semicolon() -> None:
    with pytest.raises(SystemExit, match="Disallowed characters"):
        validate_script_args(["foo;bar"])


def test_script_job_name_is_dns_safe_and_bounded() -> None:
    fixed = datetime(2026, 8, 23, 19, 15, 0, tzinfo=UTC)
    name = script_job_name("backfill_candidate_evals.py", now=fixed)
    assert name == "script-backfill-candidate-evals-20260823191500"
    assert len(name) <= 63


def test_render_script_job_manifest_smoke() -> None:
    manifest = render_script_job_manifest(
        job_name="script-backfill-candidate-evals-20260823191500",
        script_basename="backfill_candidate_evals.py",
        script_args=["--workers", "4"],
        image="registry.example/chess-teacher:abc123",
        image_pull_policy="Always",
        namespace="chess-teacher",
    )

    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["name"] == "script-backfill-candidate-evals-20260823191500"
    assert manifest["metadata"]["labels"]["chess-teacher.io/job-type"] == "script"
    assert (
        manifest["metadata"]["labels"]["chess-teacher.io/script"] == "backfill_candidate_evals.py"
    )
    assert manifest["spec"]["ttlSecondsAfterFinished"] == 604800

    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["python", "scripts/backfill_candidate_evals.py"]
    assert container["args"] == ["--workers", "4"]
    assert container["image"] == "registry.example/chess-teacher:abc123"
    assert container["envFrom"] == [{"secretRef": {"name": "chess-teacher-env"}}]

    env_names = {item["name"] for item in container["env"]}
    assert env_names == {"ENVIRONMENT", "HOSTNAME"}


def test_render_script_job_manifest_empty_args() -> None:
    manifest = render_script_job_manifest(
        job_name="script-baseline-train-until-caught-up-20260823191500",
        script_basename="baseline_train_until_caught_up.py",
        script_args=[],
        image="img:tag",
        image_pull_policy="IfNotPresent",
        namespace="chess-teacher",
    )
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["args"] == []


def test_dry_run_output_is_valid_yaml(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PIPELINE_JOB_IMAGE", "registry.example/chess-teacher:test")

    exit_code = run_script_job_main([
        "backfill_candidate_evals",
        "--dry-run",
        "--",
        "--workers",
        "2",
        "--limit",
        "5",
    ])
    assert exit_code == 0

    parsed = yaml.safe_load(capsys.readouterr().out)
    assert isinstance(parsed, dict)
    container = parsed["spec"]["template"]["spec"]["containers"][0]
    assert container["args"] == ["--workers", "2", "--limit", "5"]


def test_non_whitelisted_script_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_JOB_IMAGE", "registry.example/chess-teacher:test")

    with pytest.raises(SystemExit, match="not whitelisted"):
        run_script_job_main(["maintenance", "--dry-run"])


def test_allowed_scripts_match_repo_entrypoints() -> None:
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    for basename in ALLOWED_SCRIPTS:
        assert (scripts_dir / basename).is_file(), f"Missing whitelisted script: {basename}"
