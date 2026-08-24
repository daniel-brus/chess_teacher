"""Tests for scripts/run_script_job.py (validation + manifest render)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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
    assert normalize_script_basename("baseline_training") == "baseline_training.py"
    assert normalize_script_basename("baseline_training.py") == "baseline_training.py"


def test_validate_script_args_rejects_shell_metacharacters() -> None:
    with pytest.raises(SystemExit, match="Disallowed characters"):
        validate_script_args(["--workers", "4; rm -rf /"])


def test_validate_script_args_rejects_semicolon() -> None:
    with pytest.raises(SystemExit, match="Disallowed characters"):
        validate_script_args(["foo;bar"])


def test_script_job_name_is_dns_safe_and_bounded() -> None:
    fixed = datetime(2026, 8, 23, 19, 15, 0, tzinfo=UTC)
    name = script_job_name("baseline_training.py", now=fixed)
    assert name == "script-baseline-training-20260823191500"
    assert len(name) <= 63


def test_render_script_job_manifest_smoke() -> None:
    manifest = render_script_job_manifest(
        job_name="script-baseline-training-20260823191500",
        script_basename="baseline_training.py",
        script_args=["--help"],
        image="registry.example/chess-teacher:abc123",
        image_pull_policy="Always",
        namespace="chess-teacher",
    )

    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["name"] == "script-baseline-training-20260823191500"
    assert manifest["metadata"]["labels"]["chess-teacher.io/job-type"] == "script"
    assert manifest["metadata"]["labels"]["chess-teacher.io/script"] == "baseline_training.py"
    assert manifest["spec"]["ttlSecondsAfterFinished"] == 604800

    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["python", "scripts/baseline_training.py"]
    assert container["args"] == ["--help"]
    assert container["image"] == "registry.example/chess-teacher:abc123"
    assert container["envFrom"] == [{"secretRef": {"name": "chess-teacher-env"}}]

    env_names = {item["name"] for item in container["env"]}
    assert env_names == {"ENVIRONMENT", "HOSTNAME"}


def test_render_script_job_manifest_empty_args() -> None:
    manifest = render_script_job_manifest(
        job_name="script-baseline-promotion-20260823191500",
        script_basename="baseline_promotion.py",
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
    monkeypatch.setenv("IMAGE_PULL_POLICY", "Always")

    exit_code = run_script_job_main([
        "baseline_training",
        "--dry-run",
    ])
    assert exit_code == 0

    parsed = yaml.safe_load(capsys.readouterr().out)
    assert isinstance(parsed, dict)
    container = parsed["spec"]["template"]["spec"]["containers"][0]
    assert container["args"] == []
    assert container["image"] == "registry.example/chess-teacher:test"


def test_non_whitelisted_script_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_JOB_IMAGE", "registry.example/chess-teacher:test")
    monkeypatch.setenv("IMAGE_PULL_POLICY", "Always")

    with pytest.raises(SystemExit, match="not whitelisted"):
        run_script_job_main(["dispatcher", "--dry-run"])


def test_allowed_scripts_match_repo_entrypoints() -> None:
    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    for basename in ALLOWED_SCRIPTS:
        assert (scripts_dir / basename).is_file(), f"Missing whitelisted script: {basename}"


def test_bash_wrapper_whitelist_matches_python() -> None:
    wrapper = (
        Path(__file__).resolve().parent.parent.parent
        / "orchestration"
        / "k8s"
        / "run-script-job.sh"
    )
    text = wrapper.read_text(encoding="utf-8")
    for basename in ALLOWED_SCRIPTS:
        assert basename in text, f"{basename} missing from run-script-job.sh"
