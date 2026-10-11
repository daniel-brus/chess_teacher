from __future__ import annotations

from pathlib import Path

import yaml


def test_dispatcher_cronjob_uses_module_entrypoint() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "orchestration"
        / "k8s"
        / "cronjob"
        / "ingestion-dispatcher.yaml"
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    container = manifest["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]

    assert container["command"] == ["python", "-m", "scripts.entrypoints.dispatcher"]


def test_dispatcher_module_imports_shared_pipeline_jobs() -> None:
    from scripts.entrypoints import dispatcher

    assert callable(dispatcher.main)
