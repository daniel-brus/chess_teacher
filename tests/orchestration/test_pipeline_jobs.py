from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from orchestration.pipeline_jobs import (
    create_pipeline_job,
    pipeline_job_name,
    pipeline_job_state,
    render_pipeline_job_manifest,
    work_item_key,
)


def test_pipeline_job_name_is_dns_safe_and_timestamped() -> None:
    name = pipeline_job_name(
        "User @Chess.Com",
        now=datetime(2026, 10, 11, 20, 30, 40, tzinfo=UTC),
    )

    assert name == "pipeline-user-chess--20261011203040"
    assert len(name) <= 63


def test_render_pipeline_job_manifest_uses_shared_template() -> None:
    manifest = render_pipeline_job_manifest(
        job_name="pipeline-user-20261011203040",
        user_id="user-123",
        work_item=work_item_key("user-123"),
        image="chess-teacher:test",
        image_pull_policy="IfNotPresent",
    )

    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "chess-teacher:test"
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert container["args"] == [
        "--user-id",
        "user-123",
        "--progress-stdout",
    ]
    assert manifest["metadata"]["labels"]["chess-teacher.io/work-item"] == work_item_key("user-123")


def test_pipeline_job_state_recognizes_terminal_conditions() -> None:
    succeeded = SimpleNamespace(
        status=SimpleNamespace(
            conditions=[SimpleNamespace(status="True", type="Complete", message=None)],
            active=0,
            succeeded=1,
            failed=0,
        )
    )
    failed = SimpleNamespace(
        status=SimpleNamespace(
            conditions=[SimpleNamespace(status="True", type="Failed", message="OOM")],
            active=0,
            succeeded=0,
            failed=1,
        )
    )

    assert pipeline_job_state(succeeded) == ("succeeded", None)
    assert pipeline_job_state(failed) == ("failed", "OOM")


def test_create_pipeline_job_submits_rendered_job(
    monkeypatch,
) -> None:
    submitted: list[dict[str, object]] = []

    class BatchApi:
        def create_namespaced_job(self, *, namespace: str, body: dict[str, object]) -> None:
            assert namespace == "chess-teacher"
            submitted.append(body)

    client = SimpleNamespace(BatchV1Api=lambda: BatchApi())

    monkeypatch.setattr(
        "orchestration.pipeline_jobs.kubernetes_client",
        lambda: (client, Exception),
    )
    monkeypatch.setenv("PIPELINE_JOB_IMAGE", "chess-teacher:test")
    job_name = create_pipeline_job(namespace="chess-teacher", user_id="user-123")

    assert job_name.startswith("pipeline-user-123-")
    assert submitted[0]["metadata"]["name"] == job_name
