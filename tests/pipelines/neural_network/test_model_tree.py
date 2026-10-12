"""Tests for starting a new hybrid model tree."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from chess_teacher.pipelines.neural_network.model_tree import (
    eval_metrics_are_hybrid_board,
    format_model_tree_reset,
    start_new_model_tree,
)
from chess_teacher.pipelines.neural_network.models import (
    BaselineModel,
    BaselineModelStatus,
    PersonalModel,
    TrainingState,
)
from scripts.ops.start_model_tree import main

_TRAINED_AT = datetime(2026, 10, 1, tzinfo=UTC)


def _metrics(*, hybrid: bool) -> str:
    blob: dict[str, float] = {"head_candidate_style": 1.0, "move_feat_dim": 55.0}
    if hybrid:
        blob["encoder_hybrid_board"] = 1.0
    return json.dumps(blob)


def _baseline(
    version: str,
    status: BaselineModelStatus,
    *,
    hybrid: bool = False,
    parent: bool = False,
    metrics: str | None = None,
) -> BaselineModel:
    return BaselineModel(
        id=f"baseline-{version}",
        version=version,
        trained_at=_TRAINED_AT,
        status=status,
        is_parent_baseline=parent,
        model_uri=f"models/{version}.keras",
        eval_metrics=_metrics(hybrid=hybrid) if metrics is None else metrics,
    )


def _personal(
    user_id: str,
    version: str,
    status: BaselineModelStatus,
    *,
    hybrid: bool = False,
) -> PersonalModel:
    return PersonalModel(
        id=f"personal-{user_id}-{version}",
        user_id=user_id,
        version=version,
        trained_at=_TRAINED_AT,
        status=status,
        model_uri=f"models/{user_id}-{version}.keras",
        eval_metrics=_metrics(hybrid=hybrid),
    )


def test_hybrid_metric_requires_encoder_flag() -> None:
    assert eval_metrics_are_hybrid_board(_metrics(hybrid=True))
    assert eval_metrics_are_hybrid_board(json.dumps({"encoder_hybrid_board": 1}))
    assert not eval_metrics_are_hybrid_board(_metrics(hybrid=False))
    assert not eval_metrics_are_hybrid_board(None)
    assert not eval_metrics_are_hybrid_board("not-json")
    assert not eval_metrics_are_hybrid_board(json.dumps(["encoder_hybrid_board"]))


def test_start_archives_non_hybrid_and_keeps_hybrid_rows(monkeypatch) -> None:
    flat_prod = _baseline("v1", BaselineModelStatus.PRODUCTION, parent=True)
    flat_candidate = _baseline("v2", BaselineModelStatus.CANDIDATE)
    hybrid_prod = _baseline("v3", BaselineModelStatus.PRODUCTION, hybrid=True, parent=False)
    archived_flat = _baseline("v0", BaselineModelStatus.ARCHIVED)
    archived_parent = _baseline("vOld", BaselineModelStatus.ARCHIVED, parent=True)
    broken = _baseline(
        "vBad",
        BaselineModelStatus.CANDIDATE,
        metrics="{",
    )
    hybrid_personal = _personal("user-a", "v2", BaselineModelStatus.PRODUCTION, hybrid=True)
    flat_personal = _personal("user-a", "v1", BaselineModelStatus.PRODUCTION)
    flat_personal_candidate = _personal("user-b", "v1", BaselineModelStatus.CANDIDATE)
    archived_personal = _personal("user-b", "v0", BaselineModelStatus.ARCHIVED)

    saved: list[object] = []

    def _save(self, db, **kwargs):
        saved.append(self)
        return self

    monkeypatch.setattr(
        BaselineModel,
        "fetch_all_ordered",
        lambda db: [
            hybrid_prod,
            flat_candidate,
            flat_prod,
            broken,
            archived_parent,
            archived_flat,
        ],
    )
    monkeypatch.setattr(
        PersonalModel,
        "fetch_all_from_db",
        lambda db, **kwargs: [
            hybrid_personal,
            flat_personal,
            flat_personal_candidate,
            archived_personal,
        ],
    )
    monkeypatch.setattr(
        TrainingState,
        "fetch_all_from_db",
        lambda db, **kwargs: [
            TrainingState(scope="baseline", personal_queue_baseline="v1"),
            TrainingState(scope="user:user-a", personal_queue_baseline="v1"),
            TrainingState(scope="user:user-b"),
        ],
    )
    monkeypatch.setattr(BaselineModel, "save_to_db", _save)
    monkeypatch.setattr(PersonalModel, "save_to_db", _save)
    monkeypatch.setattr(TrainingState, "save_to_db", _save)
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.model_tree.reset_training_queues",
        lambda db, dry_run=False: 11,
    )

    result = start_new_model_tree(object())

    assert result.queue_rows == 11
    assert result.cleared_scopes == ("baseline", "user:user-a")
    archived_versions = {
        (change.kind, change.version, change.archived, change.cleared_parent_baseline)
        for change in result.changes
    }
    assert archived_versions == {
        ("baseline", "v2", True, False),
        ("baseline", "v1", True, True),
        ("baseline", "vBad", True, False),
        ("baseline", "vOld", False, True),
        ("personal", "v1", True, False),
    }
    personal_users = {change.user_id for change in result.changes if change.kind == "personal"}
    assert personal_users == {"user-a", "user-b"}

    saved_baselines = [row for row in saved if isinstance(row, BaselineModel)]
    assert {row.version for row in saved_baselines} == {"v2", "v1", "vBad", "vOld"}
    assert all(row.status == BaselineModelStatus.ARCHIVED for row in saved_baselines)
    assert all(row.is_parent_baseline is False for row in saved_baselines)
    assert all(row.version != "v3" for row in saved_baselines)

    saved_personal = [row for row in saved if isinstance(row, PersonalModel)]
    assert {(row.user_id, row.version) for row in saved_personal} == {
        ("user-a", "v1"),
        ("user-b", "v1"),
    }
    assert all(row.status == BaselineModelStatus.ARCHIVED for row in saved_personal)

    saved_state = [row for row in saved if isinstance(row, TrainingState)]
    assert [row.scope for row in saved_state] == ["baseline", "user:user-a"]
    assert all(row.personal_queue_baseline is None for row in saved_state)


def test_dry_run_reports_without_writing(monkeypatch) -> None:
    flat_prod = _baseline("v1", BaselineModelStatus.PRODUCTION, parent=True)
    saved: list[object] = []
    queue_calls: list[bool] = []

    monkeypatch.setattr(BaselineModel, "fetch_all_ordered", lambda db: [flat_prod])
    monkeypatch.setattr(PersonalModel, "fetch_all_from_db", lambda db, **kwargs: [])
    monkeypatch.setattr(
        TrainingState,
        "fetch_all_from_db",
        lambda db, **kwargs: [TrainingState(scope="baseline", personal_queue_baseline="v1")],
    )
    monkeypatch.setattr(BaselineModel, "save_to_db", lambda self, db, **kwargs: saved.append(self))
    monkeypatch.setattr(PersonalModel, "save_to_db", lambda self, db, **kwargs: saved.append(self))
    monkeypatch.setattr(TrainingState, "save_to_db", lambda self, db, **kwargs: saved.append(self))

    def _queue(db, dry_run=False):
        queue_calls.append(dry_run)
        return 4

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.model_tree.reset_training_queues",
        _queue,
    )

    result = start_new_model_tree(object(), dry_run=True)

    assert saved == []
    assert queue_calls == [True]
    assert result.queue_rows == 4
    assert result.changes[0].version == "v1"
    assert result.changes[0].cleared_parent_baseline is True
    report = format_model_tree_reset(result, applied=False)
    assert "Would archive 1 baseline model(s) and 0 personal model(s)." in report
    assert "baseline v1 (production, clear parent baseline)" in report
    assert "Would reset both training queues on 4 train rows." in report
    assert "Would clear personal_queue_baseline on baseline." in report


def test_cli_dry_run_does_not_apply(monkeypatch, capsys) -> None:
    db = object()
    calls: list[bool] = []

    class _Result:
        changes = ()
        queue_rows = 2
        cleared_scopes = ()

    monkeypatch.setattr("scripts.ops.start_model_tree.get_db_client", lambda: db)

    def _start(client, dry_run=False):
        assert client is db
        calls.append(dry_run)
        return _Result()

    monkeypatch.setattr("scripts.ops.start_model_tree.start_new_model_tree", _start)

    assert main(["--dry-run"]) == 0

    assert calls == [True]
    assert "2 train rows" in capsys.readouterr().out


def test_cli_yes_applies_after_preview(monkeypatch, capsys) -> None:
    calls: list[bool] = []

    class _Result:
        changes = ()
        queue_rows = 0
        cleared_scopes = ()

    monkeypatch.setattr("scripts.ops.start_model_tree.get_db_client", lambda: object())

    def _start(client, dry_run=False):
        calls.append(dry_run)
        return _Result()

    monkeypatch.setattr("scripts.ops.start_model_tree.start_new_model_tree", _start)

    assert main(["--yes"]) == 0

    assert calls == [True, False]
    out = capsys.readouterr().out
    assert "Would reset" in out
    assert "Reset both training queues" in out


def test_cli_refuses_without_yes_when_stdin_is_not_a_tty(monkeypatch) -> None:
    class _Stdin:
        def isatty(self) -> bool:
            return False

    class _Result:
        changes = ()
        queue_rows = 0
        cleared_scopes = ()

    monkeypatch.setattr("scripts.ops.start_model_tree.get_db_client", lambda: object())
    monkeypatch.setattr(
        "scripts.ops.start_model_tree.start_new_model_tree",
        lambda client, dry_run=False: _Result(),
    )
    monkeypatch.setattr("scripts.ops.start_model_tree.sys.stdin", _Stdin())

    assert main([]) == 1
