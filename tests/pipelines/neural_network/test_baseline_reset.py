"""Unit tests for baseline training reset."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from chess_teacher.pipelines.neural_network.baseline_reset import reset_baseline_training
from chess_teacher.pipelines.neural_network.models import (
    BaselineModel,
    BaselineModelStatus,
    TrainingState,
)


def _model(*, version: str, status: BaselineModelStatus) -> BaselineModel:
    return BaselineModel(
        id=f"id-{version}",
        version=version,
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        status=status,
    )


def test_reset_dry_run_reports_without_writes() -> None:
    db = MagicMock()
    prev_cutoff = datetime(2026, 7, 19, 15, 0, 25, tzinfo=UTC)
    state = TrainingState(scope="baseline", last_trained_data_cutoff=prev_cutoff)
    active = (
        _model(version="v9", status=BaselineModelStatus.CANDIDATE),
        _model(version="v13", status=BaselineModelStatus.PRODUCTION),
    )

    with (
        patch.object(TrainingState, "for_baseline", return_value=state),
        patch.object(BaselineModel, "fetch_all_from_db", return_value=list(active)),
        patch.object(TrainingState, "save_to_db") as save_state,
        patch.object(BaselineModel, "save_to_db") as save_model,
    ):
        result = reset_baseline_training(db, dry_run=True)

    assert result.previous_cutoff == prev_cutoff
    assert result.archived_versions == ("v9", "v13")
    assert result.models_archived == 2
    assert result.cutoff_cleared is True
    save_state.assert_not_called()
    save_model.assert_not_called()


def test_reset_applies_cutoff_and_archives() -> None:
    db = MagicMock()
    state = TrainingState(scope="baseline", last_trained_data_cutoff=None)
    row = _model(version="v6", status=BaselineModelStatus.CANDIDATE)

    with (
        patch.object(TrainingState, "for_baseline", return_value=state),
        patch.object(BaselineModel, "fetch_all_from_db", return_value=[row]),
        patch.object(TrainingState, "save_to_db", autospec=True) as save_state,
        patch.object(BaselineModel, "save_to_db", autospec=True) as save_model,
    ):
        result = reset_baseline_training(db, dry_run=False)

    assert result.models_archived == 1
    save_state.assert_called_once()
    save_model.assert_called_once()
    saved_model = save_model.call_args.args[0]
    assert isinstance(saved_model, BaselineModel)
    assert saved_model.status == BaselineModelStatus.ARCHIVED
    assert saved_model.version == "v6"
