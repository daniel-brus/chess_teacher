"""Unit tests for baseline training reset."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from chess_teacher.pipelines.neural_network.baseline_reset import reset_baseline_training
from chess_teacher.pipelines.neural_network.models import (
    PROCESSED_FLAG_BASELINE,
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
    active = (
        _model(version="v9", status=BaselineModelStatus.CANDIDATE),
        _model(version="v13", status=BaselineModelStatus.PRODUCTION),
    )
    registry = MagicMock()

    with (
        patch.object(BaselineModel, "fetch_all_from_db", return_value=list(active)),
        patch.object(TrainingState, "save_to_db") as save_state,
        patch.object(BaselineModel, "save_to_db") as save_model,
        patch(
            "chess_teacher.pipelines.neural_network.baseline_reset.get_split_registry",
            return_value=registry,
        ) as get_registry,
    ):
        result = reset_baseline_training(db, dry_run=True)

    assert result.archived_versions == ("v9", "v13")
    assert result.models_archived == 2
    assert result.flags_cleared == 0
    save_state.assert_not_called()
    save_model.assert_not_called()
    get_registry.assert_not_called()
    registry.clear_processed.assert_not_called()


def test_reset_archives_and_clears_flags() -> None:
    db = MagicMock()
    row = _model(version="v6", status=BaselineModelStatus.CANDIDATE)
    registry = MagicMock()
    registry.clear_processed.return_value = 8

    with (
        patch.object(BaselineModel, "fetch_all_from_db", return_value=[row]),
        patch.object(TrainingState, "save_to_db", autospec=True) as save_state,
        patch.object(BaselineModel, "save_to_db", autospec=True) as save_model,
        patch(
            "chess_teacher.pipelines.neural_network.baseline_reset.get_split_registry",
            return_value=registry,
        ) as get_registry,
    ):
        result = reset_baseline_training(db, dry_run=False)

    assert result.models_archived == 1
    assert result.flags_cleared == 8
    save_state.assert_not_called()
    save_model.assert_called_once()
    saved_model = save_model.call_args.args[0]
    assert isinstance(saved_model, BaselineModel)
    assert saved_model.status == BaselineModelStatus.ARCHIVED
    assert saved_model.version == "v6"
    get_registry.assert_called_once()
    assert get_registry.call_args.kwargs["split_version"] == "baseline-v1"
    registry.clear_processed.assert_called_once_with(flag_column=PROCESSED_FLAG_BASELINE)
