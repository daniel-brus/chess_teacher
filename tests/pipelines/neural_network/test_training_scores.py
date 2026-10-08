"""Val score row written once per scored training run."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.models import PersonalModel
from chess_teacher.pipelines.neural_network.scheme_steps import RecordCandidateStep
from chess_teacher.pipelines.neural_network.training_scheme import ModelHandle
from chess_teacher.pipelines.neural_network.training_scores import (
    TrainingScore,
    training_score_from_eval,
)
from chess_teacher.utils.pipeline_utils.pipeline_base import (
    PIPELINE_RUN_ID_EXTRA,
    PipelineContext,
)


def _eval() -> EvalMetrics:
    return EvalMetrics(
        top1_overall=0.42,
        top3_overall=0.68,
        top1_overall_weighted=0.40,
        top1_sf_agree=0.73,
        top3_sf_agree=0.92,
        top1_sf_disagree=0.25,
        top3_sf_disagree=0.54,
        n_eval=10027,
        n_dropped=0,
        n_sf_agree=3690,
        n_sf_disagree=6337,
        sf_disagree_frac=0.63,
        sf_delta_mean_pawns=-0.31,
        sf_delta_median_pawns=-0.12,
        top3_overall_weighted=0.70,
        sf_delta_mean_pawns_weighted=-0.44,
        sf_delta_median_pawns_weighted=-0.20,
        top1_sf_agree_weighted=0.71,
        top3_sf_agree_weighted=0.90,
        top1_sf_disagree_weighted=0.22,
        top3_sf_disagree_weighted=0.50,
    )


def test_training_score_metadata_matches_dataclass() -> None:
    TrainingScore.assert_metadata_sync()


def test_training_score_copies_val_slices_and_sf_gap() -> None:
    scored_at = datetime(2026, 10, 8, 2, 45, tzinfo=UTC)
    row = training_score_from_eval(
        run_id="run-1",
        pipeline_name="personal_training",
        user_id="user-1",
        version="v1",
        model_id="model-1",
        scored_at=scored_at,
        metrics=_eval(),
    )
    assert row.val_top1 == 0.42
    assert row.val_top3 == 0.68
    assert row.val_top1_weighted == 0.40
    assert row.val_top3_weighted == 0.70
    assert row.val_top1_agree == 0.73
    assert row.val_top3_agree == 0.92
    assert row.val_top1_disagree == 0.25
    assert row.val_top3_disagree == 0.54
    assert row.val_top1_agree_weighted == 0.71
    assert row.val_top3_agree_weighted == 0.90
    assert row.val_top1_disagree_weighted == 0.22
    assert row.val_top3_disagree_weighted == 0.50
    assert row.n_eval == 10027
    assert row.n_sf_agree == 3690
    assert row.n_sf_disagree == 6337
    assert row.sf_delta_mean_pawns == -0.31
    assert row.sf_delta_median_pawns == -0.12
    assert row.sf_delta_mean_pawns_weighted == -0.44
    assert row.sf_delta_median_pawns_weighted == -0.20
    assert row.user_id == "user-1"


class _Scheme:
    pipeline_name = "personal_training"
    split_version = "baseline-v1"

    def next_version(self, db_client: object) -> str:
        del db_client
        return "v1"

    def insert_candidate(self, db_client: object, **kwargs: object) -> ModelHandle:
        del db_client, kwargs
        row = PersonalModel(
            id="model-1",
            user_id="user-1",
            version="v1",
            trained_at=datetime(2026, 10, 8, tzinfo=UTC),
        )
        return ModelHandle(
            key="v1",
            weights_uri="s3://model.keras",
            compatible=True,
            kind="personal",
            payload=row,
        )


class _Tracker:
    def log_training_run(self, **kwargs: object) -> tuple[str, str]:
        del kwargs
        return "mlflow-1", "s3://model.keras"


def test_record_candidate_saves_one_score_row(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    saved: list[TrainingScore] = []

    def save(self: TrainingScore, db_client: object, **kwargs: object) -> object:
        del db_client, kwargs
        saved.append(self)
        return object()

    monkeypatch.setattr(TrainingScore, "save_to_db", save)
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.mlflow_utils.MLflowTracker",
        _Tracker,
    )
    model_path = tmp_path / "model.keras"
    model_path.write_bytes(b"weights")
    context = PipelineContext(
        user_id="user-1",
        extras={
            PIPELINE_RUN_ID_EXTRA: "run-1",
            "trained_model_path": model_path,
            "train_metrics": {"n_samples": 10.0, "epochs": 20.0},
            "candidate_eval": _eval(),
            "parent_eval": None,
            "parent": None,
        },
    )
    RecordCandidateStep(_Scheme()).run(object(), context)  # type: ignore[arg-type]
    assert len(saved) == 1
    assert saved[0].run_id == "run-1"
    assert saved[0].pipeline_name == "personal_training"
    assert saved[0].version == "v1"
    assert saved[0].model_id == "model-1"
    assert saved[0].val_top1_disagree == 0.25
    assert saved[0].sf_delta_median_pawns == -0.12


def test_record_candidate_skips_score_row_without_eval(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    saved: list[TrainingScore] = []

    def save(self: TrainingScore, db_client: object, **kwargs: object) -> object:
        del db_client, kwargs
        saved.append(self)
        return object()

    monkeypatch.setattr(TrainingScore, "save_to_db", save)
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.mlflow_utils.MLflowTracker",
        _Tracker,
    )
    model_path = tmp_path / "model.keras"
    model_path.write_bytes(b"weights")
    context = PipelineContext(
        user_id="user-1",
        extras={
            PIPELINE_RUN_ID_EXTRA: "run-1",
            "trained_model_path": model_path,
            "train_metrics": {},
            "candidate_eval": None,
            "parent": None,
        },
    )
    RecordCandidateStep(_Scheme()).run(object(), context)  # type: ignore[arg-type]
    assert saved == []
