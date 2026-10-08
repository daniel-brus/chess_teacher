"""One val-score row per completed training run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.utils.table_data_class import TableDataClass


@dataclass(frozen=True)
class TrainingScore(TableDataClass):
    """Scores of the final model after its training epochs.

    Grain is one pipeline run (``run_id``), not one epoch and not the parent model.
    """

    run_id: str
    pipeline_name: str
    version: str
    model_id: str
    scored_at: datetime
    n_eval: int
    n_sf_agree: int
    n_sf_disagree: int
    val_top1: float
    val_top3: float
    val_top1_weighted: float
    val_top3_weighted: float
    sf_delta_mean_pawns: float
    sf_delta_median_pawns: float
    sf_delta_mean_pawns_weighted: float
    sf_delta_median_pawns_weighted: float
    user_id: str | None = None
    val_top1_agree: float | None = None
    val_top3_agree: float | None = None
    val_top1_disagree: float | None = None
    val_top3_disagree: float | None = None
    val_top1_agree_weighted: float | None = None
    val_top3_agree_weighted: float | None = None
    val_top1_disagree_weighted: float | None = None
    val_top3_disagree_weighted: float | None = None

    @classmethod
    def get_yaml_path(cls) -> Path:
        return Path(__file__).parent / "metadata.yml"

    @classmethod
    def get_key(cls) -> str:
        return "training_scores"

    @classmethod
    def get_id_hash_columns(cls) -> tuple[str, ...]:
        return ()


def training_score_from_eval(
    *,
    run_id: str,
    pipeline_name: str,
    user_id: str | None,
    version: str,
    model_id: str,
    scored_at: datetime,
    metrics: EvalMetrics,
) -> TrainingScore:
    """Build the row for the model that just finished its epochs."""
    return TrainingScore(
        run_id=run_id,
        pipeline_name=pipeline_name,
        user_id=user_id,
        version=version,
        model_id=model_id,
        scored_at=scored_at,
        n_eval=metrics.n_eval,
        n_sf_agree=metrics.n_sf_agree,
        n_sf_disagree=metrics.n_sf_disagree,
        val_top1=metrics.top1_overall,
        val_top3=metrics.top3_overall,
        val_top1_weighted=metrics.top1_overall_weighted,
        val_top3_weighted=metrics.top3_overall_weighted,
        val_top1_agree=metrics.top1_sf_agree,
        val_top3_agree=metrics.top3_sf_agree,
        val_top1_disagree=metrics.top1_sf_disagree,
        val_top3_disagree=metrics.top3_sf_disagree,
        val_top1_agree_weighted=metrics.top1_sf_agree_weighted,
        val_top3_agree_weighted=metrics.top3_sf_agree_weighted,
        val_top1_disagree_weighted=metrics.top1_sf_disagree_weighted,
        val_top3_disagree_weighted=metrics.top3_sf_disagree_weighted,
        sf_delta_mean_pawns=metrics.sf_delta_mean_pawns,
        sf_delta_median_pawns=metrics.sf_delta_median_pawns,
        sf_delta_mean_pawns_weighted=metrics.sf_delta_mean_pawns_weighted,
        sf_delta_median_pawns_weighted=metrics.sf_delta_median_pawns_weighted,
    )
