"""Chart lines, score table, and lineage for the training page."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

from chess_teacher.pipelines.neural_network.candidate_eval import MAX_CANDIDATES, MOVE_FEAT_DIM
from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.models import (
    BaselineModel,
    BaselineModelStatus,
    PersonalModel,
)
from chess_teacher.pipelines.neural_network.training_progress import (
    MetricKind,
    SeriesSelection,
    SliceKind,
    SubjectKind,
    TrainingProgress,
    WeightKind,
    chart_points,
    lineage_layout,
    load_training_progress,
    score_table_records,
    series_label,
)
from chess_teacher.pipelines.neural_network.training_scores import (
    TrainingScore,
    training_score_from_eval,
)
from streamlit_utils.training_chart import build_score_chart
from streamlit_utils.training_lineage import lineage_markup


def _when(hour: int) -> datetime:
    return datetime(2026, 10, 1, hour, tzinfo=UTC)


def _style_metrics() -> str:
    return json.dumps({
        "head_candidate_style": 1.0,
        "move_feat_dim": MOVE_FEAT_DIM,
        "max_candidates": MAX_CANDIDATES,
    })


def _metrics(*, disagree: float | None) -> EvalMetrics:
    return EvalMetrics(
        top1_overall=0.40,
        top3_overall=0.70,
        top1_sf_agree=0.80,
        top3_sf_agree=0.90,
        top1_sf_disagree=disagree,
        top3_sf_disagree=None if disagree is None else disagree + 0.1,
        top1_overall_weighted=0.38,
        n_eval=20,
        n_dropped=0,
        n_sf_agree=8,
        n_sf_disagree=12,
        sf_disagree_frac=0.6,
        sf_delta_mean_pawns=-0.25,
        sf_delta_median_pawns=-0.10,
        top3_overall_weighted=0.66,
        top1_sf_agree_weighted=0.77,
        top3_sf_agree_weighted=0.88,
        top1_sf_disagree_weighted=None if disagree is None else 0.18,
        top3_sf_disagree_weighted=None if disagree is None else 0.28,
    )


def _baseline(
    version: str,
    *,
    hour: int,
    status: BaselineModelStatus,
    parent: bool = False,
    parent_version: str | None = None,
) -> BaselineModel:
    return BaselineModel(
        id=f"base-{version}",
        version=version,
        trained_at=_when(hour),
        model_uri=f"s3://models/{version}",
        status=status,
        is_parent_baseline=parent,
        parent_version=parent_version,
        eval_metrics=_style_metrics(),
    )


def _personal(
    version: str,
    *,
    hour: int,
    status: BaselineModelStatus,
    parent_version: str | None,
    parent_kind: str | None,
    parent_baseline_version: str | None,
    user_id: str = "user-1",
) -> PersonalModel:
    return PersonalModel(
        id=f"me-{version}",
        user_id=user_id,
        version=version,
        trained_at=_when(hour),
        model_uri=f"s3://models/me-{version}",
        status=status,
        parent_version=parent_version,
        parent_kind=parent_kind,
        parent_baseline_version=parent_baseline_version,
        eval_metrics=_style_metrics(),
    )


def _score(
    *,
    model_id: str,
    version: str,
    user_id: str | None,
    hour: int,
    disagree: float | None = 0.2,
) -> TrainingScore:
    pipeline = "baseline_training" if user_id is None else "personal_training"
    return training_score_from_eval(
        run_id=f"run-{model_id}",
        pipeline_name=pipeline,
        user_id=user_id,
        version=version,
        model_id=model_id,
        scored_at=_when(hour),
        metrics=_metrics(disagree=disagree),
    )


def _progress() -> TrainingProgress:
    older = _baseline("v1", hour=1, status=BaselineModelStatus.ARCHIVED)
    current = _baseline(
        "v2",
        hour=3,
        status=BaselineModelStatus.PRODUCTION,
        parent=True,
        parent_version="v1",
    )
    first = _personal(
        "v1",
        hour=4,
        status=BaselineModelStatus.ARCHIVED,
        parent_version="v2",
        parent_kind="baseline",
        parent_baseline_version="v2",
    )
    second = _personal(
        "v2",
        hour=5,
        status=BaselineModelStatus.PRODUCTION,
        parent_version="v1",
        parent_kind="personal",
        parent_baseline_version="v2",
    )
    loose = _personal(
        "v9",
        hour=2,
        status=BaselineModelStatus.CANDIDATE,
        parent_version=None,
        parent_kind=None,
        parent_baseline_version=None,
    )
    return TrainingProgress(
        baselines=(current, older),
        personals=(second, first, loose),
        scores=(
            _score(model_id=older.id, version="v1", user_id=None, hour=1, disagree=None),
            _score(model_id=current.id, version="v2", user_id=None, hour=3),
            _score(model_id=first.id, version="v1", user_id="user-1", hour=4),
            _score(model_id=second.id, version="v2", user_id="user-1", hour=5),
        ),
    )


def test_default_lines_include_both_axes_and_skip_empty_slices() -> None:
    points = chart_points(_progress(), SeriesSelection.default())
    series = {point.series for point in points}
    assert series == {
        series_label(
            SubjectKind.BASELINE, MetricKind.TOP1, SliceKind.OVERALL, WeightKind.UNWEIGHTED
        ),
        series_label(
            SubjectKind.BASELINE, MetricKind.TOP1, SliceKind.DISAGREE, WeightKind.UNWEIGHTED
        ),
        series_label(SubjectKind.BASELINE, MetricKind.SF_MEAN, None, WeightKind.UNWEIGHTED),
        series_label(SubjectKind.MINE, MetricKind.TOP1, SliceKind.OVERALL, WeightKind.UNWEIGHTED),
        series_label(SubjectKind.MINE, MetricKind.TOP1, SliceKind.DISAGREE, WeightKind.UNWEIGHTED),
        series_label(SubjectKind.MINE, MetricKind.SF_MEAN, None, WeightKind.UNWEIGHTED),
    }
    baseline_disagree = [
        point
        for point in points
        if point.series
        == series_label(
            SubjectKind.BASELINE, MetricKind.TOP1, SliceKind.DISAGREE, WeightKind.UNWEIGHTED
        )
    ]
    assert [point.version for point in baseline_disagree] == ["v2"]
    assert {point.axis.value for point in points} == {"accuracy", "pawns"}


def test_weighted_and_top3_can_be_on_with_the_defaults() -> None:
    selection = SeriesSelection(
        subjects=frozenset({SubjectKind.BASELINE}),
        metrics=frozenset({MetricKind.TOP1, MetricKind.TOP3, MetricKind.SF_MEDIAN}),
        slices=frozenset({SliceKind.AGREE}),
        weights=frozenset({WeightKind.WEIGHTED}),
    )
    series = {point.series for point in chart_points(_progress(), selection)}
    assert series == {
        "Baseline · Top-1 · agree · weighted",
        "Baseline · Top-3 · agree · weighted",
        "Baseline · SF median gap · weighted",
    }


def test_score_table_is_newest_first_and_marks_the_parent() -> None:
    records = score_table_records(_progress())
    assert [row["Model"] for row in records] == ["Mine v2", "Mine v1", "Baseline v2", "Baseline v1"]
    assert records[2]["Parent baseline"] == "Yes"
    assert records[2]["Status"] == "Promoted"
    assert records[3]["Disagree top-1"] is None


def test_lineage_puts_the_personal_chain_beside_its_baseline() -> None:
    layout = lineage_layout(_progress())
    assert [row.baseline.version for row in layout.trunk] == ["v2", "v1"]
    assert [node.version for node in layout.trunk[0].branch] == ["v1", "v2"]
    assert "Parent" in layout.trunk[0].baseline.meta
    assert layout.trunk[0].branch[0].title == "You v1"
    assert layout.trunk[0].branch[0].meta == "Replaced · from baseline v2"
    assert layout.trunk[0].branch[1].meta == "Promoted · from v1"
    assert [node.version for node in layout.unattached] == ["v9"]
    markup = lineage_markup(layout)
    assert "You v2" in markup
    assert "Your models with no baseline parent" in markup
    assert "promoted" in markup
    assert "<script" not in markup


def test_dual_axis_chart_keeps_accuracy_and_pawns_independent() -> None:
    points = chart_points(_progress(), SeriesSelection.default())
    chart = build_score_chart(points)
    assert chart is not None
    spec = chart.to_dict()
    assert spec["resolve"]["scale"]["y"] == "independent"


def test_accuracy_only_chart_uses_one_axis() -> None:
    selection = SeriesSelection(
        subjects=frozenset({SubjectKind.MINE}),
        metrics=frozenset({MetricKind.TOP3}),
        slices=frozenset({SliceKind.OVERALL}),
        weights=frozenset({WeightKind.UNWEIGHTED}),
    )
    chart = build_score_chart(chart_points(_progress(), selection))
    assert chart is not None
    spec = chart.to_dict()
    assert "layer" not in spec
    assert spec["encoding"]["y"]["title"] == "Accuracy"


def test_load_keeps_only_this_user_and_known_models(monkeypatch) -> None:
    progress = _progress()
    other = _score(model_id="someone-else", version="v1", user_id="user-2", hour=6)
    orphan = _score(model_id="missing", version="v8", user_id=None, hour=6)
    monkeypatch.setattr(
        BaselineModel,
        "fetch_all_ordered",
        lambda _db: list(progress.baselines),
    )
    monkeypatch.setattr(
        PersonalModel,
        "rows_for_user",
        lambda _db, user_id: list(progress.personals) if user_id == "user-1" else [],
    )
    monkeypatch.setattr(
        TrainingScore,
        "fetch_all_from_db",
        lambda _db, **_kwargs: [*progress.scores, other, orphan],
    )
    loaded = load_training_progress(MagicMock(), "user-1")
    assert {score.model_id for score in loaded.scores} == {
        score.model_id for score in progress.scores
    }
    assert len(loaded.personals) == 3
