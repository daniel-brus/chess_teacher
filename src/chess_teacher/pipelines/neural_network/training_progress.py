"""Viewer-scoped training scores and model lineage for the Training page.

The page shows the platform baseline to every logged-in user, plus that
user's personal models. Other users' models stay out of the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from chess_teacher.pipelines.neural_network.models import (
    BaselineModel,
    BaselineModelStatus,
    PersonalModel,
)
from chess_teacher.pipelines.neural_network.training_scores import TrainingScore
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.general_utils import generate_ident_is_literal

_PERSONAL_PARENT_KIND = "personal"
_BASELINE_PARENT_KIND = "baseline"


class SubjectKind(StrEnum):
    BASELINE = "baseline"
    MINE = "mine"


class MetricKind(StrEnum):
    TOP1 = "top1"
    TOP3 = "top3"
    SF_MEAN = "sf_mean"
    SF_MEDIAN = "sf_median"


class SliceKind(StrEnum):
    OVERALL = "overall"
    AGREE = "agree"
    DISAGREE = "disagree"


class WeightKind(StrEnum):
    UNWEIGHTED = "unweighted"
    WEIGHTED = "weighted"


class ScoreAxis(StrEnum):
    ACCURACY = "accuracy"
    PAWNS = "pawns"


SUBJECT_OPTIONS: tuple[tuple[SubjectKind, str], ...] = (
    (SubjectKind.BASELINE, "Baseline"),
    (SubjectKind.MINE, "Mine"),
)
METRIC_OPTIONS: tuple[tuple[MetricKind, str], ...] = (
    (MetricKind.TOP1, "Top-1"),
    (MetricKind.TOP3, "Top-3"),
    (MetricKind.SF_MEAN, "SF mean gap"),
    (MetricKind.SF_MEDIAN, "SF median gap"),
)
SLICE_OPTIONS: tuple[tuple[SliceKind, str], ...] = (
    (SliceKind.OVERALL, "Overall"),
    (SliceKind.AGREE, "Agree"),
    (SliceKind.DISAGREE, "Disagree"),
)
WEIGHT_OPTIONS: tuple[tuple[WeightKind, str], ...] = (
    (WeightKind.UNWEIGHTED, "Unweighted"),
    (WeightKind.WEIGHTED, "Weighted"),
)

_METRIC_ORDER: tuple[MetricKind, ...] = tuple(kind for kind, _label in METRIC_OPTIONS)
_SLICE_ORDER: tuple[SliceKind, ...] = tuple(kind for kind, _label in SLICE_OPTIONS)
_WEIGHT_ORDER: tuple[WeightKind, ...] = tuple(kind for kind, _label in WEIGHT_OPTIONS)

_TOPK_METRICS = frozenset({MetricKind.TOP1, MetricKind.TOP3})
_SUBJECT_LABEL = dict(SUBJECT_OPTIONS)
_METRIC_LABEL = dict(METRIC_OPTIONS)
_SLICE_LABEL = {
    SliceKind.OVERALL: "overall",
    SliceKind.AGREE: "agree",
    SliceKind.DISAGREE: "disagree",
}

_TOPK_FIELDS: dict[tuple[MetricKind, SliceKind, WeightKind], str] = {
    (MetricKind.TOP1, SliceKind.OVERALL, WeightKind.UNWEIGHTED): "val_top1",
    (MetricKind.TOP1, SliceKind.OVERALL, WeightKind.WEIGHTED): "val_top1_weighted",
    (MetricKind.TOP3, SliceKind.OVERALL, WeightKind.UNWEIGHTED): "val_top3",
    (MetricKind.TOP3, SliceKind.OVERALL, WeightKind.WEIGHTED): "val_top3_weighted",
    (MetricKind.TOP1, SliceKind.AGREE, WeightKind.UNWEIGHTED): "val_top1_agree",
    (MetricKind.TOP1, SliceKind.AGREE, WeightKind.WEIGHTED): "val_top1_agree_weighted",
    (MetricKind.TOP3, SliceKind.AGREE, WeightKind.UNWEIGHTED): "val_top3_agree",
    (MetricKind.TOP3, SliceKind.AGREE, WeightKind.WEIGHTED): "val_top3_agree_weighted",
    (MetricKind.TOP1, SliceKind.DISAGREE, WeightKind.UNWEIGHTED): "val_top1_disagree",
    (MetricKind.TOP1, SliceKind.DISAGREE, WeightKind.WEIGHTED): "val_top1_disagree_weighted",
    (MetricKind.TOP3, SliceKind.DISAGREE, WeightKind.UNWEIGHTED): "val_top3_disagree",
    (MetricKind.TOP3, SliceKind.DISAGREE, WeightKind.WEIGHTED): "val_top3_disagree_weighted",
}
_GAP_FIELDS: dict[tuple[MetricKind, WeightKind], str] = {
    (MetricKind.SF_MEAN, WeightKind.UNWEIGHTED): "sf_delta_mean_pawns",
    (MetricKind.SF_MEAN, WeightKind.WEIGHTED): "sf_delta_mean_pawns_weighted",
    (MetricKind.SF_MEDIAN, WeightKind.UNWEIGHTED): "sf_delta_median_pawns",
    (MetricKind.SF_MEDIAN, WeightKind.WEIGHTED): "sf_delta_median_pawns_weighted",
}

_STATUS_LABEL = {
    BaselineModelStatus.PRODUCTION: "Promoted",
    BaselineModelStatus.ARCHIVED: "Replaced",
    BaselineModelStatus.CANDIDATE: "Candidate",
}
_STATUS_KEY = {
    BaselineModelStatus.PRODUCTION: "promoted",
    BaselineModelStatus.ARCHIVED: "replaced",
    BaselineModelStatus.CANDIDATE: "candidate",
}


@dataclass(frozen=True)
class SeriesSelection:
    """Which score lines the chart draws. Every combination may be on at once."""

    subjects: frozenset[SubjectKind]
    metrics: frozenset[MetricKind]
    slices: frozenset[SliceKind]
    weights: frozenset[WeightKind]

    @classmethod
    def default(cls) -> SeriesSelection:
        return cls(
            subjects=frozenset({SubjectKind.BASELINE, SubjectKind.MINE}),
            metrics=frozenset({MetricKind.TOP1, MetricKind.SF_MEAN}),
            slices=frozenset({SliceKind.OVERALL, SliceKind.DISAGREE}),
            weights=frozenset({WeightKind.UNWEIGHTED}),
        )


@dataclass(frozen=True)
class TrainingProgress:
    baselines: tuple[BaselineModel, ...]
    personals: tuple[PersonalModel, ...]
    scores: tuple[TrainingScore, ...]


@dataclass(frozen=True)
class ChartPoint:
    scored_at: datetime
    series: str
    value: float
    axis: ScoreAxis
    version: str
    status: str
    n_eval: int


@dataclass(frozen=True)
class LineageNode:
    title: str
    meta: str
    status_key: str
    trained_at: datetime
    version: str
    parent_version: str | None
    parent_kind: str | None


@dataclass(frozen=True)
class TrunkRow:
    """One baseline, with this user's personal chain beside it."""

    baseline: LineageNode
    branch: tuple[LineageNode, ...]


@dataclass(frozen=True)
class LineageLayout:
    trunk: tuple[TrunkRow, ...]
    unattached: tuple[LineageNode, ...]


def load_training_progress(db_client: DatabaseClient, user_id: str) -> TrainingProgress:
    """Baseline models, this user's personal models, and their score rows."""
    baselines = tuple(BaselineModel.fetch_all_ordered(db_client))
    personals = tuple(PersonalModel.rows_for_user(db_client, user_id))
    where = (
        f"{generate_ident_is_literal('user_id', None)} OR "
        f"{generate_ident_is_literal('user_id', user_id)}"
    )
    scores = TrainingScore.fetch_all_from_db(
        db_client,
        where=where,
        order_by='"scored_at" ASC',
    )
    known = {model.id for model in baselines} | {model.id for model in personals}
    kept = tuple(
        score
        for score in scores
        if score.model_id in known and (score.user_id is None or score.user_id == user_id)
    )
    return TrainingProgress(baselines=baselines, personals=personals, scores=kept)


def series_label(
    subject: SubjectKind,
    metric: MetricKind,
    slice_kind: SliceKind | None,
    weight: WeightKind,
) -> str:
    parts = [_SUBJECT_LABEL[subject], _METRIC_LABEL[metric]]
    if metric in _TOPK_METRICS and slice_kind is not None:
        parts.append(_SLICE_LABEL[slice_kind])
    if weight is WeightKind.WEIGHTED:
        parts.append("weighted")
    return " · ".join(parts)


def chart_points(progress: TrainingProgress, selection: SeriesSelection) -> list[ChartPoint]:
    """Long-form points for the selected lines. Empty slices are omitted."""
    baselines = {model.id: model for model in progress.baselines}
    personals = {model.id: model for model in progress.personals}
    metrics = [metric for metric in _METRIC_ORDER if metric in selection.metrics]
    slices = [slice_kind for slice_kind in _SLICE_ORDER if slice_kind in selection.slices]
    weights = [weight for weight in _WEIGHT_ORDER if weight in selection.weights]
    points: list[ChartPoint] = []
    for score in progress.scores:
        subject = SubjectKind.BASELINE if score.user_id is None else SubjectKind.MINE
        if subject not in selection.subjects:
            continue
        model = _model_for_score(score, baselines, personals)
        if model is None:
            continue
        status = _STATUS_LABEL.get(model.status, "Candidate")
        for metric in metrics:
            for weight in weights:
                if metric in _TOPK_METRICS:
                    for slice_kind in slices:
                        value = _optional_float(
                            getattr(score, _TOPK_FIELDS[metric, slice_kind, weight])
                        )
                        if value is None:
                            continue
                        points.append(
                            _point(
                                score,
                                series_label(subject, metric, slice_kind, weight),
                                value,
                                ScoreAxis.ACCURACY,
                                status,
                            )
                        )
                    continue
                value = _optional_float(getattr(score, _GAP_FIELDS[metric, weight]))
                if value is None:
                    continue
                points.append(
                    _point(
                        score,
                        series_label(subject, metric, None, weight),
                        value,
                        ScoreAxis.PAWNS,
                        status,
                    )
                )
    points.sort(key=lambda point: (point.series, point.scored_at, point.version))
    return points


def score_table_records(progress: TrainingProgress) -> list[dict[str, object]]:
    """One row per stored score, newest first. Independent of the chart toggles."""
    baselines = {model.id: model for model in progress.baselines}
    personals = {model.id: model for model in progress.personals}
    records: list[dict[str, object]] = []
    for score in sorted(progress.scores, key=lambda row: row.scored_at, reverse=True):
        model = _model_for_score(score, baselines, personals)
        if model is None:
            continue
        model_label = (
            f"Baseline {score.version}" if score.user_id is None else f"Mine {score.version}"
        )
        parent = isinstance(model, BaselineModel) and model.is_parent_baseline
        records.append({
            "Scored": score.scored_at,
            "Model": model_label,
            "Status": _STATUS_LABEL.get(model.status, "Candidate"),
            "Parent baseline": "Yes" if parent else "",
            "n": score.n_eval,
            "n agree": score.n_sf_agree,
            "n disagree": score.n_sf_disagree,
            "Top-1": score.val_top1,
            "Top-1 weighted": score.val_top1_weighted,
            "Top-3": score.val_top3,
            "Top-3 weighted": score.val_top3_weighted,
            "Agree top-1": score.val_top1_agree,
            "Agree top-1 weighted": score.val_top1_agree_weighted,
            "Agree top-3": score.val_top3_agree,
            "Agree top-3 weighted": score.val_top3_agree_weighted,
            "Disagree top-1": score.val_top1_disagree,
            "Disagree top-1 weighted": score.val_top1_disagree_weighted,
            "Disagree top-3": score.val_top3_disagree,
            "Disagree top-3 weighted": score.val_top3_disagree_weighted,
            "SF mean": score.sf_delta_mean_pawns,
            "SF mean weighted": score.sf_delta_mean_pawns_weighted,
            "SF median": score.sf_delta_median_pawns,
            "SF median weighted": score.sf_delta_median_pawns_weighted,
        })
    return records


def lineage_layout(progress: TrainingProgress) -> LineageLayout:
    """Newest baseline at the top. Personal chains hang off ``parent_baseline_version``."""
    baseline_nodes = {model.version: _baseline_node(model) for model in progress.baselines}
    grouped: dict[str, list[LineageNode]] = {}
    unattached: list[LineageNode] = []
    for model in progress.personals:
        node = _personal_node(model)
        parent_version = model.parent_baseline_version
        if parent_version and parent_version in baseline_nodes:
            grouped.setdefault(parent_version, []).append(node)
        else:
            unattached.append(node)
    trunk_order = sorted(baseline_nodes.values(), key=lambda node: node.trained_at, reverse=True)
    trunk = tuple(
        TrunkRow(baseline=node, branch=tuple(_order_chain(grouped.get(node.version, []))))
        for node in trunk_order
    )
    return LineageLayout(trunk=trunk, unattached=tuple(_order_chain(unattached)))


def _model_for_score(
    score: TrainingScore,
    baselines: dict[str, BaselineModel],
    personals: dict[str, PersonalModel],
) -> BaselineModel | PersonalModel | None:
    if score.user_id is None:
        return baselines.get(score.model_id)
    return personals.get(score.model_id)


def _point(
    score: TrainingScore,
    series: str,
    value: float,
    axis: ScoreAxis,
    status: str,
) -> ChartPoint:
    return ChartPoint(
        scored_at=score.scored_at,
        series=series,
        value=value,
        axis=axis,
        version=score.version,
        status=status,
        n_eval=score.n_eval,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _status_meta(status: BaselineModelStatus, *, parent: bool, origin: str | None) -> str:
    parts = [_STATUS_LABEL.get(status, "Candidate")]
    if parent:
        parts.append("Parent")
    if origin:
        parts.append(origin)
    return " · ".join(parts)


def _baseline_node(model: BaselineModel) -> LineageNode:
    origin = f"from {model.parent_version}" if model.parent_version else None
    return LineageNode(
        title=model.version,
        meta=_status_meta(model.status, parent=model.is_parent_baseline, origin=origin),
        status_key=_STATUS_KEY.get(model.status, "candidate"),
        trained_at=model.trained_at,
        version=model.version,
        parent_version=model.parent_version,
        parent_kind=None,
    )


def _personal_node(model: PersonalModel) -> LineageNode:
    origin: str | None = None
    if model.parent_version and model.parent_kind == _BASELINE_PARENT_KIND:
        origin = f"from baseline {model.parent_version}"
    elif model.parent_version and model.parent_kind == _PERSONAL_PARENT_KIND:
        origin = f"from {model.parent_version}"
    elif model.parent_version:
        origin = f"from {model.parent_version}"
    return LineageNode(
        title=f"You {model.version}",
        meta=_status_meta(model.status, parent=False, origin=origin),
        status_key=_STATUS_KEY.get(model.status, "candidate"),
        trained_at=model.trained_at,
        version=model.version,
        parent_version=model.parent_version,
        parent_kind=model.parent_kind,
    )


def _order_chain(nodes: list[LineageNode]) -> list[LineageNode]:
    """Roots first, then each personal child, oldest sibling first."""
    by_version = {node.version: node for node in nodes}
    children: dict[str, list[LineageNode]] = {node.version: [] for node in nodes}
    roots: list[LineageNode] = []
    for node in nodes:
        parent = node.parent_version
        if node.parent_kind == _PERSONAL_PARENT_KIND and parent in by_version:
            children[parent].append(node)
        else:
            roots.append(node)
    roots.sort(key=lambda node: (node.trained_at, node.version))
    for kids in children.values():
        kids.sort(key=lambda node: (node.trained_at, node.version))
    ordered: list[LineageNode] = []

    def walk(node: LineageNode) -> None:
        ordered.append(node)
        for child in children[node.version]:
            walk(child)

    for root in roots:
        walk(root)
    return ordered
