"""Parent-agnostic training steps: one step list, two schemes."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.scheme_steps import (
    SKIP_KEY,
    AdvanceTrainingCursorStep,
    ApplyPromotionStep,
    PrepareEvaluationStep,
    PrepareTrainingStep,
    ScoreEvaluationStep,
    TrainModelStep,
    build_training_scheme_steps,
)
from chess_teacher.pipelines.neural_network.schemes import (
    BaselineTrainingScheme,
    PersonalTrainingScheme,
)
from chess_teacher.pipelines.neural_network.training_scheme import ModelHandle, PromotionDecision
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext


def _metrics(
    *,
    top1: float,
    agree: float | None = None,
    disagree: float | None = None,
) -> EvalMetrics:
    return EvalMetrics(
        top1_overall=top1,
        top3_overall=top1,
        top1_sf_agree=agree,
        top3_sf_agree=agree,
        top1_sf_disagree=disagree,
        top3_sf_disagree=disagree,
        top1_overall_weighted=top1,
        n_eval=8,
        n_dropped=0,
        n_sf_agree=4,
        n_sf_disagree=4,
        sf_disagree_frac=0.5,
    )


class _Scheme:
    """Stand-in that records calls. Not a baseline or personal model."""

    pipeline_name = "shared"
    split_version = "baseline-v1"
    min_new_moves = 10

    def __init__(self) -> None:
        self.pending = 25
        self.checked = 0
        self.batch: tuple[list[object], list[str]] = ([object()], ["g1"])
        self.eval_datums: list[object] = [object(), object()]
        self.parent: ModelHandle | None = ModelHandle(
            key="p",
            weights_uri=None,
            compatible=False,
            kind="chain",
        )
        self.reference: ModelHandle | None = None
        self.fit_calls: list[Any] = []
        self.saved: list[Path] = []
        self.marked: list[list[str]] = []
        self.applied_reference: list[ModelHandle | None] = []
        self.decision = PromotionDecision(True, "ok")

    def note_checked(self, db_client: object) -> None:
        del db_client
        self.checked += 1

    def count_pending(self, db_client: object) -> int:
        del db_client
        return self.pending

    def resolve_parent(self, db_client: object) -> ModelHandle | None:
        del db_client
        return self.parent

    def resolve_reference(self, db_client: object) -> ModelHandle | None:
        del db_client
        return self.reference

    def load_train_batch(self, db_client: object) -> tuple[list[object], list[str]]:
        del db_client
        return self.batch

    def load_eval_datums(self, db_client: object) -> list[object]:
        del db_client
        return list(self.eval_datums)

    def fit(
        self,
        datums: list[object],
        *,
        weights_path: Path | None,
    ) -> tuple[object, dict[str, float]]:
        self.fit_calls.append((datums, weights_path))
        return object(), {"n_samples": float(len(datums))}

    def save(self, model: object, path: Path) -> None:
        del model
        self.saved.append(path)

    def mark_trained(self, db_client: object, game_ids: list[str]) -> None:
        del db_client
        self.marked.append(game_ids)

    def decide_promotion(self, **kwargs: object) -> PromotionDecision:
        del kwargs
        return self.decision

    def apply_promotion(
        self,
        db_client: object,
        *,
        candidate: ModelHandle,
        reference: ModelHandle | None,
        eval_metrics_json: str | None,
    ) -> ModelHandle:
        del db_client, eval_metrics_json
        self.applied_reference.append(reference)
        return candidate


def test_prepare_skips_without_loading_when_pending_is_low() -> None:
    scheme = _Scheme()
    scheme.pending = 3
    context = PipelineContext()
    PrepareTrainingStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert context.extras[SKIP_KEY] is True
    assert scheme.checked == 1
    assert "train_datums" not in context.extras


def test_prepare_loads_parent_reference_and_batch() -> None:
    scheme = _Scheme()
    scheme.reference = ModelHandle(
        key="served",
        weights_uri="s3://m",
        compatible=True,
        kind="chain",
    )
    context = PipelineContext()
    PrepareTrainingStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert context.extras[SKIP_KEY] is False
    assert context.extras["parent"] is scheme.parent
    assert context.extras["reference"] is scheme.reference
    assert context.extras["train_game_ids"] == ["g1"]


def test_train_step_fits_and_drops_the_batch() -> None:
    scheme = _Scheme()
    context = PipelineContext()
    context.extras["train_datums"] = ["move"]
    context.extras[SKIP_KEY] = False
    context.extras["parent"] = ModelHandle(
        key="p", weights_uri=None, compatible=False, kind="chain"
    )
    TrainModelStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.fit_calls[0][0] == ["move"]
    assert scheme.fit_calls[0][1] is None
    assert "train_datums" not in context.extras
    assert context.extras["trained_model_path"].name == "model.keras"


def test_train_step_skips_when_prepare_skipped() -> None:
    scheme = _Scheme()
    context = PipelineContext()
    context.extras[SKIP_KEY] = True
    TrainModelStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.fit_calls == []


def test_prepare_evaluation_reuses_datums_already_in_context() -> None:
    scheme = _Scheme()
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    held = ["already"]
    context.extras["eval_datums"] = held
    PrepareEvaluationStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert context.extras["eval_datums"] is held


def test_prepare_evaluation_loads_once_when_missing() -> None:
    scheme = _Scheme()
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    PrepareEvaluationStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert context.extras["eval_datums"] == scheme.eval_datums


def test_score_packs_eval_once_for_candidate_and_reference(monkeypatch) -> None:
    packed = object()
    loads: list[Path] = []
    packs: list[list[object]] = []
    scores: list[tuple[object, object]] = []

    def pack_datums_for_eval(datums: list[object]) -> object:
        packs.append(datums)
        return packed

    def load_candidate_style_keras(path: Path, *, compile_model: bool = False) -> object:
        del compile_model
        loads.append(path)
        return path

    def evaluate_packed(model: object, packed_arg: object) -> EvalMetrics:
        scores.append((model, packed_arg))
        return _metrics(top1=0.4 if len(scores) == 1 else 0.3, agree=0.5, disagree=0.2)

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.eval_metrics.pack_datums_for_eval",
        pack_datums_for_eval,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.eval_metrics.evaluate_packed",
        evaluate_packed,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.train.load_candidate_style_keras",
        load_candidate_style_keras,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.mlflow_utils.MLflowTracker.require_keras_weights",
        lambda self, uri: Path("/tmp/served.keras"),
    )

    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["eval_datums"] = ["val-move"]
    context.extras["trained_model_path"] = Path("/tmp/new.keras")
    context.extras["reference"] = ModelHandle(
        key="served",
        weights_uri="s3://served",
        compatible=True,
        kind="chain",
    )
    ScoreEvaluationStep().run(MagicMock(), context)  # type: ignore[arg-type]
    assert packs == [["val-move"]]
    assert len(loads) == 2
    assert scores[0][1] is packed and scores[1][1] is packed
    assert context.extras["candidate_eval"].top1_overall == 0.4
    assert context.extras["reference_eval"].top1_overall == 0.3


def test_advance_marks_prepared_games() -> None:
    scheme = _Scheme()
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["train_game_ids"] = ["g1", "g2"]
    AdvanceTrainingCursorStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.marked == [["g1", "g2"]]


def test_apply_does_not_archive_a_different_chain() -> None:
    scheme = _Scheme()
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["promotion_decision"] = PromotionDecision(True, "promote")
    candidate = ModelHandle(key="v2", weights_uri="s3://c", compatible=True, kind="personal")
    reference = ModelHandle(key="v9", weights_uri="s3://b", compatible=True, kind="baseline")
    context.extras["candidate"] = candidate
    context.extras["reference"] = reference
    ApplyPromotionStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.applied_reference == [None]


def test_both_schemes_build_the_same_step_classes() -> None:
    baseline = build_training_scheme_steps(BaselineTrainingScheme(), promote=True)
    personal = build_training_scheme_steps(PersonalTrainingScheme("acct-1"), promote=True)
    assert [type(step) for step in baseline] == [type(step) for step in personal]
    assert [step.name for step in baseline] == [
        "PrepareTraining",
        "TrainModel",
        "PrepareEvaluation",
        "ScoreEvaluation",
        "RecordCandidate",
        "AdvanceTrainingCursor",
        "DecideFromScores",
        "ApplyPromotion",
    ]


def test_baseline_promotion_blocks_a_disagree_drop() -> None:
    decision = BaselineTrainingScheme().decide_promotion(
        candidate_eval=_metrics(top1=0.50, agree=0.8, disagree=0.19),
        reference=ModelHandle(key="v1", weights_uri="s3://x", compatible=True, kind="baseline"),
        reference_eval=_metrics(top1=0.48, agree=0.8, disagree=0.20),
    )
    assert decision.should_promote is False


def test_baseline_promotion_accepts_held_overall_and_disagree() -> None:
    decision = BaselineTrainingScheme().decide_promotion(
        candidate_eval=_metrics(top1=0.50, agree=0.7, disagree=0.21),
        reference=ModelHandle(key="v1", weights_uri="s3://x", compatible=True, kind="baseline"),
        reference_eval=_metrics(top1=0.48, agree=0.9, disagree=0.20),
    )
    assert decision.should_promote is True


def test_personal_promotion_blocks_a_large_agree_drop() -> None:
    decision = PersonalTrainingScheme("acct-1").decide_promotion(
        candidate_eval=_metrics(top1=0.40, agree=0.60, disagree=0.22),
        reference=ModelHandle(key="v1", weights_uri="s3://x", compatible=True, kind="baseline"),
        reference_eval=_metrics(top1=0.42, agree=0.70, disagree=0.20),
    )
    assert decision.should_promote is False


def test_personal_promotion_accepts_a_disagree_gain_inside_the_agree_limit() -> None:
    decision = PersonalTrainingScheme("acct-1").decide_promotion(
        candidate_eval=_metrics(top1=0.40, agree=0.68, disagree=0.22),
        reference=ModelHandle(key="v1", weights_uri="s3://x", compatible=True, kind="baseline"),
        reference_eval=_metrics(top1=0.42, agree=0.70, disagree=0.20),
    )
    assert decision.should_promote is True


def test_scheme_steps_do_not_name_either_model_chain() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "src/chess_teacher/pipelines/neural_network/scheme_steps.py"
    ).read_text()
    lowered = source.lower()
    assert "baselinemodel" not in lowered
    assert "personalmodel" not in lowered
    assert "personaltraining" not in lowered
    assert "baselinetraining" not in lowered
