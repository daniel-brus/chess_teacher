"""Parent-agnostic training steps: one step list, two schemes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.models import (
    PROCESSED_FLAG_BASELINE,
    PROCESSED_FLAG_PERSONAL,
    BaselineModel,
    BaselineModelStatus,
    PersonalModel,
)
from chess_teacher.pipelines.neural_network.scheme_steps import (
    SKIP_KEY,
    AdoptParentBaselineStep,
    AdvanceTrainingCursorStep,
    ApplyPromotionStep,
    PrepareEvaluationStep,
    PrepareTrainingStep,
    ScoreEvaluationStep,
    TrainModelStep,
    build_training_scheme_steps,
)
from chess_teacher.pipelines.neural_network.schemes import ModelTraining, resolve_training_parent
from chess_teacher.pipelines.neural_network.training_scheme import (
    ModelHandle,
    ParentBaselineDecision,
    PromotionDecision,
)
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext


def _metrics(
    *,
    top1: float,
    agree: float | None = None,
    disagree: float | None = None,
    disagree_top3: float | None = None,
) -> EvalMetrics:
    return EvalMetrics(
        top1_overall=top1,
        top3_overall=top1,
        top1_sf_agree=agree,
        top3_sf_agree=agree,
        top1_sf_disagree=disagree,
        top3_sf_disagree=disagree if disagree_top3 is None else disagree_top3,
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
    considers_parent_baseline = False

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
        self.fit_calls: list[Any] = []
        self.saved: list[Path] = []
        self.marked: list[list[str]] = []
        self.applied_current: list[ModelHandle | None] = []
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
        current: ModelHandle | None,
        eval_metrics_json: str | None,
    ) -> ModelHandle:
        del db_client, eval_metrics_json
        self.applied_current.append(current)
        return candidate


def test_prepare_skips_without_loading_when_pending_is_low() -> None:
    scheme = _Scheme()
    scheme.pending = 3
    context = PipelineContext()
    PrepareTrainingStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert context.extras[SKIP_KEY] is True
    assert scheme.checked == 1
    assert "train_datums" not in context.extras


def test_prepare_loads_parent_and_batch() -> None:
    scheme = _Scheme()
    context = PipelineContext()
    PrepareTrainingStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert context.extras[SKIP_KEY] is False
    assert context.extras["parent"] is scheme.parent
    assert "reference" not in context.extras
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
    context.extras["parent"] = ModelHandle(
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
    assert context.extras["parent_eval"].top1_overall == 0.3


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
    context.extras["parent"] = reference
    ApplyPromotionStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.applied_current == [None]


def test_both_scopes_build_the_same_step_classes() -> None:
    baseline = build_training_scheme_steps(ModelTraining(), promote=True)
    personal = build_training_scheme_steps(ModelTraining("user-1"), promote=True)
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
        "AdoptParentBaseline",
    ]


def test_promotion_blocks_a_disagree_drop() -> None:
    parent = ModelHandle(key="v1", weights_uri="s3://x", compatible=True, kind="baseline")
    decision = ModelTraining("user-1").decide_promotion(
        candidate_eval=_metrics(top1=0.50, agree=0.8, disagree=0.19),
        parent=parent,
        parent_eval=_metrics(top1=0.48, agree=0.8, disagree=0.20),
    )
    assert decision.should_promote is False


def test_promotion_accepts_held_overall_and_disagree_for_either_scope() -> None:
    parent = ModelHandle(key="v1", weights_uri="s3://x", compatible=True, kind="baseline")
    kwargs = {
        "candidate_eval": _metrics(top1=0.50, agree=0.60, disagree=0.21),
        "parent": parent,
        "parent_eval": _metrics(top1=0.48, agree=0.90, disagree=0.20),
    }
    assert ModelTraining().decide_promotion(**kwargs).should_promote is True
    assert ModelTraining("user-1").decide_promotion(**kwargs).should_promote is True


def _weights_row(
    version: str,
    uri: str | None,
    compatible: bool,
    *,
    parent_baseline_version: str | None = None,
) -> MagicMock:
    row = MagicMock()
    row.version = version
    row.model_uri = uri
    row.looks_like_candidate_style.return_value = compatible
    row.parent_baseline_version = parent_baseline_version
    return row


def _no_parent_baseline(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.BaselineModel.current_parent_baseline",
        lambda db: None,
    )


def test_parent_is_the_users_latest_promotion(monkeypatch: Any) -> None:
    _no_parent_baseline(monkeypatch)
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.PersonalModel.latest_promotion_for_user",
        lambda db, user_id: _weights_row("v3", "s3://user", True),
    )
    baseline = MagicMock()
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.BaselineModel.latest_with_status",
        baseline,
    )
    handle = resolve_training_parent(MagicMock(), "user-1")
    assert handle is not None
    assert handle.key == "v3"
    assert handle.kind == "personal"
    baseline.assert_not_called()


def test_parent_falls_back_to_latest_baseline_promotion(monkeypatch: Any) -> None:
    _no_parent_baseline(monkeypatch)
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.PersonalModel.latest_promotion_for_user",
        lambda db, user_id: _weights_row("v1", None, False),
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.BaselineModel.latest_with_status",
        lambda db, status: _weights_row("v9", "s3://base", True),
    )
    handle = resolve_training_parent(MagicMock(), "user-1")
    assert handle is not None
    assert handle.key == "v9"
    assert handle.kind == "baseline"


def test_parent_cold_starts_when_no_promotion_exists(monkeypatch: Any) -> None:
    _no_parent_baseline(monkeypatch)
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.PersonalModel.latest_promotion_for_user",
        lambda db, user_id: None,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.BaselineModel.latest_with_status",
        lambda db, status: None,
    )
    assert resolve_training_parent(MagicMock(), "user-1") is None
    assert resolve_training_parent(MagicMock(), None) is None


def test_platform_parent_ignores_user_models(monkeypatch) -> None:
    user_lookup = MagicMock()
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.PersonalModel.latest_promotion_for_user",
        user_lookup,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.BaselineModel.latest_with_status",
        lambda db, status: _weights_row("v2", "s3://base", True),
    )
    handle = resolve_training_parent(MagicMock(), None)
    user_lookup.assert_not_called()
    assert handle is not None
    assert handle.kind == "baseline"


def test_user_run_pools_every_linked_account(monkeypatch) -> None:
    user = MagicMock()
    user.get_linked_accounts.return_value = [
        MagicMock(account_id="acct-a"),
        MagicMock(account_id="acct-b"),
    ]
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.User.fetch_from_db",
        lambda db, id: user,
    )
    seen: dict[str, object] = {}

    def count(self, *, split_version: str, flag_column: str, extra_where: str | None = None) -> int:
        del self, split_version
        seen["flag"] = flag_column
        seen["where"] = extra_where
        return 4

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.TrainingDataStore.count_unprocessed_train",
        count,
    )
    assert ModelTraining("user-1").count_pending(MagicMock()) == 4
    assert seen["flag"] == PROCESSED_FLAG_PERSONAL
    where = str(seen["where"])
    assert "acct-a" in where and "acct-b" in where


def test_platform_run_uses_every_account(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def count(self, *, split_version: str, flag_column: str, extra_where: str | None = None) -> int:
        del self, split_version
        seen["flag"] = flag_column
        seen["where"] = extra_where
        return 1

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.TrainingDataStore.count_unprocessed_train",
        count,
    )
    assert ModelTraining().count_pending(MagicMock()) == 1
    assert seen["flag"] == PROCESSED_FLAG_BASELINE
    assert seen["where"] is None


def _parent_handle() -> ModelHandle:
    return ModelHandle(
        key="v1",
        weights_uri="s3://parent",
        compatible=True,
        kind="baseline",
    )


def test_parent_baseline_adopts_on_a_clear_disagree_gain() -> None:
    decision = ModelTraining().decide_parent_baseline(
        candidate_eval=_metrics(top1=0.40, agree=0.70, disagree=0.22, disagree_top3=0.55),
        parent=_parent_handle(),
        parent_eval=_metrics(top1=0.40, agree=0.70, disagree=0.20, disagree_top3=0.55),
    )
    assert decision.should_adopt is True


def test_parent_baseline_rejects_an_agree_drop() -> None:
    decision = ModelTraining().decide_parent_baseline(
        candidate_eval=_metrics(top1=0.40, agree=0.6999, disagree=0.22, disagree_top3=0.55),
        parent=_parent_handle(),
        parent_eval=_metrics(top1=0.40, agree=0.70, disagree=0.20, disagree_top3=0.55),
    )
    assert decision.should_adopt is False


def test_parent_baseline_rejects_a_small_disagree_gain() -> None:
    decision = ModelTraining().decide_parent_baseline(
        candidate_eval=_metrics(top1=0.40, agree=0.70, disagree=0.219, disagree_top3=0.55),
        parent=_parent_handle(),
        parent_eval=_metrics(top1=0.40, agree=0.70, disagree=0.20, disagree_top3=0.55),
    )
    assert decision.should_adopt is False


def test_parent_baseline_rejects_a_disagree_top3_drop() -> None:
    decision = ModelTraining().decide_parent_baseline(
        candidate_eval=_metrics(top1=0.40, agree=0.70, disagree=0.22, disagree_top3=0.549),
        parent=_parent_handle(),
        parent_eval=_metrics(top1=0.40, agree=0.70, disagree=0.20, disagree_top3=0.55),
    )
    assert decision.should_adopt is False


def test_parent_baseline_rejects_an_overall_drop() -> None:
    decision = ModelTraining().decide_parent_baseline(
        candidate_eval=_metrics(top1=0.399, agree=0.70, disagree=0.22, disagree_top3=0.55),
        parent=_parent_handle(),
        parent_eval=_metrics(top1=0.40, agree=0.70, disagree=0.20, disagree_top3=0.55),
    )
    assert decision.should_adopt is False


def test_promotion_can_pass_when_the_parent_baseline_gate_fails() -> None:
    parent = _parent_handle()
    candidate_eval = _metrics(top1=0.50, agree=0.60, disagree=0.22, disagree_top3=0.55)
    parent_eval = _metrics(top1=0.48, agree=0.90, disagree=0.20, disagree_top3=0.55)
    kwargs = {
        "candidate_eval": candidate_eval,
        "parent": parent,
        "parent_eval": parent_eval,
    }
    assert ModelTraining().decide_promotion(**kwargs).should_promote is True
    assert ModelTraining().decide_parent_baseline(**kwargs).should_adopt is False


def test_first_promoted_model_becomes_the_parent_baseline() -> None:
    decision = ModelTraining().decide_parent_baseline(
        candidate_eval=None,
        parent=None,
        parent_eval=None,
    )
    assert decision.should_adopt is True


def test_personal_run_does_not_adopt_a_parent_baseline() -> None:
    decision = ModelTraining("user-1").decide_parent_baseline(
        candidate_eval=_metrics(top1=0.9, agree=0.9, disagree=0.9, disagree_top3=0.9),
        parent=None,
        parent_eval=None,
    )
    assert decision.should_adopt is False


def test_personal_run_keeps_a_model_trained_on_the_current_parent(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.BaselineModel.current_parent_baseline",
        lambda db: _weights_row("v2", "s3://base", True),
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.PersonalModel.latest_promotion_for_user",
        lambda db, user_id: _weights_row(
            "v5",
            "s3://user",
            True,
            parent_baseline_version="v2",
        ),
    )
    latest = MagicMock()
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.BaselineModel.latest_with_status",
        latest,
    )
    handle = resolve_training_parent(MagicMock(), "user-1")
    assert handle is not None
    assert handle.key == "v5"
    assert handle.kind == "personal"
    latest.assert_not_called()


def test_personal_run_resets_when_its_model_descends_from_an_older_parent(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.BaselineModel.current_parent_baseline",
        lambda db: _weights_row("v2", "s3://base", True),
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.PersonalModel.latest_promotion_for_user",
        lambda db, user_id: _weights_row(
            "v5",
            "s3://user",
            True,
            parent_baseline_version="v1",
        ),
    )
    handle = resolve_training_parent(MagicMock(), "user-1")
    assert handle is not None
    assert handle.key == "v2"
    assert handle.kind == "baseline"
    assert handle.parent_baseline_version == "v2"


def test_platform_run_ignores_the_parent_baseline_pointer(monkeypatch: Any) -> None:
    pointer = MagicMock()
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.BaselineModel.current_parent_baseline",
        pointer,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.BaselineModel.latest_with_status",
        lambda db, status: _weights_row("v8", "s3://prod", True),
    )
    handle = resolve_training_parent(MagicMock(), None)
    pointer.assert_not_called()
    assert handle is not None
    assert handle.key == "v8"


def test_personal_candidate_records_parent_baseline_lineage(monkeypatch: Any) -> None:
    saved: list[PersonalModel] = []

    def save_new(self: PersonalModel, db: object) -> bool:
        del db
        saved.append(self)
        return True

    monkeypatch.setattr(PersonalModel, "save_new_to_db", save_new)
    monkeypatch.setattr(BaselineModel, "current_git_commit", lambda: None)
    parent = ModelHandle(
        key="v3",
        weights_uri="s3://personal",
        compatible=True,
        kind="personal",
        parent_baseline_version="v1",
    )
    ModelTraining("user-1").insert_candidate(
        MagicMock(),
        version="v4",
        model_uri="s3://new",
        mlflow_run_id=None,
        metrics_json="{}",
        parent=parent,
    )
    assert saved[0].parent_baseline_version == "v1"
    assert saved[0].parent_version == "v3"
    assert saved[0].parent_kind == "personal"


def test_adopt_clears_the_previous_pointer(monkeypatch: Any) -> None:
    previous = BaselineModel(
        id="id-v1",
        version="v1",
        trained_at=datetime.now(UTC),
        model_uri="s3://old",
        status=BaselineModelStatus.ARCHIVED,
        is_parent_baseline=True,
    )
    candidate = BaselineModel(
        id="id-v2",
        version="v2",
        trained_at=datetime.now(UTC),
        model_uri="s3://new",
        status=BaselineModelStatus.PRODUCTION,
        is_parent_baseline=False,
    )
    saved: list[BaselineModel] = []

    def fetch(db: object, **kwargs: object) -> list[BaselineModel]:
        del db, kwargs
        return [previous]

    def save(self: BaselineModel, db: object, **kwargs: object) -> MagicMock:
        del db, kwargs
        saved.append(self)
        return MagicMock()

    monkeypatch.setattr(BaselineModel, "fetch_all_from_db", fetch)
    monkeypatch.setattr(BaselineModel, "save_to_db", save)
    adopted = candidate.adopt_as_parent_baseline(MagicMock())
    assert [row.version for row in saved] == ["v1", "v2"]
    assert saved[0].is_parent_baseline is False
    assert saved[0].status == BaselineModelStatus.ARCHIVED
    assert adopted.is_parent_baseline is True
    assert adopted.status == BaselineModelStatus.PRODUCTION


class _AdoptingScheme(_Scheme):
    considers_parent_baseline = True

    def __init__(self) -> None:
        super().__init__()
        self.pointer: ModelHandle | None = None
        self.parent_decision = ParentBaselineDecision(True, "clear")
        self.adopted: list[ModelHandle] = []
        self.seen_parent_eval: object = "unset"

    def current_parent_baseline(self, db_client: object) -> ModelHandle | None:
        del db_client
        return self.pointer

    def decide_parent_baseline(self, **kwargs: object) -> ParentBaselineDecision:
        self.seen_parent_eval = kwargs.get("parent_eval")
        return self.parent_decision

    def apply_parent_baseline(
        self,
        db_client: object,
        *,
        candidate: ModelHandle,
    ) -> ModelHandle:
        del db_client
        self.adopted.append(candidate)
        return candidate


def _promoted(key: str = "v2") -> ModelHandle:
    return ModelHandle(key=key, weights_uri="s3://cand", compatible=True, kind="baseline")


def test_adopt_step_skips_a_personal_run() -> None:
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["promoted"] = _promoted()
    AdoptParentBaselineStep(_Scheme()).run(MagicMock(), context)  # type: ignore[arg-type]
    assert "parent_baseline_decision" not in context.extras
    assert "parent_baseline" not in context.extras


def test_adopt_step_requires_promotion() -> None:
    scheme = _AdoptingScheme()
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    AdoptParentBaselineStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert context.extras["parent_baseline_decision"].should_adopt is False
    assert scheme.adopted == []


def test_adopt_step_reuses_the_training_parent_score() -> None:
    scheme = _AdoptingScheme()
    pointer = _parent_handle()
    scheme.pointer = pointer
    held = _metrics(top1=0.4, agree=0.5, disagree=0.2)
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["promoted"] = _promoted()
    context.extras["parent"] = pointer
    context.extras["parent_eval"] = held
    context.extras["candidate_eval"] = held
    AdoptParentBaselineStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.seen_parent_eval is held
    assert scheme.adopted[0].key == "v2"
    assert context.extras["parent_baseline"].key == "v2"


def test_adopt_step_scores_a_different_parent_baseline(monkeypatch: Any) -> None:
    scheme = _AdoptingScheme()
    scheme.pointer = ModelHandle(
        key="v1",
        weights_uri="s3://old",
        compatible=True,
        kind="baseline",
    )
    scored = _metrics(top1=0.3, agree=0.4, disagree=0.1)
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.eval_metrics.pack_datums_for_eval",
        lambda datums: datums,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.eval_metrics.evaluate_packed",
        lambda model, packed: scored,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.train.load_candidate_style_keras",
        lambda path, compile_model=False: path,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.mlflow_utils.MLflowTracker.require_keras_weights",
        lambda self, uri: Path("/tmp/old.keras"),
    )
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["promoted"] = _promoted("v3")
    context.extras["parent"] = ModelHandle(
        key="v2",
        weights_uri="s3://prod",
        compatible=True,
        kind="baseline",
    )
    context.extras["parent_eval"] = _metrics(top1=0.5, agree=0.5, disagree=0.5)
    context.extras["eval_datums"] = ["move"]
    context.extras["candidate_eval"] = _metrics(top1=0.5, agree=0.5, disagree=0.5)
    AdoptParentBaselineStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.seen_parent_eval is scored
    assert context.extras["parent_baseline"].key == "v3"


def test_adopt_step_leaves_the_pointer_when_the_gate_fails() -> None:
    scheme = _AdoptingScheme()
    scheme.parent_decision = ParentBaselineDecision(False, "agree dropped")
    scheme.pointer = _parent_handle()
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["promoted"] = _promoted()
    context.extras["parent"] = scheme.pointer
    context.extras["parent_eval"] = _metrics(top1=0.4, agree=0.5, disagree=0.2)
    AdoptParentBaselineStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.adopted == []
    assert "parent_baseline" not in context.extras


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
