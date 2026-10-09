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
    TrainingState,
)
from chess_teacher.pipelines.neural_network.scheme_steps import (
    SKIP_KEY,
    AdoptParentBaselineStep,
    AdvanceTrainingCursorStep,
    ApplyPromotionStep,
    PrepareEvaluationStep,
    PrepareTrainingStep,
    ScoreEvaluationStep,
    ScoreParentBaselineStep,
    TrainModelStep,
    build_training_scheme_steps,
)
from chess_teacher.pipelines.neural_network.schemes import ModelTraining, resolve_training_parent
from chess_teacher.pipelines.neural_network.split_registry import SplitRegistry
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
        self.rejected: list[list[str]] = []
        self.aligned = 0
        self.applied_current: list[ModelHandle | None] = []
        self.decision = PromotionDecision(True, "ok")

    def align_personal_queue(self, db_client: object) -> None:
        del db_client
        self.aligned += 1

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

    def note_rejected(self, db_client: object, game_ids: list[str]) -> None:
        del db_client
        self.rejected.append(game_ids)

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


def test_prepare_does_not_load_when_the_round_was_already_skipped() -> None:
    scheme = _Scheme()
    context = PipelineContext()
    context.extras[SKIP_KEY] = True
    PrepareTrainingStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.checked == 0
    assert "train_datums" not in context.extras


def test_prepare_skips_without_loading_when_pending_is_low() -> None:
    scheme = _Scheme()
    scheme.pending = 3
    context = PipelineContext()
    PrepareTrainingStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert context.extras[SKIP_KEY] is True
    assert scheme.checked == 1
    assert scheme.aligned == 1
    assert "train_datums" not in context.extras


def test_prepare_loads_parent_and_batch() -> None:
    scheme = _Scheme()
    context = PipelineContext()
    PrepareTrainingStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert context.extras[SKIP_KEY] is False
    assert context.extras["parent"] is scheme.parent
    assert "reference" not in context.extras
    assert context.extras["train_game_ids"] == ["g1"]
    assert scheme.aligned == 1


class _RecordingProcess:
    """Stand-in for the child. Records one call and returns a fixed result."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call(self, op: str, **payload: object) -> object:
        self.calls.append((op, payload))
        return self.result


def test_train_step_fits_and_drops_the_batch() -> None:
    context = PipelineContext()
    context.extras["train_datums"] = ["move"]
    context.extras[SKIP_KEY] = False
    context.extras["parent"] = ModelHandle(
        key="p", weights_uri=None, compatible=False, kind="chain"
    )
    process = _RecordingProcess({
        "model_path": "/tmp/scheme_model_x/model.keras",
        "metrics": {"n_samples": 1.0},
    })
    TrainModelStep(process=process).run(MagicMock(), context)  # type: ignore[arg-type]
    assert process.calls[0][0] == "fit"
    assert process.calls[0][1]["datums"] == ["move"]
    assert process.calls[0][1]["weights_path"] is None
    assert "train_datums" not in context.extras
    assert context.extras["trained_model_path"].name == "model.keras"


def test_train_step_skips_when_prepare_skipped() -> None:
    context = PipelineContext()
    context.extras[SKIP_KEY] = True
    TrainModelStep().run(MagicMock(), context)  # type: ignore[arg-type]


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


def _stub_mlflow_tracker(monkeypatch: Any) -> None:
    """Avoid the tracker's Postgres tracking-URI lookup."""

    def init(
        self: object,
        *,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        del tracking_uri, experiment_name
        return None

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.mlflow_utils.MLflowTracker.__init__",
        init,
    )


def test_score_sends_candidate_and_parent_to_the_process(monkeypatch: Any) -> None:
    _stub_mlflow_tracker(monkeypatch)
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.mlflow_utils.MLflowTracker.require_keras_weights",
        lambda self, uri: Path("/tmp/served.keras"),
    )
    process = _RecordingProcess({
        "candidate_eval": _metrics(top1=0.4, agree=0.5, disagree=0.2),
        "parent_eval": _metrics(top1=0.3, agree=0.5, disagree=0.2),
    })

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
    ScoreEvaluationStep(process=process).run(MagicMock(), context)  # type: ignore[arg-type]
    op, payload = process.calls[0]
    assert op == "score"
    assert payload["datums"] == ["val-move"]
    assert payload["model_path"] == Path("/tmp/new.keras")
    assert payload["parent_weights_path"] == Path("/tmp/served.keras")
    assert context.extras["candidate_eval"].top1_overall == 0.4
    assert context.extras["parent_eval"].top1_overall == 0.3


def test_advance_marks_prepared_games() -> None:
    scheme = _Scheme()
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["train_game_ids"] = ["g1", "g2"]
    AdvanceTrainingCursorStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.marked == [["g1", "g2"]]
    assert scheme.rejected == []


def test_advance_marks_a_promoted_batch() -> None:
    scheme = _Scheme()
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["train_game_ids"] = ["g1", "g2"]
    context.extras["promotion_decision"] = PromotionDecision(True, "promote")
    AdvanceTrainingCursorStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.marked == [["g1", "g2"]]
    assert scheme.rejected == []


def test_advance_moves_a_rejected_batch_down_the_queue() -> None:
    scheme = _Scheme()
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["train_game_ids"] = ["g1", "g2"]
    context.extras["promotion_decision"] = PromotionDecision(False, "miss")
    AdvanceTrainingCursorStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.marked == []
    assert scheme.rejected == [["g1", "g2"]]


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
    shared = [
        "PrepareTraining",
        "WaitForTrainingSlot",
        "TrainModel",
        "PrepareEvaluation",
        "ScoreEvaluation",
        "ReleaseTrainingSlot",
        "RecordCandidate",
        "DecideFromScores",
        "ApplyPromotion",
        "AdvanceTrainingCursor",
        "AdoptParentBaseline",
    ]
    assert [step.name for step in personal] == shared
    assert [step.name for step in baseline] == [
        *shared[:5],
        "ScoreParentBaseline",
        *shared[5:],
    ]
    assert baseline[2].process is baseline[4].process
    assert baseline[2].process is baseline[5].process
    assert baseline[2].process is not None
    assert baseline[3].process is None
    assert personal[2].process is personal[4].process
    assert personal[5].name == "ReleaseTrainingSlot"
    assert personal[5].process is None
    assert personal[2].process is not baseline[2].process


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

    def count(
        self,
        *,
        split_version: str,
        flag_column: str,
        extra_where: str | None = None,
        max_attempts: int | None = None,
    ) -> int:
        del self, split_version, max_attempts
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

    def count(
        self,
        *,
        split_version: str,
        flag_column: str,
        extra_where: str | None = None,
        max_attempts: int | None = None,
    ) -> int:
        del self, split_version, max_attempts
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


def test_pending_stays_on_never_tried_games_while_that_front_is_large(monkeypatch) -> None:
    calls: list[int | None] = []

    def count(self, *, max_attempts: int | None = None, **kwargs: object) -> int:
        del self, kwargs
        calls.append(max_attempts)
        return 5000

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.TrainingDataStore.count_unprocessed_train",
        count,
    )
    assert ModelTraining().count_pending(MagicMock()) == 5000
    assert calls == [0]


def test_pending_includes_the_back_after_the_front_is_short(monkeypatch) -> None:
    calls: list[int | None] = []

    def count(self, *, max_attempts: int | None = None, **kwargs: object) -> int:
        del self, kwargs
        calls.append(max_attempts)
        return 100 if max_attempts == 0 else 1400

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.TrainingDataStore.count_unprocessed_train",
        count,
    )
    assert ModelTraining().count_pending(MagicMock()) == 1400
    assert calls == [0, None]


def test_batch_keeps_one_miss_behind_a_full_front(monkeypatch) -> None:
    fetched: dict[str, object] = {}

    def count(self, *, max_attempts: int | None = None, **kwargs: object) -> int:
        del self, kwargs, max_attempts
        return 5000

    def fetch(self, **kwargs: object) -> tuple[list[object], list[str]]:
        del self
        fetched.update(kwargs)
        return [object()], ["g-new"]

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.TrainingDataStore.count_unprocessed_train",
        count,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.TrainingDataStore.fetch_unprocessed_train_batch",
        fetch,
    )
    datums, game_ids = ModelTraining().load_train_batch(MagicMock())
    assert game_ids == ["g-new"]
    assert len(datums) == 1
    assert fetched["max_attempts"] == 0


def test_batch_reaches_the_back_when_the_front_is_short(monkeypatch) -> None:
    fetched: dict[str, object] = {}

    def count(self, *, max_attempts: int | None = None, **kwargs: object) -> int:
        del self, kwargs, max_attempts
        return 10

    def fetch(self, **kwargs: object) -> tuple[list[object], list[str]]:
        del self
        fetched.update(kwargs)
        return [object()], ["g-miss"]

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.TrainingDataStore.count_unprocessed_train",
        count,
    )
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.TrainingDataStore.fetch_unprocessed_train_batch",
        fetch,
    )
    ModelTraining().load_train_batch(MagicMock())
    assert fetched["max_attempts"] is None


def _parent_row(version: str) -> MagicMock:
    row = MagicMock()
    row.version = version
    return row


def _install_align_stubs(
    monkeypatch: Any,
    *,
    parent_version: str | None,
    stored: str | None,
    served_parent: str | None = None,
) -> dict[str, Any]:
    """Patch the lookups align_personal_queue uses. No served model when ``served_parent`` is omitted."""
    seen: dict[str, Any] = {"saved": [], "cleared": None}
    if parent_version is None:
        parent = None
    else:
        parent = _parent_row(parent_version)
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.BaselineModel.current_parent_baseline",
        lambda db: parent,
    )
    if served_parent is None:
        served = None
    else:
        served = MagicMock()
        served.parent_baseline_version = served_parent
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.PersonalModel.latest_promotion_for_user",
        lambda db, user_id: served,
    )
    state = TrainingState(scope="user:user-1", personal_queue_baseline=stored)

    def for_scope(db: object, scope: str) -> TrainingState:
        del db
        assert scope == "user:user-1"
        return state

    def save(self: TrainingState, db: object) -> None:
        del db
        seen["saved"].append(self)

    def clear(self: SplitRegistry, account_ids: list[str]) -> int:
        del self
        seen["cleared"] = list(account_ids)
        return 3

    monkeypatch.setattr(TrainingState, "for_scope", for_scope)
    monkeypatch.setattr(TrainingState, "save_to_db", save)
    monkeypatch.setattr(SplitRegistry, "clear_personal_queue_for_accounts", clear)
    user = MagicMock()
    user.get_linked_accounts.return_value = [
        MagicMock(account_id="acct-b"),
        MagicMock(account_id="acct-a"),
    ]
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.User.fetch_from_db",
        lambda db, id: user,
    )
    return seen


def test_align_is_a_noop_for_the_platform_queue(monkeypatch: Any) -> None:
    lookup = MagicMock()
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.BaselineModel.current_parent_baseline",
        lookup,
    )
    ModelTraining().align_personal_queue(MagicMock())
    lookup.assert_not_called()


def test_align_waits_when_there_is_no_parent_baseline(monkeypatch: Any) -> None:
    seen = _install_align_stubs(monkeypatch, parent_version=None, stored=None)
    ModelTraining("user-1").align_personal_queue(MagicMock())
    assert seen["saved"] == []
    assert seen["cleared"] is None


def test_align_leaves_a_queue_already_on_this_baseline(monkeypatch: Any) -> None:
    seen = _install_align_stubs(monkeypatch, parent_version="v2", stored="v2")
    ModelTraining("user-1").align_personal_queue(MagicMock())
    assert seen["saved"] == []
    assert seen["cleared"] is None


def test_align_records_without_clearing_when_the_served_model_already_matches(
    monkeypatch: Any,
) -> None:
    seen = _install_align_stubs(
        monkeypatch,
        parent_version="v1",
        stored=None,
        served_parent="v1",
    )
    ModelTraining("user-1").align_personal_queue(MagicMock())
    assert seen["cleared"] is None
    assert [row.personal_queue_baseline for row in seen["saved"]] == ["v1"]


def test_align_replays_once_when_the_parent_baseline_moves(monkeypatch: Any) -> None:
    seen = _install_align_stubs(
        monkeypatch,
        parent_version="v2",
        stored="v1",
        served_parent="v1",
    )
    ModelTraining("user-1").align_personal_queue(MagicMock())
    assert seen["cleared"] == ["acct-b", "acct-a"]
    assert [row.personal_queue_baseline for row in seen["saved"]] == ["v2"]


def test_align_clears_a_queue_that_was_never_aligned(monkeypatch: Any) -> None:
    seen = _install_align_stubs(monkeypatch, parent_version="v1", stored=None, served_parent=None)
    ModelTraining("user-1").align_personal_queue(MagicMock())
    assert seen["cleared"] == ["acct-b", "acct-a"]
    assert [row.personal_queue_baseline for row in seen["saved"]] == ["v1"]


def test_scheme_eval_loads_capped_registry_val(monkeypatch: Any) -> None:
    from chess_teacher.pipelines.neural_network.pipeline_steps import (
        MAX_MOVES_PER_REGISTRY_VAL_EVAL,
    )

    seen: dict[str, object] = {}

    def fake_load(
        db_client: object,
        *,
        split_version: str,
        limit: int | None = None,
        full: bool = False,
        extra_where: str | None = None,
        assign_if_missing: bool = True,
    ) -> list[object]:
        del db_client, assign_if_missing
        seen["split_version"] = split_version
        seen["limit"] = limit
        seen["full"] = full
        seen["extra_where"] = extra_where
        return ["datum"]

    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.schemes.load_registry_val_datums",
        fake_load,
    )
    out = ModelTraining().load_eval_datums(MagicMock())
    assert out == ["datum"]
    assert seen["full"] is False
    assert seen["limit"] == MAX_MOVES_PER_REGISTRY_VAL_EVAL == 10_000
    assert seen["extra_where"] is None


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
    context.extras["parent_baseline_eval"] = held
    context.extras["candidate_eval"] = held
    AdoptParentBaselineStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.seen_parent_eval is held
    assert scheme.adopted[0].key == "v2"
    assert context.extras["parent_baseline"].key == "v2"


def test_score_parent_baseline_uses_the_child_for_a_different_model(monkeypatch: Any) -> None:
    scheme = _AdoptingScheme()
    scheme.pointer = ModelHandle(
        key="v1",
        weights_uri="s3://old",
        compatible=True,
        kind="baseline",
    )
    scored = _metrics(top1=0.3, agree=0.4, disagree=0.1)
    process = _RecordingProcess(scored)
    _stub_mlflow_tracker(monkeypatch)
    monkeypatch.setattr(
        "chess_teacher.pipelines.neural_network.mlflow_utils.MLflowTracker.require_keras_weights",
        lambda self, uri: Path("/tmp/old.keras"),
    )
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["parent"] = ModelHandle(
        key="v2",
        weights_uri="s3://prod",
        compatible=True,
        kind="baseline",
    )
    context.extras["parent_eval"] = _metrics(top1=0.5, agree=0.5, disagree=0.5)
    context.extras["eval_datums"] = ["move"]
    ScoreParentBaselineStep(scheme, process=process).run(MagicMock(), context)  # type: ignore[arg-type]
    op, payload = process.calls[0]
    assert op == "score_weights"
    assert payload["weights_path"] == Path("/tmp/old.keras")
    assert payload["datums"] == ["move"]
    assert context.extras["parent_baseline_eval"] is scored


def test_score_parent_baseline_reuses_the_training_parent() -> None:
    scheme = _AdoptingScheme()
    pointer = _parent_handle()
    scheme.pointer = pointer
    held = _metrics(top1=0.4, agree=0.5, disagree=0.2)
    process = _RecordingProcess(held)
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["parent"] = pointer
    context.extras["parent_eval"] = held
    context.extras["eval_datums"] = ["move"]
    ScoreParentBaselineStep(scheme, process=process).run(MagicMock(), context)  # type: ignore[arg-type]
    assert process.calls == []
    assert context.extras["parent_baseline_eval"] is held


def test_adopt_step_uses_the_score_from_the_child() -> None:
    scheme = _AdoptingScheme()
    scheme.pointer = ModelHandle(
        key="v1",
        weights_uri="s3://old",
        compatible=True,
        kind="baseline",
    )
    scored = _metrics(top1=0.3, agree=0.4, disagree=0.1)
    context = PipelineContext()
    context.extras[SKIP_KEY] = False
    context.extras["promoted"] = _promoted("v3")
    context.extras["parent_baseline_eval"] = scored
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
    context.extras["parent_baseline_eval"] = _metrics(top1=0.4, agree=0.5, disagree=0.2)
    AdoptParentBaselineStep(scheme).run(MagicMock(), context)  # type: ignore[arg-type]
    assert scheme.adopted == []
    assert "parent_baseline" not in context.extras


def test_user_promotion_archives_production_when_the_parent_is_a_baseline(
    monkeypatch: Any,
) -> None:
    trained_at = datetime.now(UTC)
    existing = PersonalModel(
        id="old",
        user_id="user-1",
        version="v1",
        trained_at=trained_at,
        status=BaselineModelStatus.PRODUCTION,
    )
    candidate = PersonalModel(
        id="new",
        user_id="user-1",
        version="v2",
        trained_at=trained_at,
        status=BaselineModelStatus.CANDIDATE,
    )
    seen: dict[str, PersonalModel | None] = {}

    def promote_over(
        self: PersonalModel,
        db: object,
        *,
        current_production: PersonalModel | None,
        eval_metrics: str | None,
    ) -> PersonalModel:
        del self, db, eval_metrics
        seen["current"] = current_production
        return candidate

    monkeypatch.setattr(PersonalModel, "promote_over", promote_over)
    monkeypatch.setattr(
        PersonalModel,
        "latest_promotion_for_user",
        lambda db, user_id: existing,
    )
    ModelTraining("user-1").apply_promotion(
        MagicMock(),
        candidate=ModelHandle(
            key="v2",
            weights_uri="s3://cand",
            compatible=True,
            kind="personal",
            payload=candidate,
        ),
        current=None,
        eval_metrics_json=None,
    )
    assert seen["current"] is existing


def test_platform_promotion_archives_production_when_the_parent_handle_is_missing(
    monkeypatch: Any,
) -> None:
    trained_at = datetime.now(UTC)
    existing = BaselineModel(
        id="old",
        version="v1",
        trained_at=trained_at,
        status=BaselineModelStatus.PRODUCTION,
    )
    candidate = BaselineModel(
        id="new",
        version="v2",
        trained_at=trained_at,
        status=BaselineModelStatus.CANDIDATE,
    )
    seen: dict[str, BaselineModel | None] = {}

    def promote_over(
        self: BaselineModel,
        db: object,
        *,
        current_production: BaselineModel | None,
        eval_metrics: str | None,
    ) -> BaselineModel:
        del self, db, eval_metrics
        seen["current"] = current_production
        return candidate

    monkeypatch.setattr(BaselineModel, "promote_over", promote_over)
    monkeypatch.setattr(
        BaselineModel,
        "latest_with_status",
        lambda db, status: existing,
    )
    ModelTraining().apply_promotion(
        MagicMock(),
        candidate=ModelHandle(
            key="v2",
            weights_uri="s3://cand",
            compatible=True,
            kind="baseline",
            payload=candidate,
        ),
        current=None,
        eval_metrics_json=None,
    )
    assert seen["current"] is existing


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
