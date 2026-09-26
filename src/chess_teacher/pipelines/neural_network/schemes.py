"""One training run, scoped by user id.

``user_id`` none trains the platform model on every account. A user id pools
that user's linked accounts into one model. The parent for the round is that
user's latest promotion, then the latest platform promotion, then a cold start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingDataStore,
    TrainingDatum,
    account_id_in_sql,
)
from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.models import (
    BASELINE_TRAINING_SCOPE,
    PROCESSED_FLAG_BASELINE,
    PROCESSED_FLAG_PERSONAL,
    BaselineModel,
    BaselineModelStatus,
    PersonalModel,
    TrainingState,
)
from chess_teacher.pipelines.neural_network.offline_eval import load_registry_val_datums
from chess_teacher.pipelines.neural_network.pipeline_steps import (
    MAX_MOVES_PER_BASELINE_BATCH,
    MIN_NEW_MOVES_BASELINE,
)
from chess_teacher.pipelines.neural_network.split_registry import SplitRegistry
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT
from chess_teacher.pipelines.neural_network.training_scheme import ModelHandle, PromotionDecision
from chess_teacher.platform.user import User
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.general_utils import get_current_datetime

# Users have less new data than the platform pool.
MIN_USER_TRAIN_MOVES = 300

KIND_BASELINE = "baseline"
KIND_PERSONAL = "personal"


def _touch_scope(db_client: DatabaseClient, scope: str) -> None:
    state = TrainingState.for_scope(db_client, scope)
    state.with_check_at().save_to_db(db_client)


def _baseline_handle(row: BaselineModel | None) -> ModelHandle | None:
    if row is None:
        return None
    return ModelHandle(
        key=row.version,
        weights_uri=row.model_uri,
        compatible=bool(row.model_uri) and row.looks_like_candidate_style(),
        kind=KIND_BASELINE,
        payload=row,
    )


def _personal_handle(row: PersonalModel | None) -> ModelHandle | None:
    if row is None:
        return None
    return ModelHandle(
        key=row.version,
        weights_uri=row.model_uri,
        compatible=bool(row.model_uri) and row.looks_like_candidate_style(),
        kind=KIND_PERSONAL,
        payload=row,
    )


def _usable(handle: ModelHandle | None) -> ModelHandle | None:
    if handle is not None and handle.compatible and handle.weights_uri:
        return handle
    return None


def resolve_training_parent(
    db_client: DatabaseClient,
    user_id: str | None,
) -> ModelHandle | None:
    """User's latest promotion, else the latest platform promotion, else cold start."""
    if user_id:
        user_parent = _usable(
            _personal_handle(PersonalModel.latest_promotion_for_user(db_client, user_id))
        )
        if user_parent is not None:
            return user_parent
    return _usable(
        _baseline_handle(
            BaselineModel.latest_with_status(db_client, BaselineModelStatus.PRODUCTION)
        )
    )


def _promotion_decision(
    *,
    candidate_eval: EvalMetrics | None,
    parent: ModelHandle | None,
    parent_eval: EvalMetrics | None,
) -> PromotionDecision:
    """Same gate for platform and user runs."""
    if parent is None:
        return PromotionDecision(True, "Cold start: no promoted model to replace.")
    if not parent.compatible:
        return PromotionDecision(
            True,
            f"Parent {parent.key} is incompatible; promoting the new candidate.",
        )
    if candidate_eval is None or parent_eval is None:
        return PromotionDecision(False, "Missing registry-val scores; not promoting.")
    if candidate_eval.top1_overall < parent_eval.top1_overall:
        return PromotionDecision(
            False,
            (f"overall top1 {candidate_eval.top1_overall:.4f} < {parent_eval.top1_overall:.4f}"),
        )
    drop = candidate_eval.top1_sf_disagree
    served = parent_eval.top1_sf_disagree
    if drop is not None and served is not None and drop < served:
        return PromotionDecision(
            False,
            f"disagree top1 {drop:.4f} < parent {served:.4f}",
        )
    return PromotionDecision(True, "overall top1 held and disagree top1 did not drop.")


class ModelTraining:
    """Platform run when ``user_id`` is none. Otherwise one model for that user."""

    split_version = DEFAULT_SPLIT_SALT

    def __init__(
        self,
        user_id: str | None = None,
        *,
        split_version: str = DEFAULT_SPLIT_SALT,
    ) -> None:
        self.user_id = (user_id or "").strip() or None
        self.split_version = split_version
        if self.user_id is None:
            self.pipeline_name = "baseline_training"
            self.min_new_moves = MIN_NEW_MOVES_BASELINE
        else:
            self.pipeline_name = "personal_training"
            self.min_new_moves = MIN_USER_TRAIN_MOVES

    @property
    def _scope(self) -> str:
        if self.user_id is None:
            return BASELINE_TRAINING_SCOPE
        return f"user:{self.user_id}"

    @property
    def _flag(self) -> str:
        if self.user_id is None:
            return PROCESSED_FLAG_BASELINE
        return PROCESSED_FLAG_PERSONAL

    def _account_filter(self, db_client: DatabaseClient) -> str | None:
        """None for the platform pool. Otherwise every account linked to the user."""
        if self.user_id is None:
            return None
        user = User.fetch_from_db(db_client, id=self.user_id)
        account_ids = [account.account_id for account in user.get_linked_accounts(db_client)]
        return account_id_in_sql(account_ids)

    def note_checked(self, db_client: DatabaseClient) -> None:
        _touch_scope(db_client, self._scope)

    def count_pending(self, db_client: DatabaseClient) -> int:
        return TrainingDataStore(db_client).count_unprocessed_train(
            split_version=self.split_version,
            flag_column=self._flag,
            extra_where=self._account_filter(db_client),
        )

    def resolve_parent(self, db_client: DatabaseClient) -> ModelHandle | None:
        return resolve_training_parent(db_client, self.user_id)

    def load_train_batch(self, db_client: DatabaseClient) -> tuple[list[TrainingDatum], list[str]]:
        return TrainingDataStore(db_client).fetch_unprocessed_train_batch(
            split_version=self.split_version,
            limit=MAX_MOVES_PER_BASELINE_BATCH,
            flag_column=self._flag,
            extra_where=self._account_filter(db_client),
        )

    def load_eval_datums(self, db_client: DatabaseClient) -> list[TrainingDatum]:
        return load_registry_val_datums(
            db_client,
            split_version=self.split_version,
            full=True,
            extra_where=self._account_filter(db_client),
        )

    def fit(
        self,
        datums: list[TrainingDatum],
        *,
        weights_path: Path | None,
    ) -> tuple[Any, dict[str, float]]:
        from chess_teacher.pipelines.neural_network.train import BaselineTrainer

        return BaselineTrainer().fit(datums, weights_path=weights_path)

    def save(self, model: Any, path: Path) -> None:
        from chess_teacher.pipelines.neural_network.train import BaselineTrainer

        BaselineTrainer.save(model, path)

    def next_version(self, db_client: DatabaseClient) -> str:
        if self.user_id is None:
            return BaselineModel.next_version(db_client)
        return PersonalModel.next_version(db_client, self.user_id)

    def insert_candidate(
        self,
        db_client: DatabaseClient,
        *,
        version: str,
        model_uri: str,
        mlflow_run_id: str | None,
        metrics_json: str,
        parent: ModelHandle | None,
    ) -> ModelHandle:
        trained_at = get_current_datetime()
        git_commit = BaselineModel.current_git_commit()
        if self.user_id is None:
            row = BaselineModel(
                id=BaselineModel.generate_id({"version": version}),
                version=version,
                trained_at=trained_at,
                mlflow_run_id=mlflow_run_id,
                model_uri=model_uri,
                status=BaselineModelStatus.CANDIDATE,
                parent_version=parent.key if parent else None,
                eval_metrics=metrics_json,
                git_commit_hash=git_commit,
            )
            row.save_new_to_db(db_client)
            handle = _baseline_handle(row)
        else:
            personal = PersonalModel(
                id=PersonalModel.generate_id({"user_id": self.user_id, "version": version}),
                user_id=self.user_id,
                version=version,
                trained_at=trained_at,
                mlflow_run_id=mlflow_run_id,
                model_uri=model_uri,
                status=BaselineModelStatus.CANDIDATE,
                parent_version=parent.key if parent else None,
                parent_kind=parent.kind if parent else None,
                eval_metrics=metrics_json,
                git_commit_hash=git_commit,
            )
            personal.save_new_to_db(db_client)
            handle = _personal_handle(personal)
        if handle is None:
            raise RuntimeError("insert_candidate failed to build a model handle")
        return handle

    def mark_trained(self, db_client: DatabaseClient, game_ids: list[str]) -> None:
        SplitRegistry(db_client, split_version=self.split_version).mark_processed(
            game_ids,
            flag_column=self._flag,
        )
        _touch_scope(db_client, self._scope)

    def decide_promotion(
        self,
        *,
        candidate_eval: EvalMetrics | None,
        parent: ModelHandle | None,
        parent_eval: EvalMetrics | None,
    ) -> PromotionDecision:
        return _promotion_decision(
            candidate_eval=candidate_eval,
            parent=parent,
            parent_eval=parent_eval,
        )

    def apply_promotion(
        self,
        db_client: DatabaseClient,
        *,
        candidate: ModelHandle,
        current: ModelHandle | None,
        eval_metrics_json: str | None,
    ) -> ModelHandle:
        row = candidate.payload
        if self.user_id is None:
            if not isinstance(row, BaselineModel):
                raise TypeError("Platform run can only promote a platform row")
            served_platform = current.payload if current is not None else None
            if not isinstance(served_platform, BaselineModel):
                served_platform = None
            promoted_platform = row.promote_over(
                db_client,
                current_production=served_platform,
                eval_metrics=eval_metrics_json,
            )
            handle = _baseline_handle(promoted_platform)
        else:
            if not isinstance(row, PersonalModel):
                raise TypeError("User run can only promote a user row")
            if row.user_id != self.user_id:
                raise ValueError(f"Candidate user {row.user_id} does not match run {self.user_id}")
            served_user = current.payload if current is not None else None
            if not isinstance(served_user, PersonalModel):
                served_user = None
            promoted_user = row.promote_over(
                db_client,
                current_production=served_user,
                eval_metrics=eval_metrics_json,
            )
            handle = _personal_handle(promoted_user)
        if handle is None:
            raise RuntimeError("apply_promotion failed to build a model handle")
        return handle
