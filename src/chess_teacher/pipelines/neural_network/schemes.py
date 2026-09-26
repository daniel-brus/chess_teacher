"""Platform and per-account schemes for the shared training steps."""

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
from chess_teacher.pipelines.neural_network.offline_eval import (
    load_account_registry_bucket_datums,
    load_registry_val_datums,
)
from chess_teacher.pipelines.neural_network.pipeline_steps import (
    MAX_MOVES_PER_BASELINE_BATCH,
    MIN_NEW_MOVES_BASELINE,
)
from chess_teacher.pipelines.neural_network.split_registry import SplitRegistry
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT, SplitBucket
from chess_teacher.pipelines.neural_network.training_scheme import ModelHandle, PromotionDecision
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.general_utils import get_current_datetime

# Phase 3a gate: below this, keep serving the platform model for the account.
MIN_PERSONAL_TRAIN_MOVES = 300
# Phase 3a finetune sample-weight boost. Platform training keeps the trainer default.
PERSONAL_STYLE_DISAGREE_BOOST = 4.0
# Agree top1 may drop this far and still promote a personal model.
PERSONAL_AGREE_DROP_LIMIT = 0.03


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
        kind="baseline",
        payload=row,
    )


def _personal_handle(row: PersonalModel | None) -> ModelHandle | None:
    if row is None:
        return None
    return ModelHandle(
        key=row.version,
        weights_uri=row.model_uri,
        compatible=bool(row.model_uri) and row.looks_like_candidate_style(),
        kind="personal",
        payload=row,
    )


def _missing_scores(candidate_eval: EvalMetrics | None, reference_eval: EvalMetrics | None) -> bool:
    return candidate_eval is None or reference_eval is None


class BaselineTrainingScheme:
    """Platform chain: all accounts, registry-train queue, registry val."""

    pipeline_name = "baseline_training"
    split_version = DEFAULT_SPLIT_SALT
    min_new_moves = MIN_NEW_MOVES_BASELINE

    def note_checked(self, db_client: DatabaseClient) -> None:
        _touch_scope(db_client, BASELINE_TRAINING_SCOPE)

    def count_pending(self, db_client: DatabaseClient) -> int:
        return TrainingDataStore(db_client).count_unprocessed_train(
            split_version=self.split_version,
            flag_column=PROCESSED_FLAG_BASELINE,
        )

    def resolve_parent(self, db_client: DatabaseClient) -> ModelHandle | None:
        return _baseline_handle(BaselineModel.resolve_parent(db_client))

    def resolve_reference(self, db_client: DatabaseClient) -> ModelHandle | None:
        return _baseline_handle(
            BaselineModel.latest_with_status(db_client, BaselineModelStatus.PRODUCTION)
        )

    def load_train_batch(self, db_client: DatabaseClient) -> tuple[list[TrainingDatum], list[str]]:
        return TrainingDataStore(db_client).fetch_unprocessed_train_batch(
            split_version=self.split_version,
            limit=MAX_MOVES_PER_BASELINE_BATCH,
            flag_column=PROCESSED_FLAG_BASELINE,
        )

    def load_eval_datums(self, db_client: DatabaseClient) -> list[TrainingDatum]:
        return load_registry_val_datums(
            db_client,
            split_version=self.split_version,
            full=True,
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
        return BaselineModel.next_version(db_client)

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
        row = BaselineModel(
            id=BaselineModel.generate_id({"version": version}),
            version=version,
            trained_at=get_current_datetime(),
            mlflow_run_id=mlflow_run_id,
            model_uri=model_uri,
            status=BaselineModelStatus.CANDIDATE,
            parent_version=parent.key if parent else None,
            eval_metrics=metrics_json,
            git_commit_hash=BaselineModel.current_git_commit(),
        )
        row.save_new_to_db(db_client)
        handle = _baseline_handle(row)
        if handle is None:
            raise RuntimeError("insert_candidate failed to build a baseline handle")
        return handle

    def mark_trained(self, db_client: DatabaseClient, game_ids: list[str]) -> None:
        SplitRegistry(db_client, split_version=self.split_version).mark_processed(
            game_ids,
            flag_column=PROCESSED_FLAG_BASELINE,
        )
        _touch_scope(db_client, BASELINE_TRAINING_SCOPE)

    def decide_promotion(
        self,
        *,
        candidate_eval: EvalMetrics | None,
        reference: ModelHandle | None,
        reference_eval: EvalMetrics | None,
    ) -> PromotionDecision:
        if reference is None:
            return PromotionDecision(True, "No served model yet.")
        if not reference.compatible:
            return PromotionDecision(
                True,
                f"Served model {reference.key} is incompatible; promoting the new candidate.",
            )
        if _missing_scores(candidate_eval, reference_eval):
            return PromotionDecision(False, "Missing registry-val scores; not promoting.")
        assert candidate_eval is not None and reference_eval is not None
        if candidate_eval.top1_overall < reference_eval.top1_overall:
            return PromotionDecision(
                False,
                (
                    f"overall top1 {candidate_eval.top1_overall:.4f} "
                    f"< {reference_eval.top1_overall:.4f}"
                ),
            )
        drop = candidate_eval.top1_sf_disagree
        served = reference_eval.top1_sf_disagree
        if drop is not None and served is not None and drop < served:
            return PromotionDecision(
                False,
                f"disagree top1 {drop:.4f} < served {served:.4f}",
            )
        return PromotionDecision(True, "overall top1 held and disagree top1 did not drop.")

    def apply_promotion(
        self,
        db_client: DatabaseClient,
        *,
        candidate: ModelHandle,
        reference: ModelHandle | None,
        eval_metrics_json: str | None,
    ) -> ModelHandle:
        row = candidate.payload
        if not isinstance(row, BaselineModel):
            raise TypeError("Baseline scheme can only promote a baseline row")
        current = reference.payload if reference is not None else None
        if current is not None and not isinstance(current, BaselineModel):
            current = None
        promoted = row.promote_over(
            db_client,
            current_production=current,
            eval_metrics=eval_metrics_json,
        )
        handle = _baseline_handle(promoted)
        if handle is None:
            raise RuntimeError("apply_promotion failed to build a baseline handle")
        return handle


class PersonalTrainingScheme:
    """One account: that account ∩ the same hash registry. Served fallback is the platform model."""

    pipeline_name = "personal_training"
    split_version = DEFAULT_SPLIT_SALT
    min_new_moves = MIN_PERSONAL_TRAIN_MOVES

    def __init__(self, account_id: str, *, split_version: str = DEFAULT_SPLIT_SALT) -> None:
        account = (account_id or "").strip()
        if not account:
            raise ValueError("account_id is required")
        self.account_id = account
        self.split_version = split_version

    @property
    def _scope(self) -> str:
        return f"account:{self.account_id}"

    def note_checked(self, db_client: DatabaseClient) -> None:
        _touch_scope(db_client, self._scope)

    def count_pending(self, db_client: DatabaseClient) -> int:
        return TrainingDataStore(db_client).count_unprocessed_train(
            split_version=self.split_version,
            flag_column=PROCESSED_FLAG_PERSONAL,
            extra_where=account_id_in_sql([self.account_id]),
        )

    def resolve_parent(self, db_client: DatabaseClient) -> ModelHandle | None:
        options = (
            _personal_handle(
                PersonalModel.latest_for_account(
                    db_client, self.account_id, BaselineModelStatus.CANDIDATE
                )
            ),
            _personal_handle(
                PersonalModel.latest_for_account(
                    db_client, self.account_id, BaselineModelStatus.PRODUCTION
                )
            ),
            _baseline_handle(
                BaselineModel.latest_with_status(db_client, BaselineModelStatus.PRODUCTION)
            ),
        )
        for handle in options:
            if handle is not None and handle.compatible:
                return handle
        for handle in options:
            if handle is not None:
                return handle
        return None

    def resolve_reference(self, db_client: DatabaseClient) -> ModelHandle | None:
        personal = _personal_handle(
            PersonalModel.latest_for_account(
                db_client, self.account_id, BaselineModelStatus.PRODUCTION
            )
        )
        if personal is not None:
            return personal
        return _baseline_handle(
            BaselineModel.latest_with_status(db_client, BaselineModelStatus.PRODUCTION)
        )

    def load_train_batch(self, db_client: DatabaseClient) -> tuple[list[TrainingDatum], list[str]]:
        return TrainingDataStore(db_client).fetch_unprocessed_train_batch(
            split_version=self.split_version,
            limit=MAX_MOVES_PER_BASELINE_BATCH,
            flag_column=PROCESSED_FLAG_PERSONAL,
            extra_where=account_id_in_sql([self.account_id]),
        )

    def load_eval_datums(self, db_client: DatabaseClient) -> list[TrainingDatum]:
        return load_account_registry_bucket_datums(
            self.account_id,
            db_client,
            bucket=SplitBucket.VAL,
            split_version=self.split_version,
            limit=None,
        )

    def fit(
        self,
        datums: list[TrainingDatum],
        *,
        weights_path: Path | None,
    ) -> tuple[Any, dict[str, float]]:
        from chess_teacher.pipelines.neural_network.board_encoder import HybridBoardTrainer

        trainer = HybridBoardTrainer(style_disagree_boost=PERSONAL_STYLE_DISAGREE_BOOST)
        return trainer.fit(datums, weights_path=weights_path)

    def save(self, model: Any, path: Path) -> None:
        from chess_teacher.pipelines.neural_network.board_encoder import HybridBoardTrainer

        HybridBoardTrainer.save(model, path)

    def next_version(self, db_client: DatabaseClient) -> str:
        return PersonalModel.next_version(db_client, self.account_id)

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
        row = PersonalModel(
            id=PersonalModel.generate_id({"account_id": self.account_id, "version": version}),
            account_id=self.account_id,
            version=version,
            trained_at=get_current_datetime(),
            mlflow_run_id=mlflow_run_id,
            model_uri=model_uri,
            status=BaselineModelStatus.CANDIDATE,
            parent_version=parent.key if parent else None,
            parent_kind=parent.kind if parent else None,
            eval_metrics=metrics_json,
            git_commit_hash=BaselineModel.current_git_commit(),
        )
        row.save_new_to_db(db_client)
        handle = _personal_handle(row)
        if handle is None:
            raise RuntimeError("insert_candidate failed to build a personal handle")
        return handle

    def mark_trained(self, db_client: DatabaseClient, game_ids: list[str]) -> None:
        SplitRegistry(db_client, split_version=self.split_version).mark_processed(
            game_ids,
            flag_column=PROCESSED_FLAG_PERSONAL,
        )
        _touch_scope(db_client, self._scope)

    def decide_promotion(
        self,
        *,
        candidate_eval: EvalMetrics | None,
        reference: ModelHandle | None,
        reference_eval: EvalMetrics | None,
    ) -> PromotionDecision:
        if reference is None:
            return PromotionDecision(True, "No served model yet.")
        if not reference.compatible:
            return PromotionDecision(
                True,
                f"Served model {reference.key} is incompatible; promoting the new candidate.",
            )
        if _missing_scores(candidate_eval, reference_eval):
            return PromotionDecision(False, "Missing registry-val scores; not promoting.")
        assert candidate_eval is not None and reference_eval is not None
        disagree = candidate_eval.top1_sf_disagree
        served_disagree = reference_eval.top1_sf_disagree
        if disagree is not None and served_disagree is not None:
            if disagree < served_disagree:
                return PromotionDecision(
                    False,
                    f"disagree top1 {disagree:.4f} < served {served_disagree:.4f}",
                )
        elif candidate_eval.top1_overall < reference_eval.top1_overall:
            return PromotionDecision(
                False,
                (
                    f"overall top1 {candidate_eval.top1_overall:.4f} "
                    f"< {reference_eval.top1_overall:.4f}"
                ),
            )
        agree = candidate_eval.top1_sf_agree
        served_agree = reference_eval.top1_sf_agree
        if (
            agree is not None
            and served_agree is not None
            and agree < served_agree - PERSONAL_AGREE_DROP_LIMIT
        ):
            return PromotionDecision(
                False,
                (
                    f"agree top1 {agree:.4f} dropped more than "
                    f"{PERSONAL_AGREE_DROP_LIMIT} below {served_agree:.4f}"
                ),
            )
        return PromotionDecision(True, "disagree top1 held and agree drop is within the limit.")

    def apply_promotion(
        self,
        db_client: DatabaseClient,
        *,
        candidate: ModelHandle,
        reference: ModelHandle | None,
        eval_metrics_json: str | None,
    ) -> ModelHandle:
        row = candidate.payload
        if not isinstance(row, PersonalModel):
            raise TypeError("Personal scheme can only promote a personal row")
        if row.account_id != self.account_id:
            raise ValueError(
                f"Candidate account {row.account_id} does not match scheme {self.account_id}"
            )
        current = reference.payload if reference is not None else None
        if current is not None and not isinstance(current, PersonalModel):
            current = None
        promoted = row.promote_over(
            db_client,
            current_production=current,
            eval_metrics=eval_metrics_json,
        )
        handle = _personal_handle(promoted)
        if handle is None:
            raise RuntimeError("apply_promotion failed to build a personal handle")
        return handle
