"""Pipeline steps shared by every training scheme.

Preparation stays out of the TensorFlow steps so a failed fit or score retries
that attempt without re-querying. Eval datums are loaded once per run and
reused for the candidate and the served model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from chess_teacher.pipelines.neural_network.eval_metrics import EvalMetrics
from chess_teacher.pipelines.neural_network.training_scheme import (
    ModelHandle,
    ParentBaselineDecision,
    TrainingScheme,
)
from chess_teacher.pipelines.neural_network.training_slot import (
    TRAINING_SLOT_POLL_SECONDS,
    TRAINING_SLOT_WAIT_LIMIT,
    DbTrainingSlot,
    pipeline_host_has_memory,
)
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_base import (
    PIPELINE_RUN_ID_EXTRA,
    PipelineContext,
    PipelineStep,
)
from chess_teacher.utils.pipeline_utils.step_process import StepProcess

logger = get_logger()

SKIP_KEY = "model_run_skip"


def _skipped(context: PipelineContext) -> bool:
    return bool(context.extras.get(SKIP_KEY))


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:.4f}"


def _model_id(handle: ModelHandle) -> str:
    model_id = getattr(handle.payload, "id", None)
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("RecordCandidate model handle has no id")
    return model_id


class WaitForTrainingSlotStep(PipelineStep):
    """Wait until this run holds the global training slot and the host has memory.

    This sits immediately before the TensorFlow steps. ``ReleaseTrainingSlot`` drops
    the slot once that child has exited, including when the fit or the score failed.
    """

    def __init__(
        self,
        scheme: TrainingScheme,
        *,
        slot: DbTrainingSlot | None = None,
        memory_available: Callable[[], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        poll_seconds: float = TRAINING_SLOT_POLL_SECONDS,
        wait_seconds: float = TRAINING_SLOT_WAIT_LIMIT.total_seconds(),
    ) -> None:
        super().__init__(name="WaitForTrainingSlot", max_retries=0)
        self._scheme = scheme
        self._slot = slot
        self._memory_available = memory_available or pipeline_host_has_memory
        self._sleep = sleep
        self._monotonic = monotonic
        self._poll_seconds = poll_seconds
        self._wait_seconds = wait_seconds

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _skipped(context):
            return
        pending = self._scheme.count_pending(db_client)
        if pending < self._scheme.min_new_moves:
            self._scheme.note_checked(db_client)
            logger.info(
                "WaitForTrainingSlot skip: pending=%s < min=%s",
                pending,
                self._scheme.min_new_moves,
            )
            context.extras[SKIP_KEY] = True
            return

        run_id = context.extras.get(PIPELINE_RUN_ID_EXTRA)
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("WaitForTrainingSlot requires a pipeline run id")

        slot = self._slot or DbTrainingSlot(db_client)
        started = self._monotonic()
        while True:
            if slot.try_acquire(run_id, now=get_current_datetime()):
                if self._memory_available():
                    logger.info("Training slot acquired run_id=%s", run_id)
                    return
                slot.release(run_id)
                logger.info(
                    "Training slot released: MemAvailable is below the training minimum. run_id=%s",
                    run_id,
                )
            else:
                logger.info("Training slot busy run_id=%s", run_id)

            elapsed = self._monotonic() - started
            if elapsed >= self._wait_seconds:
                self._scheme.note_checked(db_client)
                logger.warning(
                    "Training slot wait exceeded %.0fs; skipping this training round. run_id=%s",
                    self._wait_seconds,
                    run_id,
                )
                context.extras[SKIP_KEY] = True
                context.progress_warning(
                    "Training waited too long for a free slot; skipping this round."
                )
                return

            delay = min(self._poll_seconds, self._wait_seconds - elapsed)
            context.progress_update("Waiting for another training run to finish...")
            self._sleep(delay)


class ReleaseTrainingSlotStep(PipelineStep):
    """Drop the training slot after the TensorFlow child has exited."""

    def __init__(self, *, slot: DbTrainingSlot | None = None) -> None:
        super().__init__(
            name="ReleaseTrainingSlot",
            max_retries=0,
            critical=False,
            run_if_earlier_step_failed=True,
        )
        self._slot = slot

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        run_id = context.extras.get(PIPELINE_RUN_ID_EXTRA)
        if not isinstance(run_id, str) or not run_id:
            logger.info("ReleaseTrainingSlot skipped: no pipeline run id")
            return
        slot = self._slot or DbTrainingSlot(db_client)
        slot.release(run_id)
        logger.info("Training slot released run_id=%s", run_id)


class PrepareTrainingStep(PipelineStep):
    """Count pending moves, then load the parent, the served model, and the train batch."""

    def __init__(self, scheme: TrainingScheme) -> None:
        super().__init__(name="PrepareTraining")
        self._scheme = scheme

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _skipped(context):
            return
        self._scheme.note_checked(db_client)
        pending = self._scheme.count_pending(db_client)
        context.extras["pending_count"] = pending
        if pending < self._scheme.min_new_moves:
            logger.info(
                "PrepareTraining skip: pending=%s < min=%s",
                pending,
                self._scheme.min_new_moves,
            )
            context.extras[SKIP_KEY] = True
            return

        parent = self._scheme.resolve_parent(db_client)
        datums, game_ids = self._scheme.load_train_batch(db_client)
        context.extras["parent"] = parent
        context.extras["train_datums"] = datums
        context.extras["train_game_ids"] = game_ids
        if not datums:
            logger.info("PrepareTraining skip: pending=%s but the batch was empty.", pending)
            context.extras[SKIP_KEY] = True
            return

        context.extras[SKIP_KEY] = False
        logger.info(
            "PrepareTraining pending=%s batch=%s games=%s parent=%s",
            pending,
            len(datums),
            len(game_ids),
            parent.key if parent else None,
        )


class TrainModelStep(PipelineStep):
    """Download parent weights if needed, then fit inside this step's child process."""

    def __init__(self, *, process: StepProcess | None = None) -> None:
        super().__init__(name="TrainModel", process=process)

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        del db_client
        if _skipped(context):
            return

        datums = context.extras.get("train_datums") or []
        if not datums:
            context.extras[SKIP_KEY] = True
            return
        if self.process is None:
            raise ValueError("TrainModel requires a step process")

        parent: ModelHandle | None = context.extras.get("parent")
        weights_path: Path | None = None
        if parent is not None and parent.compatible and parent.weights_uri:
            from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker

            logger.info("Downloading parent weights uri=%s", parent.weights_uri)
            weights_path = MLflowTracker().download_keras_weights(parent.weights_uri)

        fitted = self.process.call("fit", datums=datums, weights_path=weights_path)
        context.extras["trained_model_path"] = Path(fitted["model_path"])
        context.extras["train_metrics"] = fitted["metrics"]
        # The child keeps the batch until it finishes the fit. Drop the parent's copy
        # before the eval load. A retry of this step already finished only after this
        # assignment, so a later score retry still has the file.
        context.extras.pop("train_datums", None)
        logger.info(
            "TrainModel saved path=%s n_samples=%s",
            context.extras["trained_model_path"],
            fitted["metrics"].get("n_samples"),
        )


class PrepareEvaluationStep(PipelineStep):
    """Load registry-val moves once. A second call in the same run reuses them."""

    def __init__(self, scheme: TrainingScheme) -> None:
        super().__init__(name="PrepareEvaluation")
        self._scheme = scheme

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _skipped(context):
            return
        existing = context.extras.get("eval_datums")
        if existing is not None:
            logger.info("PrepareEvaluation reuse n=%s", len(existing))
            return
        datums = self._scheme.load_eval_datums(db_client)
        context.extras["eval_datums"] = datums
        logger.info("PrepareEvaluation loaded n=%s", len(datums))


class ScoreEvaluationStep(PipelineStep):
    """Score the new file and the served model inside this step's child process."""

    def __init__(self, *, process: StepProcess | None = None) -> None:
        super().__init__(name="ScoreEvaluation", process=process)

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        del db_client
        if _skipped(context):
            return

        datums = context.extras.get("eval_datums") or []
        model_path: Path | None = context.extras.get("trained_model_path")
        if model_path is None:
            raise ValueError("ScoreEvaluation requires trained_model_path")
        if not datums:
            logger.info("ScoreEvaluation skipped scores: empty eval set.")
            context.extras["candidate_eval"] = None
            context.extras["parent_eval"] = None
            return
        if self.process is None:
            raise ValueError("ScoreEvaluation requires a step process")

        from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker

        parent_weights_path: Path | None = None
        parent: ModelHandle | None = context.extras.get("parent")
        if parent is not None and parent.compatible and parent.weights_uri:
            parent_weights_path = MLflowTracker().require_keras_weights(parent.weights_uri)

        scored = self.process.call(
            "score",
            model_path=model_path,
            parent_weights_path=parent_weights_path,
            datums=datums,
        )
        context.extras["candidate_eval"] = scored["candidate_eval"]
        context.extras["parent_eval"] = scored["parent_eval"]
        if context.extras["parent_eval"] is None:
            logger.info(
                "ScoreEvaluation candidate only (cold start). n_eval=%s",
                context.extras["candidate_eval"].n_eval,
            )
            return
        logger.info(
            "ScoreEvaluation candidate_top1=%s parent_top1=%s n=%s",
            context.extras["candidate_eval"].top1_overall,
            context.extras["parent_eval"].top1_overall,
            context.extras["candidate_eval"].n_eval,
        )


class RecordCandidateStep(PipelineStep):
    """Log the run and insert the candidate row. Does not change what is served."""

    def __init__(self, scheme: TrainingScheme) -> None:
        super().__init__(name="RecordCandidate")
        self._scheme = scheme

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _skipped(context):
            return
        model_path: Path | None = context.extras.get("trained_model_path")
        if model_path is None:
            raise ValueError("RecordCandidate requires trained_model_path")

        from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker

        train_metrics: dict[str, float] = dict(context.extras.get("train_metrics") or {})
        candidate_eval = context.extras.get("candidate_eval")
        parent_eval = context.extras.get("parent_eval")
        parent: ModelHandle | None = context.extras.get("parent")
        version = self._scheme.next_version(db_client)

        blob: dict[str, float | str] = {
            key: value for key, value in train_metrics.items() if isinstance(value, (int, float))
        }
        blob["split_version"] = self._scheme.split_version
        if candidate_eval is not None:
            for key, value in candidate_eval.as_dict().items():
                blob[f"val_{key}"] = value
        if parent_eval is not None:
            for key, value in parent_eval.as_dict().items():
                blob[f"parent_val_{key}"] = value
        metrics_json = json.dumps(blob)

        mlflow_metrics = {
            key: float(value) for key, value in blob.items() if isinstance(value, (int, float))
        }
        run_id, artifact_uri = MLflowTracker().log_training_run(
            run_name=f"{self._scheme.pipeline_name}-{version}",
            model_path=model_path,
            params={
                "version": version,
                "parent_key": parent.key if parent else "",
                "parent_kind": parent.kind if parent else "",
                "split_version": self._scheme.split_version,
                "n_samples": int(train_metrics.get("n_samples", 0)),
                "epochs": int(train_metrics.get("epochs", 0)),
            },
            metrics=mlflow_metrics,
        )
        handle = self._scheme.insert_candidate(
            db_client,
            version=version,
            model_uri=artifact_uri,
            mlflow_run_id=run_id,
            metrics_json=metrics_json,
            parent=parent,
        )
        context.extras["candidate"] = handle
        logger.info("RecordCandidate version=%s uri=%s", handle.key, artifact_uri)
        self._save_training_score(
            db_client,
            context,
            version=version,
            model_id=_model_id(handle),
            candidate_eval=candidate_eval,
        )

    def _save_training_score(
        self,
        db_client: DatabaseClient,
        context: PipelineContext,
        *,
        version: str,
        model_id: str,
        candidate_eval: EvalMetrics | None,
    ) -> None:
        """One row for this run's final model. Skipped when the eval set was empty."""
        if candidate_eval is None:
            logger.info("RecordCandidate: no val scores to store.")
            return

        from chess_teacher.pipelines.neural_network.training_scores import (
            training_score_from_eval,
        )

        run_id = context.extras.get(PIPELINE_RUN_ID_EXTRA)
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("RecordCandidate requires pipeline_run_id to store training scores")
        row = training_score_from_eval(
            run_id=run_id,
            pipeline_name=self._scheme.pipeline_name,
            user_id=context.user_id,
            version=version,
            model_id=model_id,
            scored_at=get_current_datetime(),
            metrics=candidate_eval,
        )
        row.save_to_db(db_client)
        logger.info(
            "TrainingScore run_id=%s version=%s n_eval=%s val_top1=%.4f "
            "val_top1_disagree=%s sf_delta_mean_pawns=%.4f sf_delta_median_pawns=%.4f",
            row.run_id,
            row.version,
            row.n_eval,
            row.val_top1,
            _fmt_optional(row.val_top1_disagree),
            row.sf_delta_mean_pawns,
            row.sf_delta_median_pawns,
        )


class AdvanceTrainingCursorStep(PipelineStep):
    """Mark the fitted games processed. Safe to retry after the candidate row exists."""

    def __init__(self, scheme: TrainingScheme) -> None:
        super().__init__(name="AdvanceTrainingCursor")
        self._scheme = scheme

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _skipped(context):
            return
        game_ids = list(context.extras.get("train_game_ids") or [])
        self._scheme.mark_trained(db_client, game_ids)
        logger.info("AdvanceTrainingCursor games=%s", len(game_ids))


class DecideFromScoresStep(PipelineStep):
    """Apply the scheme's promotion gate to scores already in memory."""

    def __init__(self, scheme: TrainingScheme) -> None:
        super().__init__(name="DecideFromScores")
        self._scheme = scheme

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        del db_client
        if _skipped(context):
            return
        decision = self._scheme.decide_promotion(
            candidate_eval=context.extras.get("candidate_eval"),
            parent=context.extras.get("parent"),
            parent_eval=context.extras.get("parent_eval"),
        )
        context.extras["promotion_decision"] = decision
        logger.info(
            "DecideFromScores should_promote=%s reason=%s",
            decision.should_promote,
            decision.reason,
        )


class ApplyPromotionStep(PipelineStep):
    """Flip the candidate to served when the decision says so."""

    def __init__(self, scheme: TrainingScheme) -> None:
        super().__init__(name="ApplyPromotion")
        self._scheme = scheme

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _skipped(context):
            return
        decision = context.extras.get("promotion_decision")
        if decision is None or not decision.should_promote:
            logger.info("ApplyPromotion: no change to the served model.")
            return
        candidate: ModelHandle | None = context.extras.get("candidate")
        if candidate is None:
            raise ValueError("ApplyPromotion requires a recorded candidate")
        parent: ModelHandle | None = context.extras.get("parent")
        # A user run may warm-start from the platform model. Do not archive that.
        current = parent if parent is not None and parent.kind == candidate.kind else None
        promoted = self._scheme.apply_promotion(
            db_client,
            candidate=candidate,
            current=current,
            eval_metrics_json=None,
        )
        context.extras["promoted"] = promoted
        logger.info("ApplyPromotion served=%s", promoted.key)


class ScoreParentBaselineStep(PipelineStep):
    """Score a parent baseline that is not the model we just trained.

    This holds the same child as fit and score, so the extra load stays inside
    the training slot. When the pointer already is the training parent, the
    score from ``ScoreEvaluation`` is reused and the child is not called.
    """

    def __init__(self, scheme: TrainingScheme, *, process: StepProcess | None = None) -> None:
        super().__init__(name="ScoreParentBaseline", process=process)
        self._scheme = scheme

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _skipped(context) or not self._scheme.considers_parent_baseline:
            return

        current = self._scheme.current_parent_baseline(db_client)
        if current is None or not current.compatible or not current.weights_uri:
            context.extras["parent_baseline_eval"] = None
            return

        scored = context.extras.get("parent")
        if scored is not None and scored.key == current.key and scored.kind == current.kind:
            context.extras["parent_baseline_eval"] = context.extras.get("parent_eval")
            logger.info("ScoreParentBaseline reuse key=%s", current.key)
            return

        datums = context.extras.get("eval_datums") or []
        if not datums:
            logger.info("ScoreParentBaseline skipped: empty eval set.")
            context.extras["parent_baseline_eval"] = None
            return
        if self.process is None:
            raise ValueError("ScoreParentBaseline requires a step process")

        from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker

        logger.info("Scoring parent baseline key=%s", current.key)
        weights_path = MLflowTracker().require_keras_weights(current.weights_uri)
        context.extras["parent_baseline_eval"] = self.process.call(
            "score_weights",
            weights_path=weights_path,
            datums=datums,
        )


class AdoptParentBaselineStep(PipelineStep):
    """Move the parent baseline when a promoted platform model clears the strict gate.

    Promotion can still succeed when this step does not. Personal runs no-op.
    """

    def __init__(self, scheme: TrainingScheme) -> None:
        super().__init__(name="AdoptParentBaseline")
        self._scheme = scheme

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _skipped(context) or not self._scheme.considers_parent_baseline:
            return

        promoted: ModelHandle | None = context.extras.get("promoted")
        if promoted is None:
            decision = ParentBaselineDecision(False, "Candidate was not promoted.")
            context.extras["parent_baseline_decision"] = decision
            logger.info("AdoptParentBaseline: %s", decision.reason)
            return

        current = self._scheme.current_parent_baseline(db_client)
        parent_eval = context.extras.get("parent_baseline_eval")
        decision = self._scheme.decide_parent_baseline(
            candidate_eval=context.extras.get("candidate_eval"),
            parent=current,
            parent_eval=parent_eval,
        )
        context.extras["parent_baseline_decision"] = decision
        logger.info(
            "AdoptParentBaseline should_adopt=%s reason=%s",
            decision.should_adopt,
            decision.reason,
        )
        if not decision.should_adopt:
            return
        adopted = self._scheme.apply_parent_baseline(db_client, candidate=promoted)
        context.extras["parent_baseline"] = adopted
        logger.info("AdoptParentBaseline version=%s", adopted.key)


def build_training_scheme_steps(
    scheme: TrainingScheme,
    *,
    promote: bool = False,
) -> list[PipelineStep]:
    """Same step classes for every scheme. ``promote`` adds the write steps.

    Train and score share one child interpreter. The runner starts it at TrainModel
    and exits it after the last step that holds that object. On a platform run
    that can adopt a parent baseline, that last step scores the baseline inside
    the same child. The training slot wraps that window and is dropped on the
    next step.
    """
    tensorflow = StepProcess("chess_teacher.pipelines.neural_network.tf_worker")
    steps: list[PipelineStep] = [
        PrepareTrainingStep(scheme),
        WaitForTrainingSlotStep(scheme),
        TrainModelStep(process=tensorflow),
        PrepareEvaluationStep(scheme),
        ScoreEvaluationStep(process=tensorflow),
    ]
    if promote and scheme.considers_parent_baseline:
        steps.append(ScoreParentBaselineStep(scheme, process=tensorflow))
    steps.extend([
        ReleaseTrainingSlotStep(),
        RecordCandidateStep(scheme),
        AdvanceTrainingCursorStep(scheme),
    ])
    if promote:
        steps.extend([
            DecideFromScoresStep(scheme),
            ApplyPromotionStep(scheme),
            AdoptParentBaselineStep(scheme),
        ])
    return steps
