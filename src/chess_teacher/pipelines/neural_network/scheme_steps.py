"""Pipeline steps shared by every training scheme.

Preparation stays out of the TensorFlow steps so a failed fit or score retries
that attempt without re-querying. Eval datums are loaded once per run and
reused for the candidate and the served model.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from chess_teacher.pipelines.neural_network.training_scheme import (
    ModelHandle,
    TrainingScheme,
)
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext, PipelineStep

logger = get_logger()

SKIP_KEY = "model_run_skip"


def _skipped(context: PipelineContext) -> bool:
    return bool(context.extras.get(SKIP_KEY))


class PrepareTrainingStep(PipelineStep):
    """Count pending moves, then load the parent, the served model, and the train batch."""

    def __init__(self, scheme: TrainingScheme) -> None:
        super().__init__(name="PrepareTraining")
        self._scheme = scheme

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
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
        reference = self._scheme.resolve_reference(db_client)
        datums, game_ids = self._scheme.load_train_batch(db_client)
        context.extras["parent"] = parent
        context.extras["reference"] = reference
        context.extras["train_datums"] = datums
        context.extras["train_game_ids"] = game_ids
        if not datums:
            logger.info("PrepareTraining skip: pending=%s but the batch was empty.", pending)
            context.extras[SKIP_KEY] = True
            return

        context.extras[SKIP_KEY] = False
        logger.info(
            "PrepareTraining pending=%s batch=%s games=%s parent=%s reference=%s",
            pending,
            len(datums),
            len(game_ids),
            parent.key if parent else None,
            reference.key if reference else None,
        )


class TrainModelStep(PipelineStep):
    """Download parent weights if needed, fit, and keep only the saved file."""

    def __init__(self, scheme: TrainingScheme) -> None:
        super().__init__(name="TrainModel")
        self._scheme = scheme

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        del db_client
        if _skipped(context):
            return

        datums = context.extras.get("train_datums") or []
        if not datums:
            context.extras[SKIP_KEY] = True
            return

        parent: ModelHandle | None = context.extras.get("parent")
        weights_path: Path | None = None
        if parent is not None and parent.compatible and parent.weights_uri:
            from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker

            logger.info("Downloading parent weights uri=%s", parent.weights_uri)
            weights_path = MLflowTracker().download_keras_weights(parent.weights_uri)

        model, metrics = self._scheme.fit(datums, weights_path=weights_path)
        out_path = Path(tempfile.mkdtemp(prefix="scheme_model_")) / "model.keras"
        self._scheme.save(model, out_path)
        context.extras["trained_model_path"] = out_path
        context.extras["train_metrics"] = metrics
        # Free the batch before the eval load. A retry of this step already
        # finished only after this assignment, so a later score retry still
        # has the file.
        context.extras.pop("train_datums", None)
        logger.info("TrainModel saved path=%s n_samples=%s", out_path, metrics.get("n_samples"))


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
    """Score the new file and the served model on the prepared eval list."""

    def __init__(self) -> None:
        super().__init__(name="ScoreEvaluation")

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        del db_client
        if _skipped(context):
            return

        from chess_teacher.pipelines.neural_network.eval_metrics import (
            evaluate_packed,
            pack_datums_for_eval,
        )
        from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker
        from chess_teacher.pipelines.neural_network.train import load_candidate_style_keras

        datums = context.extras.get("eval_datums") or []
        model_path: Path | None = context.extras.get("trained_model_path")
        if model_path is None:
            raise ValueError("ScoreEvaluation requires trained_model_path")
        if not datums:
            logger.info("ScoreEvaluation skipped scores: empty eval set.")
            context.extras["candidate_eval"] = None
            context.extras["reference_eval"] = None
            return

        packed = pack_datums_for_eval(datums)
        candidate_model = load_candidate_style_keras(model_path, compile_model=False)
        context.extras["candidate_eval"] = evaluate_packed(candidate_model, packed)

        reference: ModelHandle | None = context.extras.get("reference")
        if reference is None or not reference.compatible or not reference.weights_uri:
            context.extras["reference_eval"] = None
            logger.info(
                "ScoreEvaluation candidate only (no compatible served weights). n_eval=%s",
                context.extras["candidate_eval"].n_eval,
            )
            return

        reference_path = MLflowTracker().require_keras_weights(reference.weights_uri)
        reference_model = load_candidate_style_keras(reference_path, compile_model=False)
        context.extras["reference_eval"] = evaluate_packed(reference_model, packed)
        logger.info(
            "ScoreEvaluation candidate_top1=%s reference_top1=%s n=%s",
            context.extras["candidate_eval"].top1_overall,
            context.extras["reference_eval"].top1_overall,
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
        reference_eval = context.extras.get("reference_eval")
        parent: ModelHandle | None = context.extras.get("parent")
        version = self._scheme.next_version(db_client)

        blob: dict[str, float | str] = {
            key: value for key, value in train_metrics.items() if isinstance(value, (int, float))
        }
        blob["split_version"] = self._scheme.split_version
        if candidate_eval is not None:
            for key, value in candidate_eval.as_dict().items():
                blob[f"val_{key}"] = value
        if reference_eval is not None:
            for key, value in reference_eval.as_dict().items():
                blob[f"reference_val_{key}"] = value
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
            reference=context.extras.get("reference"),
            reference_eval=context.extras.get("reference_eval"),
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
        reference: ModelHandle | None = context.extras.get("reference")
        # Archive only a served row from this same chain. The score may have
        # used another chain's weights as the reference.
        current = reference if reference is not None and reference.kind == candidate.kind else None
        promoted = self._scheme.apply_promotion(
            db_client,
            candidate=candidate,
            reference=current,
            eval_metrics_json=None,
        )
        context.extras["promoted"] = promoted
        logger.info("ApplyPromotion served=%s", promoted.key)


def build_training_scheme_steps(
    scheme: TrainingScheme,
    *,
    promote: bool = False,
) -> list[PipelineStep]:
    """Same step classes for every scheme. ``promote`` adds the two write steps."""
    steps: list[PipelineStep] = [
        PrepareTrainingStep(scheme),
        TrainModelStep(scheme),
        PrepareEvaluationStep(scheme),
        ScoreEvaluationStep(),
        RecordCandidateStep(scheme),
        AdvanceTrainingCursorStep(scheme),
    ]
    if promote:
        steps.extend([
            DecideFromScoresStep(scheme),
            ApplyPromotionStep(scheme),
        ])
    return steps
