"""Pipeline steps for baseline candidate training."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

from chess_teacher.pipelines.neural_network.candidate_eval import HEAD_TYPE_CANDIDATE_STYLE
from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingDataStore,
    TrainingDatum,
)
from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker
from chess_teacher.pipelines.neural_network.models import (
    BaselineModel,
    BaselineModelStatus,
    TrainingState,
)
from chess_teacher.pipelines.neural_network.train import BaselineTrainer
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext, PipelineStep

logger = get_logger()

# Train only when at least this many new moves exist since last cutoff.
MIN_NEW_MOVES_BASELINE = 1000
# Cap each incremental train batch (oldest-first); avoids first-run OOM on huge backlog.
MAX_MOVES_PER_BASELINE_BATCH = 10_000


def _should_skip(context: PipelineContext) -> bool:
    return bool(context.extras.get("baseline_skip"))


class CheckSufficientNewDataStep(PipelineStep):
    """No-op remaining steps when new moves since cutoff < MIN_NEW_MOVES_BASELINE."""

    def __init__(self) -> None:
        super().__init__(name="CheckSufficientNewData")

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        db_client.ensure_metadata(BaselineModel.get_metadata())
        db_client.ensure_metadata(TrainingState.get_metadata())
        state = TrainingState.for_baseline(db_client)
        cutoff = state.last_trained_data_cutoff
        store = TrainingDataStore(db_client)
        logger.info(
            "Counting eligible training moves (needs characteristics + candidate_evaluations; "
            "cutoff=%s) — may take a while…",
            cutoff,
        )
        n_new = store.count_since(cutoff)
        min_needed = MIN_NEW_MOVES_BASELINE
        updated = state.with_check_at(get_current_datetime())
        updated.save_to_db(db_client)

        context.extras["training_state"] = updated
        context.extras["new_move_count"] = n_new
        context.extras["min_new_moves"] = min_needed

        if n_new < min_needed:
            logger.info(
                "Baseline training skip: new_moves=%s < min=%s cutoff=%s",
                n_new,
                min_needed,
                cutoff,
            )
            context.extras["baseline_skip"] = True
            return

        logger.info(
            "Baseline training proceed: new_moves=%s >= min=%s cutoff=%s",
            n_new,
            min_needed,
            cutoff,
        )
        context.extras["baseline_skip"] = False


class LoadPreviousCandidateWeightsStep(PipelineStep):
    """Resolve parent baseline weights path (candidate, else production)."""

    def __init__(self) -> None:
        super().__init__(name="LoadPreviousCandidateWeights")

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _should_skip(context):
            return

        parent = BaselineModel.resolve_parent(db_client)
        context.extras["parent_model"] = parent
        context.extras["parent_version"] = parent.version if parent else None
        # Only resume same-family weights; policy/MSE parents cold-start.
        if parent is not None and parent.looks_like_candidate_style() and parent.model_uri:
            context.extras["parent_model_uri"] = parent.model_uri
            logger.info(
                "Parent baseline version=%s status=%s uri=%s (candidate_style)",
                parent.version,
                parent.status,
                parent.model_uri,
            )
        else:
            context.extras["parent_model_uri"] = None
            if parent is None:
                logger.info("No previous baseline weights; cold start.")
            else:
                logger.info(
                    "Parent version=%s not candidate_style-compatible; cold start (uri ignored).",
                    parent.version,
                )


class LoadNewDataStep(PipelineStep):
    """Load incremental training datums since last cutoff (oldest first)."""

    def __init__(self) -> None:
        super().__init__(name="LoadNewData")

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _should_skip(context):
            return

        state: TrainingState = context.extras["training_state"]
        limit = MAX_MOVES_PER_BASELINE_BATCH
        logger.info(
            "Loading training datums from DB (oldest-first, limit=%s, cutoff=%s) — "
            "SQL + hydrate characteristics…",
            limit,
            state.last_trained_data_cutoff,
        )
        datums, max_end_time = TrainingDataStore(db_client).fetch_since(
            state.last_trained_data_cutoff,
            limit=limit,
        )
        context.extras["training_datums"] = datums
        context.extras["batch_data_cutoff_at"] = max_end_time
        logger.info(
            "Loaded training datums=%s batch_cutoff=%s limit=%s",
            len(datums),
            max_end_time,
            limit,
        )
        if not datums:
            context.extras["baseline_skip"] = True


class TrainIncrementalStep(PipelineStep):
    """Finetune (or cold-start) Keras baseline on the new batch."""

    def __init__(self) -> None:
        super().__init__(name="TrainIncremental")
        self._tracker = MLflowTracker()
        self._trainer = BaselineTrainer()

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _should_skip(context):
            return

        datums: list[TrainingDatum] = context.extras["training_datums"]
        parent_uri: str | None = context.extras.get("parent_model_uri")
        if parent_uri:
            logger.info("Downloading parent Keras weights from MLflow/S3 uri=%s…", parent_uri)
        weights_path = self._tracker.download_keras_weights(parent_uri)

        logger.info(
            "Preparing candidate-style tensors + training (n_datums=%s, epochs=%s) — "
            "on-the-fly move features can take a while…",
            len(datums),
            self._trainer.epochs,
        )
        model, metrics = self._trainer.fit(datums, weights_path=weights_path)
        out_path = Path(tempfile.mkdtemp(prefix="baseline_model_")) / "model.keras"
        logger.info("Saving trained Keras model to %s…", out_path)
        BaselineTrainer.save(model, out_path)
        context.extras["trained_model_path"] = out_path
        context.extras["train_metrics"] = metrics
        logger.info("Train metrics=%s path=%s", metrics, out_path)


class LogToMLflowStep(PipelineStep):
    """Log run to MLflow and insert a new candidate BaselineModel row."""

    def __init__(self) -> None:
        super().__init__(name="LogToMLflow")
        self._tracker = MLflowTracker()

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _should_skip(context):
            return

        model_path: Path = context.extras["trained_model_path"]
        metrics: dict[str, float] = context.extras.get("train_metrics") or {}
        version = BaselineModel.next_version(db_client)
        parent_version: str | None = context.extras.get("parent_version")
        data_cutoff_at: datetime | None = context.extras.get("batch_data_cutoff_at")
        trained_at = get_current_datetime()

        logger.info(
            "Logging baseline %s to MLflow + inserting candidate row…",
            version,
        )
        run_id, artifact_uri = self._tracker.log_training_run(
            run_name=f"baseline-{version}",
            model_path=model_path,
            params={
                "version": version,
                "parent_version": parent_version or "",
                "n_samples": int(metrics.get("n_samples", 0)),
                "min_new_moves": context.extras.get("min_new_moves"),
                "head": HEAD_TYPE_CANDIDATE_STYLE,
                "max_candidates": int(metrics.get("max_candidates", 0)),
                "move_feat_dim": int(metrics.get("move_feat_dim", 0)),
                "move_feat_version": int(metrics.get("move_feat_version", 0)),
                "style_disagree_boost": float(metrics.get("style_disagree_boost", 1.0)),
                "style_disagree_scale": float(metrics.get("style_disagree_scale", 2.0)),
                "epochs": int(metrics.get("epochs", 0)),
            },
            metrics=metrics,
        )

        row = BaselineModel(
            id=BaselineModel.generate_id({"version": version}),
            version=version,
            trained_at=trained_at,
            mlflow_run_id=run_id,
            model_uri=artifact_uri,
            status=BaselineModelStatus.CANDIDATE,
            parent_version=parent_version,
            data_cutoff_at=data_cutoff_at,
            eval_metrics=json.dumps(metrics),
            git_commit_hash=BaselineModel.current_git_commit(),
        )
        row.save_new_to_db(db_client)
        context.extras["new_baseline_model"] = row
        logger.info(
            "Logged baseline candidate version=%s run_id=%s uri=%s",
            version,
            run_id,
            artifact_uri,
        )


class UpdateTrainingStateStep(PipelineStep):
    """Advance baseline data cutoff after a successful train."""

    def __init__(self) -> None:
        super().__init__(name="UpdateTrainingState")

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _should_skip(context):
            return

        data_cutoff_at: datetime | None = context.extras.get("batch_data_cutoff_at")
        if data_cutoff_at is None:
            logger.warning("No batch_data_cutoff_at; training_state cutoff unchanged.")
            return

        previous: TrainingState = context.extras.get(
            "training_state"
        ) or TrainingState.for_baseline(db_client)
        state = previous.with_cutoff(data_cutoff_at)
        state.save_to_db(db_client)
        context.extras["training_state"] = state
        logger.info("Updated baseline training_state cutoff=%s", data_cutoff_at)
