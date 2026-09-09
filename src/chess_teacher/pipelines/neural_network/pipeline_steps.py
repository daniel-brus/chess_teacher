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
    GameSplitAssignment,
    TrainingState,
)
from chess_teacher.pipelines.neural_network.split_registry import SplitRegistry
from chess_teacher.pipelines.neural_network.splits import DEFAULT_SPLIT_SALT
from chess_teacher.pipelines.neural_network.train import BaselineTrainer
from chess_teacher.pipelines.preprocessing.games import Game
from chess_teacher.pipelines.preprocessing.moves import Move, MoveCharacteristics
from chess_teacher.utils.db.client import DatabaseClient
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.pipeline_utils.pipeline_base import PipelineContext, PipelineStep

logger = get_logger()

# Train only when at least this many unprocessed train moves sit on the queue.
MIN_NEW_MOVES_BASELINE = 1000
# Cap each incremental train batch (complete games by game_id); avoids first-run OOM.
MAX_MOVES_PER_BASELINE_BATCH = 10_000


def _should_skip(context: PipelineContext) -> bool:
    return bool(context.extras.get("baseline_skip"))


class CheckSufficientNewDataStep(PipelineStep):
    """No-op remaining steps when unprocessed train moves < MIN_NEW_MOVES_BASELINE."""

    def __init__(self, *, split_version: str = DEFAULT_SPLIT_SALT) -> None:
        super().__init__(name="CheckSufficientNewData")
        self.split_version = split_version

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        db_client.ensure_metadata(BaselineModel.get_metadata())
        db_client.ensure_metadata(TrainingState.get_metadata())
        db_client.ensure_metadata(GameSplitAssignment.get_metadata())
        db_client.ensure_tables(
            Move.get_metadata(),
            Game.get_metadata(),
            MoveCharacteristics.get_metadata(),
        )
        state = TrainingState.for_baseline(db_client)
        store = TrainingDataStore(db_client)
        logger.info(
            "Counting unprocessed train moves (bucket=train, flag NULL, "
            "split_version=%s) - may take a while...",
            self.split_version,
        )
        n_new = store.count_unprocessed_train(split_version=self.split_version)
        min_needed = MIN_NEW_MOVES_BASELINE
        updated = state.with_check_at(get_current_datetime())
        updated.save_to_db(db_client)

        context.extras["training_state"] = updated
        context.extras["split_version"] = self.split_version
        context.extras["new_move_count"] = n_new
        context.extras["min_new_moves"] = min_needed

        if n_new < min_needed:
            logger.info(
                "Baseline training skip: unprocessed_train=%s < min=%s split_version=%s",
                n_new,
                min_needed,
                self.split_version,
            )
            context.extras["baseline_skip"] = True
            return

        logger.info(
            "Baseline training proceed: unprocessed_train=%s >= min=%s split_version=%s",
            n_new,
            min_needed,
            self.split_version,
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
    """Load next unprocessed train games in game_id order (complete games)."""

    def __init__(self, *, split_version: str = DEFAULT_SPLIT_SALT) -> None:
        super().__init__(name="LoadNewData")
        self.split_version = split_version

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _should_skip(context):
            return

        split_version = str(context.extras.get("split_version") or self.split_version)
        limit = MAX_MOVES_PER_BASELINE_BATCH
        logger.info(
            "Loading unprocessed train datums (game_id order, limit=%s, "
            "split_version=%s) - SQL + hydrate characteristics...",
            limit,
            split_version,
        )
        datums, game_ids = TrainingDataStore(db_client).fetch_unprocessed_train_batch(
            split_version=split_version,
            limit=limit,
        )
        context.extras["training_datums"] = datums
        context.extras["batch_game_ids"] = game_ids
        context.extras["split_version"] = split_version
        logger.info(
            "Loaded training datums=%s games=%s limit=%s",
            len(datums),
            len(game_ids),
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
    """Mark batch game_ids processed after a successful fit. Skip must not mark."""

    def __init__(self, *, split_version: str = DEFAULT_SPLIT_SALT) -> None:
        super().__init__(name="UpdateTrainingState")
        self.split_version = split_version

    def run(self, db_client: DatabaseClient, context: PipelineContext) -> None:
        if _should_skip(context):
            return

        game_ids: list[str] = list(context.extras.get("batch_game_ids") or [])
        if not game_ids:
            logger.warning("No batch_game_ids; processed flags unchanged.")
            return

        split_version = str(context.extras.get("split_version") or self.split_version)
        registry = SplitRegistry(db_client, split_version=split_version)
        updated = registry.mark_processed(game_ids)
        context.extras["marked_game_count"] = updated
        logger.info(
            "Marked already_processed_baseline on %s/%s train games (split_version=%s)",
            updated,
            len(game_ids),
            split_version,
        )
