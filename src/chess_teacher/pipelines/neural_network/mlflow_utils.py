"""MLflow tracking + artifact access for neural-network pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.logging import get_logger

logger = get_logger()


class MLflowTracker:
    """Owns MLflow env wiring, experiment setup, runs, and artifact download."""

    DEFAULT_TRACKING_URI = "file:./storage/mlflow/tracking"
    DEFAULT_EXPERIMENT = "baseline"

    _configured = False

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", self.DEFAULT_TRACKING_URI
        )
        self.experiment_name = experiment_name or os.getenv(
            "MLFLOW_EXPERIMENT_NAME", self.DEFAULT_EXPERIMENT
        )

    @staticmethod
    def artifact_root() -> str:
        """S3 URI under existing bucket + STORAGE_ROOT/mlflow."""
        explicit = os.getenv("MLFLOW_ARTIFACT_ROOT")
        if explicit:
            return explicit.rstrip("/")
        bucket = get_env_variable("S3_BUCKET")
        root = get_env_variable("STORAGE_ROOT").strip("/")
        return f"s3://{bucket}/{root}/mlflow"

    def configure(self) -> None:
        """Map project S3_* into MLflow/boto env and set tracking URI (idempotent)."""
        if MLflowTracker._configured:
            return

        os.environ.setdefault("AWS_ACCESS_KEY_ID", get_env_variable("S3_ACCESS_KEY_ID"))
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", get_env_variable("S3_SECRET_ACCESS_KEY"))
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", get_env_variable("S3_ENDPOINT_URL"))
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

        if self.tracking_uri.startswith("file:"):
            Path(self.tracking_uri.removeprefix("file:")).mkdir(parents=True, exist_ok=True)

        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)
        artifact_root = self.artifact_root()
        existing = mlflow.get_experiment_by_name(self.experiment_name)
        if existing is None:
            mlflow.create_experiment(self.experiment_name, artifact_location=artifact_root)
            logger.info(
                "Created MLflow experiment=%s artifact_root=%s tracking_uri=%s",
                self.experiment_name,
                artifact_root,
                self.tracking_uri,
            )
        mlflow.set_experiment(self.experiment_name)
        MLflowTracker._configured = True

    def start_run(self, *, run_name: str | None = None) -> Any:
        """Configure and start a run (caller must end/close)."""
        self.configure()
        import mlflow

        return mlflow.start_run(run_name=run_name)

    def download_keras_weights(self, model_uri: str | None) -> Path | None:
        """Resolve a local ``.keras`` file from a file path or MLflow/S3 artifact URI."""
        if not model_uri:
            return None
        if model_uri.startswith("file:"):
            path = Path(model_uri.removeprefix("file:"))
            return path if path.is_file() else None
        local_candidate = Path(model_uri)
        if local_candidate.is_file():
            return local_candidate

        self.configure()
        import mlflow

        downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
        if downloaded.is_file():
            return downloaded
        if downloaded.is_dir():
            matches = list(downloaded.glob("*.keras")) + list(downloaded.rglob("*.keras"))
            return matches[0] if matches else None
        return None

    def log_training_run(
        self,
        *,
        run_name: str,
        model_path: Path,
        params: dict[str, Any],
        metrics: dict[str, float],
    ) -> tuple[str, str]:
        """Log params/metrics/artifact; return ``(run_id, artifact_uri)``."""
        import mlflow

        with self.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            mlflow.log_artifact(str(model_path), artifact_path="model")
            run_id = run.info.run_id
            artifact_uri = f"{run.info.artifact_uri}/model/{model_path.name}"
        return run_id, artifact_uri
