"""MLflow tracking + artifact access for neural-network pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.keras_weights import (
    resolve_keras_weights_path,
)
from chess_teacher.utils.db.engine import postgres_url_string
from chess_teacher.utils.env_utils import get_optional_env_variable
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.object_storage.factory import (
    build_s3_storage_settings,
    s3_url_string,
)

logger = get_logger()


def _log_tracking_uri(uri: str) -> str:
    """Hide password in tracking URI logs (SQLAlchemy render when possible)."""
    try:
        from sqlalchemy.engine import make_url

        return make_url(uri).render_as_string(hide_password=True)
    except Exception:
        return uri


def _apply_mlflow_s3_env() -> None:
    """MLflow artifact store uses boto env vars; our ObjectStorage passes keys explicitly.

    Maps project ``S3_*`` into what MLflow/boto expect (setdefault only).
    """
    cfg = build_s3_storage_settings()
    os.environ.setdefault("AWS_ACCESS_KEY_ID", cfg.access_key)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", cfg.secret_key)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", cfg.endpoint_url)
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


class MLflowTracker:
    """Owns MLflow env wiring, experiment setup, runs, and artifact download."""

    DEFAULT_EXPERIMENT = "baseline"

    _configured = False

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        self.tracking_uri = (
            tracking_uri
            or get_optional_env_variable("MLFLOW_TRACKING_URI")
            or postgres_url_string()
        )
        self.experiment_name: str = (
            experiment_name
            or get_optional_env_variable("MLFLOW_EXPERIMENT_NAME")
            or self.DEFAULT_EXPERIMENT
        )

    @staticmethod
    def artifact_root() -> str:
        """S3 URI under existing bucket + STORAGE_ROOT/mlflow."""
        explicit = get_optional_env_variable("MLFLOW_ARTIFACT_ROOT")
        if explicit:
            return explicit.rstrip("/")
        return s3_url_string("mlflow")

    def configure(self) -> None:
        """Set tracking URI + artifact root; adapt S3 env for MLflow boto (idempotent)."""
        if MLflowTracker._configured:
            return

        _apply_mlflow_s3_env()

        import mlflow

        log_uri = _log_tracking_uri(self.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)
        artifact_root = self.artifact_root()
        existing = mlflow.get_experiment_by_name(self.experiment_name)
        if existing is None:
            mlflow.create_experiment(self.experiment_name, artifact_location=artifact_root)
            logger.info(
                "Created MLflow experiment=%s artifact_root=%s tracking_uri=%s",
                self.experiment_name,
                artifact_root,
                log_uri,
            )
        else:
            logger.info(
                "Using MLflow experiment=%s tracking_uri=%s",
                self.experiment_name,
                log_uri,
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

        direct = resolve_keras_weights_path(model_uri)
        if direct is not None:
            return direct

        # ``runs:/`` and other MLflow-only URIs (training / promotion tooling).
        self.configure()
        import mlflow

        downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))
        if downloaded.is_file():
            return downloaded
        if downloaded.is_dir():
            matches = list(downloaded.glob("*.keras")) + list(downloaded.rglob("*.keras"))
            return matches[0] if matches else None
        return None

    def require_keras_weights(self, model_uri: str) -> Path:
        """Like ``download_keras_weights`` but raises when the artifact cannot be resolved."""
        weights_path = self.download_keras_weights(model_uri)
        if weights_path is None or not weights_path.is_file():
            raise FileNotFoundError(f"Could not resolve Keras weights from uri={model_uri!r}")
        return weights_path

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
