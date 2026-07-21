"""Keras baseline model trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingBatch,
    TrainingDatum,
)
from chess_teacher.utils.logging import get_logger

logger = get_logger()


class BaselineTrainer:
    """Build, load, fit, and persist the stub baseline MLP."""

    DEFAULT_EPOCHS = 3
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_HIDDEN = 128

    def __init__(
        self,
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        hidden: int = DEFAULT_HIDDEN,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden = hidden

    def build(self, input_dim: int, output_dim: int) -> Any:
        """Small MLP: state → action coords + piece one-hot."""
        from tensorflow import keras
        from tensorflow.keras import layers

        inputs = keras.Input(shape=(input_dim,), name="state")
        x = layers.Dense(self.hidden, activation="relu")(inputs)
        x = layers.Dense(self.hidden, activation="relu")(x)
        outputs = layers.Dense(output_dim, activation="linear", name="action")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_move_mlp")
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
        return model

    def load_or_build(
        self,
        *,
        input_dim: int,
        output_dim: int,
        weights_path: Path | None = None,
    ) -> Any:
        """Load Keras weights if present; else cold-start a new model."""
        from tensorflow import keras

        if weights_path is not None and weights_path.is_file():
            logger.info("Loading baseline weights from %s", weights_path)
            return keras.models.load_model(weights_path)
        logger.info(
            "Cold-start baseline model input_dim=%s output_dim=%s",
            input_dim,
            output_dim,
        )
        return self.build(input_dim, output_dim)

    def fit(
        self,
        datums: list[TrainingDatum],
        *,
        weights_path: Path | None = None,
    ) -> tuple[Any, dict[str, float]]:
        """Incremental fit on a batch of datums; returns ``(model, metrics)``."""
        if not datums:
            raise ValueError("BaselineTrainer.fit requires a non-empty batch")

        batch = TrainingBatch(datums)
        x = batch.state_matrix()
        y = batch.action_matrix()
        model = self.load_or_build(
            input_dim=int(x.shape[1]),
            output_dim=int(y.shape[1]),
            weights_path=weights_path,
        )
        history = model.fit(
            x,
            y,
            epochs=self.epochs,
            batch_size=min(self.batch_size, len(datums)),
            verbose=0,
        )
        metrics: dict[str, float] = {}
        for key, values in history.history.items():
            if values:
                metrics[key] = float(values[-1])
        metrics["n_samples"] = float(len(datums))
        return model, metrics

    @staticmethod
    def save(model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        logger.info("Saved baseline model to %s", path)
