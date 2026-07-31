"""Keras baseline model trainer — fixed-vocab move policy head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingBatch,
    TrainingDatum,
)
from chess_teacher.pipelines.neural_network.move_encoding import POLICY_VOCAB_SIZE
from chess_teacher.utils.logging import get_logger

logger = get_logger()

HEAD_TYPE_POLICY = "policy"


def _masked_sparse_categorical_crossentropy(vocab_size: int = POLICY_VOCAB_SIZE):
    """Build loss: ``y_true`` is ``(batch, V+1)`` = legal mask floats + class index."""
    import tensorflow as tf  # type: ignore[import-untyped]

    def loss_fn(y_true: Any, y_pred: Any) -> Any:
        mask = y_true[:, :vocab_size]
        indices = tf.cast(y_true[:, vocab_size], tf.int32)
        neg_inf = tf.constant(-1.0e9, dtype=y_pred.dtype)
        masked_logits = tf.where(mask > 0.5, y_pred, neg_inf)
        return tf.keras.losses.sparse_categorical_crossentropy(
            indices, masked_logits, from_logits=True
        )

    loss_fn.__name__ = "masked_sparse_ce"
    return loss_fn


def _masked_top_k_accuracy(k: int, vocab_size: int = POLICY_VOCAB_SIZE):
    import tensorflow as tf  # type: ignore[import-untyped]

    def metric_fn(y_true: Any, y_pred: Any) -> Any:
        mask = y_true[:, :vocab_size]
        indices = tf.cast(y_true[:, vocab_size], tf.int32)
        neg_inf = tf.constant(-1.0e9, dtype=y_pred.dtype)
        masked_logits = tf.where(mask > 0.5, y_pred, neg_inf)
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(indices, masked_logits, k=k)

    metric_fn.__name__ = f"masked_top{k}"
    return metric_fn


def pack_policy_targets(y_index: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Pack ``(N,)`` indices + ``(N, V)`` mask into ``(N, V+1)`` float32 for Keras."""
    y_index = np.asarray(y_index, dtype=np.float32).reshape(-1, 1)
    mask = np.asarray(legal_mask, dtype=np.float32)
    return np.concatenate([mask, y_index], axis=1)


def policy_custom_objects(vocab_size: int = POLICY_VOCAB_SIZE) -> dict[str, Any]:
    """Keras ``custom_objects`` for masked policy loss/metrics."""
    return {
        "masked_sparse_ce": _masked_sparse_categorical_crossentropy(vocab_size),
        "masked_top1": _masked_top_k_accuracy(1, vocab_size),
        "masked_top5": _masked_top_k_accuracy(5, vocab_size),
    }


def load_policy_keras(
    path: Path,
    *,
    vocab_size: int = POLICY_VOCAB_SIZE,
    compile_model: bool = False,
) -> Any:
    """Load a ``.keras`` policy model with masked CE custom objects."""
    from tensorflow import keras  # type: ignore[import-untyped]

    return keras.models.load_model(
        path,
        custom_objects=policy_custom_objects(vocab_size),
        compile=compile_model,
    )


def load_policy_from_uri(
    model_uri: str,
    *,
    tracker: Any | None = None,
    vocab_size: int = POLICY_VOCAB_SIZE,
    require_compatible: bool = True,
) -> Any:
    """Download weights via MLflow tracker (if needed) and load a policy Keras model."""
    from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker

    mlflow_tracker = tracker or MLflowTracker()
    weights_path = mlflow_tracker.require_keras_weights(model_uri)
    model = load_policy_keras(weights_path, vocab_size=vocab_size, compile_model=False)
    if require_compatible and not model_is_policy_compatible(model, vocab_size=vocab_size):
        raise ValueError(
            f"Model at {model_uri!r} is not policy-compatible "
            f"(output_shape={getattr(model, 'output_shape', None)}, want V={vocab_size})"
        )
    return model


def model_is_policy_compatible(model: Any, *, vocab_size: int = POLICY_VOCAB_SIZE) -> bool:
    """True when model output dim matches the fixed policy vocab."""
    try:
        shape = model.output_shape
    except Exception:
        return False
    if not shape:
        return False
    # Handle multi-output tuple or single TensorShape
    if isinstance(shape, (list, tuple)) and shape and not isinstance(shape[-1], int):
        # e.g. (None, V)
        last = shape[-1] if not isinstance(shape[0], (list, tuple)) else shape[0][-1]
    else:
        last = shape[-1]
    try:
        return int(last) == int(vocab_size)
    except (TypeError, ValueError):
        return False


class BaselineTrainer:
    """MLP policy head: state → logits[V], masked sparse CE (user-move imitation).

    MSE action-vector path is gone. Parent weights load only when output dim == V.
    """

    DEFAULT_EPOCHS = 3
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_HIDDEN = 128

    def __init__(
        self,
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        hidden: int = DEFAULT_HIDDEN,
        vocab_size: int = POLICY_VOCAB_SIZE,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden = hidden
        self.vocab_size = vocab_size

    def build(self, input_dim: int, output_dim: int | None = None) -> Any:
        """Small MLP: state → policy logits over fixed move vocab."""
        from tensorflow import keras  # type: ignore[import-untyped]

        layers = keras.layers
        out_dim = int(output_dim or self.vocab_size)
        inputs = keras.Input(shape=(input_dim,), name="state")
        x = layers.Dense(self.hidden, activation="relu")(inputs)
        x = layers.Dense(self.hidden, activation="relu")(x)
        outputs = layers.Dense(out_dim, activation="linear", name="policy_logits")(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_move_policy")
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=_masked_sparse_categorical_crossentropy(out_dim),
            metrics=[
                _masked_top_k_accuracy(1, out_dim),
                _masked_top_k_accuracy(5, out_dim),
            ],
        )
        return model

    def load_or_build(
        self,
        *,
        input_dim: int,
        output_dim: int | None = None,
        weights_path: Path | None = None,
    ) -> Any:
        """Load Keras weights if policy-compatible; else cold-start."""
        from tensorflow import keras  # type: ignore[import-untyped]

        out_dim = int(output_dim or self.vocab_size)
        if weights_path is not None and weights_path.is_file():
            logger.info("Loading baseline weights from %s", weights_path)
            try:
                model = load_policy_keras(weights_path, vocab_size=out_dim, compile_model=False)
            except Exception:
                logger.exception("Failed to load baseline weights; cold-starting policy model")
                return self.build(input_dim, out_dim)
            if not model_is_policy_compatible(model, vocab_size=out_dim):
                logger.warning(
                    "Parent weights not policy-compatible (output_shape=%s, want V=%s); "
                    "cold-starting instead of resuming MSE/other head",
                    getattr(model, "output_shape", None),
                    out_dim,
                )
                return self.build(input_dim, out_dim)
            # Re-compile so metrics/loss match current trainer (safe after load).
            model.compile(
                optimizer=keras.optimizers.Adam(1e-3),
                loss=_masked_sparse_categorical_crossentropy(out_dim),
                metrics=[
                    _masked_top_k_accuracy(1, out_dim),
                    _masked_top_k_accuracy(5, out_dim),
                ],
            )
            return model
        logger.info(
            "Cold-start baseline policy model input_dim=%s vocab_size=%s",
            input_dim,
            out_dim,
        )
        return self.build(input_dim, out_dim)

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
        y_index, legal_mask = batch.policy_targets()
        y = pack_policy_targets(y_index, legal_mask)
        model = self.load_or_build(
            input_dim=int(x.shape[1]),
            output_dim=self.vocab_size,
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
        metrics["vocab_size"] = float(self.vocab_size)
        metrics["head_policy"] = 1.0
        return model, metrics

    @staticmethod
    def save(model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        logger.info("Saved baseline model to %s", path)
