"""Keras baseline trainer — candidate-aware style scorer (SF eval features per move).

Replaces the fixed-vocab policy head. Parent weights load only when compatible with
``head=candidate_style`` (state tower + per-candidate scorer). See
``candidate_eval.py`` for delta convention.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_VERSION,
    MAX_CANDIDATES,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingBatch,
    TrainingDatum,
)
from chess_teacher.pipelines.neural_network.ply_weights import (
    candidate_style_sample_weights,
    style_disagree_boost_from_env,
    style_disagree_scale_from_env,
    user_not_sf_best_mask,
    user_sf_disagree_strength,
)
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.logging import get_logger

logger = get_logger()

# Before any lazy ``import tensorflow`` in this module (and usually before other
# call sites that import train first).
ensure_tensorflow_logging()


def _import_tensorflow():
    """Import TF after quieting C++ STDERR, then re-wire Python loggers."""
    ensure_tensorflow_logging()
    import tensorflow as tf  # type: ignore[import-untyped]

    ensure_tensorflow_logging()
    return tf


def _import_keras():
    ensure_tensorflow_logging()
    from tensorflow import keras  # type: ignore[import-untyped]

    ensure_tensorflow_logging()
    return keras


# Re-export for pipeline / MLflow params.
HEAD_TYPE_POLICY = "policy"  # legacy marker only; trainer no longer builds this head.


def _masked_candidate_sparse_ce(max_candidates: int = MAX_CANDIDATES):
    """``y_true`` is ``(batch, MAX+1)`` = candidate mask floats + class index."""
    tf = _import_tensorflow()

    def loss_fn(y_true: Any, y_pred: Any) -> Any:
        mask = y_true[:, :max_candidates]
        indices = tf.cast(y_true[:, max_candidates], tf.int32)
        neg_inf = tf.constant(-1.0e9, dtype=y_pred.dtype)
        masked_logits = tf.where(mask > 0.5, y_pred, neg_inf)
        return tf.keras.losses.sparse_categorical_crossentropy(
            indices, masked_logits, from_logits=True
        )

    loss_fn.__name__ = "masked_candidate_sparse_ce"
    return loss_fn


def _masked_candidate_top_k(k: int, max_candidates: int = MAX_CANDIDATES):
    tf = _import_tensorflow()

    def metric_fn(y_true: Any, y_pred: Any) -> Any:
        mask = y_true[:, :max_candidates]
        indices = tf.cast(y_true[:, max_candidates], tf.int32)
        neg_inf = tf.constant(-1.0e9, dtype=y_pred.dtype)
        masked_logits = tf.where(mask > 0.5, y_pred, neg_inf)
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(indices, masked_logits, k=k)

    metric_fn.__name__ = f"masked_cand_top{k}"
    return metric_fn


def pack_candidate_targets(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pack ``(N,)`` labels + ``(N, MAX)`` mask into ``(N, MAX+1)`` float32."""
    y_index = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    m = np.asarray(mask, dtype=np.float32)
    return np.concatenate([m, y_index], axis=1)


def candidate_style_custom_objects(max_candidates: int = MAX_CANDIDATES) -> dict[str, Any]:
    return {
        "masked_candidate_sparse_ce": _masked_candidate_sparse_ce(max_candidates),
        "masked_cand_top1": _masked_candidate_top_k(1, max_candidates),
        "masked_cand_top3": _masked_candidate_top_k(3, max_candidates),
    }


def load_candidate_style_keras(
    path: Path,
    *,
    max_candidates: int = MAX_CANDIDATES,
    compile_model: bool = False,
) -> Any:
    keras = _import_keras()

    return keras.models.load_model(
        path,
        custom_objects=candidate_style_custom_objects(max_candidates),
        compile=compile_model,
    )


def model_is_candidate_style_compatible(
    model: Any,
    *,
    max_candidates: int = MAX_CANDIDATES,
    move_feat_dim: int = MOVE_FEAT_DIM,
) -> bool:
    """True when outputs MAX slots and ``move_feats`` input has current feat dim."""
    try:
        shape = model.output_shape
    except Exception:
        return False
    if not shape:
        return False
    if isinstance(shape, (list, tuple)) and shape and not isinstance(shape[-1], int):
        last = shape[-1] if not isinstance(shape[0], (list, tuple)) else shape[0][-1]
    else:
        last = shape[-1]
    try:
        if int(last) != int(max_candidates):
            return False
    except (TypeError, ValueError):
        return False

    # Input: list of tensors; find move_feats by name or second input shape.
    try:
        inputs = model.inputs
    except Exception:
        return False
    if not inputs:
        return False
    feat_input = None
    for inp in inputs:
        name = getattr(inp, "name", "") or ""
        if "move_feats" in name:
            feat_input = inp
            break
    if feat_input is None and len(inputs) >= 2:
        feat_input = inputs[1]
    if feat_input is None:
        return False
    try:
        in_shape = tuple(feat_input.shape)
        # (None, MAX, F)
        return int(in_shape[-1]) == int(move_feat_dim) and int(in_shape[-2]) == int(max_candidates)
    except (TypeError, ValueError, IndexError):
        return False


_LOADED_CANDIDATE_STYLE_MODELS: dict[tuple[str, int], Any] = {}


def clear_candidate_style_model_cache() -> None:
    """Drop in-process Keras models (tests / memory pressure)."""
    _LOADED_CANDIDATE_STYLE_MODELS.clear()


def load_candidate_style_from_uri(
    model_uri: str,
    *,
    tracker: Any | None = None,
    max_candidates: int = MAX_CANDIDATES,
    require_compatible: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> Any:
    progress = on_progress or (lambda _message: None)
    cache_key = (model_uri, int(max_candidates))
    cached = _LOADED_CANDIDATE_STYLE_MODELS.get(cache_key)
    if cached is not None:
        progress("Using cached TensorFlow model…")
        return cached

    from chess_teacher.pipelines.neural_network.mlflow_utils import MLflowTracker

    progress("Fetching model weights from storage…")
    mlflow_tracker = tracker or MLflowTracker()
    weights_path = mlflow_tracker.require_keras_weights(model_uri)
    progress("Loading model into TensorFlow…")
    model = load_candidate_style_keras(
        weights_path, max_candidates=max_candidates, compile_model=False
    )
    if require_compatible and not model_is_candidate_style_compatible(
        model, max_candidates=max_candidates, move_feat_dim=MOVE_FEAT_DIM
    ):
        raise ValueError(
            f"Model at {model_uri!r} is not candidate_style-compatible "
            f"(output_shape={getattr(model, 'output_shape', None)}, "
            f"want MAX={max_candidates} feat_dim={MOVE_FEAT_DIM})"
        )
    _LOADED_CANDIDATE_STYLE_MODELS[cache_key] = model
    return model


class BaselineTrainer:
    """Shared state tower + per-candidate MLP scorer; listwise masked CE.

    Inputs: ``state`` (D,), ``move_feats`` (MAX, F). Output: logits (MAX,).
    Sample weights: ply * continuous SF-disagree style boost (see ``ply_weights``).
    """

    DEFAULT_EPOCHS = 8
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_HIDDEN = 128
    DEFAULT_SCORE_HIDDEN = 64

    def __init__(
        self,
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        hidden: int = DEFAULT_HIDDEN,
        score_hidden: int = DEFAULT_SCORE_HIDDEN,
        max_candidates: int = MAX_CANDIDATES,
        move_feat_dim: int = MOVE_FEAT_DIM,
        style_disagree_boost: float | None = None,
        style_disagree_scale: float | None = None,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden = hidden
        self.score_hidden = score_hidden
        self.max_candidates = max_candidates
        self.move_feat_dim = move_feat_dim
        self.style_disagree_boost = (
            style_disagree_boost_from_env()
            if style_disagree_boost is None
            else float(style_disagree_boost)
        )
        self.style_disagree_scale = (
            style_disagree_scale_from_env()
            if style_disagree_scale is None
            else float(style_disagree_scale)
        )

    def build(self, input_dim: int) -> Any:
        keras = _import_keras()

        layers = keras.layers
        state_in = keras.Input(shape=(input_dim,), name="state")
        feats_in = keras.Input(
            shape=(self.max_candidates, self.move_feat_dim),
            name="move_feats",
        )

        h = layers.Dense(self.hidden, activation="relu", name="state_h1")(state_in)
        h = layers.Dense(self.hidden, activation="relu", name="state_h2")(h)
        h_tile = layers.RepeatVector(self.max_candidates, name="state_tile")(h)
        x = layers.Concatenate(axis=-1, name="state_move_concat")([h_tile, feats_in])
        x = layers.TimeDistributed(
            layers.Dense(self.score_hidden, activation="relu"),
            name="score_h",
        )(x)
        scores = layers.TimeDistributed(
            layers.Dense(1, activation="linear"),
            name="score_out",
        )(x)
        logits = layers.Reshape((self.max_candidates,), name="candidate_logits")(scores)

        model = keras.Model(
            inputs=[state_in, feats_in],
            outputs=logits,
            name="baseline_candidate_style",
        )
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=_masked_candidate_sparse_ce(self.max_candidates),
            metrics=[
                _masked_candidate_top_k(1, self.max_candidates),
                _masked_candidate_top_k(3, self.max_candidates),
            ],
        )
        return model

    def load_or_build(
        self,
        *,
        input_dim: int,
        weights_path: Path | None = None,
    ) -> Any:
        if weights_path is not None and weights_path.is_file():
            logger.info("Loading baseline weights from %s", weights_path)
            try:
                model = load_candidate_style_keras(
                    weights_path,
                    max_candidates=self.max_candidates,
                    compile_model=False,
                )
            except Exception:
                logger.exception(
                    "Failed to load baseline weights; cold-starting candidate_style model"
                )
                return self.build(input_dim)
            if not model_is_candidate_style_compatible(
                model,
                max_candidates=self.max_candidates,
                move_feat_dim=self.move_feat_dim,
            ):
                logger.warning(
                    "Parent weights not candidate_style-compatible "
                    "(output_shape=%s, want MAX=%s feat_dim=%s / version=%s); "
                    "cold-starting instead of resuming old feat layout",
                    getattr(model, "output_shape", None),
                    self.max_candidates,
                    self.move_feat_dim,
                    CANDIDATE_MOVE_FEAT_VERSION,
                )
                return self.build(input_dim)
            from tensorflow import keras  # type: ignore[import-untyped]

            ensure_tensorflow_logging()
            model.compile(
                optimizer=keras.optimizers.Adam(1e-3),
                loss=_masked_candidate_sparse_ce(self.max_candidates),
                metrics=[
                    _masked_candidate_top_k(1, self.max_candidates),
                    _masked_candidate_top_k(3, self.max_candidates),
                ],
            )
            return model
        logger.info(
            "Cold-start candidate_style model input_dim=%s max_candidates=%s feat_dim=%s",
            input_dim,
            self.max_candidates,
            self.move_feat_dim,
        )
        return self.build(input_dim)

    def fit(
        self,
        datums: list[TrainingDatum],
        *,
        weights_path: Path | None = None,
    ) -> tuple[Any, dict[str, float]]:
        if not datums:
            raise ValueError("BaselineTrainer.fit requires a non-empty batch")

        logger.info(
            "Building candidate move features for %s datums "
            "(SF evals from DB + on-the-fly geometry/material/openness; feat_dim=%s)…",
            len(datums),
            self.move_feat_dim,
        )
        batch = TrainingBatch(datums)
        feats, mask, labels, kept = batch.candidate_style_targets()
        if not kept:
            raise ValueError(
                "BaselineTrainer.fit: no datums with usable candidate_evaluations "
                "(user move must be in evals)"
            )
        kept_datums = [datums[i] for i in kept]
        logger.info(
            "Candidate features ready kept=%s dropped=%s; building state matrix…",
            len(kept_datums),
            len(datums) - len(kept_datums),
        )
        x_state = TrainingBatch(kept_datums).state_matrix()
        y = pack_candidate_targets(labels, mask)
        disagree_mask = user_not_sf_best_mask(feats, labels)
        strength = user_sf_disagree_strength(feats, labels, scale_pawns=self.style_disagree_scale)
        sample_w = candidate_style_sample_weights(
            [d.ply for d in kept_datums],
            feats,
            labels,
            style_disagree_boost=self.style_disagree_boost,
            style_disagree_scale=self.style_disagree_scale,
        )
        disagree_frac = float(np.mean(disagree_mask))
        mean_strength = float(np.mean(strength))

        model = self.load_or_build(
            input_dim=int(x_state.shape[1]),
            weights_path=weights_path,
        )
        logger.info(
            "Starting Keras fit samples=%s epochs=%s batch_size=%s "
            "style_disagree_boost=%s scale_pawns=%s disagree_frac=%.3f "
            "mean_strength=%.3f…",
            len(kept_datums),
            self.epochs,
            min(self.batch_size, len(kept_datums)),
            self.style_disagree_boost,
            self.style_disagree_scale,
            disagree_frac,
            mean_strength,
        )
        total_epochs = self.epochs
        from tensorflow.keras.callbacks import Callback  # type: ignore[import-untyped]

        class _EpochInfoCallback(Callback):
            def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
                logger.info(
                    "Keras epoch %s/%s metrics=%s",
                    epoch + 1,
                    total_epochs,
                    {k: round(float(v), 6) for k, v in (logs or {}).items()},
                )

        # Prefer our logger over Keras STDERR progress bars.
        history = model.fit(
            {"state": x_state, "move_feats": feats},
            y,
            sample_weight=sample_w,
            epochs=self.epochs,
            batch_size=min(self.batch_size, len(kept_datums)),
            verbose=0,
            callbacks=[_EpochInfoCallback()],
        )
        metrics: dict[str, float] = {}
        for key, values in history.history.items():
            if values:
                metrics[key] = float(values[-1])
        metrics["n_samples"] = float(len(kept_datums))
        metrics["n_dropped_missing_candidates"] = float(len(datums) - len(kept_datums))
        metrics["max_candidates"] = float(self.max_candidates)
        metrics["move_feat_dim"] = float(self.move_feat_dim)
        metrics["move_feat_version"] = float(CANDIDATE_MOVE_FEAT_VERSION)
        metrics["head_candidate_style"] = 1.0
        metrics["style_disagree_boost"] = float(self.style_disagree_boost)
        metrics["style_disagree_scale"] = float(self.style_disagree_scale)
        metrics["sf_disagree_frac"] = disagree_frac
        metrics["sf_disagree_mean_strength"] = mean_strength
        metrics["epochs"] = float(self.epochs)
        return model, metrics

    @staticmethod
    def save(model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        logger.info("Saved baseline model to %s", path)


# Back-compat aliases used by older call sites / tests.
def load_policy_keras(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "Policy head removed; use load_candidate_style_keras / load_candidate_style_from_uri"
    )


def load_policy_from_uri(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("Policy head removed; use load_candidate_style_from_uri")


def model_is_policy_compatible(*args: Any, **kwargs: Any) -> bool:
    return False


def policy_custom_objects(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return candidate_style_custom_objects()
