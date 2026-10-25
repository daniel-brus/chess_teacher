"""Small Maia-shaped policy: current-position board in, 8x8x73 policy out.

No Stockfish channels and no candidate MLP. The trunk is six residual blocks
at 64 filters. Training is masked cross-entropy on the played move. Scores for
the architecture comparison are gathered back onto the same 128 candidate
slots the hybrid scorer uses, so top-1, agree, disagree, and pawn gap match.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from chess_teacher.pipelines.neural_network.board_tensor import (
    BOARD_TENSOR_CHANNELS,
    BOARD_TENSOR_SHAPE,
    pack_board_tensors,
)
from chess_teacher.pipelines.neural_network.candidate_eval import (
    MAX_CANDIDATES,
    capped_candidate_rows,
    parse_candidate_evaluations,
)
from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum
from chess_teacher.pipelines.neural_network.eval_metrics import (
    EvalMetrics,
    compute_candidate_style_metrics,
)
from chess_teacher.pipelines.neural_network.move_encoding import POLICY_VOCAB_SIZE
from chess_teacher.pipelines.neural_network.oriented_policy import (
    ILLEGAL_LOGIT,
    datum_board_flip,
    oriented_legal_mask,
    oriented_policy_index,
)
from chess_teacher.pipelines.neural_network.ply_weights import (
    style_disagree_boost_from_env,
    style_disagree_scale_from_env,
)
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.pipelines.neural_network.train import (
    global_sample_weights,
    scan_candidate_stream,
)
from chess_teacher.utils.logging import get_logger

logger = get_logger()


def _import_keras():
    ensure_tensorflow_logging()
    from tensorflow import keras  # type: ignore[import-untyped]

    ensure_tensorflow_logging()
    return keras


def _import_tf():
    ensure_tensorflow_logging()
    import tensorflow as tf  # type: ignore[import-untyped]

    ensure_tensorflow_logging()
    return tf


def _residual_block(x: Any, filters: int, name: str) -> Any:
    keras = _import_keras()
    layers = keras.layers
    y = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)
    y = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2",
    )(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = layers.Add(name=f"{name}_add")([x, y])
    return layers.ReLU(name=f"{name}_out")(y)


def masked_candidate_log_loss(
    logits: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Mean cross-entropy of the played candidate slot. Same 128-way space for both arms."""
    scores = np.asarray(logits, dtype=np.float64)
    legal = np.asarray(mask, dtype=np.float64) > 0.5
    y = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or y.shape[0] != scores.shape[0]:
        raise ValueError(f"logits {scores.shape} do not match labels {y.shape}")
    masked = np.where(legal, scores, ILLEGAL_LOGIT)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=1)
    chosen = exp[np.arange(y.shape[0]), y]
    probs = np.clip(chosen / np.maximum(denom, 1e-12), 1e-12, 1.0)
    return float(-np.mean(np.log(probs)))


def _rows_for_datum(
    datum: TrainingDatum,
    *,
    max_candidates: int,
) -> list[tuple[str, float, float]] | None:
    payload = parse_candidate_evaluations(datum.candidate_evaluations)
    if payload is None:
        return None
    return capped_candidate_rows(
        payload["evals_white_pov"],
        color_is_white=not datum_board_flip(datum.color),
        user_move_uci=datum.move_uci,
        legal_ucis=datum.legal_move_ucis,
        max_candidates=max_candidates,
    )


def gather_candidate_logits(
    policy_logits: np.ndarray,
    datums: list[TrainingDatum],
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> np.ndarray:
    """Copy oriented policy logits into the candidate-slot layout.

    Slot ``i`` is the same UCI ``pack_candidate_tensors`` writes at ``i``.
    Missing or unsupported moves stay at the illegal fill.
    """
    flat = np.asarray(policy_logits, dtype=np.float64)
    if flat.ndim != 2 or flat.shape[1] != POLICY_VOCAB_SIZE:
        raise ValueError(f"Expected policy logits (N, {POLICY_VOCAB_SIZE}), got {flat.shape}")
    if flat.shape[0] != len(datums):
        raise ValueError(f"logit rows {flat.shape[0]} != datums {len(datums)}")
    out = np.full((len(datums), max_candidates), ILLEGAL_LOGIT, dtype=np.float64)
    for i, datum in enumerate(datums):
        rows = _rows_for_datum(datum, max_candidates=max_candidates)
        if rows is None:
            continue
        flip = datum_board_flip(datum.color)
        for slot, (uci, _ev, _delta) in enumerate(rows):
            try:
                index = oriented_policy_index(uci, flip=flip)
            except ValueError:
                continue
            out[i, slot] = flat[i, index]
    return out


class SpatialPolicyTrainer:
    """Cold-start spatial policy. Board planes only; masked human-move CE."""

    DEFAULT_FILTERS = 64
    DEFAULT_BLOCKS = 6
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_LEARNING_RATE = 1e-3

    def __init__(
        self,
        *,
        epochs: int = 2,
        batch_size: int = DEFAULT_BATCH_SIZE,
        filters: int = DEFAULT_FILTERS,
        blocks: int = DEFAULT_BLOCKS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_candidates: int = MAX_CANDIDATES,
        style_disagree_boost: float | None = None,
        style_disagree_scale: float | None = None,
    ) -> None:
        if blocks < 1:
            raise ValueError(f"blocks must be >= 1, got {blocks}")
        if filters < 1:
            raise ValueError(f"filters must be >= 1, got {filters}")
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.filters = int(filters)
        self.blocks = int(blocks)
        self.learning_rate = float(learning_rate)
        self.max_candidates = int(max_candidates)
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

    def build(self) -> Any:
        keras = _import_keras()
        layers = keras.layers
        board_in = keras.Input(shape=BOARD_TENSOR_SHAPE, name="board")
        mask_in = keras.Input(shape=(POLICY_VOCAB_SIZE,), name="legal_mask")

        x = layers.Conv2D(self.filters, 3, padding="same", use_bias=False, name="stem_conv")(
            board_in
        )
        x = layers.BatchNormalization(name="stem_bn")(x)
        x = layers.ReLU(name="stem_relu")(x)
        for i in range(self.blocks):
            x = _residual_block(x, self.filters, f"res{i}")
        policy = layers.Conv2D(73, 1, name="policy_conv")(x)
        logits = layers.Reshape((POLICY_VOCAB_SIZE,), name="policy_logits")(policy)

        def _apply_mask(inputs: list[Any]) -> Any:
            raw, legal = inputs
            return raw + (1.0 - legal) * ILLEGAL_LOGIT

        masked = layers.Lambda(_apply_mask, name="masked_policy_logits")([logits, mask_in])
        model = keras.Model(inputs=[board_in, mask_in], outputs=masked, name="spatial_policy")
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        return model

    def fit(self, datums: list[TrainingDatum]) -> tuple[Any, dict[str, Any]]:
        """Train on datums that have usable candidate evals. Cold start only."""
        if not datums:
            raise ValueError("SpatialPolicyTrainer.fit requires a non-empty batch")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")

        scan = scan_candidate_stream(
            datums,
            style_disagree_boost=self.style_disagree_boost,
            style_disagree_scale=self.style_disagree_scale,
            batch_dir=None,
            include_board=False,
        )
        kept = list(scan.datums)
        if not kept:
            raise ValueError(
                "SpatialPolicyTrainer.fit: no datums with usable candidate_evaluations"
            )
        sample_w = global_sample_weights(scan.raw_weights, finetune=False).astype(np.float32)
        model = self.build()
        batch_size = max(1, min(self.batch_size, len(kept)))
        dataset = self._dataset(kept, sample_w, batch_size=batch_size)
        logger.info(
            "Starting spatial policy fit samples=%s epochs=%s filters=%s blocks=%s "
            "batch_size=%s lr=%s style_disagree_boost=%s",
            len(kept),
            self.epochs,
            self.filters,
            self.blocks,
            batch_size,
            self.learning_rate,
            self.style_disagree_boost,
        )
        started = time.monotonic()
        history = model.fit(dataset, epochs=self.epochs, verbose=0)
        elapsed = time.monotonic() - started
        epoch_losses = [float(v) for v in history.history.get("loss", [])]
        metrics: dict[str, Any] = {
            "n_samples": float(len(kept)),
            "n_dropped_missing_candidates": float(scan.n_dropped),
            "epochs": float(self.epochs),
            "loss": float(epoch_losses[-1]) if epoch_losses else float("nan"),
            "epoch_losses": epoch_losses,
            "param_count": int(model.count_params()),
            "fit_seconds": float(elapsed),
            "filters": float(self.filters),
            "blocks": float(self.blocks),
            "learning_rate": float(self.learning_rate),
            "style_disagree_boost": float(self.style_disagree_boost),
            "style_disagree_scale": float(self.style_disagree_scale),
            "policy_vocab": float(POLICY_VOCAB_SIZE),
            "board_channels": float(BOARD_TENSOR_CHANNELS),
        }
        logger.info(
            "Spatial policy fit finished duration_s=%.2f n_samples=%s last_loss=%.4f",
            elapsed,
            len(kept),
            metrics["loss"],
        )
        return model, metrics

    def _dataset(
        self, datums: list[TrainingDatum], sample_w: np.ndarray, *, batch_size: int
    ) -> Any:
        tf = _import_tf()

        def generate() -> Iterator[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]]:
            for start in range(0, len(datums), batch_size):
                end = min(len(datums), start + batch_size)
                chunk = datums[start:end]
                yield _batch_arrays(chunk), _batch_labels(chunk), sample_w[start:end]

        return tf.data.Dataset.from_generator(
            generate,
            output_signature=(
                {
                    "board": tf.TensorSpec(
                        shape=(None, 8, 8, BOARD_TENSOR_CHANNELS),
                        dtype=tf.float32,
                    ),
                    "legal_mask": tf.TensorSpec(
                        shape=(None, POLICY_VOCAB_SIZE),
                        dtype=tf.float32,
                    ),
                },
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ),
        )


def _batch_arrays(datums: list[TrainingDatum]) -> dict[str, np.ndarray]:
    boards = pack_board_tensors(datums)
    masks = np.stack([
        oriented_legal_mask(d.legal_move_ucis, flip=datum_board_flip(d.color)) for d in datums
    ])
    return {"board": boards, "legal_mask": masks}


def _batch_labels(datums: list[TrainingDatum]) -> np.ndarray:
    return np.asarray(
        [oriented_policy_index(d.move_uci, flip=datum_board_flip(d.color)) for d in datums],
        dtype=np.int32,
    )


def predict_policy_logits(
    model: Any,
    datums: list[TrainingDatum],
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Policy logits ``(N, 4672)`` with illegal moves already filled."""
    if not datums:
        return np.zeros((0, POLICY_VOCAB_SIZE), dtype=np.float64)
    parts: list[np.ndarray] = []
    step = max(1, int(batch_size))
    for start in range(0, len(datums), step):
        chunk = datums[start : start + step]
        pred = model.predict(_batch_arrays(chunk), verbose=0)
        parts.append(np.asarray(pred, dtype=np.float64))
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=0)


def score_spatial_datums(
    model: Any,
    datums: list[TrainingDatum],
    *,
    batch_size: int = 64,
    max_candidates: int = MAX_CANDIDATES,
) -> tuple[EvalMetrics, float, np.ndarray, np.ndarray, np.ndarray]:
    """Candidate-slot metrics plus 128-way log-loss for datums with evals.

    Returns ``(metrics, log_loss, logits, mask, labels)``.
    """
    from chess_teacher.pipelines.neural_network.create_training_set import TrainingBatch

    batch = TrainingBatch(datums)
    feats, mask, labels, kept_idx = batch.candidate_style_targets()
    if not kept_idx:
        raise ValueError("score_spatial_datums: no datums with usable candidate_evaluations")
    kept = [datums[i] for i in kept_idx]
    policy = predict_policy_logits(model, kept, batch_size=batch_size)
    logits = gather_candidate_logits(policy, kept, max_candidates=max_candidates)
    metrics = compute_candidate_style_metrics(
        logits=logits,
        mask=mask,
        labels=labels,
        move_feats=feats,
        plies=[d.ply for d in kept],
        n_input=len(datums),
        max_candidates=max_candidates,
    )
    log_loss = masked_candidate_log_loss(logits, mask, labels)
    return metrics, log_loss, logits, mask, labels
