"""Deprecated state-vector-only candidate-style trainer and shared utilities.

``BaselineTrainer`` is retained for offline A/B comparisons and older model
artifacts. The active production chessboard state model is
``HybridBoardTrainer`` from ``board_encoder``: board convolutions plus the
state-vector tower and per-candidate scorer. See ``candidate_eval.py`` for
the feature delta convention.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from chess_teacher.pipelines.neural_network.candidate_eval import (
    CANDIDATE_MOVE_FEAT_VERSION,
    MAX_CANDIDATES,
    MOVE_FEAT_DIM,
)
from chess_teacher.pipelines.neural_network.candidate_losses import (
    DEFAULT_LOSS_KIND,
    DEFAULT_SF_MIX_ALPHA,
    DEFAULT_SOFT_TEMPERATURE_PAWNS,
    LossKind,
    candidate_loss_custom_objects,
    masked_candidate_sparse_ce,
    masked_candidate_top_k,
    pack_candidate_targets_for_loss,
    pack_sparse_candidate_targets,
    resolve_candidate_loss,
)
from chess_teacher.pipelines.neural_network.create_training_set import TrainingDatum
from chess_teacher.pipelines.neural_network.ply_weights import (
    DEFAULT_PLY_WEIGHT_CLIP,
    DEFAULT_PLY_WEIGHT_LAMBDA,
    DEFAULT_RECENCY_BOOST,
    baseline_disagree_strength,
    forced_move_downweight_factor,
    normalize_sample_weights,
    ply_weight_raw,
    recency_weight_raw,
    style_disagree_boost_from_env,
    style_disagree_scale_from_env,
    user_not_sf_best_mask,
    user_sf_disagree_strength,
)
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import snapshot_host_pressure

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

# Backward-compatible aliases — E22 implementations live in ``candidate_losses``.
_masked_candidate_sparse_ce = masked_candidate_sparse_ce
_masked_candidate_top_k = masked_candidate_top_k
pack_candidate_targets = pack_sparse_candidate_targets


def candidate_style_custom_objects(max_candidates: int = MAX_CANDIDATES) -> dict[str, Any]:
    return candidate_loss_custom_objects(max_candidates)


def numpy_batches_to_tf_dataset(
    *,
    x_inputs: dict[str, np.ndarray],
    y: np.ndarray,
    sample_weight: np.ndarray,
    batch_size: int,
) -> tuple[Any, int, int]:
    """Build a batched ``tf.data`` dataset from already-materialized NumPy tensors.

    Returns ``(dataset, n_samples, batch_size_used)``. ``from_tensor_slices`` copies
    the full arrays into TensorFlow while the NumPy inputs are still alive.
    ``BaselineTrainer.fit`` does not use this. ``HybridBoardTrainer`` still does.
    """
    tf = _import_tensorflow()
    n = int(y.shape[0])
    if n < 1:
        raise ValueError("numpy_batches_to_tf_dataset requires n_samples >= 1")
    bs = min(int(batch_size), n)
    weights = np.asarray(sample_weight, dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((dict(x_inputs), y, weights))
    ds = ds.batch(bs).prefetch(1)
    return ds, n, bs


def candidate_target_width(loss_kind: str, max_candidates: int) -> int:
    """Width of ``pack_candidate_targets_for_loss`` for ``loss_kind``."""
    kind = str(loss_kind)
    width = int(max_candidates)
    if kind == "sparse":
        return width + 1
    if kind == "soft":
        return 2 * width + 1
    if kind == "sf_mix":
        return width + 2
    raise ValueError(f"unknown loss_kind={loss_kind!r}")


@dataclass
class CandidateStreamScan:
    """Kept rows and per-row raw sample weights. Feature tensors are not kept.

    When ``batch_paths`` is set, each file is one training batch
    (``state``, ``feats``, ``mask``, ``labels``) written during the scan.
    """

    datums: list[TrainingDatum]
    raw_weights: np.ndarray
    end_times: list[datetime | None]
    n_input: int
    n_dropped: int
    n_missing_end: int
    disagree_frac: float
    mean_strength: float
    state_dim: int
    max_candidates: int
    move_feat_dim: int
    batch_paths: list[str]


def _row_training_stats(
    ply: int,
    feats: np.ndarray,
    mask: np.ndarray,
    label: int,
    *,
    style_disagree_boost: float,
    style_disagree_scale: float,
    forced_scale_pawns: float | None,
    lam: float = DEFAULT_PLY_WEIGHT_LAMBDA,
) -> tuple[float, bool, float]:
    """Ply x style x optional forced raw weight, plus disagree stats, for one row.

    Matches the pre-normalize product inside ``candidate_style_sample_weights``.
    The ``(1, MAX, F)`` view is discarded by the caller.
    """
    row_feats = np.asarray(feats, dtype=np.float32)[None, ...]
    row_mask = np.asarray(mask, dtype=np.float32)[None, ...]
    row_label = np.asarray([label], dtype=np.int64)
    strength = float(
        user_sf_disagree_strength(
            row_feats,
            row_label,
            scale_pawns=style_disagree_scale,
        )[0]
    )
    disagree = bool(user_not_sf_best_mask(row_feats, row_label)[0])
    raw = float(ply_weight_raw([ply], lam=lam)[0])
    if style_disagree_boost != 1.0:
        raw *= 1.0 + (style_disagree_boost - 1.0) * strength
    if forced_scale_pawns is not None:
        factor = forced_move_downweight_factor(
            row_feats,
            row_mask,
            scale_pawns=float(forced_scale_pawns),
        )
        raw *= float(factor[0])
    return raw, disagree, strength


def _empty_candidate_scan(n_input: int) -> CandidateStreamScan:
    return CandidateStreamScan(
        datums=[],
        raw_weights=np.zeros((0,), dtype=np.float64),
        end_times=[],
        n_input=n_input,
        n_dropped=n_input,
        n_missing_end=0,
        disagree_frac=0.0,
        mean_strength=0.0,
        state_dim=0,
        max_candidates=0,
        move_feat_dim=0,
        batch_paths=[],
    )


def scan_candidate_stream(
    datums: Sequence[TrainingDatum],
    *,
    style_disagree_boost: float,
    style_disagree_scale: float,
    forced_scale_pawns: float | None = None,
    end_time_by_game_id: Mapping[str, datetime] | None = None,
    lam: float = DEFAULT_PLY_WEIGHT_LAMBDA,
    batch_dir: str | None = None,
    batch_size: int = 64,
) -> CandidateStreamScan:
    """Walk datums once. Keep the row objects and one raw weight each.

    ``candidate_style_target`` runs per row. With ``batch_dir``, each full batch
    is written to disk and the feature arrays are dropped before the next batch.
    Global mean-normalization happens later, on the scalar vector only, so it
    matches a full-tensor ``normalize_sample_weights`` call.
    """
    n_input = len(datums)
    mapping = end_time_by_game_id or {}
    kept: list[TrainingDatum] = []
    raw: list[float] = []
    end_times: list[datetime | None] = []
    n_missing_end = 0
    disagree_n = 0
    strength_sum = 0.0
    state_dim = 0
    max_candidates = 0
    move_feat_dim = 0
    spill = batch_dir is not None
    spill_bs = max(1, int(batch_size))
    batch_paths: list[str] = []
    buf_n = 0
    state_buf: np.ndarray | None = None
    feats_buf: np.ndarray | None = None
    mask_buf: np.ndarray | None = None
    labels_buf: np.ndarray | None = None
    progress_every = max(1, n_input // 5) if n_input else 1
    logger.info(
        "Scanning candidate move features for %s datums "
        "(batch_size=%s spill=%s style_disagree_boost=%s forced_scale_pawns=%s)",
        n_input,
        spill_bs if spill else None,
        spill,
        style_disagree_boost,
        forced_scale_pawns,
    )

    def _flush_batch() -> None:
        nonlocal buf_n
        if not spill or buf_n == 0:
            return
        if (
            state_buf is None
            or feats_buf is None
            or mask_buf is None
            or labels_buf is None
            or batch_dir is None
        ):
            raise RuntimeError("candidate batch buffer was not allocated")
        path = Path(batch_dir) / f"batch-{len(batch_paths):05d}.npz"
        np.savez(
            path,
            state=np.ascontiguousarray(state_buf[:buf_n]),
            feats=np.ascontiguousarray(feats_buf[:buf_n]),
            mask=np.ascontiguousarray(mask_buf[:buf_n]),
            labels=np.ascontiguousarray(labels_buf[:buf_n]),
        )
        batch_paths.append(str(path))
        buf_n = 0

    for i, datum in enumerate(datums):
        packed = datum.candidate_style_target()
        done = i + 1
        if packed is None:
            if done == n_input or done % progress_every == 0:
                logger.info(
                    "Candidate feature progress %s/%s kept=%s",
                    done,
                    n_input,
                    len(kept),
                )
            continue
        feats, mask, label = packed
        row_raw, disagree, strength = _row_training_stats(
            int(datum.ply),
            feats,
            mask,
            int(label),
            style_disagree_boost=float(style_disagree_boost),
            style_disagree_scale=float(style_disagree_scale),
            forced_scale_pawns=forced_scale_pawns,
            lam=lam,
        )
        state_row = np.asarray(datum.state_vector(), dtype=np.float32)
        if not kept:
            state_dim = int(state_row.shape[0])
            max_candidates = int(np.asarray(feats).shape[0])
            move_feat_dim = int(np.asarray(feats).shape[1])
            if spill:
                state_buf = np.empty((spill_bs, state_dim), dtype=np.float32)
                feats_buf = np.empty((spill_bs, max_candidates, move_feat_dim), dtype=np.float32)
                mask_buf = np.empty((spill_bs, max_candidates), dtype=np.float32)
                labels_buf = np.empty((spill_bs,), dtype=np.int32)
        if spill:
            if state_buf is None or feats_buf is None or mask_buf is None or labels_buf is None:
                raise RuntimeError("candidate batch buffer was not allocated")
            state_buf[buf_n] = state_row
            feats_buf[buf_n] = feats
            mask_buf[buf_n] = mask
            labels_buf[buf_n] = int(label)
            buf_n += 1
            if buf_n == spill_bs:
                _flush_batch()
        del feats, mask
        kept.append(datum)
        raw.append(row_raw)
        end = mapping.get(datum.game_id)
        if end is None:
            n_missing_end += 1
            end_times.append(None)
        else:
            end_times.append(end)
        disagree_n += int(disagree)
        strength_sum += strength
        if done == n_input or done % progress_every == 0:
            logger.info(
                "Candidate feature progress %s/%s kept=%s",
                done,
                n_input,
                len(kept),
            )
    _flush_batch()
    n_kept = len(kept)
    if n_kept == 0:
        return _empty_candidate_scan(n_input)
    return CandidateStreamScan(
        datums=kept,
        raw_weights=np.asarray(raw, dtype=np.float64),
        end_times=end_times,
        n_input=n_input,
        n_dropped=n_input - n_kept,
        n_missing_end=n_missing_end,
        disagree_frac=float(disagree_n / n_kept),
        mean_strength=float(strength_sum / n_kept),
        state_dim=state_dim,
        max_candidates=max_candidates,
        move_feat_dim=move_feat_dim,
        batch_paths=batch_paths,
    )


def global_sample_weights(
    raw: np.ndarray,
    *,
    finetune: bool,
    end_times: Sequence[datetime | None] | None = None,
    recency_lambda: float = 0.0,
    recency_boost: float = 1.0,
    baseline_strength: np.ndarray | None = None,
    baseline_disagree_boost: float = 1.0,
    now: datetime | None = None,
) -> np.ndarray:
    """Mean-normalize raw ply x style weights the same way the full-tensor helpers do.

    ``finetune=False`` is ``candidate_style_sample_weights`` (clip, then re-mean).
    ``finetune=True`` is ``user_finetune_sample_weights``: clip the ply x style
    vector, apply recency and baseline-disagree, then mean-normalize with no clip.
    """
    if not finetune:
        return normalize_sample_weights(raw, clip=DEFAULT_PLY_WEIGHT_CLIP)
    styled = normalize_sample_weights(raw, clip=DEFAULT_PLY_WEIGHT_CLIP)
    weights = np.asarray(styled, dtype=np.float64)
    recency_b = float(recency_boost)
    if recency_b != 1.0:
        if end_times is None or now is None:
            raise ValueError("recency_boost != 1.0 requires end_times and now")
        recency = recency_weight_raw(end_times, lam=float(recency_lambda), now=now)
        if recency.shape[0] != weights.shape[0]:
            raise ValueError(
                f"end_times length {recency.shape[0]} != raw length {weights.shape[0]}"
            )
        weights = weights * (1.0 + (recency_b - 1.0) * recency)
    baseline_b = float(baseline_disagree_boost)
    if baseline_b != 1.0:
        if baseline_strength is None:
            raise ValueError("baseline_disagree_boost != 1.0 requires baseline_disagree_strength")
        baseline_s = np.asarray(baseline_strength, dtype=np.float64).reshape(-1)
        if baseline_s.shape[0] != weights.shape[0]:
            raise ValueError(
                f"baseline_disagree_strength length {baseline_s.shape[0]} != N {weights.shape[0]}"
            )
        weights = weights * (1.0 + (baseline_b - 1.0) * baseline_s)
    return normalize_sample_weights(weights, clip=None)


def _load_candidate_batch_file(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        state = np.asarray(data["state"])
        feats = np.asarray(data["feats"])
        mask = np.asarray(data["mask"])
        labels = np.asarray(data["labels"])
    return state, feats, mask, labels


def baseline_disagree_from_batches(
    model: Any,
    batch_paths: Sequence[str],
) -> np.ndarray:
    """Predict baseline argmax disagreement from spilled batch files.

    One file is loaded at a time. Logits are reduced to one strength float per
    row and discarded.
    """
    if not batch_paths:
        return np.zeros((0,), dtype=np.float64)
    parts: list[np.ndarray] = []
    for path in batch_paths:
        state, feats, mask, labels = _load_candidate_batch_file(path)
        logits = np.asarray(
            model.predict({"state": state, "move_feats": feats}, verbose=0),
            dtype=np.float64,
        )
        parts.append(baseline_disagree_strength(logits, mask, labels))
        del state, feats, mask, labels, logits
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=0)


def candidate_batch_dataset(
    batch_paths: Sequence[str],
    *,
    sample_weight: np.ndarray,
    batch_size: int,
    loss_kind: LossKind,
    soft_temperature_pawns: float,
    state_dim: int,
    max_candidates: int,
    move_feat_dim: int,
) -> tuple[Any, int, int]:
    """``tf.data`` dataset that yields spilled batches when training pulls them.

    Loss targets are packed from the batch file at read time. The dataset is one
    finite pass. Callers that train for more than one epoch should ``repeat`` it
    and pass ``steps_per_epoch``; each epoch re-reads the files.
    """
    tf = _import_tensorflow()
    paths = list(batch_paths)
    n_batches = len(paths)
    if n_batches < 1:
        raise ValueError("candidate_batch_dataset requires at least one batch file")
    weights = np.asarray(sample_weight, dtype=np.float32)
    n = int(weights.shape[0])
    if n < 1:
        raise ValueError("candidate_batch_dataset requires n_samples >= 1")
    bs = min(int(batch_size), n)
    y_width = candidate_target_width(str(loss_kind), int(max_candidates))

    def generate() -> Any:
        offset = 0
        for path in paths:
            state, feats, mask, labels = _load_candidate_batch_file(path)
            if (
                int(state.shape[1]) != int(state_dim)
                or int(feats.shape[1]) != int(max_candidates)
                or int(feats.shape[2]) != int(move_feat_dim)
            ):
                raise RuntimeError(
                    "candidate batch shape "
                    f"state={tuple(state.shape)} feats={tuple(feats.shape)} "
                    f"!= expected state_dim={state_dim} "
                    f"max_candidates={max_candidates} feat_dim={move_feat_dim}"
                )
            rows = int(state.shape[0])
            if offset + rows > n:
                raise RuntimeError(
                    f"batch files have more rows than sample weights ({offset + rows} > {n})"
                )
            y = pack_candidate_targets_for_loss(
                loss_kind=loss_kind,
                labels=labels,
                mask=mask,
                move_feats=feats,
                soft_temperature_pawns=soft_temperature_pawns,
            )
            yield (
                {"state": state, "move_feats": feats},
                y,
                weights[offset : offset + rows],
            )
            offset += rows
        if offset != n:
            raise RuntimeError(f"batch files covered {offset} rows, sample weights have {n}")

    signature = (
        {
            "state": tf.TensorSpec(shape=(None, int(state_dim)), dtype=tf.float32),
            "move_feats": tf.TensorSpec(
                shape=(None, int(max_candidates), int(move_feat_dim)),
                dtype=tf.float32,
            ),
        },
        tf.TensorSpec(shape=(None, y_width), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(generate, output_signature=signature)
    dataset = dataset.take(n_batches)
    return dataset, n, bs


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
    """Deprecated state-vector-only tower + per-candidate scorer.

    The active production model is ``HybridBoardTrainer``, which also encodes
    the spatial board state with convolutions. This trainer remains available
    for offline comparisons and legacy state-vector model artifacts.

    Inputs: ``state`` (D,), ``move_feats`` (MAX, F). Output: logits (MAX,).
    Sample weights: ply * SF-style, optional recency and baseline-disagree
    (see ``ply_weights``). ``baseline_disagree_boost`` defaults to 1.0 so
    platform baseline catch-up is unchanged. When boost is on, predict the
    mask from ``baseline_weights_path`` (frozen production baseline) one
    batch at a time and resume ``fit`` from ``weights_path`` (last personal
    checkpoint).

    ``fit`` keeps the datum list and per-row weight scalars. The scan writes
    each feature batch to disk and drops it. TensorFlow reads those files, so
    the full candidate matrix is never resident and later epochs do not repack.
    """

    # Justified 2a pick: 10k registry-val sweep still climbing at 20; peak
    # disagree_t1 on the 3-20 grid. Val only 32 games -- revisit if 2b replay
    # or a larger val set plateaus earlier.
    DEFAULT_EPOCHS = 20
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
        baseline_disagree_boost: float = 1.0,
        recency_boost: float = DEFAULT_RECENCY_BOOST,
        forced_scale_pawns: float | None = None,
        loss_kind: LossKind = DEFAULT_LOSS_KIND,
        soft_temperature_pawns: float = DEFAULT_SOFT_TEMPERATURE_PAWNS,
        sf_mix_alpha: float = DEFAULT_SF_MIX_ALPHA,
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
        self.baseline_disagree_boost = float(baseline_disagree_boost)
        self.recency_boost = float(recency_boost)
        self.forced_scale_pawns = None if forced_scale_pawns is None else float(forced_scale_pawns)
        self.loss_kind: LossKind = str(loss_kind)  # type: ignore[assignment]
        self.soft_temperature_pawns = float(soft_temperature_pawns)
        self.sf_mix_alpha = float(sf_mix_alpha)

    def _loss_fn(self) -> Any:
        return resolve_candidate_loss(
            self.loss_kind,
            max_candidates=self.max_candidates,
            sf_mix_alpha=self.sf_mix_alpha,
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
            loss=self._loss_fn(),
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
        require_compatible_parent: bool = False,
    ) -> Any:
        if require_compatible_parent:
            if weights_path is None or not weights_path.is_file():
                raise FileNotFoundError(
                    "require_compatible_parent=True needs an existing Keras weights file"
                )
            logger.info("Loading required parent weights from %s", weights_path)
            try:
                model = load_candidate_style_keras(
                    weights_path,
                    max_candidates=self.max_candidates,
                    compile_model=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load parent Keras weights from {weights_path}"
                ) from exc
            if not model_is_candidate_style_compatible(
                model,
                max_candidates=self.max_candidates,
                move_feat_dim=self.move_feat_dim,
            ):
                raise RuntimeError(
                    "Parent weights not candidate_style-compatible "
                    f"(output_shape={getattr(model, 'output_shape', None)}, "
                    f"want MAX={self.max_candidates} feat_dim={self.move_feat_dim} "
                    f"/ version={CANDIDATE_MOVE_FEAT_VERSION})"
                )
            from tensorflow import keras  # type: ignore[import-untyped]

            ensure_tensorflow_logging()
            model.compile(
                optimizer=keras.optimizers.Adam(1e-3),
                loss=self._loss_fn(),
                metrics=[
                    _masked_candidate_top_k(1, self.max_candidates),
                    _masked_candidate_top_k(3, self.max_candidates),
                ],
            )
            return model
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
                loss=self._loss_fn(),
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
        baseline_weights_path: Path | None = None,
        recency_lambda: float | None = None,
        end_time_by_game_id: Mapping[str, datetime] | None = None,
        now: datetime | None = None,
        require_parent_weights: bool = False,
    ) -> tuple[Any, dict[str, float]]:
        if not datums:
            raise ValueError("BaselineTrainer.fit requires a non-empty batch")

        use_baseline = self.baseline_disagree_boost != 1.0
        use_recency = recency_lambda is not None
        if use_baseline and baseline_weights_path is None:
            raise ValueError("baseline_disagree_boost != 1.0 requires baseline_weights_path")

        n_input = len(datums)
        logger.info(
            "Building candidate move features for %s datums "
            "(SF evals from DB + on-the-fly geometry/material/openness; feat_dim=%s) %s",
            n_input,
            self.move_feat_dim,
            snapshot_host_pressure().format_fields(),
        )
        # Forced downweight is part of candidate_style_sample_weights only.
        # The finetune helper does not take forced_scale_pawns.
        dataset: Any = None
        batch_dir = tempfile.mkdtemp(prefix="ct-fit-")
        try:
            scan = scan_candidate_stream(
                datums,
                style_disagree_boost=self.style_disagree_boost,
                style_disagree_scale=self.style_disagree_scale,
                forced_scale_pawns=None
                if (use_recency or use_baseline)
                else self.forced_scale_pawns,
                end_time_by_game_id=end_time_by_game_id,
                batch_dir=batch_dir,
                batch_size=self.batch_size,
            )
            n_kept = len(scan.datums)
            n_dropped = scan.n_dropped
            if n_kept < 1:
                raise ValueError(
                    "BaselineTrainer.fit: no datums with usable candidate_evaluations "
                    "(user move must be in evals)"
                )
            if (
                scan.max_candidates != self.max_candidates
                or scan.move_feat_dim != self.move_feat_dim
            ):
                raise ValueError(
                    "packed candidate shape "
                    f"(max_candidates={scan.max_candidates}, feat_dim={scan.move_feat_dim}) "
                    f"!= trainer (max_candidates={self.max_candidates}, "
                    f"feat_dim={self.move_feat_dim})"
                )
            # Feature batches are on disk. Drop our kept-row list; the caller may
            # still hold the original 10k datums.
            batch_paths = list(scan.batch_paths)
            state_dim = scan.state_dim
            raw_weights = scan.raw_weights
            end_times = scan.end_times
            n_missing_end = scan.n_missing_end
            disagree_frac = scan.disagree_frac
            mean_strength = scan.mean_strength
            del scan
            del datums
            gc.collect()
            logger.info(
                "Candidate feature scan kept=%s dropped=%s batch_files=%s "
                "(full move tensor not retained) %s",
                n_kept,
                n_dropped,
                len(batch_paths),
                snapshot_host_pressure().format_fields(),
            )

            model = None
            baseline_strength = None
            if use_baseline:
                baseline_model = self.load_or_build(
                    input_dim=state_dim,
                    weights_path=baseline_weights_path,
                    require_compatible_parent=True,
                )
                baseline_strength = baseline_disagree_from_batches(baseline_model, batch_paths)
                if weights_path is not None and weights_path == baseline_weights_path:
                    model = baseline_model
                gc.collect()

            if use_recency and n_missing_end:
                logger.warning(
                    "recency: %s/%s kept datums missing end_time; strength=0",
                    n_missing_end,
                    n_kept,
                )
            recency_now = now if now is not None else get_current_datetime()
            sample_w = global_sample_weights(
                raw_weights,
                finetune=use_recency or use_baseline,
                end_times=end_times,
                recency_lambda=float(recency_lambda) if recency_lambda is not None else 0.0,
                recency_boost=self.recency_boost if use_recency else 1.0,
                baseline_strength=baseline_strength,
                baseline_disagree_boost=self.baseline_disagree_boost,
                now=recency_now,
            )
            del raw_weights, end_times, baseline_strength
            gc.collect()

            if model is None:
                model = self.load_or_build(
                    input_dim=state_dim,
                    weights_path=weights_path,
                    require_compatible_parent=require_parent_weights,
                )
            fit_started = snapshot_host_pressure()
            logger.info(
                "Starting Keras fit samples=%s epochs=%s batch_size=%s "
                "style_disagree_boost=%s scale_pawns=%s forced_scale_pawns=%s "
                "disagree_frac=%.3f mean_strength=%.3f %s",
                n_kept,
                self.epochs,
                min(self.batch_size, n_kept),
                self.style_disagree_boost,
                self.style_disagree_scale,
                self.forced_scale_pawns,
                disagree_frac,
                mean_strength,
                fit_started.format_fields(),
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

            fit_t0 = time.monotonic()
            dataset, n_ds, bs_used = candidate_batch_dataset(
                batch_paths,
                sample_weight=sample_w,
                batch_size=self.batch_size,
                loss_kind=self.loss_kind,
                soft_temperature_pawns=self.soft_temperature_pawns,
                state_dim=state_dim,
                max_candidates=self.max_candidates,
                move_feat_dim=self.move_feat_dim,
            )
            del sample_w
            gc.collect()
            n_batches = len(batch_paths)
            if self.epochs > 1:
                dataset = dataset.repeat()
            dataset = dataset.prefetch(1)
            logger.info(
                "Streaming Keras fit via generated batches n_samples=%s "
                "batch_size=%s epochs=%s steps_per_epoch=%s batch_files=%s %s",
                n_ds,
                bs_used,
                self.epochs,
                n_batches,
                n_batches,
                snapshot_host_pressure().format_fields(),
            )
            history = model.fit(
                dataset,
                epochs=self.epochs,
                steps_per_epoch=n_batches,
                verbose=0,
                callbacks=[_EpochInfoCallback()],
            )
        finally:
            del dataset
            gc.collect()
            shutil.rmtree(batch_dir, ignore_errors=True)
        metrics: dict[str, float] = {}
        for key, values in history.history.items():
            if values:
                metrics[key] = float(values[-1])
        metrics["n_samples"] = float(n_kept)
        metrics["n_dropped_missing_candidates"] = float(n_dropped)
        metrics["max_candidates"] = float(self.max_candidates)
        metrics["move_feat_dim"] = float(self.move_feat_dim)
        metrics["move_feat_version"] = float(CANDIDATE_MOVE_FEAT_VERSION)
        metrics["head_candidate_style"] = 1.0
        metrics["style_disagree_boost"] = float(self.style_disagree_boost)
        metrics["style_disagree_scale"] = float(self.style_disagree_scale)
        metrics["baseline_disagree_boost"] = float(self.baseline_disagree_boost)
        metrics["sf_disagree_frac"] = disagree_frac
        metrics["sf_disagree_mean_strength"] = mean_strength
        metrics["epochs"] = float(self.epochs)
        metrics["sf_mix_alpha"] = float(self.sf_mix_alpha)
        if recency_lambda is not None:
            metrics["recency_lambda"] = float(recency_lambda)
            metrics["recency_boost"] = float(self.recency_boost)
        fit_ended = snapshot_host_pressure()
        logger.info(
            "Keras fit finished duration_s=%.2f delta_rss_mb=%.1f n_samples=%s %s",
            time.monotonic() - fit_t0,
            fit_ended.rss_mb - fit_started.rss_mb,
            n_kept,
            fit_ended.format_fields(),
        )
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
