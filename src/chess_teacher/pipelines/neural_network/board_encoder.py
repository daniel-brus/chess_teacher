"""Hybrid board-encoder + flat state + candidate scorer (Phase 2c offline).

Board conv trunk is an **addition** to the baseline flat ``state`` tower (not a
replacement). Both embeddings fuse before the shared candidate head.
Does **not** wire production entrypoints.

POC: intended *successor candidate* for ``BaselineTrainer`` if registry-val
A/B wins — still disposable with the rest of ``neural_network`` until Phase 4
promotes a greenfield design. Not production-wired yet.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from chess_teacher.pipelines.neural_network.board_tensor import (
    BOARD_TENSOR_CHANNELS,
    BOARD_TENSOR_SHAPE,
    BOARD_TENSOR_VERSION,
    pack_board_tensors,
)
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
    DEFAULT_RECENCY_BOOST,
    baseline_disagree_strength,
    candidate_style_sample_weights,
    style_disagree_boost_from_env,
    style_disagree_scale_from_env,
    user_finetune_sample_weights,
    user_not_sf_best_mask,
    user_sf_disagree_strength,
)
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.pipelines.neural_network.train import (
    BaselineTrainer,
    _masked_candidate_sparse_ce,
    _masked_candidate_top_k,
    candidate_style_custom_objects,
    pack_candidate_targets,
)
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import snapshot_host_pressure

logger = get_logger()
ensure_tensorflow_logging()


def _import_keras():
    ensure_tensorflow_logging()
    from tensorflow import keras  # type: ignore[import-untyped]

    ensure_tensorflow_logging()
    return keras


def load_hybrid_board_keras(
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


def model_is_hybrid_board_compatible(
    model: Any,
    *,
    max_candidates: int = MAX_CANDIDATES,
    move_feat_dim: int = MOVE_FEAT_DIM,
    board_channels: int = BOARD_TENSOR_CHANNELS,
) -> bool:
    """True when inputs include board, state, and move_feats."""
    try:
        shape = model.output_shape
        last = shape[-1] if not isinstance(shape[0], (list, tuple)) else shape[0][-1]
        if int(last) != int(max_candidates):
            return False
    except Exception:
        return False

    try:
        inputs = model.inputs
    except Exception:
        return False
    if not inputs or len(inputs) < 3:
        return False

    board_in = None
    state_in = None
    feats_in = None
    for inp in inputs:
        name = (getattr(inp, "name", "") or "").split(":")[0]
        if name == "board" or name.startswith("board"):
            board_in = inp
        elif name == "state" or name.startswith("state"):
            state_in = inp
        elif name == "move_feats" or "move_feats" in name:
            feats_in = inp
    if board_in is None or state_in is None or feats_in is None:
        return False
    try:
        b_shape = tuple(board_in.shape)
        f_shape = tuple(feats_in.shape)
        return (
            int(b_shape[-1]) == int(board_channels)
            and int(b_shape[-2]) == 8
            and int(b_shape[-3]) == 8
            and int(f_shape[-1]) == int(move_feat_dim)
            and int(f_shape[-2]) == int(max_candidates)
        )
    except (TypeError, ValueError, IndexError):
        return False


class HybridBoardTrainer:
    """Board conv + flat state tower fused, then candidate scorer (offline Phase 2c).

    State tower matches ``BaselineTrainer`` widths; conv trunk is additive.
    ``DEFAULT_CONV_FILTERS`` bumped vs first A/B (32 → 64).

    POC / intended successor for flat-state-only ``BaselineTrainer`` if A/B wins.
    Still offline-only; package remains deletable until Phase 4 greenfield.
    """

    DEFAULT_EPOCHS = BaselineTrainer.DEFAULT_EPOCHS
    DEFAULT_BATCH_SIZE = BaselineTrainer.DEFAULT_BATCH_SIZE
    DEFAULT_HIDDEN = BaselineTrainer.DEFAULT_HIDDEN
    DEFAULT_SCORE_HIDDEN = BaselineTrainer.DEFAULT_SCORE_HIDDEN
    DEFAULT_CONV_FILTERS = 64

    def __init__(
        self,
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        hidden: int = DEFAULT_HIDDEN,
        score_hidden: int = DEFAULT_SCORE_HIDDEN,
        conv_filters: int = DEFAULT_CONV_FILTERS,
        max_candidates: int = MAX_CANDIDATES,
        move_feat_dim: int = MOVE_FEAT_DIM,
        style_disagree_boost: float | None = None,
        style_disagree_scale: float | None = None,
        baseline_disagree_boost: float = 1.0,
        recency_boost: float = DEFAULT_RECENCY_BOOST,
        forced_scale_pawns: float | None = None,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden = hidden
        self.score_hidden = score_hidden
        self.conv_filters = conv_filters
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
        self.forced_scale_pawns = (
            None if forced_scale_pawns is None else float(forced_scale_pawns)
        )

    def build(self, state_dim: int) -> Any:
        keras = _import_keras()
        layers = keras.layers

        board_in = keras.Input(shape=BOARD_TENSOR_SHAPE, name="board")
        state_in = keras.Input(shape=(state_dim,), name="state")
        feats_in = keras.Input(
            shape=(self.max_candidates, self.move_feat_dim),
            name="move_feats",
        )

        # Spatial trunk (additive geometry).
        x = layers.Conv2D(
            self.conv_filters,
            3,
            padding="same",
            activation="relu",
            name="board_conv1",
        )(board_in)
        x = layers.Conv2D(
            self.conv_filters,
            3,
            padding="same",
            activation="relu",
            name="board_conv2",
        )(x)
        x = layers.GlobalAveragePooling2D(name="board_gap")(x)
        board_emb = layers.Dense(self.hidden, activation="relu", name="board_emb")(x)

        # Same flat-state tower as BaselineTrainer.
        s = layers.Dense(self.hidden, activation="relu", name="state_h1")(state_in)
        state_emb = layers.Dense(self.hidden, activation="relu", name="state_h2")(s)

        fused = layers.Concatenate(axis=-1, name="board_state_concat")([board_emb, state_emb])
        h = layers.Dense(self.hidden, activation="relu", name="fused_emb")(fused)
        h_tile = layers.RepeatVector(self.max_candidates, name="fused_tile")(h)
        scored_in = layers.Concatenate(axis=-1, name="fused_move_concat")([h_tile, feats_in])
        scored = layers.TimeDistributed(
            layers.Dense(self.score_hidden, activation="relu"),
            name="score_h",
        )(scored_in)
        scores = layers.TimeDistributed(
            layers.Dense(1, activation="linear"),
            name="score_out",
        )(scored)
        logits = layers.Reshape((self.max_candidates,), name="candidate_logits")(scores)

        model = keras.Model(
            inputs=[board_in, state_in, feats_in],
            outputs=logits,
            name="baseline_hybrid_board",
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
        state_dim: int,
        weights_path: Path | None = None,
        require_compatible_parent: bool = False,
    ) -> Any:
        if require_compatible_parent:
            if weights_path is None or not weights_path.is_file():
                raise FileNotFoundError(
                    "require_compatible_parent=True needs an existing Keras weights file"
                )
            logger.info("Loading required hybrid parent weights from %s", weights_path)
            try:
                model = load_hybrid_board_keras(
                    weights_path,
                    max_candidates=self.max_candidates,
                    compile_model=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load parent hybrid Keras weights from {weights_path}"
                ) from exc
            if not model_is_hybrid_board_compatible(
                model,
                max_candidates=self.max_candidates,
                move_feat_dim=self.move_feat_dim,
            ):
                raise RuntimeError(
                    "Parent weights not hybrid board+state compatible "
                    f"(output_shape={getattr(model, 'output_shape', None)})"
                )
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
        if weights_path is not None and weights_path.is_file():
            logger.info("Loading hybrid board weights from %s", weights_path)
            try:
                model = load_hybrid_board_keras(
                    weights_path,
                    max_candidates=self.max_candidates,
                    compile_model=False,
                )
            except Exception:
                logger.exception(
                    "Failed to load hybrid weights; cold-starting hybrid board+state model"
                )
                return self.build(state_dim)
            if not model_is_hybrid_board_compatible(
                model,
                max_candidates=self.max_candidates,
                move_feat_dim=self.move_feat_dim,
            ):
                logger.warning(
                    "Parent weights not hybrid board+state compatible; cold-starting"
                )
                return self.build(state_dim)
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
            "Cold-start hybrid board+state model state_dim=%s conv_filters=%s "
            "max_candidates=%s feat_dim=%s",
            state_dim,
            self.conv_filters,
            self.max_candidates,
            self.move_feat_dim,
        )
        return self.build(state_dim)

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
            raise ValueError("HybridBoardTrainer.fit requires a non-empty batch")

        logger.info(
            "Hybrid board+state encoder: packing candidates for %s datums "
            "(board_tensor_version=%s C=%s conv_filters=%s). %s",
            len(datums),
            BOARD_TENSOR_VERSION,
            BOARD_TENSOR_CHANNELS,
            self.conv_filters,
            snapshot_host_pressure().format_fields(),
        )
        batch = TrainingBatch(datums)
        feats, mask, labels, kept = batch.candidate_style_targets()
        if not kept:
            raise ValueError(
                "HybridBoardTrainer.fit: no datums with usable candidate_evaluations"
            )
        kept_datums = [datums[i] for i in kept]
        x_board = pack_board_tensors(kept_datums)
        x_state = TrainingBatch(kept_datums).state_matrix()
        y = pack_candidate_targets(labels, mask)
        disagree_mask = user_not_sf_best_mask(feats, labels)
        strength = user_sf_disagree_strength(feats, labels, scale_pawns=self.style_disagree_scale)
        plies = [d.ply for d in kept_datums]
        mapping = end_time_by_game_id or {}
        aligned: list[datetime | None] = []
        missing = 0
        for d in kept_datums:
            end = mapping.get(d.game_id)
            if end is None:
                missing += 1
                aligned.append(None)
            else:
                aligned.append(end)

        use_baseline = self.baseline_disagree_boost != 1.0
        use_recency = recency_lambda is not None
        model = None
        baseline_strength = None
        if use_baseline:
            if baseline_weights_path is None:
                raise ValueError("baseline_disagree_boost != 1.0 requires baseline_weights_path")
            baseline_model = self.load_or_build(
                state_dim=int(x_state.shape[1]),
                weights_path=baseline_weights_path,
                require_compatible_parent=True,
            )
            logits = np.asarray(
                baseline_model.predict(
                    {"board": x_board, "state": x_state, "move_feats": feats},
                    verbose=0,
                ),
                dtype=np.float64,
            )
            baseline_strength = baseline_disagree_strength(logits, mask, labels)
            if weights_path is not None and weights_path == baseline_weights_path:
                model = baseline_model

        if use_recency or use_baseline:
            if use_recency and missing:
                logger.warning(
                    "recency: %s/%s kept datums missing end_time; strength=0",
                    missing,
                    len(kept_datums),
                )
            recency_now = now if now is not None else get_current_datetime()
            sample_w = user_finetune_sample_weights(
                plies,
                feats,
                labels,
                aligned,
                recency_lambda=float(recency_lambda) if use_recency else 0.0,
                recency_boost=self.recency_boost if use_recency else 1.0,
                style_disagree_boost=self.style_disagree_boost,
                style_disagree_scale=self.style_disagree_scale,
                baseline_disagree_strength=baseline_strength,
                baseline_disagree_boost=self.baseline_disagree_boost,
                now=recency_now,
            )
        else:
            sample_w = candidate_style_sample_weights(
                plies,
                feats,
                labels,
                style_disagree_boost=self.style_disagree_boost,
                style_disagree_scale=self.style_disagree_scale,
                candidate_mask=mask,
                forced_scale_pawns=self.forced_scale_pawns,
            )
        disagree_frac = float(np.mean(disagree_mask))
        mean_strength = float(np.mean(strength))

        if model is None:
            model = self.load_or_build(
                state_dim=int(x_state.shape[1]),
                weights_path=weights_path,
                require_compatible_parent=require_parent_weights,
            )
        fit_started = snapshot_host_pressure()
        logger.info(
            "Starting hybrid Keras fit samples=%s epochs=%s batch_size=%s "
            "conv_filters=%s style_disagree_boost=%s scale_pawns=%s "
            "forced_scale_pawns=%s disagree_frac=%.3f mean_strength=%.3f "
            "parent=%s %s",
            len(kept_datums),
            self.epochs,
            min(self.batch_size, len(kept_datums)),
            self.conv_filters,
            self.style_disagree_boost,
            self.style_disagree_scale,
            self.forced_scale_pawns,
            disagree_frac,
            mean_strength,
            weights_path,
            fit_started.format_fields(),
        )
        total_epochs = self.epochs
        from tensorflow.keras.callbacks import Callback  # type: ignore[import-untyped]

        class _EpochInfoCallback(Callback):
            def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
                logger.info(
                    "Hybrid Keras epoch %s/%s metrics=%s",
                    epoch + 1,
                    total_epochs,
                    {k: round(float(v), 6) for k, v in (logs or {}).items()},
                )

        fit_t0 = time.monotonic()
        history = model.fit(
            {"board": x_board, "state": x_state, "move_feats": feats},
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
        metrics["board_tensor_version"] = float(BOARD_TENSOR_VERSION)
        metrics["board_channels"] = float(BOARD_TENSOR_CHANNELS)
        metrics["state_dim"] = float(x_state.shape[1])
        metrics["conv_filters"] = float(self.conv_filters)
        metrics["head_candidate_style"] = 1.0
        metrics["encoder_hybrid_board"] = 1.0
        metrics["encoder_fuses_state"] = 1.0
        metrics["style_disagree_boost"] = float(self.style_disagree_boost)
        metrics["style_disagree_scale"] = float(self.style_disagree_scale)
        metrics["baseline_disagree_boost"] = float(self.baseline_disagree_boost)
        metrics["sf_disagree_frac"] = disagree_frac
        metrics["sf_disagree_mean_strength"] = mean_strength
        metrics["epochs"] = float(self.epochs)
        if self.forced_scale_pawns is not None:
            metrics["forced_scale_pawns"] = float(self.forced_scale_pawns)
        if recency_lambda is not None:
            metrics["recency_lambda"] = float(recency_lambda)
            metrics["recency_boost"] = float(self.recency_boost)
        fit_ended = snapshot_host_pressure()
        logger.info(
            "Hybrid Keras fit finished duration_s=%.2f delta_rss_mb=%.1f n_samples=%s %s",
            time.monotonic() - fit_t0,
            fit_ended.rss_mb - fit_started.rss_mb,
            len(kept_datums),
            fit_ended.format_fields(),
        )
        return model, metrics

    @staticmethod
    def save(model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        logger.info("Saved hybrid board model to %s", path)
