"""Active hybrid board-state encoder + candidate scorer.

The production model combines a board convolution trunk with the flat state
tower; both embeddings fuse before the shared candidate head. It replaces the
deprecated state-vector-only ``BaselineTrainer`` for platform and personal
training and live chessboard scoring.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from chess_teacher.pipelines.neural_network.board_tensor import (
    BOARD_TENSOR_CHANNELS,
    BOARD_TENSOR_SHAPE,
    BOARD_TENSOR_VERSION,
)
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
    masked_candidate_top_k,
    resolve_candidate_loss,
)
from chess_teacher.pipelines.neural_network.create_training_set import (
    TrainingDatum,
)
from chess_teacher.pipelines.neural_network.ply_weights import (
    DEFAULT_RECENCY_BOOST,
    style_disagree_boost_from_env,
    style_disagree_scale_from_env,
)
from chess_teacher.pipelines.neural_network.tf_runtime import ensure_tensorflow_logging
from chess_teacher.pipelines.neural_network.train import (
    BaselineTrainer,
    baseline_disagree_from_batches,
    candidate_batch_dataset,
    candidate_style_custom_objects,
    global_sample_weights,
    scan_candidate_stream,
)
from chess_teacher.pipelines.neural_network.training_slot import (
    FitMemoryError,
    fit_host_has_memory,
    min_fit_mem_available_mb,
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
    """Active board-convolution + flat-state candidate scorer.

    State tower matches ``BaselineTrainer`` widths; conv trunk is additive.
    ``DEFAULT_CONV_FILTERS`` bumped vs first A/B (32 → 64).
    Training features are spilled to bounded-size disk batches and streamed into
    TensorFlow instead of materializing the full move tensor in RAM.

    Used to train production platform and personal models. The flat-state-only
    ``BaselineTrainer`` is deprecated and remains only for offline comparisons
    and legacy artifacts.

    A cold start trains for ``DEFAULT_EPOCHS``. A warm start, where parent
    weights actually load, trains for ``WARM_START_EPOCHS``. Pass ``epochs``
    to use that count for either path.
    """

    DEFAULT_EPOCHS = BaselineTrainer.DEFAULT_EPOCHS
    WARM_START_EPOCHS = 10
    DEFAULT_BATCH_SIZE = BaselineTrainer.DEFAULT_BATCH_SIZE
    DEFAULT_HIDDEN = BaselineTrainer.DEFAULT_HIDDEN
    DEFAULT_SCORE_HIDDEN = BaselineTrainer.DEFAULT_SCORE_HIDDEN
    DEFAULT_CONV_FILTERS = 64

    def __init__(
        self,
        *,
        epochs: int | None = None,
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
        loss_kind: LossKind = DEFAULT_LOSS_KIND,
        soft_temperature_pawns: float = DEFAULT_SOFT_TEMPERATURE_PAWNS,
        sf_mix_alpha: float = DEFAULT_SF_MIX_ALPHA,
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
            loss=self._loss_fn(),
            metrics=[
                masked_candidate_top_k(1, self.max_candidates),
                masked_candidate_top_k(3, self.max_candidates),
            ],
        )
        return model

    def resolve_epochs(self, *, cold_start: bool) -> int:
        """Epochs for this fit. An explicit ``epochs`` value wins either way."""
        if self.epochs is not None:
            return self.epochs
        if cold_start:
            return self.DEFAULT_EPOCHS
        return self.WARM_START_EPOCHS

    def describe_epochs(self) -> str:
        """Log label. The real count is chosen once fit knows if weights loaded."""
        if self.epochs is not None:
            return str(self.epochs)
        return f"{self.WARM_START_EPOCHS} warm-start / {self.DEFAULT_EPOCHS} cold-start"

    def load_or_build(
        self,
        *,
        state_dim: int,
        weights_path: Path | None = None,
        require_compatible_parent: bool = False,
    ) -> tuple[Any, bool]:
        """Return ``(model, cold_started)``.

        ``cold_started`` is false only when compatible hybrid weights were loaded.
        """
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
                loss=self._loss_fn(),
                metrics=[
                    masked_candidate_top_k(1, self.max_candidates),
                    masked_candidate_top_k(3, self.max_candidates),
                ],
            )
            return model, False
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
                return self.build(state_dim), True
            if not model_is_hybrid_board_compatible(
                model,
                max_candidates=self.max_candidates,
                move_feat_dim=self.move_feat_dim,
            ):
                logger.warning("Parent weights not hybrid board+state compatible; cold-starting")
                return self.build(state_dim), True
            from tensorflow import keras  # type: ignore[import-untyped]

            ensure_tensorflow_logging()
            model.compile(
                optimizer=keras.optimizers.Adam(1e-3),
                loss=self._loss_fn(),
                metrics=[
                    masked_candidate_top_k(1, self.max_candidates),
                    masked_candidate_top_k(3, self.max_candidates),
                ],
            )
            return model, False
        logger.info(
            "Cold-start hybrid board+state model state_dim=%s conv_filters=%s "
            "max_candidates=%s feat_dim=%s",
            state_dim,
            self.conv_filters,
            self.max_candidates,
            self.move_feat_dim,
        )
        return self.build(state_dim), True

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
            "Hybrid board+state encoder: scanning candidates for %s datums "
            "(board_tensor_version=%s C=%s conv_filters=%s). %s",
            len(datums),
            BOARD_TENSOR_VERSION,
            BOARD_TENSOR_CHANNELS,
            self.conv_filters,
            snapshot_host_pressure().format_fields(),
        )
        use_baseline = self.baseline_disagree_boost != 1.0
        use_recency = recency_lambda is not None
        if use_baseline and baseline_weights_path is None:
            raise ValueError("baseline_disagree_boost != 1.0 requires baseline_weights_path")

        dataset: Any = None
        batch_dir = tempfile.mkdtemp(prefix="ct-hybrid-fit-")
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
                include_board=True,
            )
            n_kept = len(scan.datums)
            n_dropped = scan.n_dropped
            if n_kept < 1:
                raise ValueError(
                    "HybridBoardTrainer.fit: no datums with usable candidate_evaluations"
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

            batch_paths = list(scan.batch_paths)
            state_dim = scan.state_dim
            raw_weights = scan.raw_weights
            end_times = scan.end_times
            missing = scan.n_missing_end
            disagree_frac = scan.disagree_frac
            mean_strength = scan.mean_strength
            del scan, datums
            gc.collect()
            logger.info(
                "Hybrid candidate scan kept=%s dropped=%s batch_files=%s "
                "(board + move tensors spilled to disk) %s",
                n_kept,
                n_dropped,
                len(batch_paths),
                snapshot_host_pressure().format_fields(),
            )

            model = None
            cold_started = True
            baseline_strength = None
            if use_baseline:
                baseline_model, _baseline_cold = self.load_or_build(
                    state_dim=state_dim,
                    weights_path=baseline_weights_path,
                    require_compatible_parent=True,
                )
                baseline_strength = baseline_disagree_from_batches(
                    baseline_model,
                    batch_paths,
                    include_board=True,
                )
                if weights_path is not None and weights_path == baseline_weights_path:
                    model = baseline_model
                    cold_started = False
                gc.collect()

            if use_recency and missing:
                logger.warning(
                    "recency: %s/%s kept datums missing end_time; strength=0",
                    missing,
                    n_kept,
                )
            sample_w = global_sample_weights(
                raw_weights,
                finetune=use_recency or use_baseline,
                end_times=end_times,
                recency_lambda=float(recency_lambda) if recency_lambda is not None else 0.0,
                recency_boost=self.recency_boost if use_recency else 1.0,
                baseline_strength=baseline_strength,
                baseline_disagree_boost=self.baseline_disagree_boost,
                now=now if now is not None else get_current_datetime(),
            )
            del raw_weights, end_times, baseline_strength
            gc.collect()

            if model is None:
                model, cold_started = self.load_or_build(
                    state_dim=state_dim,
                    weights_path=weights_path,
                    require_compatible_parent=require_parent_weights,
                )
            epochs = self.resolve_epochs(cold_start=cold_started)
            fit_started = snapshot_host_pressure()
            logger.info(
                "Starting hybrid Keras fit samples=%s epochs=%s cold_start=%s batch_size=%s "
                "conv_filters=%s style_disagree_boost=%s scale_pawns=%s "
                "forced_scale_pawns=%s disagree_frac=%.3f mean_strength=%.3f "
                "parent=%s %s",
                n_kept,
                epochs,
                cold_started,
                min(self.batch_size, n_kept),
                self.conv_filters,
                self.style_disagree_boost,
                self.style_disagree_scale,
                self.forced_scale_pawns,
                disagree_frac,
                mean_strength,
                weights_path,
                fit_started.format_fields(),
            )
            fit_t0 = time.monotonic()
            n_batches = len(batch_paths)
            logger.info(
                "Streaming hybrid Keras fit via generated batches n_samples=%s "
                "batch_size=%s epochs=%s steps_per_epoch=%s batch_files=%s %s",
                n_kept,
                min(self.batch_size, n_kept),
                epochs,
                n_batches,
                n_batches,
                snapshot_host_pressure().format_fields(),
            )
            # One epoch at a time so a growing allocator can be collected, and so
            # a tight host stops before the kernel OOM-kills the pod.
            completed = 0
            history = None
            for _epoch_index in range(epochs):
                if not fit_host_has_memory():
                    if completed == 0:
                        raise FitMemoryError(
                            "Refusing hybrid fit: MemAvailable is below "
                            f"{min_fit_mem_available_mb():.0f} MB after the model loaded."
                        )
                    logger.warning(
                        "Stopping hybrid fit after %s/%s epochs to avoid an OOM kill.",
                        completed,
                        epochs,
                    )
                    break
                dataset, _n_ds, _bs_used = candidate_batch_dataset(
                    batch_paths,
                    sample_weight=sample_w,
                    batch_size=self.batch_size,
                    loss_kind=self.loss_kind,
                    soft_temperature_pawns=self.soft_temperature_pawns,
                    state_dim=state_dim,
                    max_candidates=self.max_candidates,
                    move_feat_dim=self.move_feat_dim,
                    include_board=True,
                )
                dataset = dataset.prefetch(1)
                history = model.fit(
                    dataset,
                    epochs=1,
                    steps_per_epoch=n_batches,
                    verbose=0,
                )
                completed += 1
                epoch_logs = {
                    key: round(float(value[-1]), 6)
                    for key, value in history.history.items()
                    if value
                }
                logger.info(
                    "Hybrid Keras epoch %s/%s metrics=%s",
                    completed,
                    epochs,
                    epoch_logs,
                )
                del dataset
                dataset = None
                gc.collect()
            del sample_w
            gc.collect()
        finally:
            del dataset
            gc.collect()
            shutil.rmtree(batch_dir, ignore_errors=True)
        if history is None:
            raise RuntimeError("hybrid fit finished without a Keras history")
        metrics: dict[str, float] = {}
        for key, values in history.history.items():
            if values:
                metrics[key] = float(values[-1])
        metrics["n_samples"] = float(n_kept)
        metrics["n_dropped_missing_candidates"] = float(n_dropped)
        metrics["max_candidates"] = float(self.max_candidates)
        metrics["move_feat_dim"] = float(self.move_feat_dim)
        metrics["move_feat_version"] = float(CANDIDATE_MOVE_FEAT_VERSION)
        metrics["board_tensor_version"] = float(BOARD_TENSOR_VERSION)
        metrics["board_channels"] = float(BOARD_TENSOR_CHANNELS)
        metrics["state_dim"] = float(state_dim)
        metrics["conv_filters"] = float(self.conv_filters)
        metrics["head_candidate_style"] = 1.0
        metrics["encoder_hybrid_board"] = 1.0
        metrics["encoder_fuses_state"] = 1.0
        metrics["style_disagree_boost"] = float(self.style_disagree_boost)
        metrics["style_disagree_scale"] = float(self.style_disagree_scale)
        metrics["baseline_disagree_boost"] = float(self.baseline_disagree_boost)
        metrics["sf_disagree_frac"] = disagree_frac
        metrics["sf_disagree_mean_strength"] = mean_strength
        metrics["epochs"] = float(completed)
        metrics["sf_mix_alpha"] = float(self.sf_mix_alpha)
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
            n_kept,
            fit_ended.format_fields(),
        )
        return model, metrics

    @staticmethod
    def save(model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        logger.info("Saved hybrid board model to %s", path)
