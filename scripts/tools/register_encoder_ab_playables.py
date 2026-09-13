"""Upload Phase 2c encoder A/B hybrid checkpoints as archived Play presets.

Does **not** touch production. Inserts/updates ``ml.baseline_models`` with
``status=archived`` so Streamlit Play can load them (requires hybrid-aware
``NeuralBaselineBot`` board feed).

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/register_encoder_ab_playables.py ^
      --run-dir storage/tmp/encoder_ab_fuse64_r3_ep10 --rounds 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from chess_teacher.bots.presets import invalidate_baseline_presets_cache, list_baseline_presets
from chess_teacher.pipelines.neural_network.board_tensor import (
    BOARD_TENSOR_CHANNELS,
    BOARD_TENSOR_VERSION,
)
from chess_teacher.pipelines.neural_network.candidate_eval import MAX_CANDIDATES, MOVE_FEAT_DIM
from chess_teacher.pipelines.neural_network.models import BaselineModel, BaselineModelStatus
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.object_storage.factory import get_raw_storage, s3_url_string
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

logger = get_logger()

_S3_PREFIX = "experiments/encoder_ab_dev"

# Fallback metrics if val_curve.jsonl is missing (older fuse64 A/B).
_HYBRID_METRICS: dict[int, dict[str, float]] = {
    1: {
        "top1_overall": 0.3829,
        "top3_overall": 0.6414,
        "top1_sf_agree": 0.6561,
        "top1_sf_disagree": 0.2145,
        "n_eval": 45027.0,
    },
    2: {
        "top1_overall": 0.4088,
        "top3_overall": 0.6630,
        "top1_sf_agree": 0.7177,
        "top1_sf_disagree": 0.2184,
        "n_eval": 45027.0,
    },
    3: {
        "top1_overall": 0.4102,
        "top3_overall": 0.6628,
        "top1_sf_agree": 0.7154,
        "top1_sf_disagree": 0.2221,
        "n_eval": 45027.0,
    },
}


def _read_curve_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    path = run_dir / "val_curve.jsonl"
    if not path.is_file():
        return {}
    out: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        blob = json.loads(line)
        if not isinstance(blob, dict) or "round" not in blob:
            continue
        round_i = int(blob["round"])
        metrics = blob.get("hybrid")
        if not isinstance(metrics, dict):
            metrics = blob.get("forced_on")
        if not isinstance(metrics, dict):
            metrics = blob.get("metrics")
        if not isinstance(metrics, dict):
            continue
        picked: dict[str, float] = {}
        for key in (
            "top1_overall",
            "top3_overall",
            "top1_sf_agree",
            "top1_sf_disagree",
            "top3_sf_disagree",
            "n_eval",
        ):
            if key in metrics:
                picked[key] = float(metrics[key])
        if picked:
            out[round_i] = picked
    return out


def _version_for(run_name: str, round_i: int, *, version_override: str | None) -> str:
    if version_override:
        return version_override
    short = run_name.replace("encoder_ab_", "").replace("_", "")[:24]
    return f"v_hyb_{short}_r{round_i}"


def _metrics_blob(
    round_i: int,
    *,
    run_name: str,
    curve_metrics: dict[int, dict[str, float]],
    label_round: int | None,
) -> dict[str, float | str | int]:
    blob: dict[str, float | str | int] = {
        "head_candidate_style": 1.0,
        "move_feat_dim": float(MOVE_FEAT_DIM),
        "max_candidates": float(MAX_CANDIDATES),
        "encoder_hybrid_board": 1.0,
        "encoder_fuses_state": 1.0,
        "board_tensor_version": float(BOARD_TENSOR_VERSION),
        "board_channels": float(BOARD_TENSOR_CHANNELS),
        "conv_filters": 64.0,
        "source": f"encoder_ab_{run_name}",
        "round": int(label_round if label_round is not None else round_i),
        "epochs": 10,
    }
    blob.update(_HYBRID_METRICS.get(round_i, {}))
    blob.update(curve_metrics.get(round_i, {}))
    return blob


def register_hybrid_rounds(
    *,
    run_dir: Path,
    rounds: list[int],
    overwrite_s3: bool,
    version_override: str | None = None,
    label_round: int | None = None,
    checkpoint_name: str = "hybrid.keras",
) -> list[str]:
    run_dir = run_dir.resolve()
    run_name = run_dir.name
    curve_metrics = _read_curve_metrics(run_dir)
    storage = get_raw_storage()
    db = get_db_client()
    existing = {row.version: row for row in BaselineModel.fetch_all_ordered(db)}
    registered: list[str] = []

    for round_i in rounds:
        local = run_dir / f"round_{round_i}" / checkpoint_name
        if not local.is_file():
            raise FileNotFoundError(f"missing checkpoint: {local}")

        rel_key = f"{_S3_PREFIX}/{run_name}/round_{round_i}/{checkpoint_name}"
        storage.write_bytes(rel_key, local.read_bytes(), overwrite=overwrite_s3)
        model_uri = s3_url_string(rel_key)
        version = _version_for(run_name, round_i, version_override=version_override)
        metrics_json = json.dumps(
            _metrics_blob(
                round_i,
                run_name=run_name,
                curve_metrics=curve_metrics,
                label_round=label_round,
            )
        )

        if version in existing:
            updated = replace(
                existing[version],
                model_uri=model_uri,
                status=BaselineModelStatus.ARCHIVED,
                eval_metrics=metrics_json,
            )
            updated.save_to_db(db)
            logger.info("Updated archived hybrid version=%s uri=%s", version, model_uri)
        else:
            new_row = BaselineModel(
                id=BaselineModel.generate_id({"version": version}),
                version=version,
                trained_at=get_current_datetime(),
                model_uri=model_uri,
                status=BaselineModelStatus.ARCHIVED,
                eval_metrics=metrics_json,
                git_commit_hash=BaselineModel.current_git_commit(),
            )
            inserted = new_row.save_new_to_db(db)
            if not inserted:
                logger.warning("Insert skipped (conflict) version=%s", version)
            else:
                logger.info("Inserted archived hybrid version=%s uri=%s", version, model_uri)
        registered.append(version)

    invalidate_baseline_presets_cache()
    return registered


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("storage/tmp/encoder_ab_fuse64_r3_ep10"),
        help="A/B output dir with round_N/hybrid.keras",
    )
    p.add_argument(
        "--rounds",
        type=str,
        default="3",
        help="Comma rounds to register (default: 3 = latest catch-up).",
    )
    p.add_argument(
        "--version",
        type=str,
        default=None,
        help="Optional exact ml.baseline_models.version (single-round register).",
    )
    p.add_argument(
        "--label-round",
        type=int,
        default=None,
        help="Optional overall round number stored in eval_metrics.",
    )
    p.add_argument("--overwrite-s3", action="store_true")
    p.add_argument(
        "--checkpoint-name",
        type=str,
        default="hybrid.keras",
        help="Filename under round_N/ (hybrid.keras or forced_on.keras).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    rounds = sorted({int(x.strip()) for x in str(args.rounds).split(",") if x.strip()})
    if not rounds:
        logger.error("No rounds specified")
        return 1
    if args.version and len(rounds) != 1:
        logger.error("--version requires exactly one --rounds value")
        return 1
    logger.info(
        "Registering hybrid run=%s rounds=%s checkpoint=%s (archived only)",
        args.run_dir,
        rounds,
        args.checkpoint_name,
    )
    versions = register_hybrid_rounds(
        run_dir=Path(args.run_dir),
        rounds=rounds,
        overwrite_s3=bool(args.overwrite_s3),
        version_override=str(args.version) if args.version else None,
        label_round=int(args.label_round) if args.label_round is not None else None,
        checkpoint_name=str(args.checkpoint_name),
    )
    db = get_db_client()
    presets = list_baseline_presets(db, force_refresh=True)
    keys = [p.key for p in presets if p.key.startswith("baseline:v_hyb_")]
    print(f"registered versions={versions}")
    print(f"play baseline hybrid keys now visible={keys}")
    return 0


if __name__ == "__main__":
    log_script_runtime_context(logger, script="register_encoder_ab_playables")
    run_script_main(main)
