"""Upload Phase 3a hybrid parent + user FT checkpoints as archived Play presets.

Does **not** touch production. Inserts/updates ``ml.baseline_models`` with
``status=archived`` so Streamlit Play can load them under Baseline.

Product note (Phase 4): personal bot will be keyed to the **current user**,
combining all linked accounts. These archived rows are a local Play bridge
only (per-artifact labels), not the final Personal UX.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/register_phase3a_playables.py ^
      --overwrite-s3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
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

_S3_PREFIX = "experiments/phase3a_dev"
_DEFAULT_DIR = Path("storage/tmp/phase3a")


@dataclass(frozen=True)
class Phase3aArtifact:
    filename: str
    version: str
    label: str
    eval_metrics: dict[str, float | str | int]


# Val metrics from overnight FT logs (parent_user_val / user_ft_user_val).
_ARTIFACTS: tuple[Phase3aArtifact, ...] = (
    Phase3aArtifact(
        filename="hybrid_parent_25k25k.keras",
        version="phase3a_hybrid_parent_25k25k",
        label="phase3a_balanced_parent",
        eval_metrics={
            "note": "cold hybrid 25k ikbendaniel + 25k RebeccaHarris registry-train",
            "epochs": 20,
            "train_moves_per_account": 25000,
        },
    ),
    Phase3aArtifact(
        filename="hybrid_parent_50k50k.keras",
        version="phase3a_hybrid_parent_50k50k",
        label="phase3a_balanced_parent_50k50k",
        eval_metrics={
            "note": "cold hybrid 50k ikbendaniel + 50k RebeccaHarris registry-train",
            "epochs": 20,
            "train_moves_per_account": 50000,
            "train_top1": 0.4848,
            "n_train": 100005,
        },
    ),
    Phase3aArtifact(
        filename="ikbendaniel_ft_30k.keras",
        version="phase3a_ikbendaniel_ft_30k",
        label="phase3a_ikbendaniel_ft",
        eval_metrics={
            "account": "ikbendaniel",
            "epochs": 20,
            "train_n": 30036,
            "val_n": 8022,
            "top1_overall": 0.4068,
            "top3_overall": 0.6452,
            "top1_sf_agree": 0.6190,
            "top1_sf_disagree": 0.2777,
            "delta_disagree_t1_vs_parent": 0.0150,
            "n_eval": 8022,
        },
    ),
    Phase3aArtifact(
        filename="RebeccaHarris_ft_30k.keras",
        version="phase3a_RebeccaHarris_ft_30k",
        label="phase3a_RebeccaHarris_ft",
        eval_metrics={
            "account": "RebeccaHarris",
            "epochs": 20,
            "train_n": 30005,
            "val_n": 8009,
            "top1_overall": 0.3728,
            "top3_overall": 0.6309,
            "top1_sf_agree": 0.6036,
            "top1_sf_disagree": 0.2122,
            "delta_disagree_t1_vs_parent": 0.0121,
            "n_eval": 8009,
        },
    ),
)


def _metrics_blob(art: Phase3aArtifact) -> dict[str, float | str | int]:
    blob: dict[str, float | str | int] = {
        "head_candidate_style": 1.0,
        "move_feat_dim": float(MOVE_FEAT_DIM),
        "max_candidates": float(MAX_CANDIDATES),
        "encoder_hybrid_board": 1.0,
        "encoder_fuses_state": 1.0,
        "board_tensor_version": float(BOARD_TENSOR_VERSION),
        "board_channels": float(BOARD_TENSOR_CHANNELS),
        "conv_filters": 64.0,
        "source": art.label,
        "phase": "3a",
    }
    blob.update(art.eval_metrics)
    return blob


def register_keras_playable(
    *,
    local_path: Path,
    version: str,
    label: str,
    eval_metrics: dict[str, float | str | int] | None = None,
    overwrite_s3: bool = True,
    s3_filename: str | None = None,
) -> str:
    """Upload one ``.keras`` and upsert ``ml.baseline_models`` archived row."""
    local_path = Path(local_path).resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {local_path}")
    art = Phase3aArtifact(
        filename=local_path.name,
        version=version,
        label=label,
        eval_metrics=dict(eval_metrics or {}),
    )
    storage = get_raw_storage()
    db = get_db_client()
    existing = {row.version: row for row in BaselineModel.fetch_all_ordered(db)}
    rel_key = f"{_S3_PREFIX}/{s3_filename or local_path.name}"
    storage.write_bytes(rel_key, local_path.read_bytes(), overwrite=overwrite_s3)
    model_uri = s3_url_string(rel_key)
    metrics_json = json.dumps(_metrics_blob(art))
    if version in existing:
        updated = replace(
            existing[version],
            model_uri=model_uri,
            status=BaselineModelStatus.ARCHIVED,
            eval_metrics=metrics_json,
        )
        updated.save_to_db(db)
        logger.info("Updated archived playable version=%s uri=%s", version, model_uri)
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
            logger.info("Inserted archived playable version=%s uri=%s", version, model_uri)
    invalidate_baseline_presets_cache()
    return version


def register_phase3a_playables(
    *,
    artifact_dir: Path,
    overwrite_s3: bool,
    which: list[str] | None = None,
) -> list[str]:
    artifact_dir = artifact_dir.resolve()
    wanted = set(which) if which else {a.version for a in _ARTIFACTS}
    storage = get_raw_storage()
    db = get_db_client()
    existing = {row.version: row for row in BaselineModel.fetch_all_ordered(db)}
    registered: list[str] = []

    for art in _ARTIFACTS:
        if art.version not in wanted:
            continue
        local = artifact_dir / art.filename
        if not local.is_file():
            raise FileNotFoundError(f"missing checkpoint: {local}")

        rel_key = f"{_S3_PREFIX}/{art.filename}"
        storage.write_bytes(rel_key, local.read_bytes(), overwrite=overwrite_s3)
        model_uri = s3_url_string(rel_key)
        metrics_json = json.dumps(_metrics_blob(art))

        if art.version in existing:
            updated = replace(
                existing[art.version],
                model_uri=model_uri,
                status=BaselineModelStatus.ARCHIVED,
                eval_metrics=metrics_json,
            )
            updated.save_to_db(db)
            logger.info("Updated archived phase3a version=%s uri=%s", art.version, model_uri)
        else:
            new_row = BaselineModel(
                id=BaselineModel.generate_id({"version": art.version}),
                version=art.version,
                trained_at=get_current_datetime(),
                model_uri=model_uri,
                status=BaselineModelStatus.ARCHIVED,
                eval_metrics=metrics_json,
                git_commit_hash=BaselineModel.current_git_commit(),
            )
            inserted = new_row.save_new_to_db(db)
            if not inserted:
                logger.warning("Insert skipped (conflict) version=%s", art.version)
            else:
                logger.info(
                    "Inserted archived phase3a version=%s uri=%s",
                    art.version,
                    model_uri,
                )
        registered.append(art.version)

    invalidate_baseline_presets_cache()
    return registered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=_DEFAULT_DIR,
        help="Dir with phase3a *.keras (default: storage/tmp/phase3a)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Comma versions to register (default: all). "
            "e.g. phase3a_ikbendaniel_ft_30k,phase3a_RebeccaHarris_ft_30k"
        ),
    )
    parser.add_argument(
        "--register-one",
        type=Path,
        default=None,
        help="Register a single .keras path (with --version / --label).",
    )
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument(
        "--metrics-json",
        type=str,
        default="{}",
        help="Extra eval_metrics JSON object for --register-one.",
    )
    parser.add_argument("--overwrite-s3", action="store_true")
    args = parser.parse_args()

    if args.register_one is not None:
        if not args.version or not args.label:
            logger.error("--register-one requires --version and --label")
            return 1
        try:
            metrics = json.loads(args.metrics_json or "{}")
        except json.JSONDecodeError as exc:
            logger.error("Bad --metrics-json: %s", exc)
            return 1
        if not isinstance(metrics, dict):
            logger.error("--metrics-json must be a JSON object")
            return 1
        version = register_keras_playable(
            local_path=Path(args.register_one),
            version=str(args.version),
            label=str(args.label),
            eval_metrics=metrics,
            overwrite_s3=bool(args.overwrite_s3),
        )
        presets = list_baseline_presets(get_db_client())
        key = f"baseline:{version}"
        print(
            f"registered version={version} preset_key={key} "
            f"in_play_list={key in {p.key for p in presets}}"
        )
        return 0

    which = [p.strip() for p in str(args.only).split(",") if p.strip()] if args.only else None
    versions = register_phase3a_playables(
        artifact_dir=Path(args.artifact_dir),
        overwrite_s3=bool(args.overwrite_s3),
        which=which,
    )
    db = get_db_client()
    presets = list_baseline_presets(db)
    preset_keys = {p.key for p in presets}
    print("\n=== phase3a playables registered ===")
    for v in versions:
        key = f"baseline:{v}"
        print(f"  version={v} preset_key={key} in_play_list={key in preset_keys}")
    print("Play -> Baseline bot -> pick phase3a_* versions.")
    return 0


if __name__ == "__main__":
    log_script_runtime_context(logger, script="register_phase3a_playables")
    run_script_main(main)
