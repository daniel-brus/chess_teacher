"""Upload HP-grid checkpoints to local MinIO and register as archived Play presets.

Does **not** touch production status. Inserts ``ml.baseline_models`` rows with
``status=archived`` so Streamlit Play can load them.

Default selection: peak overall top1 round(s) on the cell val curve, plus the
latest scored round.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/register_hp_grid_playables.py ^
      --cell-dir storage/experiments/hp_grid_batch_epochs_2026-09-05/cell_bs64_ep20
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from chess_teacher.bots.presets import invalidate_baseline_presets_cache, list_baseline_presets
from chess_teacher.pipelines.neural_network.candidate_eval import MAX_CANDIDATES, MOVE_FEAT_DIM
from chess_teacher.pipelines.neural_network.models import BaselineModel, BaselineModelStatus
from chess_teacher.utils.db.client import get_db_client
from chess_teacher.utils.general_utils import get_current_datetime
from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.object_storage.factory import get_raw_storage, s3_url_string
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

logger = get_logger()

_S3_PREFIX = "experiments/hp_grid_dev"


def _read_curve(cell_dir: Path) -> list[dict[str, object]]:
    path = cell_dir / "val_curve.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing val curve: {path}")
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        blob = json.loads(line)
        if not isinstance(blob, dict) or "round" not in blob:
            continue
        rows.append(blob)
    if not rows:
        raise ValueError(f"empty val curve: {path}")
    return rows


def _overall_top1(row: dict[str, object]) -> float:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return float("-inf")
    return float(metrics.get("top1_overall", float("-inf")))


def select_peak_and_latest_rounds(
    rows: list[dict[str, object]],
    *,
    peak_tol: float = 0.001,
) -> list[int]:
    """Near-peak overall top1 (within ``peak_tol``) + latest scored round."""
    peak = max(_overall_top1(r) for r in rows)
    peak_rounds = sorted(
        {int(r["round"]) for r in rows if (peak - _overall_top1(r)) <= peak_tol}
    )
    latest = max(int(r["round"]) for r in rows)
    return sorted(set(peak_rounds) | {latest})


def _version_for(cell_name: str, round_i: int) -> str:
    # cell_bs64_ep20 -> hp64; keep short Play labels.
    short = cell_name.replace("cell_", "").replace("bs", "").replace("_ep", "e")
    return f"v_{short}_r{round_i}"


def _metrics_blob(row: dict[str, object], *, cell_name: str) -> dict[str, float | str | int]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    assert isinstance(metrics, dict)
    blob: dict[str, float | str | int] = {
        "head_candidate_style": 1.0,
        "move_feat_dim": float(MOVE_FEAT_DIM),
        "max_candidates": float(MAX_CANDIDATES),
        "source": f"hp_grid_{cell_name}",
        "round": int(row["round"]),
        "batch_size": int(row.get("batch_size") or 0),
        "epochs": int(row.get("epochs") or 0),
        "style_disagree_boost": float(row.get("style_disagree_boost") or 1.0),
    }
    for key in (
        "top1_overall",
        "top3_overall",
        "top1_sf_agree",
        "top1_sf_disagree",
        "n_eval",
    ):
        if key in metrics:
            blob[key] = float(metrics[key])
    return blob


def register_rounds(
    *,
    cell_dir: Path,
    rounds: list[int],
    overwrite_s3: bool,
) -> list[str]:
    cell_dir = cell_dir.resolve()
    cell_name = cell_dir.name
    curve_by_round = {int(r["round"]): r for r in _read_curve(cell_dir)}
    storage = get_raw_storage()
    db = get_db_client()
    existing = {row.version: row for row in BaselineModel.fetch_all_ordered(db)}
    registered: list[str] = []

    for round_i in rounds:
        row = curve_by_round.get(round_i)
        if row is None:
            raise ValueError(f"round {round_i} not in val_curve.jsonl under {cell_dir}")
        local = cell_dir / f"round_{round_i}" / "model.keras"
        if not local.is_file():
            raise FileNotFoundError(f"missing checkpoint: {local}")

        rel_key = f"{_S3_PREFIX}/{cell_name}/round_{round_i}/model.keras"
        data = local.read_bytes()
        storage.write_bytes(rel_key, data, overwrite=overwrite_s3)
        model_uri = s3_url_string(rel_key)
        version = _version_for(cell_name, round_i)
        metrics_json = json.dumps(_metrics_blob(row, cell_name=cell_name))

        if version in existing:
            updated = replace(
                existing[version],
                model_uri=model_uri,
                status=BaselineModelStatus.ARCHIVED,
                eval_metrics=metrics_json,
            )
            updated.save_to_db(db)
            logger.info("Updated archived preset version=%s uri=%s", version, model_uri)
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
                logger.info("Inserted archived preset version=%s uri=%s", version, model_uri)
        registered.append(version)

    invalidate_baseline_presets_cache()
    return registered


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cell-dir",
        type=Path,
        required=True,
        help="Path to one HP-grid cell (contains val_curve.jsonl + round_N/)",
    )
    p.add_argument(
        "--rounds",
        type=str,
        default="",
        help="Comma rounds to register (default: peak overall + latest)",
    )
    p.add_argument(
        "--overwrite-s3",
        action="store_true",
        help="Overwrite MinIO objects if they already exist",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    cell_dir = Path(args.cell_dir)
    curve = _read_curve(cell_dir)
    if str(args.rounds).strip():
        rounds = sorted({int(x.strip()) for x in str(args.rounds).split(",") if x.strip()})
    else:
        rounds = select_peak_and_latest_rounds(curve)
    logger.info("Registering cell=%s rounds=%s (archived only)", cell_dir.name, rounds)
    versions = register_rounds(
        cell_dir=cell_dir,
        rounds=rounds,
        overwrite_s3=bool(args.overwrite_s3),
    )
    presets = list_baseline_presets(get_db_client(), force_refresh=True)
    for preset in presets:
        if any(v in preset.key for v in versions):
            logger.info("Play preset ready key=%s label=%s", preset.key, preset.label)
    print(json.dumps({"registered_versions": versions, "rounds": rounds}, indent=2))
    return 0


if __name__ == "__main__":
    log_script_runtime_context(logger, script="register_hp_grid_playables")
    run_script_main(main)
