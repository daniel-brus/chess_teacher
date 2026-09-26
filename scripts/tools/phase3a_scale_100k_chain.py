"""Phase 3a scale chain: 100k+100k parent, then multi-round user FTs + Play register.

Leaves the enrich k8s job alone. Writes status under ``storage/tmp/phase3a/scale100k/``.

Run (dev)::

    doppler run --project chess-teacher --config dev_local -- ^
      .venv\\Scripts\\python.exe scripts/tools/phase3a_scale_100k_chain.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from chess_teacher.utils.logging import get_logger
from chess_teacher.utils.process_utils import log_script_runtime_context, run_script_main

logger = get_logger()

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "storage" / "tmp" / "phase3a"
_SCALE = _OUT / "scale100k"

_DANIEL = "5c3f069730e098aedc61b978c92d6448eebfdf234e575552e86343a10bfb698b"
_REBECCA = "d984e60e96a24fc53262d2579bf80cf9fb31290d806869a1d567dcb12e5e0c01"

# Growing registry-train prefixes (complete-game caps). ep10 x many rounds.
_ROUND_TRAIN_LIMITS = (25_000, 45_000, 65_000, 85_000, 110_000, 140_000, 170_000, 200_000)
_VAL_LIMIT = 8_000
_FT_EPOCHS = 10
_PARENT_EPOCHS = 20


def _py() -> Path:
    return _REPO / ".venv" / "Scripts" / "python.exe"


def _run_tool(args: list[str], log_path: Path) -> int:
    cmd = [
        "doppler",
        "run",
        "--project",
        "chess-teacher",
        "--config",
        "dev_local",
        "--",
        str(_py()),
        *args,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running: %s -> %s", " ".join(args[:4]), log_path)
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(cmd, cwd=str(_REPO), stdout=fh, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def _register(
    local: Path,
    *,
    version: str,
    label: str,
    metrics: dict[str, float | str | int],
) -> None:
    # Subprocess avoids importlib+dataclass sys.modules crash when loading
    # scripts/tools as a loose file (cls.__module__ lookup → None).
    metrics_path = _SCALE / f"_metrics_{version}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    cmd = [
        "doppler",
        "run",
        "--project",
        "chess-teacher",
        "--config",
        "dev_local",
        "--",
        str(_py()),
        "scripts/tools/register_phase3a_playables.py",
        "--register-one",
        str(local),
        "--version",
        version,
        "--label",
        label,
        "--metrics-json",
        metrics_path.read_text(encoding="utf-8"),
        "--overwrite-s3",
    ]
    logger.info("Register Play version=%s path=%s", version, local)
    proc = subprocess.run(cmd, cwd=str(_REPO))
    if proc.returncode != 0:
        raise RuntimeError(f"register failed version={version} exit={proc.returncode}")


def _write_status(blob: dict[str, object]) -> None:
    _SCALE.mkdir(parents=True, exist_ok=True)
    path = _SCALE / "STATUS.json"
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def run_chain(
    *,
    max_rounds: int,
    skip_parent: bool,
    parent_out: Path,
    account_names: tuple[str, ...] | None = None,
) -> int:
    all_accounts = (
        ("ikbendaniel", _DANIEL),
        ("RebeccaHarris", _REBECCA),
    )
    if account_names:
        wanted = {n.strip() for n in account_names if n.strip()}
        accounts = tuple(a for a in all_accounts if a[0] in wanted)
        unknown = wanted - {a[0] for a in all_accounts}
        if unknown:
            logger.error("Unknown --accounts: %s", sorted(unknown))
            return 1
        if not accounts:
            logger.error("No accounts selected")
            return 1
    else:
        accounts = all_accounts
    rounds = list(_ROUND_TRAIN_LIMITS[: max(1, max_rounds)])
    steps: list[dict[str, object]] = []
    status: dict[str, object] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent": str(parent_out),
        "rounds_train_limits": rounds,
        "ft_epochs": _FT_EPOCHS,
        "accounts": [n for n, _ in accounts],
        "steps": steps,
    }
    _write_status(status)

    if not skip_parent:
        parent_log = _SCALE / "parent_100k100k_run.log"
        steps.append({"step": "parent_100k100k", "state": "running"})
        _write_status(status)
        code = _run_tool(
            [
                "scripts/tools/offline_user_finetune_eval.py",
                "--train-parent",
                "--parent-out",
                str(parent_out),
                "--parent-train-limit",
                "100000",
                "--parent-balance-account-ids",
                f"{_DANIEL},{_REBECCA}",
                "--parent-epochs",
                str(_PARENT_EPOCHS),
                "--parent-style-disagree-boost",
                "2.0",
                "--split-version",
                "baseline-v1",
            ],
            parent_log,
        )
        if code != 0 or not parent_out.is_file():
            steps[-1] = {"step": "parent_100k100k", "state": "failed", "exit": code}
            _write_status(status)
            return 1
        steps[-1] = {"step": "parent_100k100k", "state": "done"}
        _write_status(status)

        _register(
            parent_out,
            version="phase3a_hybrid_parent_100k100k",
            label="phase3a_balanced_parent_100k100k",
            metrics={
                "note": "cold hybrid 100k+100k registry-train",
                "epochs": _PARENT_EPOCHS,
                "train_moves_per_account": 100000,
            },
        )
        steps.append({
            "step": "register_parent",
            "version": "phase3a_hybrid_parent_100k100k",
            "state": "done",
        })
        _write_status(status)
    elif not parent_out.is_file():
        logger.error("skip_parent but missing %s", parent_out)
        return 1

    for name, account_id in accounts:
        weights = parent_out
        for round_i, train_limit in enumerate(rounds, start=1):
            child = _SCALE / name / f"round_{round_i}.keras"
            child.parent.mkdir(parents=True, exist_ok=True)
            version = f"phase3a_{name}_from100k_r{round_i}"
            step: dict[str, object] = {
                "step": f"ft_{name}_r{round_i}",
                "train_limit": train_limit,
                "parent": str(weights),
                "child": str(child),
                "version": version,
                "state": "running",
            }
            steps.append(step)
            _write_status(status)
            log = _SCALE / name / f"round_{round_i}_run.log"
            if child.is_file():
                logger.info("Skip train (checkpoint exists): %s", child)
                step["skipped_train"] = True
            else:
                code = _run_tool(
                    [
                        "scripts/tools/offline_user_finetune_eval.py",
                        "--account-id",
                        account_id,
                        "--parent-weights",
                        str(weights),
                        "--child-out",
                        str(child),
                        "--train-limit",
                        str(train_limit),
                        "--val-limit",
                        str(_VAL_LIMIT),
                        "--epochs",
                        str(_FT_EPOCHS),
                        "--style-disagree-boost",
                        "4.0",
                        "--split-version",
                        "baseline-v1",
                    ],
                    log,
                )
                if code != 0 or not child.is_file():
                    step["state"] = "failed"
                    step["exit"] = code
                    _write_status(status)
                    # Continue other account; skip remaining rounds for this one.
                    logger.error("%s round %s failed exit=%s", name, round_i, code)
                    break
            step["state"] = "done"
            _write_status(status)
            _register(
                child,
                version=version,
                label=f"phase3a_{name}_r{round_i}",
                metrics={
                    "account": name,
                    "round": round_i,
                    "train_limit": train_limit,
                    "val_limit": _VAL_LIMIT,
                    "epochs": _FT_EPOCHS,
                    "from_parent": "phase3a_hybrid_parent_100k100k",
                },
            )
            step["registered"] = True
            _write_status(status)
            weights = child  # warm-start next round

    status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status["state"] = "complete"
    _write_status(status)
    (_SCALE / "DONE.txt").write_text(
        f"done {status['finished_at']}\n",
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-rounds", type=int, default=len(_ROUND_TRAIN_LIMITS))
    p.add_argument("--skip-parent", action="store_true")
    p.add_argument(
        "--accounts",
        type=str,
        default="",
        help="Comma names (ikbendaniel,RebeccaHarris). Default: both.",
    )
    p.add_argument(
        "--parent-out",
        type=Path,
        default=_OUT / "hybrid_parent_100k100k.keras",
    )
    args = p.parse_args()
    names = tuple(x.strip() for x in str(args.accounts).split(",") if x.strip()) or None
    return run_chain(
        max_rounds=max(1, int(args.max_rounds)),
        skip_parent=bool(args.skip_parent),
        parent_out=Path(args.parent_out),
        account_names=names,
    )


if __name__ == "__main__":
    log_script_runtime_context(logger, script="phase3a_scale_100k_chain")
    run_script_main(main)
