"""Export a reproducible FEN sample from games.moves for Stockfish benchmarks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import polars as pl

from chess_teacher.pipelines.preprocessing.moves import Move
from chess_teacher.utils.db.client import get_db_client


def _default_output_path() -> Path:
    return Path("storage/benchmarks/fen_sample.json")


def export_fen_sample(
    *,
    account_id: str,
    limit: int,
    output: Path,
    seed: int,
) -> int:
    db_client = get_db_client()
    moves_table = Move.get_metadata()
    rows = db_client.read(
        moves_table,
        columns=["fen_before", "fen_after"],
        where=f"account_id = '{account_id}'",
        as_polars=True,
    )
    if rows.height == 0:
        raise SystemExit(f"No moves found for account_id={account_id!r}")

    unique_fens = pl.concat([rows["fen_before"], rows["fen_after"]]).unique().to_list()
    rng = random.Random(seed)
    rng.shuffle(unique_fens)
    sample = unique_fens[: min(limit, len(unique_fens))]

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "account_id": account_id,
        "seed": seed,
        "total_unique_fens_available": len(unique_fens),
        "sample_size": len(sample),
        "fens": sample,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(sample)} FEN(s) to {output} ({len(unique_fens)} unique available).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export unique FENs from games.moves.")
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    return export_fen_sample(
        account_id=args.account_id,
        limit=args.limit,
        output=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
