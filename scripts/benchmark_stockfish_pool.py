"""Benchmark ProcessPoolExecutor + Stockfish settings for FEN evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@dataclass(frozen=True)
class BenchmarkConfig:
    workers: int
    threads_per_engine: int
    hash_mb: int
    depth: int

    @property
    def total_threads(self) -> int:
        return self.workers * self.threads_per_engine


@dataclass(frozen=True)
class BenchmarkResult:
    timestamp: str
    workers: int
    threads_per_engine: int
    hash_mb: int
    depth: int
    sample_size: int
    repeat_index: int
    wall_time_sec: float
    fens_per_sec: float
    cpu_count: int
    total_threads: int


_MIN_PARALLEL_FENS = 100


def _default_sample_path() -> Path:
    return Path("storage/benchmarks/fen_sample.json")


def _default_output_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path("storage/benchmarks") / f"stockfish_pool_{stamp}.csv"


def load_fen_sample(path: Path, sample_size: int) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fens: list[str] = payload["fens"]
    if sample_size <= 0 or sample_size >= len(fens):
        return fens
    return fens[:sample_size]


def fens_to_benchmark_df(fens: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"fen_before": fens, "fen_after": fens})


@contextmanager
def _stockfish_env(
    *,
    workers: int,
    threads_per_engine: int,
    hash_mb: int,
) -> Iterator[None]:
    keys = ("STOCKFISH_WORKERS", "STOCKFISH_THREADS_PER_ENGINE", "STOCKFISH_HASH_MB")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["STOCKFISH_WORKERS"] = str(workers)
    os.environ["STOCKFISH_THREADS_PER_ENGINE"] = str(threads_per_engine)
    os.environ["STOCKFISH_HASH_MB"] = str(hash_mb)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_single_benchmark(
    df: pl.DataFrame,
    config: BenchmarkConfig,
) -> float:
    from chess_teacher.pipelines.preprocessing.move_characteristics.stockfish_evaluation import (
        StockfishEvaluationTransformation,
    )

    with _stockfish_env(
        workers=config.workers,
        threads_per_engine=config.threads_per_engine,
        hash_mb=config.hash_mb,
    ):
        transformation = StockfishEvaluationTransformation(
            depth=config.depth,
            log_progress_percent=None,
            n_workers=config.workers,
        )
        started = time.perf_counter()
        transformation.transform(df)
        return time.perf_counter() - started


def build_config_grid(
    *,
    workers: list[int],
    threads_per_engine: list[int],
    hash_mb_values: list[int],
    depth: int,
    max_total_threads: int | None,
) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    seen: set[tuple[int, int, int, int]] = set()
    for worker_count in sorted(set(workers)):
        for thread_count in sorted(set(threads_per_engine)):
            total_threads = worker_count * thread_count
            if max_total_threads is not None and total_threads > max_total_threads:
                continue
            for hash_mb in sorted(set(hash_mb_values)):
                key = (worker_count, thread_count, hash_mb, depth)
                if key in seen:
                    continue
                seen.add(key)
                configs.append(
                    BenchmarkConfig(
                        workers=worker_count,
                        threads_per_engine=thread_count,
                        hash_mb=hash_mb,
                        depth=depth,
                    )
                )
    return configs


def write_results(path: Path, results: list[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def summarize_results(results: list[BenchmarkResult]) -> list[dict[str, float | int]]:
    grouped: dict[tuple[int, int, int, int], list[float]] = {}
    for result in results:
        key = (result.workers, result.threads_per_engine, result.hash_mb, result.depth)
        grouped.setdefault(key, []).append(result.fens_per_sec)

    summary: list[dict[str, float | int]] = []
    for (workers, threads_per_engine, hash_mb, depth), rates in grouped.items():
        summary.append({
            "workers": workers,
            "threads_per_engine": threads_per_engine,
            "hash_mb": hash_mb,
            "depth": depth,
            "total_threads": workers * threads_per_engine,
            "runs": len(rates),
            "fens_per_sec_mean": statistics.mean(rates),
            "fens_per_sec_median": statistics.median(rates),
            "fens_per_sec_min": min(rates),
            "fens_per_sec_max": max(rates),
        })
    summary.sort(key=lambda row: float(row["fens_per_sec_mean"]), reverse=True)
    return summary


def print_summary(summary: list[dict[str, float | int]], cpu_count: int) -> None:
    print()
    print(f"CPU count: {cpu_count}")
    print("Top configurations (mean fens/sec):")
    for index, row in enumerate(summary[:5], start=1):
        print(
            f"  {index}. workers={row['workers']} "
            f"threads/engine={row['threads_per_engine']} "
            f"hash_mb={row['hash_mb']} "
            f"total_threads={row['total_threads']} "
            f"-> {row['fens_per_sec_mean']:.2f} fens/sec "
            f"(median {row['fens_per_sec_median']:.2f})"
        )
    best = summary[0]
    print()
    print("Recommended .env:")
    print(f"  STOCKFISH_WORKERS={best['workers']}")
    print(f"  STOCKFISH_THREADS_PER_ENGINE={best['threads_per_engine']}")
    print(f"  STOCKFISH_HASH_MB={best['hash_mb']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Stockfish ProcessPool settings on a fixed FEN sample.",
    )
    parser.add_argument("--sample-file", type=Path, default=_default_sample_path())
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--workers", default="1,2,4,6,8")
    parser.add_argument("--threads-per-engine", default="1,2,4")
    parser.add_argument("--hash-mb", default="32")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-size", type=int, default=150)
    parser.add_argument(
        "--max-total-threads",
        type=int,
        default=None,
        help="Skip configs where workers * threads_per_engine exceeds this value.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.sample_file.is_file():
        raise SystemExit(
            f"Sample file not found: {args.sample_file}. Run scripts/export_fen_sample.py first."
        )

    cpu_count = _cpu_count()
    max_total_threads = args.max_total_threads
    if max_total_threads is None:
        max_total_threads = max(cpu_count, cpu_count + 2)

    fens = load_fen_sample(args.sample_file, args.sample_size)
    df = fens_to_benchmark_df(fens)
    configs = build_config_grid(
        workers=_parse_int_list(args.workers),
        threads_per_engine=_parse_int_list(args.threads_per_engine),
        hash_mb_values=_parse_int_list(args.hash_mb),
        depth=args.depth,
        max_total_threads=max_total_threads,
    )
    if not configs:
        raise SystemExit("No benchmark configurations to run.")

    output = args.output or _default_output_path()
    timestamp = datetime.now(UTC).isoformat()
    results: list[BenchmarkResult] = []

    warmup_fens = fens[: min(args.warmup_size, len(fens))]
    if warmup_fens and len(warmup_fens) < _MIN_PARALLEL_FENS:
        print(
            f"Warning: warmup-size {len(warmup_fens)} < {_MIN_PARALLEL_FENS}; "
            f"bumping to {_MIN_PARALLEL_FENS} so ProcessPool path is exercised."
        )
        warmup_fens = fens[: min(_MIN_PARALLEL_FENS, len(fens))]
    if warmup_fens:
        warmup_config = max(configs, key=lambda config: config.workers)
        print(
            f"Warm-up: {len(warmup_fens)} FEN(s) with workers={warmup_config.workers} "
            f"threads={warmup_config.threads_per_engine} hash={warmup_config.hash_mb}MB "
            f"(parallel path)"
        )
        run_single_benchmark(fens_to_benchmark_df(warmup_fens), warmup_config)

    print(
        f"Benchmarking {len(configs)} config(s) on {len(fens)} FEN(s), "
        f"{args.repeats} repeat(s), max_total_threads={max_total_threads}"
    )

    for config_index, config in enumerate(configs, start=1):
        for repeat_index in range(1, args.repeats + 1):
            print(
                f"[{config_index}/{len(configs)} repeat {repeat_index}/{args.repeats}] "
                f"workers={config.workers} threads/engine={config.threads_per_engine} "
                f"hash={config.hash_mb}MB total_threads={config.total_threads}"
            )
            wall_time_sec = run_single_benchmark(df, config)
            fens_per_sec = len(fens) / wall_time_sec if wall_time_sec > 0 else 0.0
            results.append(
                BenchmarkResult(
                    timestamp=timestamp,
                    workers=config.workers,
                    threads_per_engine=config.threads_per_engine,
                    hash_mb=config.hash_mb,
                    depth=config.depth,
                    sample_size=len(fens),
                    repeat_index=repeat_index,
                    wall_time_sec=wall_time_sec,
                    fens_per_sec=fens_per_sec,
                    cpu_count=cpu_count,
                    total_threads=config.total_threads,
                )
            )
            print(f"  -> {wall_time_sec:.1f}s ({fens_per_sec:.2f} fens/sec)")

    write_results(output, results)
    summary = summarize_results(results)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} run(s) to {output}")
    print(f"Wrote summary to {summary_path}")
    print_summary(summary, cpu_count)
    return 0


if __name__ == "__main__":
    from chess_teacher.utils.process_utils import run_script_main

    run_script_main(main)
