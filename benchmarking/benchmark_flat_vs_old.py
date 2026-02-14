#!/usr/bin/env python3
"""
Benchmark flat-buffer solver vs element-by-element solver.

Compares use_flat_buffers=True (run_flat / bulk memcpy) against
use_flat_buffers=False (run / nested vector copy loop) across
varying dataset sizes and sparsity levels.

Usage:
    uv run benchmarking/benchmark_flat_vs_old.py
    uv run benchmarking/benchmark_flat_vs_old.py -n 500000 -k 200 -p 0.05 0.1 --runs 3
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from generate_data import generate_data_sparse_maq

from sparse_maq import Solver


def run_solver(data: pl.DataFrame, use_flat_buffers: bool, runs: int) -> dict:
    """Time a single solver configuration over multiple runs, returning averaged metrics."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        solver = Solver()
        solver.fit_from_polars(data, use_flat_buffers=use_flat_buffers)
        times.append(time.perf_counter() - t0)
    return {
        "mean_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
    }


def run_comparison(
    n: int,
    k: int,
    p: float,
    runs: int,
) -> dict:
    print(f"  Generating data (n={n:,}, k={k}, p={p})...")
    _, _, data = generate_data_sparse_maq(n=n, k=k, p=p)

    print(f"  Running flat-buffer solver ({runs} run(s))...")
    flat = run_solver(data, use_flat_buffers=True, runs=runs)

    print(f"  Running element-by-element solver ({runs} run(s))...")
    old = run_solver(data, use_flat_buffers=False, runs=runs)

    speedup = old["mean_s"] / flat["mean_s"] if flat["mean_s"] > 0 else float("inf")

    return {
        "n": n,
        "k": k,
        "p": p,
        "runs": runs,
        "flat_mean_s": flat["mean_s"],
        "flat_min_s": flat["min_s"],
        "flat_max_s": flat["max_s"],
        "old_mean_s": old["mean_s"],
        "old_min_s": old["min_s"],
        "old_max_s": old["max_s"],
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark flat-buffer vs element-by-element solver")
    parser.add_argument("-n", default=1_000_000, type=int, help="Number of patients")
    parser.add_argument("-k", default=500, type=int, help="Number of treatments")
    parser.add_argument("-p", nargs="+", default=[0.05], type=float, help="Sparsity level(s)")
    parser.add_argument("--runs", default=3, type=int, help="Runs to average over")
    parser.add_argument("--output-dir", default="benchmarking/results", type=Path)
    args = parser.parse_args()

    print("=== Flat-buffer vs Element-by-element Benchmark ===")
    print(f"n={args.n:,}  k={args.k}  p={args.p}  runs={args.runs}")
    print()

    results = []
    for p in args.p:
        print(f"[p={p}]")
        result = run_comparison(
            n=args.n,
            k=args.k,
            p=p,
            runs=args.runs,
        )
        results.append(result)
        print(
            f"  flat:  {result['flat_mean_s']:.3f}s mean  "
            f"(min {result['flat_min_s']:.3f}s, max {result['flat_max_s']:.3f}s)\n"
            f"  old:   {result['old_mean_s']:.3f}s mean  "
            f"(min {result['old_min_s']:.3f}s, max {result['old_max_s']:.3f}s)\n"
            f"  speedup: {result['speedup']:.2f}x"
        )
        print()

    # Summary table
    df = pl.DataFrame(results)
    print("=== Summary ===")
    print(
        df.select("n", "k", "p", "flat_mean_s", "old_mean_s", "speedup")
        .with_columns(
            pl.col("flat_mean_s").round(3),
            pl.col("old_mean_s").round(3),
            pl.col("speedup").round(2),
        )
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"flat_vs_old_{timestamp}.csv"
    df.write_csv(output_file)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
