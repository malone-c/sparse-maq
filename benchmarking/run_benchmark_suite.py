#!/usr/bin/env python3

import itertools
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import re
from typing import Any
import argparse

import polars as pl

from generate_data import generate_data


def parse_time_output(stderr: str) -> dict[str, Any]:
    """
    Extract metrics from GNU time -v output.

    Parses stderr output from /usr/bin/time -v to extract:
    - User time (seconds)
    - System time (seconds)
    - Elapsed wall clock time (converted to seconds)
    - Maximum resident set size (kilobytes)

    Args:
        stderr: Standard error output from GNU time -v

    Returns:
        Dictionary with parsed metrics
    """
    metrics = {
    }

    # Parse user time
    match = re.search(r"User time \(seconds\): ([\d.]+)", stderr)
    if match:
        metrics["user_time_seconds"] = float(match.group(1))

    # Parse system time
    match = re.search(r"System time \(seconds\): ([\d.]+)", stderr)
    if match:
        metrics["system_time_seconds"] = float(match.group(1))

    # Parse elapsed time (can be in format h:mm:ss or m:ss or just ss.ss)
    match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)", stderr)
    if match:
        time_str = match.group(1).strip()
        # Parse different time formats
        if ":" in time_str:
            parts = time_str.split(":")
            if len(parts) == 2:  # m:ss.ss format
                minutes, seconds = parts
                metrics["wall_time_seconds"] = float(minutes) * 60 + float(seconds)
            elif len(parts) == 3:  # h:mm:ss.ss format
                hours, minutes, seconds = parts
                metrics["wall_time_seconds"] = (
                    float(hours) * 3600 + float(minutes) * 60 + float(seconds)
                )
        else:
            # Just seconds
            metrics["wall_time_seconds"] = float(time_str)

    # Parse maximum resident set size (in kilobytes)
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    if match:
        metrics["peak_memory_kb"] = int(match.group(1))

    return metrics

def run_single_benchmark(
    n: int, k: int, p: float, solver: str, temp_dir: Path
) -> dict[str, Any]:
    """
    Execute a single benchmark with GNU time wrapper.

    Prepares data files (NOT timed), then times only the solver execution
    to ensure fair comparison without data preprocessing overhead.

    Args:
        n: Number of patients
        k: Number of treatments
        solver: Solver name ('maq' or 'sparse_maq')
        timeout: Maximum execution time in seconds
        temp_dir: Directory for temporary data files

    Returns:
        Dictionary containing benchmark results and metrics
    """
    result = {
        "p": p,
        "solver": solver,
        "wall_time_seconds": None,
        "user_time_seconds": None,
        "system_time_seconds": None,
        "peak_memory_kb": None,
        "exit_code": None,
        "error_message": None,
    }

    try:
        # Step 1: Prepare data (NOT timed)
        generate_data(n=n, k=k, p=p, temp_dir=temp_dir, solver=solver)

        # Step 2: Time only the solver execution
        cmd = [
            "/usr/bin/time",
            "-v",
            "uv",
            "run",
            f"benchmarking/run_{solver}.py",
            "--base-path",
            str(temp_dir),
        ]

        # Run command with timeout
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        result["exit_code"] = process.returncode

        # GNU time writes to stderr
        if process.stderr:
            metrics = parse_time_output(process.stderr)
            result.update(metrics)

        # Capture error messages if process failed
        if process.returncode != 0:
            # Filter out GNU time output from error message
            stderr_lines = process.stderr.split("\n")
            error_lines = [
                line
                for line in stderr_lines
                if not line.strip().startswith(("Command being timed:", "User time", "System time", "Percent", "Elapsed", "Average", "Maximum", "Major", "Minor", "Voluntary", "Involuntary", "Swaps", "File system", "Socket", "Signals", "Page size", "Exit status"))
            ]
            result["error_message"] = "\n".join(error_lines).strip()[:500]  # Limit length

    except subprocess.TimeoutExpired:
        result["exit_code"] = -1
    except Exception as e:
        result["exit_code"] = -2
        result["error_message"] = str(e)[:500]

    return result


def run_benchmark_suite(
    n: int, k: int, sparsity_levels: list[float], output_dir: Path, runs: int = 1,
) -> Path:
    """
    Execute full benchmark suite with progress tracking.

    Loads configuration, runs all benchmark combinations, and saves
    results incrementally to CSV.

    Args:
        config_path: Path to YAML configuration file
        output_dir: Directory to save results CSV
        verbose: Whether to print progress information

    Returns:
        Path to output CSV file
    """
    print(f"Results will be saved to {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for benchmark data
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{timestamp}.csv"

    results = []

    # Run each benchmark
    for p, solver in itertools.product(sparsity_levels, ['maq', 'sparse_maq']):
        print(f"Running p={p}, solver={solver}" + (f" ({runs} runs)" if runs > 1 else ""))

        run_timestamp = datetime.now().isoformat()

        run_results = []
        for i in range(runs):
            if runs > 1:
                print(f"  Run {i + 1}/{runs}...")
            run_results.append(run_single_benchmark(
                n=n,
                k=k,
                p=p,
                solver=solver,
                temp_dir=temp_dir,
            ))

        # Average numeric metrics across successful runs; report first error if any failed
        numeric_keys = ["wall_time_seconds", "user_time_seconds", "system_time_seconds", "peak_memory_kb"]
        successful_runs = [r for r in run_results if r["exit_code"] == 0]
        failed_runs = [r for r in run_results if r["exit_code"] != 0]

        result = {
            "p": p,
            "solver": solver,
            "exit_code": run_results[-1]["exit_code"],
            "error_message": next((r["error_message"] for r in failed_runs), None),
        }
        for key in numeric_keys:
            values = [r[key] for r in successful_runs if r[key] is not None]
            result[key] = sum(values) / len(values) if values else None

        result["timestamp"] = run_timestamp

        results.append(result)

        df = pl.DataFrame(results)

        column_order = [
            "timestamp",
            "p",
            "solver",
            "wall_time_seconds",
            "user_time_seconds",
            "system_time_seconds",
            "peak_memory_kb",
            "exit_code",
            "error_message",
        ]
        df = df.select(column_order)

        # Save to CSV (overwrite each time for incremental updates)
        df.write_csv(output_file)

        if result["exit_code"] == 0:
            print(
                f"  ✓ Completed in {result['wall_time_seconds']:.2f}s, "
                f"Peak memory: {result['peak_memory_kb']:,} KB"
            )
        else:
            print(f"  ✗ Failed with exit code {result['exit_code']}")
            if result["error_message"]:
                print(f"    Error: {result['error_message'][:100]}")
        print()

    print(f"Benchmark suite completed!")
    print(f"Results saved to: {output_file}")

    # Print summary statistics
    df = pl.DataFrame(results)
    successful = df.filter(pl.col("exit_code") == 0)
    failed = df.filter(pl.col("exit_code") != 0)

    print()
    print("=== Summary ===")
    print(f"Total runs: {len(df)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if len(successful) > 0:
        print()
        print("Performance by solver:")
        summary = successful.group_by("solver").agg(
            [
                pl.col("wall_time_seconds").mean().alias("avg_time_s"),
                pl.col("peak_memory_kb").mean().alias("avg_memory_kb"),
                pl.len().alias("count"),
            ]
        )
        print(summary)

    return output_file

def main():
    pass

if __name__ == '__main__':
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', nargs='+', default=[0.05], type=float)
    parser.add_argument('-n', default=1_000_000, type=int)
    parser.add_argument('-k', default=500, type=int)
    parser.add_argument('--runs', default=1, type=int, help='Number of runs to average over')
    args = parser.parse_args()

    # Run benchmark suite
    try:
        output_file = run_benchmark_suite(
            n=args.n,
            k=args.k,
            sparsity_levels=args.p,
            output_dir=Path('benchmarking/results'),
            runs=args.runs,
        )
    except Exception as e:
        print(f"Error running benchmark suite: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


