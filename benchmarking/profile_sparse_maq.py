#!/usr/bin/env python3
"""Profile sparse_maq at n=1M, k=500, p=0.05."""

import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl

# Allow running from repo root or benchmarking/
sys.path.insert(0, str(Path(__file__).parent))

from generate_data import generate_data_sparse_maq
from run_benchmark_suite import parse_time_output

N = 1_000_000
K = 500
P = 0.05


def parse_stage_timings(stdout: str) -> list[dict]:
    """Parse SPARSE_MAQ_PROFILE=1 stage lines from stdout.

    Each stage line looks like:
        sort: 1.23s, Peak: 0.45 GB
        cpp_solver: 5.67s, Peak: 1.23 GB
    """
    stages = []
    stage_pattern = re.compile(
        r"^(\w+):\s+([\d.]+)s,\s+Peak:\s+([\d.]+)\s+GB", re.MULTILINE
    )
    for m in stage_pattern.finditer(stdout):
        stages.append(
            {
                "stage": m.group(1),
                "elapsed_s": float(m.group(2)),
                "peak_mem_gb": float(m.group(3)),
            }
        )
    return stages


def bottleneck_narrative(
    stages: list[dict], sys_metrics: dict, total_s: float | None
) -> str:
    """Generate a short bottleneck analysis paragraph."""
    if not stages:
        return "No stage-level data available."

    lines = []

    slowest = max(stages, key=lambda s: s["elapsed_s"])
    pct = (slowest["elapsed_s"] / total_s * 100) if total_s else float("nan")
    lines.append(
        f"The dominant stage is **{slowest['stage']}** "
        f"({slowest['elapsed_s']:.2f}s, {pct:.1f}% of total solver time)."
    )

    most_mem = max(stages, key=lambda s: s["peak_mem_gb"])
    lines.append(
        f"Peak tracemalloc memory is reported at the **{most_mem['stage']}** stage "
        f"({most_mem['peak_mem_gb']:.2f} GB)."
    )

    peak_rss_kb = sys_metrics.get("peak_memory_kb")
    if peak_rss_kb:
        lines.append(
            f"OS-level peak RSS is {peak_rss_kb / 1024**2:.2f} GB "
            f"({peak_rss_kb:,} KB), which includes Python interpreter overhead."
        )

    return " ".join(lines)


def build_report(
    stages: list[dict],
    sys_metrics: dict,
    stdout: str,
    stderr: str,
    avg_treatments: float,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract total time from stdout
    total_match = re.search(r"Total time:\s+([\d.]+)s", stdout)
    total_s = float(total_match.group(1)) if total_match else None

    # --- System-level metrics section ---
    wall = sys_metrics.get("wall_time_seconds")
    user = sys_metrics.get("user_time_seconds")
    sys_ = sys_metrics.get("system_time_seconds")
    rss_kb = sys_metrics.get("peak_memory_kb")

    def fmt(v, unit="s"):
        return f"{v:.2f}{unit}" if v is not None else "n/a"

    sys_section = f"""\
## System-level metrics

| Metric | Value |
|--------|-------|
| Wall time | {fmt(wall)} |
| User CPU time | {fmt(user)} |
| System CPU time | {fmt(sys_)} |
| Peak RSS | {f"{rss_kb / 1024**2:.2f} GB ({rss_kb:,} KB)" if rss_kb else "n/a"} |
"""

    # --- Stage-level breakdown ---
    if stages:
        total_elapsed = sum(s["elapsed_s"] for s in stages)
        rows = []
        for s in stages:
            pct = s["elapsed_s"] / total_elapsed * 100 if total_elapsed else 0
            rows.append(
                f"| {s['stage']} | {s['elapsed_s']:.2f} | {pct:.1f}% | {s['peak_mem_gb']:.2f} |"
            )
        stage_section = (
            "## Stage-level breakdown\n\n"
            "| Stage | Elapsed (s) | % of total | Peak mem (GB) |\n"
            "|-------|-------------|------------|---------------|\n"
            + "\n".join(rows)
            + "\n"
        )
        if total_s:
            stage_section += f"\n_Solver total (from profiler): {total_s:.2f}s_\n"
    else:
        stage_section = "## Stage-level breakdown\n\nNo stage data captured.\n"

    # --- Bottleneck analysis ---
    bottleneck = bottleneck_narrative(stages, sys_metrics, total_s)

    report = f"""\
# sparse_maq Profiling Report

**Date:** {now}
**Config:** n={N:,}, k={K}, p={P}, avg treatments/patient ≈ {avg_treatments:.1f}

{sys_section}
{stage_section}
## Bottleneck analysis

{bottleneck}

## Raw output

### stdout

```
{stdout.strip()}
```

### stderr

```
{stderr.strip()}
```
"""
    return report


def main() -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Generate data ---
    print(f"Generating data: n={N:,}, k={K}, p={P} ...")
    with tempfile.TemporaryDirectory(prefix="sparse_maq_profile_") as tmp:
        tmp_path = Path(tmp)

        treatments, patients, df = generate_data_sparse_maq(n=N, k=K, p=P)

        avg_treatments = df["treatment_id"].list.len().mean()
        print(f"  avg treatments/patient: {avg_treatments:.1f}")

        treatments.write_parquet(tmp_path / "treatments.parquet")
        patients.write_parquet(tmp_path / "patients.parquet")
        df.write_parquet(tmp_path / "data.parquet")
        print("  Data written to temp dir.")

        # --- Step 2: Run solver with profiling ---
        print("Running solver with SPARSE_MAQ_PROFILE=1 ...")
        env = {**os.environ, "SPARSE_MAQ_PROFILE": "1"}
        repo_root = Path(__file__).parent.parent

        cmd = [
            "/usr/bin/time",
            "-v",
            "uv",
            "run",
            "benchmarking/run_sparse_maq.py",
            "--base-path",
            str(tmp_path),
        ]

        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(repo_root),
        )

    stdout = process.stdout
    stderr = process.stderr

    if process.returncode != 0:
        print("WARNING: solver exited with non-zero status", process.returncode, file=sys.stderr)

    # --- Step 3: Parse outputs ---
    stages = parse_stage_timings(stdout)
    sys_metrics = parse_time_output(stderr)

    # --- Step 4: Build and print report ---
    report = build_report(stages, sys_metrics, stdout, stderr, avg_treatments)

    print("\n" + report)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = results_dir / f"profile_report_{timestamp}.md"
    report_path.write_text(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
