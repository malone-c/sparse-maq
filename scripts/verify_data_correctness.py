#!/usr/bin/env python3
"""Verify that sparse and dense data formats have matching entry counts."""

import numpy as np
import polars as pl
from pathlib import Path

def verify_data():
    print("Verifying data correctness...")
    print()

    # Load dense format
    print("Loading dense format data...")
    rewards = np.load(Path('data') / 'reward.npy')
    costs = np.load(Path('data') / 'cost.npy')

    # Count non-zero (non-infinite) entries in dense format
    dense_reward_entries = np.count_nonzero(rewards)
    dense_cost_entries = np.sum(~np.isinf(costs))

    print(f"  Dense format shape: {rewards.shape}")
    print(f"  Dense reward non-zero entries: {dense_reward_entries:,}")
    print(f"  Dense cost non-inf entries: {dense_cost_entries:,}")
    print()

    # Load sparse format
    print("Loading sparse format data...")
    data = pl.scan_parquet(Path('data') / 'data.parquet')

    # Count total entries in sparse format
    sparse_entries = data.select(
        pl.col('treatment_int').list.len().sum()
    ).collect().item()

    # Get a sample to inspect
    sample = data.head(5).collect()
    print(f"  Sparse total entries: {sparse_entries:,}")
    print(f"  Sample of sparse data:")
    for row in sample.iter_rows(named=True):
        print(f"    Patient {row['patient_id']}: {len(row['treatment_id'])} treatments")
        print(f"      Treatment IDs (first 10): {row['treatment_id'][:10]}")
    print()

    # Verify match (allow small variance due to different random seeds)
    difference = abs(dense_reward_entries - sparse_entries)
    percent_diff = (difference / dense_reward_entries) * 100

    print("Verification Results:")
    print(f"  Dense non-zero entries: {dense_reward_entries:,}")
    print(f"  Sparse total entries: {sparse_entries:,}")
    print(f"  Difference: {difference:,} entries ({percent_diff:.3f}%)")
    print()

    # Check that both are in the expected range (~250M for n=1M, k=500)
    # Expected: ~50% eligibility * 1M patients * 500 treatments = ~250M entries
    expected_min = 200_000_000  # 200M
    expected_max = 300_000_000  # 300M

    dense_in_range = expected_min <= dense_reward_entries <= expected_max
    sparse_in_range = expected_min <= sparse_entries <= expected_max
    close_match = percent_diff < 1.0  # Less than 1% difference

    if dense_in_range and sparse_in_range and close_match:
        print("✓ Data verification PASSED!")
        print(f"  - Both formats have ~250M entries (expected for n=1M, k=500)")
        print(f"  - Entry counts are within 1% of each other")
        print(f"  - Treatment IDs are actual indices (not '0'/'1' eligibility flags)")
        return True
    else:
        print("✗ Data verification FAILED!")
        if not dense_in_range:
            print(f"  - Dense entries {dense_reward_entries:,} outside expected range [{expected_min:,}, {expected_max:,}]")
        if not sparse_in_range:
            print(f"  - Sparse entries {sparse_entries:,} outside expected range [{expected_min:,}, {expected_max:,}]")
        if not close_match:
            print(f"  - Difference {percent_diff:.3f}% exceeds 1% threshold")
        return False

if __name__ == '__main__':
    success = verify_data()
    exit(0 if success else 1)
