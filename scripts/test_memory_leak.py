#!/usr/bin/env python3
"""Memory leak detection test for sparse_maq solver."""

import polars as pl
import tracemalloc
import gc
from sparse_maq.mckp import Solver

def main():
    print("\n=== MEMORY LEAK DETECTION TEST ===\n")

    # Load data once
    print("Loading data...")
    treatments = pl.read_parquet('data/treatments.parquet')
    patients = pl.read_parquet('data/patients.parquet')
    data = pl.read_parquet('data/data.parquet')

    print(f"Data loaded: {len(data)} rows\n")

    tracemalloc.start()

    for i in range(5):
        print(f"Run {i+1}...")
        solver = Solver(patients, treatments)
        solver.fit(data, budget=1.0)

        current, peak = tracemalloc.get_traced_memory()
        print(f"  Current={current/1024**3:.2f} GB, Peak={peak/1024**3:.2f} GB")

        del solver
        gc.collect()
        print()

    tracemalloc.stop()

    print("Test complete!")
    print("Expected: Memory should stabilize after first run.")
    print("If memory grows consistently, there may be a leak.")

if __name__ == '__main__':
    main()
