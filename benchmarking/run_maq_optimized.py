import numpy as np
from maq import MAQ
import time
import tracemalloc

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--base-path', type=str)
    args = parser.parse_args()

    print("\n=== MAQ OPTIMIZED PROFILING ===")

    tracemalloc.start()
    t0 = time.perf_counter()

    # Load data (sparse parquet format)
    import polars as pl
    df = pl.read_parquet(f'{args.base_path}/data.parquet')

    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    print(f"Load data: {t1-t0:.2f}s, Peak Memory: {peak/1024**3:.2f} GB")

    # Transform: build condensed dense matrices + mapping
    treatment_id_lists = df["treatment_int"].to_list()
    reward_lists = df["reward"].to_list()
    cost_lists = df["cost"].to_list()

    n = len(treatment_id_lists)
    n_eligible_per_patient = np.array([len(t) for t in treatment_id_lists], dtype=np.int64)
    max_eligible_per_patient = int(n_eligible_per_patient.max())

    treatment_ids_flat = np.concatenate(treatment_id_lists).astype(np.int32)
    rewards_flat = np.concatenate(reward_lists)
    costs_flat = np.concatenate(cost_lists)

    patient_ids_flattened = np.repeat(np.arange(n, dtype=np.int64), n_eligible_per_patient)
    patient_offsets = np.zeros(n + 1, dtype=np.int64)
    patient_offsets[1:] = n_eligible_per_patient.cumsum()
    patient_specific_treatment_ids = (
        np.arange(len(patient_ids_flattened), dtype=np.int64)
        - patient_offsets[patient_ids_flattened]
    )

    rewards_mat = np.zeros((n, max_eligible_per_patient))
    costs_mat = np.full((n, max_eligible_per_patient), np.inf)
    treatment_ids_mat = np.full((n, max_eligible_per_patient), -1, dtype=np.int32)

    rewards_mat[patient_ids_flattened, patient_specific_treatment_ids] = rewards_flat
    costs_mat[patient_ids_flattened, patient_specific_treatment_ids] = costs_flat
    treatment_ids_mat[patient_ids_flattened, patient_specific_treatment_ids] = treatment_ids_flat

    t2 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    dense_size = n * max_eligible_per_patient
    full_dense_size = n * treatment_ids_flat.max() if len(treatment_ids_flat) > 0 else 1
    k = int(treatment_ids_flat.max()) + 1 if len(treatment_ids_flat) > 0 else 1
    savings = k / max_eligible_per_patient
    print(f"Transform (condensed matrix + mapping): {t2-t1:.2f}s, Peak Memory: {peak/1024**3:.2f} GB")
    print(f"  Condensed shape: {n} x {max_eligible_per_patient} (vs {n} x {k} dense)")
    print(f"  Memory savings: ~{savings:.1f}x vs dense")

    # Fit solver
    solver = MAQ()
    t3 = time.perf_counter()
    solver.fit(rewards_mat, costs_mat, rewards_mat)
    t4 = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()
    print(f"Fit solver: {t4-t3:.2f}s, Peak Memory: {peak/1024**3:.2f} GB")
    print(f"\nTotal time: {t4-t0:.2f}s")
    print(f"Total peak memory: {peak/1024**3:.2f} GB")
    tracemalloc.stop()
