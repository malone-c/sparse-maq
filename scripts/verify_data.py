import numpy as np
import polars as pl

print("=== MAQ Data (Dense Format) ===")
rewards = np.load('data/temp/rewards.npy')
costs = np.load('data/temp/costs.npy')
print(f"Rewards shape: {rewards.shape}")
print(f"Costs shape: {costs.shape}")
print(f"Rewards non-zero elements: {np.count_nonzero(rewards)}")
print(f"Costs finite elements: {np.sum(~np.isinf(costs))}")
print(f"Rewards sample values: {rewards[rewards != 0][:5]}")

print("\n=== Sparse MAQ Data (Sparse Format) ===")
# Use scan_parquet for efficiency
data_lazy = pl.scan_parquet('data/temp/data.parquet')
patients_count = pl.scan_parquet('data/temp/patients.parquet').select(pl.len()).collect().item()
treatments_count = pl.scan_parquet('data/temp/treatments.parquet').select(pl.len()).collect().item()
data_count = data_lazy.select(pl.len()).collect().item()

print(f"Patients: {patients_count}")
print(f"Treatments: {treatments_count}")
print(f"Data rows: {data_count}")

# Count total entries in sparse format
total_entries = data_lazy.select(
    pl.col('treatment_int').list.len().sum()
).collect().item()
print(f"Total sparse entries: {total_entries}")

print("\n=== Comparison ===")
print(f"Dense format: {rewards.shape[0]} x {rewards.shape[1]} = {rewards.shape[0] * rewards.shape[1]:,} total elements")
print(f"Dense non-zero/finite: {np.count_nonzero(rewards):,} entries")
print(f"Sparse format: {total_entries:,} entries")
print(f"Match: {np.count_nonzero(rewards) == total_entries}")
