# %% imports
from pathlib import Path
import polars as pl
import numpy as np

# %% data generation fn
def sample_treatment_eligibility_set(k: int) -> np.ndarray:
    subset_size = np.random.randint(1, k + 1)
    indices = np.random.choice(k, subset_size, replace=False)
    treatments = np.zeros(k, dtype=int)
    treatments[indices] = 1
    return treatments


def generate_data_sparse_maq(n: int, k: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    unique_treatment_ids = np.array([str(i) for i in range(k)])
    unique_treatment_ids_df = pl.DataFrame({'treatment_id': [str(i) for i in range(k)]})
    unique_patient_ids_df = pl.DataFrame({'patient_id': np.arange(n)})
    df = (
        pl.DataFrame({'patient_id': np.arange(n)})
            .with_columns(
                treatment_id=pl.col('patient_id').map_elements(lambda _: unique_treatment_ids[sample_treatment_eligibility_set(k)])
            )
            .with_columns(
                treatment_int=pl.col('treatment_id').list.eval(pl.element().cast(pl.Int32)),
                reward=pl.col('treatment_id').map_elements(lambda x: np.random.standard_exponential(len(x))),
                cost=pl.col('treatment_id').map_elements(lambda x: np.random.standard_exponential(len(x))),
            )
    )
    return unique_treatment_ids_df, unique_patient_ids_df, df


def generate_data_maq(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    eligibility = np.concatenate([sample_treatment_eligibility_set(k).reshape(1, -1) for _ in range(n)], axis=0)
    n_to_generate = np.sum(eligibility)
    random_rewards = np.random.standard_exponential(n_to_generate)
    random_costs = np.random.standard_exponential(n_to_generate)

    reward = np.zeros((n, k))
    reward[eligibility == 1] = random_rewards

    cost = np.full((n, k), np.inf)
    cost[:] = np.inf
    cost[eligibility == 1] = random_costs

    return reward, cost


def generate_data(n: int, k: int) -> None:
    print(f"Generating data for n={n}, k={k}")
    base_path = Path('data') / f'{n=}_{k=}'

    (base_path / 'sparse_maq').mkdir(parents=True, exist_ok=True)
    (base_path / 'maq').mkdir(parents=True, exist_ok=True)

    print("  Generating MAQ data...")
    reward, cost = generate_data_maq(n, k)
    np.save(base_path / 'maq' / 'reward.npy', reward)
    np.save(base_path / 'maq' / 'cost.npy', cost)
    print("  MAQ data generation complete")

    print("  Generating sparse MAQ data...")
    treatments, patients, df = generate_data_sparse_maq(n, k)
    treatments.write_parquet(base_path / 'sparse_maq' / 'treatments.parquet')
    patients.write_parquet(base_path / 'sparse_maq' / 'patients.parquet')
    df.write_parquet(base_path / 'sparse_maq' / 'data.parquet')
    print("  Sparse MAQ data generation complete")

if __name__ == '__main__':
    for n in range(100_000, 1_000_001, 100_000):
        for k in range(100, 1_001, 100):
            generate_data(n, k)



