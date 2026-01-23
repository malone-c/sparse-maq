# %% imports
from pathlib import Path
import polars as pl
import numpy as np

# %% data generation fn
def sample_treatment_eligibility_set(k: int, p: float) -> np.ndarray:
    return np.random.binomial(n=1, p=p, size=k)

def generate_data_sparse_maq(n: int, k: int, p: float) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    unique_treatment_ids = np.array([str(i) for i in range(k)])
    unique_treatment_ids_df = pl.DataFrame({'treatment_id': [str(i) for i in range(k)]})
    unique_patient_ids_df = pl.DataFrame({'patient_id': np.arange(n)})
    df = (
        pl.DataFrame({'patient_id': np.arange(n)})
            .with_columns(
                treatment_id=pl.col('patient_id').map_elements(
                    lambda _: unique_treatment_ids[sample_treatment_eligibility_set(k, p).astype(bool)]
                )
            )
            .with_columns(
                treatment_int=pl.col('treatment_id').list.eval(pl.element().cast(pl.Int32)),
                reward=pl.col('treatment_id').map_elements(lambda x: np.random.standard_exponential(len(x))),
                cost=pl.col('treatment_id').map_elements(lambda x: np.random.standard_exponential(len(x))),
            )
    )
    return unique_treatment_ids_df, unique_patient_ids_df, df


def generate_data_maq(n: int, k: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    eligibility = np.concatenate([sample_treatment_eligibility_set(k, p).reshape(1, -1) for _ in range(n)], axis=0)
    n_to_generate = np.sum(eligibility)
    random_rewards = np.random.standard_exponential(n_to_generate)
    random_costs = np.random.standard_exponential(n_to_generate)

    reward = np.zeros((n, k))
    reward[eligibility == 1] = random_rewards

    cost = np.full((n, k), np.inf)
    cost[:] = np.inf
    cost[eligibility == 1] = random_costs

    return reward, cost


def generate_data(n: int, k: int, p: float, temp_dir: Path, solver: str) -> None:
    print(f"Generating data for n={n}, k={k}")

    temp_dir.mkdir(exist_ok=True)
    if solver == 'maq':
        print("  Generating MAQ data...")
        reward, cost = generate_data_maq(n, k, p)
        np.save(temp_dir / 'reward.npy', reward)
        np.save(temp_dir / 'cost.npy', cost)
        print("  MAQ data generation complete")

    elif solver == 'sparse_maq':
        print("  Generating sparse MAQ data...")
        treatments, patients, df = generate_data_sparse_maq(n, k, p)
        treatments.write_parquet(temp_dir / 'treatments.parquet')
        patients.write_parquet(temp_dir / 'patients.parquet')
        df.write_parquet(temp_dir / 'data.parquet')
        print("  Sparse MAQ data generation complete")

    else:
        raise Exception('Give a proper value of solver')
if __name__ == '__main__':
    pass
    # generate_data(n=100_000, k=100, p=0.1, temp_dir=Path('data') / 'temp', solver='maq')



