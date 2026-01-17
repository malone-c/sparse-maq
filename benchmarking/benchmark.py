# %% imports
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

# %% sparse_maq connection
from sparse_maq import Solver

treatments, patients, df = generate_data_sparse_maq(100, 10)
solver = Solver(treatments, patients)
solver.fit(df)

# %% maq connection
from maq import MAQ

# %% generate data
def generate_data_maq(n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    eligibility = np.concatenate([sample_treatment_eligibility_set(k).reshape(1, -1) for _ in range(n)], axis=0)
    reward = eligibility * np.random.standard_exponential((n, k))
    cost = np.zeros((n, k))
    cost[:] = np.inf
    cost[eligibility == 1] = 1
    cost *= np.random.standard_exponential((n, k))

    return reward, cost

# reward and cost must be nxk matrices
solver = MAQ()
reward, cost = generate_data_maq(100, 10)
print(cost)
# solver.fit(reward, cost, reward)

# %% fit MAQ
solver.fit(reward, cost, reward)

# %% look at outputs
solver.plot()
import matplotlib.pyplot as plt
plt.show()

# %% script
def generate_data(
    n_patients: int, 
    n_treatments: int,
    eligible_proportion: float
) -> pl.DataFrame:
    np.rand.exp
    return pl.DataFrame()

if __name__ == '__main__':
    print(generate_data(100, 10, 0.3))
