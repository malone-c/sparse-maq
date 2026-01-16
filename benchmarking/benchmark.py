# %% imports
import polars as pl
import numpy as np

# %% experiment
# for each person get the ...j
n = 100
p = 10
eligibility_prob = 0.2
unique_treatment_ids = pl.DataFrame({'treatment_id': [str(i) for i in range(p)]})

df = (
    pl.DataFrame({'patient_id': np.arange(n)})
        .with_columns(
            treatment_id=pl.col('patient_id').map_elements(lambda x: np.arange(p)[
                np.random.binomial(1, eligibility_prob, p).astype(bool)
            ].astype(str))
        )
        .filter(pl.col('treatment_id').list.len() > 0)
        .with_columns(
            reward=pl.col('treatment_id').map_elements(lambda x: np.random.standard_exponential(len(x))),
            cost=pl.col('treatment_id').map_elements(lambda x: np.random.standard_exponential(len(x))),
        )
)

# %% sparse_maq connection
from sparse_maq import Solver
unique_treatment_ids = pl.DataFrame({'treatment_id': np.arange(p)}).cast({'treatment_id': pl.String})
unique_patient_ids = pl.DataFrame({'patient_id': np.arange(n)})
solver = Solver(unique_patient_ids, unique_treatment_ids)
solver.fit(df)



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
