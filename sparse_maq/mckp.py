import numpy as np
import pyarrow as pa
import polars as pl
from typing import cast
import gc

from .ext import solver_cpp

class Solver:
    def __init__(self, unique_patients: pl.DataFrame, unique_treatments: pl.DataFrame):
        assert unique_patients.columns == ['patient_id']
        assert unique_treatments.columns == ['treatment_id']

        # TODO: Handle this stuff inside the C++ extension
        self.patient_id_mapping = unique_patients.with_columns(patient_num=pl.col("patient_id").rank("dense") - 1)

        self.treatment_id_mapping = (
            unique_treatments
                .filter(pl.col('treatment_id').str.to_lowercase() != 'dns') # remove DNS
                .with_columns(treatment_num=pl.col("treatment_id").rank("dense").cast(pl.Int64))
                .vstack(pl.DataFrame({'treatment_id': 'dns', 'treatment_num': 0})) # Add DNS back as 0
        )

        self._path = None
        self._is_fit = False
        self.budget = None

    def fit(
        self,
        data: pl.DataFrame,
        budget: np.float64 = np.finfo(np.float64).max,
        n_threads: int = 0,
    ):
        """Solve the multi-armed knapsack problem using a path algorithm."""
        # TODO: Data validation
            # Check typing of columns (list types)
            # Check inner arrays have consistent lengths across outer arrays
                # e.g. all(len(treatment_id_arrays[i]) == len(reward_arrays[i]) == len(cost_arrays[i]) for i in range(len(treatment_id_arrays)))
            # Check for nans/nulls

        assert np.isscalar(budget), "budget should be a scalar."
        assert n_threads >= 0, "n_threads should be >=0."

        self.budget = budget

        # convert treatment ID to treatment number
        treatment_nums = (
            data
                .select('patient_id', 'treatment_id')
                .explode('treatment_id')
                .join(self.treatment_id_mapping, on='treatment_id')
                .select('patient_id', treatment_id='treatment_num')
                .group_by('patient_id')
                .agg('treatment_id')
        )

        data = (
            data
                .drop('treatment_id') 
                .join(treatment_nums, on='patient_id') # replace treatment_id with treatment_num
                .sort('patient_id')
        )
        del treatment_nums
        gc.collect()  # Force garbage collection to free memory

        table: pa.Table = pl.DataFrame(data).to_arrow()
        
        assert 'treatment_id' in table.column_names, "table must contain a treatment_id column."
        assert 'reward' in table.column_names, "table must contain a reward column."
        assert 'cost' in table.column_names, "table must contain a cost column."

        # cast from large_list to list and combine the chunks so everything is contiguous in memory
        # (helps us iterate over them faster)
        treatment_id_arrays = table.column("treatment_id").cast(pa.list_(pa.uint32())).combine_chunks()
        reward_arrays = table.column("reward").cast(pa.list_(pa.float64())).combine_chunks()
        cost_arrays = table.column("cost").cast(pa.list_(pa.float64())).combine_chunks()

        self._path = solver_cpp(
            treatment_id_arrays,
            reward_arrays,
            cost_arrays,
            budget,
            n_threads,
        )

        self._is_fit = True
        return self._path

    def predict(self, budget):
        """Get the treatment allocation for a given budget."""
        assert np.isscalar(budget), "spend should be a scalar."
        assert self._is_fit, "Solver object is not fit."
        if not self._path["complete_path"] and budget > self.budget:
            raise ValueError("Path is not fit beyond given spend level. Refit with a larger budget.")
        
        # Idea: Get the last setting for each patient, before the spend exceeds the budget
        # For any patient we don't see, set their treatment to 0 (control)
        return (
            pl.DataFrame({
                'spend': self._path["spend"],
                'gain': self._path["gain"],
                'patient_num': self._path["ipath"],
                'treatment_num': self._path["kpath"],
                'time': list(range(len(self._path["spend"]))),
            })
            .filter(pl.col('spend') <= budget)
            .group_by('patient_num')
            .agg(pl.col('treatment_num').sort_by('time').last()) # get the last treatment_num for each patient
            .join(self.patient_id_mapping, on='patient_num', how='right') # get patient IDs and append control (right join)
            .select('patient_id', treatment_num=pl.coalesce('treatment_num', 0)) # replace nulls with 0
            .join(self.treatment_id_mapping, on='treatment_num') # replace treatment_num with treatment_id
            .select('patient_id', 'treatment_id')
        )

    @property
    def path_(self):
        """Get the path of the solver."""
        assert self._is_fit, "Solver object is not fit."
        return cast(dict, self._path)

    @property
    def path_spend_(self):
        assert self._is_fit, "Solver object is not fit."
        return self._path["spend"]

    @property
    def path_gain_(self):
        assert self._is_fit, "Solver object is not fit."
        return self._path["gain"]

    @property
    def path_std_err_(self):
        assert self._is_fit, "Solver object is not fit."
        return self._path["std_err"]

    @property
    def path_allocated_unit_(self):
        assert self._is_fit, "Solver object is not fit."
        return self._path["ipath"]

    @property
    def path_allocated_arm_(self):
        assert self._is_fit, "Solver object is not fit."
        return self._path["kpath"]