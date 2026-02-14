import numpy as np
from numpy import typing as npt
import pyarrow as pa
import polars as pl
from typing import cast

import gc
import time
import tracemalloc
import os
from dataclasses import dataclass
from beartype import beartype

from .ext import solver_cpp, solver_cpp_old

@beartype
@dataclass
class SolverOutput():
    spend: npt.NDArray[np.float64]
    gain: npt.NDArray[np.float64]
    ipath: npt.NDArray[np.int64]
    kpath: npt.NDArray[np.int64]
    complete_path: bool
    treatment_id_mapping: npt.NDArray

class Solver:
    def __init__(self):
        self._is_fit = False

    def fit_from_polars(
        self,
        data: pl.DataFrame,
        budget: float = 0.0,
        n_threads: int = 0,
        use_flat_buffers: bool = True,
    ) -> SolverOutput:
        """Solve the multi-armed knapsack problem using a path algorithm."""
        # TODO: Data validation
        # Check typing of columns (list types)
        # Check inner arrays have consistent lengths across outer arrays
        # e.g. all(len(treatment_id_arrays[i]) == len(reward_arrays[i]) == len(cost_arrays[i]) for i in range(len(treatment_id_arrays)))
        # Check for nans/nulls

        assert n_threads >= 0, "n_threads should be >=0."

        self.budget = budget

        # Profiling setup
        PROFILE = os.environ.get('SPARSE_MAQ_PROFILE', '0') == '1'
        if PROFILE:
            tracemalloc.start()
            print("\n=== SPARSE_MAQ PROFILING ===")
            t_start = time.perf_counter()
            t_last = t_start
            data_original_size = data.estimated_size() / 1024**3
            print(f"Input data size: {data_original_size:.2f} GB")

        data = data.sort('patient_id')

        if PROFILE:
            t_current = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            print(f"sort: {t_current - t_last:.2f}s, Peak: {peak/1024**3:.2f} GB")
            t_last = t_current

        table: pa.Table = pl.DataFrame(data).to_arrow()

        assert 'treatment_id' in table.column_names, "table must contain a treatment_id column."
        assert 'reward' in table.column_names, "table must contain a reward column."
        assert 'cost' in table.column_names, "table must contain a cost column."

        if PROFILE:
            t_current = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            print(f"arrow_conversion: {t_current - t_last:.2f}s, Peak: {peak/1024**3:.2f} GB")
            print(f"  table size: {table.nbytes/1024**3:.2f} GB")
            t_last = t_current

        # cast from large_list to list and combine the chunks so everything is contiguous in memory
        treatment_id_arrays = table.column("treatment_id").cast(pa.list_(pa.string())).combine_chunks()
        reward_arrays = table.column("reward").cast(pa.list_(pa.float64())).combine_chunks()
        cost_arrays = table.column("cost").cast(pa.list_(pa.float64())).combine_chunks()

        if PROFILE:
            t_current = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            print(f"type_casting: {t_current - t_last:.2f}s, Peak: {peak/1024**3:.2f} GB")
            t_last = t_current
        
        _solver = solver_cpp if use_flat_buffers else solver_cpp_old
        solver_output_dict: dict = _solver(
            treatment_id_arrays,
            reward_arrays,
            cost_arrays,
            budget,
            n_threads,
        )

        self.solver_output = SolverOutput(**solver_output_dict)
        self.treatment_id_mapping = pl.DataFrame({
            'treatment_num': list(range(len(self.solver_output.treatment_id_mapping))),
            'treatment_id': self.solver_output.treatment_id_mapping,
        })

        if PROFILE:
            t_current = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            print(f"cpp_solver: {t_current - t_last:.2f}s, Peak: {peak/1024**3:.2f} GB")
            print(f"\nTotal time: {t_current - t_start:.2f}s")
            print(f"Total peak memory: {peak/1024**3:.2f} GB")
            tracemalloc.stop()

        self._is_fit = True
        return self.solver_output

    def predict(self, budget: float):
        """Get the treatment allocation for a given budget."""
        assert self._is_fit, "Solver object is not fit."
        if not self.solver_output.complete_path and budget > self.budget:
            raise ValueError("Path is not fit beyond given spend level. Refit with a larger budget.")
        
        # Idea: Get the last setting for each patient, before the spend exceeds the budget
        # For any patient we don't see, set their treatment to 0 (control)
        return (
            pl.DataFrame({
                'spend': self.solver_output.spend,
                'gain': self.solver_output.gain,
                'patient_num': self.solver_output.ipath,
                'treatment_num': self.solver_output.kpath,
                'time': list(range(len(self.solver_output.spend))),
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
        return cast(dict, self.solver_output)

    @property
    def path_spend_(self):
        assert self._is_fit, "Solver object is not fit."
        return self.solver_output.spend

    @property
    def path_gain_(self):
        assert self._is_fit, "Solver object is not fit."
        return self.solver_output.gain

    @property
    def path_allocated_unit_(self):
        assert self._is_fit, "Solver object is not fit."
        return self.solver_output.ipath

    @property
    def path_allocated_arm_(self):
        assert self._is_fit, "Solver object is not fit."
        return self.solver_output.kpath
