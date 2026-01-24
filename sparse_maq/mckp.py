import numpy as np
from numpy import typing as npt
import pyarrow as pa
import polars as pl
from typing import cast

import gc
import time
import tracemalloc
import os
import math

from dataclasses import dataclass
from beartype import beartype

from .ext import solver_cpp

@beartype
@dataclass
class SolverOutput():
    spend: npt.NDArray[np.float64]
    gain: npt.NDArray[np.float64]
    ipath: npt.NDArray[np.int64]
    kpath: npt.NDArray[np.int64]
    complete_path: bool
    # treatment_id_mapping: npt.NDArray[np.int64]

class Solver:
    budget: float = math.inf
    solver_output: SolverOutput

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

    def fit(
        self,
        data: pl.DataFrame,
        budget: float = 0.0,
        n_threads: int = 0,
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

        # PHASE 1: Treatment ID mapping (explode/join/group_by)
        # convert treatment ID to treatment number

        # Step 1a: Select columns
        treatment_nums = data.select('patient_id', 'treatment_id')

        if PROFILE:
            t_current = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            print(f"Phase 1a - select: {t_current - t_last:.2f}s, Peak: {peak/1024**3:.2f} GB")
            print(f"  size after select: {treatment_nums.estimated_size()/1024**3:.2f} GB")
            t_last = t_current

        # Step 1b: Explode treatment_id
        treatment_nums = treatment_nums.explode('treatment_id')

        if PROFILE:
            t_current = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            print(f"Phase 1b - explode: {t_current - t_last:.2f}s, Peak: {peak/1024**3:.2f} GB")
            print(f"  size after explode: {treatment_nums.estimated_size()/1024**3:.2f} GB")
            print(f"  rows after explode: {len(treatment_nums):,}")
            t_last = t_current

        # Step 1c: Join with treatment mapping
        treatment_nums = treatment_nums.join(self.treatment_id_mapping, on='treatment_id')

        if PROFILE:
            t_current = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            print(f"Phase 1c - join: {t_current - t_last:.2f}s, Peak: {peak/1024**3:.2f} GB")
            print(f"  size after join: {treatment_nums.estimated_size()/1024**3:.2f} GB")
            t_last = t_current

        # Step 1d: Select and rename
        treatment_nums = treatment_nums.select('patient_id', treatment_id='treatment_num')

        # Step 1e: Group by and aggregate
        treatment_nums = treatment_nums.group_by('patient_id').agg('treatment_id')

        if PROFILE:
            t_current = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            print(f"Phase 1d-e - select/group_by: {t_current - t_last:.2f}s, Peak: {peak/1024**3:.2f} GB")
            print(f"  final treatment_nums size: {treatment_nums.estimated_size()/1024**3:.2f} GB")
            t_last = t_current

        # PHASE 2: Data join and sort
        data = (
            data
                .drop('treatment_id')
                .join(treatment_nums, on='patient_id') # replace treatment_id with treatment_num
                .sort('patient_id')
        )
        del treatment_nums
        gc.collect()  # Force garbage collection to free memory

        if PROFILE:
            t_current = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            print(f"Phase 2 - data_join_sort: {t_current - t_last:.2f}s, Peak: {peak/1024**3:.2f} GB")
            print(f"  data size: {data.estimated_size()/1024**3:.2f} GB")
            t_last = t_current

        # PHASE 3: Arrow conversion
        table: pa.Table = pl.DataFrame(data).to_arrow()

        assert 'treatment_id' in table.column_names, "table must contain a treatment_id column."
        assert 'reward' in table.column_names, "table must contain a reward column."
        assert 'cost' in table.column_names, "table must contain a cost column."

        if PROFILE:
            t_current = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            print(f"Phase 3 - arrow_conversion: {t_current - t_last:.2f}s, Peak: {peak/1024**3:.2f} GB")
            print(f"  table size: {table.nbytes/1024**3:.2f} GB")
            t_last = t_current

        # PHASE 4: Type casting and chunk combining
        # cast from large_list to list and combine the chunks so everything is contiguous in memory
        # (helps us iterate over them faster)
        treatment_id_arrays = table.column("treatment_id").cast(pa.list_(pa.uint32())).combine_chunks()
        reward_arrays = table.column("reward").cast(pa.list_(pa.float64())).combine_chunks()
        cost_arrays = table.column("cost").cast(pa.list_(pa.float64())).combine_chunks()

        if PROFILE:
            t_current = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            print(f"Phase 4 - type_casting: {t_current - t_last:.2f}s, Peak: {peak/1024**3:.2f} GB")
            t_last = t_current
        
        solver_output_dict: dict = solver_cpp(
            treatment_id_arrays,
            reward_arrays,
            cost_arrays,
            budget,
            n_threads,
        )

        self.solver_output = SolverOutput(**solver_output_dict)

        if PROFILE:
            t_current = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            print(f"Phase 5 - cpp_solver: {t_current - t_last:.2f}s, Peak: {peak/1024**3:.2f} GB")
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
