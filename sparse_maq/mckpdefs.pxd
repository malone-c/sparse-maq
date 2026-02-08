from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t

ctypedef pair[vector[vector[double]], vector[vector[size_t]]] solution_path

cdef extern from "pipeline.h" namespace "sparse_maq":
    solution_path run(
        vector[vector[string]]& treatment_id_arrays,
        vector[vector[double]]& reward_arrays,
        vector[vector[double]]& cost_arrays,
        double budget
    )
