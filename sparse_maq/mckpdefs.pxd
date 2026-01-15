from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t

ctypedef pair[vector[vector[double]], vector[vector[size_t]]] solution_path

cdef extern from "MAQ.h" namespace "sparse_maq":
    solution_path run(
        vector[vector[uint32_t]]& treatment_id_arrays,
        vector[vector[double]]& reward_arrays,
        vector[vector[double]]& cost_arrays,
        double budget
    )