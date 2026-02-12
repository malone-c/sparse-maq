from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t

cdef extern from "compute_path.hpp" namespace "sparse_maq":
    cppclass solution_path:
        vector[double] cost_path
        vector[double] reward_path
        vector[size_t] i_path
        vector[size_t] k_path
        bool complete

cdef extern from "pipeline.h" namespace "sparse_maq":
    # TODO: Put the raw elements of the solution path in here
    # instead of the solution path object.
    cppclass solver_output:
        solution_path path
        vector[string] treatment_id_mapping

    solver_output run(
        vector[vector[string]]& treatment_id_arrays,
        vector[vector[double]]& reward_arrays,
        vector[vector[double]]& cost_arrays,
        double budget
    )
