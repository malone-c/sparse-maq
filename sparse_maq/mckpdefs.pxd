from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport int32_t, int64_t, uint8_t, uint32_t

# Arrow C++ methods not exposed in pyarrow's lib.pxd.
# We alias the same C++ types under new Cython names so we can cast and call
# the methods we need without modifying pyarrow's declarations.
cdef extern from *:
    """
    #include "arrow/array.h"
    """
    cdef cppclass CDoubleArrayRaw "arrow::DoubleArray":
        const double* raw_values()

    cdef cppclass CStringArrayRaw "arrow::StringArray":
        const int32_t* raw_value_offsets()
        const uint8_t* raw_data()

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

    solver_output run_flat(
        int64_t num_patients,
        vector[int32_t]&& list_offsets,
        vector[double]&& rewards_flat,
        vector[double]&& costs_flat,
        vector[int32_t]&& str_offsets,
        vector[uint8_t]&& str_data,
        double budget
    )
