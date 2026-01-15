import cython
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
from libcpp.memory cimport shared_ptr, static_pointer_cast
from pyarrow.lib cimport ListArray, CArray, CUInt32Array, CDoubleArray, CListArray, pyarrow_unwrap_array
import numpy as np


from sparse_maq.mckpdefs cimport pair, vector, solution_path, run

cpdef solver_cpp(
    ListArray treatment_id_lists,
    ListArray reward_lists,
    ListArray cost_lists,
    double budget,
    uint32_t num_threads
):
    # Unwrap arrays and cast to list arrays in one step
    cdef shared_ptr[CArray] treatment_id_array = pyarrow_unwrap_array(treatment_id_lists)
    cdef shared_ptr[CArray] reward_array = pyarrow_unwrap_array(reward_lists)
    cdef shared_ptr[CArray] cost_array = pyarrow_unwrap_array(cost_lists)

    cdef shared_ptr[CListArray] treatment_id_list_array = static_pointer_cast[CListArray, CArray](pyarrow_unwrap_array(treatment_id_lists))
    cdef shared_ptr[CListArray] reward_list_array = static_pointer_cast[CListArray, CArray](pyarrow_unwrap_array(reward_lists))
    cdef shared_ptr[CListArray] cost_list_array = static_pointer_cast[CListArray, CArray](pyarrow_unwrap_array(cost_lists))

    # Extract values and cast to appropriate types
    cdef shared_ptr[CUInt32Array] treatment_ids = static_pointer_cast[CUInt32Array, CArray](treatment_id_list_array.get().values())
    cdef shared_ptr[CDoubleArray] rewards = static_pointer_cast[CDoubleArray, CArray](reward_list_array.get().values())
    cdef shared_ptr[CDoubleArray] costs = static_pointer_cast[CDoubleArray, CArray](cost_list_array.get().values())

    cdef vector[vector[uint32_t]] cpp_treatment_ids
    cdef vector[vector[double]] cpp_rewards
    cdef vector[vector[double]] cpp_costs

    cpp_treatment_ids.resize(treatment_id_list_array.get().length())
    cpp_rewards.resize(reward_list_array.get().length())
    cpp_costs.resize(cost_list_array.get().length())

    cdef int i, j, offset, length
    cdef uint32_t treatment_id

    for i in range(treatment_id_list_array.get().length()):
        offset = treatment_id_list_array.get().value_offset(i) # start
        length = treatment_id_list_array.get().value_length(i) # end

        cpp_treatment_ids[i].resize(length)
        cpp_rewards[i].resize(length)
        cpp_costs[i].resize(length)
        for j in range(length):
            cpp_treatment_ids[i][j] = treatment_ids.get().Value(offset + j)    
            cpp_rewards[i][j] = rewards.get().Value(offset + j)
            cpp_costs[i][j] = costs.get().Value(offset + j)

    path = run(
        cpp_treatment_ids,
        cpp_rewards,
        cpp_costs,
        budget
    )

    res = dict()
    path_len = path.first[0].size()

    spend = np.empty(path_len, dtype="double")
    gain = np.empty(path_len, dtype="double")
    ipath = np.empty(path_len, dtype="long")
    kpath = np.empty(path_len, dtype="long")

    # faster copy into nparrays with memoryviews
    cdef double[::] view_spend = spend
    cdef double[::] view_gain = gain
    cdef long[::] view_ipath = ipath
    cdef long[::] view_kpath = kpath

    for i in range(path_len):
        view_spend[i] = path.first[0][i]
        view_gain[i] = path.first[1][i]
        view_ipath[i] = path.second[0][i]
        view_kpath[i] = path.second[1][i]

    res["spend"] = spend
    res["gain"] = gain
    res["ipath"] = ipath
    res["kpath"] = kpath
    if path.second[2][0] > 0:
        res["complete_path"] = True
    else:
        res["complete_path"] = False
    return res
