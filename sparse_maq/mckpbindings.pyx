import cython
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport int32_t, int64_t, uint8_t, uint32_t
from libcpp.memory cimport shared_ptr, static_pointer_cast
from libcpp.utility cimport move
from pyarrow.lib cimport ListArray, CArray, CStringArray, CDoubleArray, CListArray, pyarrow_unwrap_array
import numpy as np
import time
import os


from sparse_maq.mckpdefs cimport (
    vector, string, solution_path, solver_output, run_flat,
    CDoubleArrayRaw, CStringArrayRaw,
)

cpdef solver_cpp(
    ListArray treatment_id_lists,
    ListArray reward_lists,
    ListArray cost_lists,
    double budget,
    uint32_t num_threads
):
    cdef bint PROFILE = os.environ.get('SPARSE_MAQ_PROFILE', '0') == '1'
    cdef double t0, t1, t2, t3, t4

    if PROFILE:
        t0 = time.perf_counter()

    # Unwrap arrays and cast to list arrays in one step
    cdef shared_ptr[CListArray] treatment_id_list_array = static_pointer_cast[CListArray, CArray](pyarrow_unwrap_array(treatment_id_lists))
    cdef shared_ptr[CListArray] reward_list_array = static_pointer_cast[CListArray, CArray](pyarrow_unwrap_array(reward_lists))
    cdef shared_ptr[CListArray] cost_list_array = static_pointer_cast[CListArray, CArray](pyarrow_unwrap_array(cost_lists))

    # Extract values and cast to appropriate types
    cdef shared_ptr[CStringArray] treatment_ids = static_pointer_cast[CStringArray, CArray](treatment_id_list_array.get().values())
    cdef shared_ptr[CDoubleArray] rewards = static_pointer_cast[CDoubleArray, CArray](reward_list_array.get().values())
    cdef shared_ptr[CDoubleArray] costs = static_pointer_cast[CDoubleArray, CArray](cost_list_array.get().values())

    if PROFILE:
        t1 = time.perf_counter()
        print(f"  Cython: PyArrow unwrap: {t1-t0:.2f}s")

    cdef int64_t n_treatments = rewards.get().length()
    cdef int64_t n_patients = treatment_id_list_array.get().length()

    # Cast to extended types to call raw buffer methods not declared in
    # pyarrow's pxd. The underlying C++ class is identical — this is safe.
    cdef const double* rewards_ptr = (<CDoubleArrayRaw*>rewards.get()).raw_values()
    cdef const double* costs_ptr = (<CDoubleArrayRaw*>costs.get()).raw_values()
    cdef const int32_t* reward_offsets_ptr = reward_list_array.get().raw_value_offsets()

    # String IDs: raw pointer + offset from CStringArray
    cdef const int32_t* str_offsets_ptr = (<CStringArrayRaw*>treatment_ids.get()).raw_value_offsets()
    cdef const uint8_t* str_data_ptr = (<CStringArrayRaw*>treatment_ids.get()).raw_data()
    cdef int32_t str_data_len = str_offsets_ptr[n_treatments]

    # Bulk copy: .assign(begin, end) — one memcpy-equivalent per column.
    # Cython doesn't support pointer arithmetic in cdef constructor args,
    # so we declare first then assign.
    cdef vector[double] cpp_rewards_flat
    cdef vector[double] cpp_costs_flat
    cdef vector[int32_t] cpp_list_offsets
    cdef vector[int32_t] cpp_str_offsets
    cdef vector[uint8_t] cpp_str_data

    cpp_rewards_flat.assign(rewards_ptr, rewards_ptr + n_treatments)
    cpp_costs_flat.assign(costs_ptr, costs_ptr + n_treatments)
    cpp_list_offsets.assign(reward_offsets_ptr, reward_offsets_ptr + n_patients + 1)
    cpp_str_offsets.assign(str_offsets_ptr, str_offsets_ptr + n_treatments + 1)
    cpp_str_data.assign(str_data_ptr, str_data_ptr + str_data_len)

    if PROFILE:
        t2 = time.perf_counter()
        print(f"  Cython: Flat buffer extraction: {t2-t1:.2f}s")

    cdef solver_output result = run_flat(
        n_patients,
        move(cpp_list_offsets),
        move(cpp_rewards_flat),
        move(cpp_costs_flat),
        move(cpp_str_offsets),
        move(cpp_str_data),
        budget
    )

    if PROFILE:
        t3 = time.perf_counter()
        print(f"  Cython: C++ solver call: {t3-t2:.2f}s")

    # cdef solution_path path = result.path
    # cdef vector[string] treatment_id_map = result.treatment_id_mapping

    res = dict()
    path_len = result.path.cost_path.size()

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
        view_spend[i] = result.path.cost_path[i]
        view_gain[i] = result.path.reward_path[i]
        view_ipath[i] = result.path.i_path[i]
        view_kpath[i] = result.path.k_path[i]

    res["spend"] = spend
    res["gain"] = gain
    res["ipath"] = ipath
    res["kpath"] = kpath
    res["complete_path"] = True if result.path.complete else False
    res["treatment_id_mapping"] = np.array(
        [result.treatment_id_mapping[i].decode('utf-8') for i in range(result.treatment_id_mapping.size())],
        dtype=str
    )
    return res
