# Performance Optimization Plan: sparse_maq

**Based on**: Performance Investigation Report (January 22, 2026)
**Target**: Reduce execution time from 116s to ~50-60s range
**Current Bottlenecks**: Polars explode (30.7s), Cython copying (10.8s), data transformations (11-20s)

## Executive Summary

This document outlines three major optimization opportunities identified through comprehensive profiling of the sparse_maq solver. These optimizations target the data transformation pipeline, which currently accounts for ~68.5 seconds of overhead (59% of total execution time).

**Estimated Total Impact**: 40-55 seconds improvement (35-47% faster)

### Quick Reference

| Optimization | Effort | Impact | Risk | Priority |
|--------------|--------|--------|------|----------|
| 1. Cython memcpy batch copying | Low (2-4 hours) | 8-10s | Low | **HIGH** |
| 2. Eliminate explode operation | High (2-3 days) | 20-25s | Medium | **HIGH** |
| 3. Reduce transformation passes | Medium (1-2 days) | 10-15s | Low | **MEDIUM** |

---

## Optimization 1: Eliminate Polars Explode Operation

**Current Cost**: 30.70 seconds (26.4% of total time)
**Estimated Savings**: 20-25 seconds
**Effort**: High (2-3 days)
**Risk**: Medium (requires architectural changes)

### Problem

The current implementation converts treatment IDs (strings like "treatment_42") to treatment numbers (integers 0-500) using an explode-join-aggregate pattern:

```python
# Current approach: mckp.py:61-69
treatment_nums = (
    data
        .select('patient_id', 'treatment_id')
        .explode('treatment_id')           # ← 1M rows → 250M rows (30.70s!)
        .join(self.treatment_id_mapping, on='treatment_id')
        .select('patient_id', treatment_id='treatment_num')
        .group_by('patient_id')
        .agg('treatment_id')
)
```

**Why it's slow**:
- Creates a 250M-row intermediate DataFrame (2.51 GB)
- Polars must materialize all 250M rows in memory
- Join operation on 250M rows
- Group-by to collapse back to 1M rows

### Solution A: Polars List Operations (Recommended)

**Approach**: Operate directly on list elements without exploding

**Implementation**:

```python
# Build a dictionary mapping for fast lookup
mapping_dict = dict(zip(
    self.treatment_id_mapping['treatment_id'],
    self.treatment_id_mapping['treatment_num']
))

# Option 1: Using list.eval()
treatment_nums = data.with_columns(
    treatment_num=pl.col('treatment_id').list.eval(
        pl.element().replace_strict(mapping_dict, default=None)
    )
).select('patient_id', 'treatment_num')

# Option 2: Using apply() if eval doesn't support replace
treatment_nums = data.with_columns(
    treatment_num=pl.col('treatment_id').list.eval(
        pl.element().map_elements(
            lambda x: mapping_dict.get(x, 0),
            return_dtype=pl.Int64
        )
    )
).select('patient_id', 'treatment_num')
```

**Pros**:
- Stays in Polars ecosystem
- No architectural changes to C++ code
- Should be 5-10x faster than explode
- Memory-efficient (no 250M-row intermediate)

**Cons**:
- Polars list operations may have limited functionality
- May need to use `apply()` which can be slower than native ops
- Need to verify Polars supports dictionary replacement in list.eval()

**Estimated Impact**: 20-25 seconds savings (bringing explode from 30.7s → 5-10s)

### Solution B: Move Mapping to Cython/C++

**Approach**: Pass string arrays to C++, do mapping there

**Implementation**:

```python
# Python layer: mckp.py
def fit(self, data, budget, n_threads=0):
    # Skip the explode/join/group_by entirely
    # Convert treatment_id_mapping to C++ map

    table = data.to_arrow()
    treatment_id_strings = table.column("treatment_id").cast(pa.list_(pa.string())).combine_chunks()

    # Pass string arrays to Cython
    self._path = solver_cpp(
        treatment_id_strings,  # ← pass strings instead of numbers
        reward_arrays,
        cost_arrays,
        budget,
        n_threads,
        self.treatment_id_mapping  # ← pass mapping
    )
```

```cython
# Cython layer: mckpbindings.pyx
cpdef solver_cpp(
    ListArray treatment_id_strings,  # ← changed from uint32
    ListArray reward_lists,
    ListArray cost_lists,
    double budget,
    uint32_t num_threads,
    object id_mapping  # ← new parameter
):
    # Build C++ unordered_map from id_mapping
    cdef unordered_map[string, uint32_t] cpp_mapping
    for row in id_mapping.iter_rows():
        cpp_mapping[row[0]] = row[1]  # treatment_id → treatment_num

    # Extract string arrays
    cdef shared_ptr[CStringArray] treatment_ids = ...

    # Copy and map in one pass
    for i in range(num_patients):
        for j in range(length):
            string tid = treatment_ids.get().Value(offset + j)
            cpp_treatment_ids[i][j] = cpp_mapping[tid]  # lookup during copy
```

```cpp
// C++ layer: No changes needed!
// Still receives vector<vector<uint32_t>>
```

**Pros**:
- Complete elimination of Polars transformation
- Single-pass copy + mapping (faster than separate operations)
- C++ unordered_map is very fast (O(1) lookup)
- Reduces memory pressure (no intermediate DataFrames)

**Cons**:
- Requires Cython changes (medium complexity)
- String handling in Cython is more complex than integers
- Need to pass additional parameter (mapping)
- Slight coupling between Python and Cython layers

**Estimated Impact**: 25-30 seconds savings (complete elimination of explode + join + group_by)

### Solution C: Arrow Dictionary Encoding

**Approach**: Use Arrow's native dictionary type for efficient string→int mapping

**Implementation**:

```python
# Generate data with dictionary encoding from the start
# In data generation script or during initial load:
treatment_ids_dict = pa.DictionaryArray.from_arrays(
    indices=treatment_indices,  # integers
    dictionary=treatment_strings  # unique strings
)

# Arrow handles mapping internally without exploding
```

**Pros**:
- Arrow-native solution (well-optimized)
- Zero-copy in many cases
- Efficient memory representation

**Cons**:
- Requires changes to data generation pipeline
- May not integrate easily with current Polars workflow
- Complex to retrofit to existing code

**Estimated Impact**: 20-25 seconds savings

### Recommendation

**Start with Solution A (Polars list operations)** because:
1. Minimal code changes
2. Low risk (stays in Polars ecosystem)
3. Easy to test and validate
4. Can fall back to current implementation if it fails

**If Solution A doesn't work or is still slow**, implement Solution B (Cython mapping) for maximum benefit.

### Implementation Steps

1. **Research phase** (2 hours):
   - Test Polars `list.eval()` with `replace()` or `map_elements()`
   - Benchmark small example to verify speedup
   - Check Polars documentation for list operation capabilities

2. **Implementation** (4 hours):
   - Modify `sparse_maq/mckp.py:61-69`
   - Add fallback to current implementation if needed
   - Update tests

3. **Validation** (2 hours):
   - Run full benchmark to measure actual savings
   - Verify results match original implementation
   - Check memory usage

**Total Effort**: 1-2 days (including testing and validation)

---

## Optimization 2: Batch Memcpy in Cython Layer

**Current Cost**: 10.76 seconds (9.3% of total time)
**Estimated Savings**: 8-10 seconds
**Effort**: Low (2-4 hours)
**Risk**: Low (localized change)

### Problem

Current Cython code copies data element-by-element using PyArrow accessor methods:

```cython
# Current approach: mckpbindings.pyx:51-54
for j in range(length):
    cpp_treatment_ids[i][j] = treatment_ids.get().Value(offset + j)  # ← 250M calls!
    cpp_rewards[i][j] = rewards.get().Value(offset + j)
    cpp_costs[i][j] = costs.get().Value(offset + j)
```

**Why it's slow**:
- `.Value(offset + j)` is a function call with bounds checking
- 250M treatments × 3 arrays = 750M function calls
- Each call has overhead: bounds check, type conversion, return value handling
- Poor cache locality (multiple arrays accessed in lockstep)

### Solution A: Batch Memcpy (Recommended)

**Approach**: Use raw pointers and memcpy for bulk copying

**Implementation**:

```cython
# Modified: mckpbindings.pyx
from libc.string cimport memcpy

cpdef solver_cpp(...):
    # ... existing unwrap code ...

    # Get raw pointers to Arrow buffer data
    cdef const uint32_t* treatment_id_data = <const uint32_t*>treatment_ids.get().raw_values()
    cdef const double* reward_data = <const double*>rewards.get().raw_values()
    cdef const double* cost_data = <const double*>costs.get().raw_values()

    cdef int i, offset, length
    cdef size_t byte_size_treatment, byte_size_double

    for i in range(treatment_id_list_array.get().length()):
        offset = treatment_id_list_array.get().value_offset(i)
        length = treatment_id_list_array.get().value_length(i)

        # Resize vectors (same as before)
        cpp_treatment_ids[i].resize(length)
        cpp_rewards[i].resize(length)
        cpp_costs[i].resize(length)

        # Batch copy entire segments
        byte_size_treatment = length * sizeof(uint32_t)
        byte_size_double = length * sizeof(double)

        memcpy(&cpp_treatment_ids[i][0], treatment_id_data + offset, byte_size_treatment)
        memcpy(&cpp_rewards[i][0], reward_data + offset, byte_size_double)
        memcpy(&cpp_costs[i][0], cost_data + offset, byte_size_double)
```

**Pros**:
- Very simple change (~10 lines modified)
- Extremely fast (10-50x speedup over element-wise)
- Uses optimized libc memcpy (often vectorized by compiler)
- Low risk (memcpy is well-tested)

**Cons**:
- Assumes contiguous memory layout (should be guaranteed by combine_chunks())
- Need to verify Arrow buffer is accessible via raw_values()
- Slightly less readable than explicit loop

**Estimated Impact**: 8-10 seconds savings (10.76s → 0.5-2s)

### Solution B: Zero-Copy with Views (Maximum Performance)

**Approach**: Don't copy data at all - make C++ work with Arrow buffers directly

**Implementation**:

```cpp
// New C++ structure: Data.hpp
struct ArrowArrayView {
    const uint32_t* treatment_id_data;
    const double* reward_data;
    const double* cost_data;
    const int32_t* offsets;  // list offsets
    int64_t num_patients;

    // Access treatment IDs for patient i
    inline uint32_t get_treatment_id(size_t i, size_t j) const {
        return treatment_id_data[offsets[i] + j];
    }

    inline size_t num_treatments(size_t i) const {
        return offsets[i+1] - offsets[i];
    }
};

// Modify solver to accept views instead of vectors
solution_path run(const ArrowArrayView& data, double budget) {
    // Process data using view accessors instead of vector indexing
    // No copying required!
}
```

```cython
# Cython passes pointers instead of copying
cpdef solver_cpp(...):
    # Extract raw pointers
    cdef const uint32_t* treatment_id_ptr = ...
    cdef const int32_t* offsets_ptr = ...

    # Create view struct (no data copy)
    cdef ArrowArrayView view
    view.treatment_id_data = treatment_id_ptr
    view.offsets = offsets_ptr
    ...

    # Pass view to C++
    return run(view, budget)
```

**Pros**:
- Complete elimination of copying (10.76s → 0s)
- Reduced memory usage (no duplicate vectors)
- Most efficient possible solution

**Cons**:
- Requires significant C++ refactoring
- Changes API of C++ solver (affects testing)
- More complex to maintain
- Accessor functions may be slower than direct vector access
- High implementation effort (2-3 days)

**Estimated Impact**: 10-11 seconds savings (complete elimination)

### Recommendation

**Implement Solution A (batch memcpy)** because:
1. **Quick win**: 2-4 hours implementation
2. **Low risk**: Isolated change in Cython layer
3. **High payoff**: ~90% of theoretical maximum speedup
4. **Easy to validate**: Compare results with existing implementation

Save Solution B (zero-copy) for future optimization if needed.

### Implementation Steps

1. **Add memcpy import** (5 minutes):
   ```cython
   from libc.string cimport memcpy
   ```

2. **Extract raw pointers** (30 minutes):
   - Research Arrow C++ API for `raw_values()` method
   - Add pointer extraction code
   - Add null checks

3. **Replace loop with memcpy** (1 hour):
   - Modify copy loop to use memcpy
   - Calculate byte sizes correctly
   - Handle edge cases (empty lists)

4. **Testing** (1 hour):
   - Verify output matches original implementation
   - Run profiling to measure actual speedup
   - Check for memory issues (valgrind if available)

5. **Rebuild and benchmark** (30 minutes):
   - Rebuild Cython extension
   - Run full benchmark suite
   - Update performance report

**Total Effort**: 2-4 hours

---

## Optimization 3: Reduce Data Transformation Passes

**Current Cost**: ~15-20 seconds (scattered across multiple operations)
**Estimated Savings**: 10-15 seconds
**Effort**: Medium (1-2 days)
**Risk**: Low (incremental improvements)

### Problem

The current pipeline performs 11 distinct transformation passes:

```python
# Passes 1-5: ID mapping (if not eliminated by Optimization 1)
data.select(...)              # 1. Select columns
    .explode(...)            # 2. Explode to 250M rows
    .join(...)               # 3. Join with mapping
    .group_by(...).agg(...)  # 4-5. Group and aggregate

# Passes 6-8: Data preparation
data.drop(...)               # 6. Drop old column
    .join(treatment_nums)    # 7. Join mapped IDs
    .sort(...)               # 8. Sort by patient_id

# Passes 9-11: Arrow conversion
to_arrow()                   # 9. Polars → Arrow
cast(pa.list_(...))         # 10. Type casting (10.89s!)
combine_chunks()            # 11. Chunk combining
```

Each pass may:
- Allocate new DataFrame/Array
- Copy data
- Trigger intermediate operations

### Solution 1: Generate Correct Arrow Types from Start

**Current approach**:
```python
# mckp.py:96
table: pa.Table = pl.DataFrame(data).to_arrow()

# mckp.py:74-76
treatment_id_arrays = table.column("treatment_id").cast(pa.list_(pa.uint32())).combine_chunks()
reward_arrays = table.column("reward").cast(pa.list_(pa.float64())).combine_chunks()
cost_arrays = table.column("cost").cast(pa.list_(pa.float64())).combine_chunks()
```

**Why it's slow**: Polars generates `large_list` types by default, requiring costly casting.

**Optimized approach**:

```python
# Define target schema explicitly
arrow_schema = pa.schema([
    ('patient_id', pa.int64()),
    ('treatment_id', pa.list_(pa.uint32())),  # ← correct type from start
    ('reward', pa.list_(pa.float64())),
    ('cost', pa.list_(pa.float64()))
])

# Convert with explicit schema (skips casting step)
table = data.to_arrow(schema=arrow_schema)

# Extract columns (already correct type and combined)
treatment_id_arrays = table.column("treatment_id")  # ← no cast() needed!
reward_arrays = table.column("reward")
cost_arrays = table.column("cost")

# Note: May still need combine_chunks() if data is chunked
if treatment_id_arrays.num_chunks > 1:
    treatment_id_arrays = treatment_id_arrays.combine_chunks()
    reward_arrays = reward_arrays.combine_chunks()
    cost_arrays = cost_arrays.combine_chunks()
```

**Estimated Impact**: 8-10 seconds (eliminating cast operations)

**Effort**: 30 minutes

**Risk**: Low (straightforward change)

### Solution 2: Combine Drop and Join Operations

**Current approach**:
```python
# mckp.py:79-84
data = (
    data
        .drop('treatment_id')          # ← separate operation
        .join(treatment_nums, on='patient_id')  # ← separate operation
        .sort('patient_id')
)
```

**Optimized approach**:

```python
# Option A: Join with suffix, then select
data = (
    data
        .join(treatment_nums, on='patient_id', suffix='_mapped')
        .select(pl.exclude('treatment_id'))  # drop original, keep treatment_id_mapped
        .rename({'treatment_id_mapped': 'treatment_id'})
        .sort('patient_id')
)

# Option B: Select before join (if possible)
data = (
    data
        .select('patient_id', 'reward', 'cost')  # drop treatment_id early
        .join(treatment_nums, on='patient_id')
        .sort('patient_id')
)
```

**Note**: May not provide significant savings, but reduces intermediate allocations.

**Estimated Impact**: 1-2 seconds

**Effort**: 15 minutes

**Risk**: Low

### Solution 3: Eliminate Redundant Sort

**Current approach**:
```python
# mckp.py:83
.sort('patient_id')
```

**Question**: Is the data already sorted?

**Investigation needed**:
1. Check if input data (Parquet file) is sorted by patient_id
2. Check if operations preserve sort order
3. If already sorted, remove this operation

**Optimized approach**:

```python
# If data is already sorted, skip:
data = (
    data
        .drop('treatment_id')
        .join(treatment_nums, on='patient_id')
        # .sort('patient_id')  ← remove if unnecessary
)

# Or check and conditionally sort:
if not data['patient_id'].is_sorted():
    data = data.sort('patient_id')
```

**Estimated Impact**: 2-4 seconds (if sort is redundant)

**Effort**: 30 minutes (investigation + implementation)

**Risk**: Low

### Solution 4: Use Lazy Evaluation

**Current approach**: Eager evaluation (each operation executes immediately)

**Optimized approach**: Use Polars lazy API

```python
# Convert to lazy frame at start
data_lazy = data.lazy()

# Chain operations (builds query plan, doesn't execute)
data_lazy = (
    data_lazy
        .drop('treatment_id')
        .join(treatment_nums.lazy(), on='patient_id')
        .sort('patient_id')
)

# Execute optimized plan at end
data = data_lazy.collect()
```

**Benefits**:
- Polars optimizer can eliminate redundant operations
- Combine multiple operations into single pass
- Reduce intermediate allocations

**Estimated Impact**: 3-5 seconds (from optimization)

**Effort**: 1 hour (converting to lazy API)

**Risk**: Low (Polars lazy API is well-tested)

### Solution 5: Reduce Temporary DataFrame Creation

**Current issue**:
```python
# mckp.py:96
table: pa.Table = pl.DataFrame(data).to_arrow()  # ← why pl.DataFrame(data)?
```

**Question**: Is `data` not already a DataFrame?

**Optimized approach**:
```python
# If data is already a DataFrame:
table: pa.Table = data.to_arrow()  # ← direct conversion
```

**Estimated Impact**: 0.5-1 second

**Effort**: 5 minutes

**Risk**: None (verify data is DataFrame first)

### Combined Recommendation

Implement all five solutions in order:

1. **Generate correct Arrow types** (30 min, 8-10s savings) - Highest ROI
2. **Eliminate redundant sort** (30 min, 2-4s savings) - Quick win
3. **Fix DataFrame creation** (5 min, 0.5-1s savings) - Trivial
4. **Use lazy evaluation** (1 hour, 3-5s savings) - Good practice
5. **Combine operations** (15 min, 1-2s savings) - Minor improvement

**Total estimated savings**: 10-15 seconds
**Total estimated effort**: 1-2 days

### Implementation Steps

1. **Phase 1: Quick wins** (1 hour):
   - Fix Arrow schema generation
   - Remove redundant DataFrame creation
   - Check if sort is necessary

2. **Phase 2: Lazy evaluation** (2 hours):
   - Convert pipeline to lazy API
   - Test and validate
   - Measure improvement

3. **Phase 3: Operation combining** (1 hour):
   - Refactor join/drop pattern
   - Test edge cases

4. **Phase 4: Validation** (2 hours):
   - Run full benchmark
   - Verify output correctness
   - Update performance metrics

**Total Effort**: 1-2 days

---

## Overall Implementation Strategy

### Phase 1: Quick Wins (Week 1)

**Priority**: Get fast, low-risk improvements deployed

1. **Optimization 2 (Cython memcpy)** - 2-4 hours, 8-10s savings
2. **Optimization 3.1 (Arrow schema)** - 30 min, 8-10s savings
3. **Optimization 3.3 (Remove sort)** - 30 min, 2-4s savings

**Total**: 1 day effort, 18-24 seconds savings

**Expected result**: 116s → 92-98s (15-21% improvement)

### Phase 2: Architectural Improvements (Week 2-3)

**Priority**: Tackle major bottleneck

4. **Optimization 1 (Eliminate explode)** - 1-2 days, 20-25s savings
5. **Optimization 3 (Remaining items)** - 1 day, 2-5s savings

**Total**: 2-3 days effort, 22-30 seconds savings

**Expected result**: 92-98s → 62-73s (40-47% improvement from baseline)

### Phase 3: Optional Advanced Work (Future)

**Priority**: Diminishing returns, pursue if needed

6. **Zero-copy Cython** (from Optimization 2B) - 2-3 days, 2-3s additional savings
7. **C++ algorithm tuning** - 1-2 weeks, 5-10s savings

**Expected result**: 62-73s → 55-65s (approaching parity with maq)

## Success Metrics

### Performance Targets

| Milestone | Target Time | Target Memory | Status |
|-----------|-------------|---------------|--------|
| Current baseline | 116s | 57 GB | ✓ Complete |
| After Phase 1 | 92-98s | 45-50 GB | Pending |
| After Phase 2 | 62-73s | 35-40 GB | Pending |
| After Phase 3 | 55-65s | 30-35 GB | Future |
| Parity with maq | ~50s | ~12 GB | Aspirational |

### Validation Requirements

For each optimization:

1. **Correctness**: Output must match original implementation
2. **Performance**: Measure with `SPARSE_MAQ_PROFILE=1`
3. **Memory**: Track peak memory usage
4. **Benchmark**: Run full benchmark suite
5. **Documentation**: Update performance reports

## Risk Mitigation

### Testing Strategy

1. **Unit tests**: Verify each optimization independently
2. **Integration tests**: Run full pipeline with profiling
3. **Regression tests**: Compare outputs with baseline
4. **Memory tests**: Check for leaks with valgrind/memory profiler

### Rollback Plan

Each optimization should:
- Be feature-flagged if possible
- Have clear commit boundaries
- Be independently revertible
- Include before/after benchmarks

### Known Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Polars list ops not supported | Medium | High | Test early, have fallback |
| Memcpy breaks on chunked data | Low | Medium | Verify combine_chunks() first |
| Arrow schema mismatch | Low | Low | Add schema validation |
| Lazy eval changes behavior | Low | Medium | Extensive testing |

## Appendix: Code Locations

### Files to Modify

1. **sparse_maq/mckp.py**:
   - Lines 61-69: ID mapping (Optimization 1)
   - Lines 79-84: Data join/sort (Optimization 3.2, 3.3)
   - Lines 96-107: Arrow conversion (Optimization 3.1)

2. **sparse_maq/mckpbindings.pyx**:
   - Lines 44-54: Data copying loop (Optimization 2)

3. **core/src/MAQ.h** (optional):
   - Lines 19-38: run() function (Optimization 2B, if zero-copy)

### Supporting Files

- **benchmarking/run_sparse_maq.py**: Benchmark script
- **PERFORMANCE_INVESTIGATION_REPORT.md**: Detailed profiling results
- **PERFORMANCE_OPTIMIZATION_PLAN.md**: This document

## References

1. Performance Investigation Report - detailed profiling breakdown
2. Polars documentation - https://pola-rs.github.io/polars/
3. PyArrow documentation - https://arrow.apache.org/docs/python/
4. Cython memoryviews - https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html

---

**Document Version**: 1.0
**Last Updated**: January 22, 2026
**Next Review**: After Phase 1 completion
