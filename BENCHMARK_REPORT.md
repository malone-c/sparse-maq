# Benchmark Report: MAQ vs Sparse MAQ Solver Performance

**Date**: January 22, 2026
**Dataset**: n=1,000,000 patients, k=500 treatments
**Benchmark Suite Version**: 1.0

## Executive Summary

Performance comparison between the dense `maq` solver and sparse format `sparse_maq` solver on datasets with 1 million patients and 500 treatments at two sparsity levels.

**Key Findings**:
- **At 5% treatment eligibility**: sparse_maq is **2.4x faster** (11.37s vs 27.60s) and uses **38% less memory** (7.5 GB vs 12.2 GB)
- At 50% treatment eligibility: maq is 2.9x faster and uses 4.7x less memory
- The sparse format shows clear advantages in truly sparse scenarios

## Dataset Characteristics

### Data Generation

Data was generated using exponentially distributed rewards and costs with random treatment eligibility per patient:
- **Patients (n)**: 1,000,000
- **Treatments (k)**: 500
- **Reward distribution**: Standard exponential
- **Cost distribution**: Standard exponential

### Sparsity Scenarios

**High Sparsity (5% eligibility)**:
- Average of ~25M eligible treatments per patient
- Dense format: 500M entries (n×k) with 95% marked as ineligible
- Sparse format: ~25M entries

**Moderate Sparsity (50% eligibility)**:
- Average of ~250M eligible treatments per patient
- Dense format: 500M entries (n×k) with 50% marked as ineligible
- Sparse format: ~250M entries

## Benchmark Results

### Scenario 1: High Sparsity (5% Treatment Eligibility)

**Performance Metrics**:

| Solver | Execution Time | Peak Memory |
|--------|---------------|-------------|
| maq | 27.60s | 12,240,548 KB (12.2 GB) |
| sparse_maq | 11.37s | 7,525,692 KB (7.5 GB) |

**Relative Performance**:

| Metric | Result |
|--------|--------|
| Execution Time | **sparse_maq 2.43x faster** |
| Peak Memory | **sparse_maq uses 38% less memory** |

This demonstrates that the sparse format delivers significant performance benefits when treatment eligibility is truly sparse.

### Scenario 2: Moderate Sparsity (50% Treatment Eligibility)

**Performance Metrics**:

| Solver | Execution Time | Peak Memory |
|--------|---------------|-------------|
| maq | 49.67s | 12,238,164 KB (12.2 GB) |
| sparse_maq | 143.35s | 57,051,580 KB (57.1 GB) |

**Relative Performance**:

| Metric | Result |
|--------|--------|
| Execution Time | maq 2.89x faster |
| Peak Memory | maq uses 78% less memory |

At moderate sparsity levels, the overhead of sparse data structures outweighs the benefits.

## Analysis

### Sparsity Level Determines Performance

The results demonstrate that **sparsity level is the critical factor** in solver performance:

**At 5% eligibility (truly sparse)**:
- sparse_maq is **2.4x faster** (11.37s vs 27.60s)
- sparse_maq uses **38% less memory** (7.5 GB vs 12.2 GB)
- The sparse format avoids allocating and processing 95% of the treatment space
- Computational overhead of sparse operations is more than offset by reduced data volume

**At 50% eligibility (moderately sparse)**:
- maq is 2.9x faster (49.67s vs 143.35s)
- maq uses 78% less memory (12.2 GB vs 57.1 GB)
- The sparse format overhead (list traversal, metadata storage) dominates any benefits
- Dense matrix operations are more efficient when data isn't truly sparse

## Benchmark Configuration

### System Information
- Platform: Linux 5.10.0-37-cloud-amd64
- Python environment: uv-managed virtual environment
- Key dependencies: polars, numpy, maq package

### Benchmark Parameters
```yaml
solvers:
  - maq
  - sparse_maq

datasets:
  - n: 1000000
    k: 500

replicates: 1
```

### Execution Details
- Results directory: `benchmarking/results/`
- Configuration file: `benchmarking/benchmark_config.yaml`
- Benchmark timestamp: 2026-01-22 03:22:10
- Results file: `benchmark_20260122_032210.csv`

## Files Generated

- **Data files**: `data/reward.npy`, `data/cost.npy`, `data/data.parquet`, `data/treatments.parquet`, `data/patients.parquet`
- **Benchmark results**: `benchmarking/results/benchmark_20260122_032210.csv`
- **Verification script**: `verify_data_correctness.py`

## Future Work

Potential areas for investigation:
1. Identify the exact crossover point between 5% and 50% eligibility
2. Test at even higher sparsity levels (1%, 0.1%) to see if advantages compound
3. Evaluate scalability with larger n and k values
4. Profile memory allocation patterns in sparse_maq
5. Compare with other sparse storage formats (CSR, COO matrices)
