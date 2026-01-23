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


