# sparse-maq

An optimised implementation of Multi-Armed Qini (MAQ) for treatment allocation under budget constraints. This is a fork of [grf-labs/maq](https://github.com/grf-labs/maq) with significant optimisations for sparse treatment eligibility and support for prospective evaluation.

## Key Features

### Sparse Data Structure
Instead of using a dense matrix representation, this implementation uses variable-length arrays (array of arrays) for treatment data. This is optimal when patients have different eligible treatment sets—similar to using a sparse matrix format but with better performance characteristics. In memory, this is represented as a single contiguous array with offset indices. This minimises cache misses and memory overhead.

### Prospective Evaluation
Unlike the original MAQ implementation which uses off-policy evaluation (OPE) with historical data, this implementation performs prospective evaluation using the predictions that drive the allocation. As such, this implementation is **not** suitable for model evaluation.

For better memory management, only prospective allocations are supported. Advanced off-policy evaluation methods such as inverse propensity score (IPS) weighting for historical data are not available in this implementation.

### Memory Efficiency
- Minimised data copying throughout the allocation pipeline
- Arrow-based interface allows zero-copy data transfer from inputs
- Contiguous memory layout for faster iteration over treatment options

## Use Case

This implementation is ideal when:
- Patients have variable treatment eligibility (not all treatments available to all patients)
- Number of patients is large (>1,000,000)
- You want prospective allocation based on predictions rather than historical counterfactuals

# Benchmarks

A simple benchmark performed with simulated data:
* `n = 1_000_000` patients
* `k = 500` treatments
* All rewards and costs distributed as iid standard exponential
  * Patient treatment eligibility distributed as Bernoulli(0.05) -- i.e. each patient eligible for ~25 randomly selected treatments


| Solver | Execution Time | Peak Memory |
|--------|---------------|-------------|
| `maq` | 33.35s | 12,240,548 KB (12.2 GB) |
| `sparse_maq` | 11.0s | 5,904,220 KB (5.9 GB) |


# References

Erik Sverdrup, Han Wu, Susan Athey, and Stefan Wager.
Qini Curves for Multi-Armed Treatment Rules. 2023.
[arxiv](https://arxiv.org/abs/2306.11979)
