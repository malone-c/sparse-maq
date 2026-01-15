# sparse-maq

An optimized implementation of Multi-Armed Qini (MAQ) for treatment allocation under budget constraints. This is a fork of [grf-labs/maq](https://github.com/grf-labs/maq) with significant architectural improvements for sparse treatment eligibility and efficient data pipelines.

## Key Features

### Sparse Data Structure
Instead of using a dense matrix representation, this implementation uses variable-length arrays (array of arrays) for treatment data. This is optimal when patients have different eligible treatment sets—similar to using a sparse matrix format but with better performance characteristics. In memory, this is represented as a single contiguous array with offset indices, minimizing cache misses and memory overhead.

### Prospective Evaluation
Unlike the original MAQ implementation which uses off-policy evaluation (OPE) with historical data, this implementation performs prospective evaluation using the predictions that drive the allocation. This makes it more suitable for forward-looking treatment allocation scenarios.

### Memory Efficiency
- Minimized data copying throughout the allocation pipeline
- Arrow-based interface allows zero-copy data transfer from data warehouses
- Contiguous memory layout for faster iteration over treatment options

### Polars Integration
- Works with Polars dataframes
- Designed to work seamlessly with data warehouses (DuckDB, BigQuery, etc.)
- Reduced I/O costs through efficient data serialization

## Use Case

This implementation is ideal when:
- Patients have variable treatment eligibility (not all treatments available to all patients)
- You're working with large-scale datasets from OLAP systems
- You want prospective allocation based on predictions rather than historical counterfactuals

# References

Erik Sverdrup, Han Wu, Susan Athey, and Stefan Wager.
Qini Curves for Multi-Armed Treatment Rules. 2023.
[arxiv](https://arxiv.org/abs/2306.11979)
