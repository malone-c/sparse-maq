This repo is a fork of https://github.com/grf-labs/maq/.

* Implementation is optimised for the case where patients are not eligible for all treatments (treatments are a list of lists rather than a matrix).
* Data copying is reduced to improve memory efficiency and processing of large datasets.
* Interface is heavily simplified and takes Arrow array list types as input to reduce i/o cost and integrate with OLAP databases.

# TODO

* Document list of optimisations made
* Change internals to take patient and treatment IDs as strings (rather than ints)
* Improve memory efficiency by further reducing data copying

# References

Erik Sverdrup, Han Wu, Susan Athey, and Stefan Wager.
Qini Curves for Multi-Armed Treatment Rules. 2023.
[arxiv](https://arxiv.org/abs/2306.11979)
