#ifndef MAQ_SOLVER_HPP
#define MAQ_SOLVER_HPP

// Fork of https://github.com/grf-labs/maq
// Distributed under the MIT License.

#include <cmath>
#include <future>
#include <random>
#include <vector>

#include "convex_hull.hpp"
#include "compute_path.hpp"

namespace sparse_maq {

class Solver {
  public:
    Solver(
      std::vector<std::vector<TreatmentView>>& treatment_arrays,
      double budget
    ) : treatment_arrays(treatment_arrays), budget(budget) {}

    solution_path fit() {
      convex_hull(treatment_arrays); // Prune in-place
      return compute_path(treatment_arrays, budget);
    }

    std::vector<std::vector<TreatmentView>> treatment_arrays;
    double budget;
};

} // namespace sparse_maq

#endif // MAQ_SOLVER_HPP
