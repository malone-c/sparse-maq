#ifndef MAQ_H
#define MAQ_H

#ifndef NDEBUG
  #define DEBUG_PRINT(x) std::cout << x << std::endl
#else
  #define DEBUG_PRINT(x)
#endif

// Fork of https://github.com/grf-labs/maq
// Distributed under the MIT License.

#include "Data.hpp"
#include "Solver.hpp"
#include <iostream>

namespace sparse_maq {

solution_path run(
  std::vector<std::vector<uint32_t>>& treatment_id_arrays,
  std::vector<std::vector<double>>& reward_arrays,
  std::vector<std::vector<double>>& cost_arrays,
  double budget
) {

  DEBUG_PRINT("Data successfully pre-processed");
  std::vector<std::vector<TreatmentView>> treatment_arrays = process_data(
    treatment_id_arrays,
    reward_arrays,
    cost_arrays
  );

  DEBUG_PRINT("Initializing solver");
  Solver solver(treatment_arrays, budget);

  DEBUG_PRINT("Fitting solver");
  return solver.fit();
}

} // namespace sparse_maq

#endif // MAQ_H
