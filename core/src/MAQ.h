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
#include <chrono>
#include <cstdlib>

namespace sparse_maq {

solution_path run(
  std::vector<std::vector<uint32_t>>& treatment_id_arrays,
  std::vector<std::vector<double>>& reward_arrays,
  std::vector<std::vector<double>>& cost_arrays,
  double budget
) {
  bool PROFILE = std::getenv("SPARSE_MAQ_PROFILE") != nullptr &&
                 std::string(std::getenv("SPARSE_MAQ_PROFILE")) == "1";

  auto t0 = std::chrono::high_resolution_clock::now();

  DEBUG_PRINT("Data successfully pre-processed");
  std::vector<std::vector<TreatmentView>> treatment_arrays = process_data(
    treatment_id_arrays,
    reward_arrays,
    cost_arrays
  );

  auto t1 = std::chrono::high_resolution_clock::now();
  if (PROFILE) {
    std::chrono::duration<double> process_time = t1 - t0;
    std::cout << "  C++: process_data: " << process_time.count() << "s" << std::endl;
  }

  DEBUG_PRINT("Initializing solver");
  Solver solver(treatment_arrays, budget);  // calls convex_hull in constructor

  auto t2 = std::chrono::high_resolution_clock::now();
  if (PROFILE) {
    std::chrono::duration<double> convex_time = t2 - t1;
    std::cout << "  C++: convex_hull (Solver init): " << convex_time.count() << "s" << std::endl;
  }

  DEBUG_PRINT("Fitting solver");
  solution_path path = solver.fit();  // calls compute_path

  auto t3 = std::chrono::high_resolution_clock::now();
  if (PROFILE) {
    std::chrono::duration<double> compute_time = t3 - t2;
    std::cout << "  C++: compute_path (solver.fit): " << compute_time.count() << "s" << std::endl;
  }

  return path;
}

} // namespace sparse_maq

#endif // MAQ_H
