#ifndef MAQ_H
#define MAQ_H

#include "preprocess_data.hpp"
#include "convex_hull.hpp"
#include "compute_path.hpp"
#include <utility>

#ifndef NDEBUG
  #define DEBUG_PRINT(x) std::cout << x << std::endl
#else
  #define DEBUG_PRINT(x)
#endif

// Fork of https://github.com/grf-labs/maq
// Distributed under the MIT License.

#include <iostream>
#include <chrono>
#include <cstdlib>

namespace sparse_maq {

struct solver_output {
  solution_path path;
  std::vector<std::string> treatment_id_mapping;
};

solver_output run_from_cpp(
  std::vector<std::vector<std::string>>&& treatment_id_arrays,
  std::vector<std::vector<double>>&& reward_arrays,
  std::vector<std::vector<double>>&& cost_arrays,
  double budget
) {
  bool PROFILE = std::getenv("SPARSE_MAQ_PROFILE") != nullptr &&
                 std::string(std::getenv("SPARSE_MAQ_PROFILE")) == "1";

  auto t0 = std::chrono::high_resolution_clock::now();

  DEBUG_PRINT("Data successfully pre-processed");
  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(treatment_id_arrays),
    std::move(reward_arrays),
    std::move(cost_arrays)
  );

  auto t1 = std::chrono::high_resolution_clock::now();
  if (PROFILE) {
    std::chrono::duration<double> process_time = t1 - t0;
    std::cout << "  C++: process_data: " << process_time.count() << "s" << std::endl;
  }

  DEBUG_PRINT("Initializing solver");

  convex_hull(treatment_arrays); // Prune in-place


  auto t2 = std::chrono::high_resolution_clock::now();
  if (PROFILE) {
    std::chrono::duration<double> convex_time = t2 - t1;
    std::cout << "  C++: convex_hull (Solver init): " << convex_time.count() << "s" << std::endl;
  }


  DEBUG_PRINT("Fitting solver");
  solution_path path = compute_path(treatment_arrays, budget);

  auto t3 = std::chrono::high_resolution_clock::now();
  if (PROFILE) {
    std::chrono::duration<double> compute_time = t3 - t2;
    std::cout << "  C++: compute_path: " << compute_time.count() << "s" << std::endl;
  }

  return solver_output{
     path,
     treatment_id_mapping
  };
}

// Cython entry point: accepts lvalue refs, forwards via move into run_from_cpp.
solver_output run(
  std::vector<std::vector<std::string>>& treatment_id_arrays,
  std::vector<std::vector<double>>& reward_arrays,
  std::vector<std::vector<double>>& cost_arrays,
  double budget
) {
  return run_from_cpp(
    std::move(treatment_id_arrays),
    std::move(reward_arrays),
    std::move(cost_arrays),
    budget
  );
}

} // namespace sparse_maq

#endif // MAQ_H
