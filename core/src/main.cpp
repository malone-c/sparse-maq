#include <iostream>
#include <iomanip>
#include "pipeline.h"

using namespace sparse_maq;

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "  Sparse-MAQ C++ Library Demo" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;

  std::vector<std::vector<std::string>> treatment_ids = {
    {"0", "1", "2", "3"},     // Patient 0: 4 treatment options (including no-treatment = 0)
    {"0", "1", "2"},        // Patient 1: 3 treatment options
    {"0", "1", "2", "3", "4"}   // Patient 2: 5 treatment options
  };

  std::vector<std::vector<double>> rewards = {
    {0.0, 10.0, 18.0, 25.0},        // Patient 0 treatment rewards
    {0.0, 12.0, 20.0},               // Patient 1 treatment rewards
    {0.0, 8.0, 14.0, 22.0, 28.0}    // Patient 2 treatment rewards
  };

  std::vector<std::vector<double>> costs = {
    {0.0, 5.0, 10.0, 15.0},          // Patient 0 treatment costs
    {0.0, 6.0, 12.0},                // Patient 1 treatment costs
    {0.0, 4.0, 8.0, 14.0, 20.0}     // Patient 2 treatment costs
  };

  double budget = 30.0;

  std::cout << "Input Data:" << std::endl;
  std::cout << "  - Number of patients: " << treatment_ids.size() << std::endl;
  std::cout << "  - Budget: $" << budget << std::endl;
  std::cout << std::endl;

  std::cout << "Patient Treatment Options:" << std::endl;
  for (size_t i = 0; i < treatment_ids.size(); ++i) {
    std::cout << "  Patient " << i << ":" << std::endl;
    for (size_t j = 0; j < treatment_ids[i].size(); ++j) {
      std::cout << "    Treatment " << treatment_ids[i][j]
                << ": Cost=$" << std::setw(5) << costs[i][j]
                << ", Reward=" << std::setw(5) << rewards[i][j];
      if (j > 0) {
        double ratio = rewards[i][j] / costs[i][j];
        std::cout << " (ratio=" << std::setw(4) << std::fixed
                  << std::setprecision(2) << ratio << ")";
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  // Run the optimization pipeline
  std::cout << "Running optimization pipeline..." << std::endl;
  solution_path result = run(treatment_ids, rewards, costs, budget);
  std::cout << std::endl;

  // Extract results
  auto& spend_gain = result.first;
  auto& i_k_path = result.second;

  // Display results
  std::cout << "========================================" << std::endl;
  std::cout << "  Optimal Treatment Allocation" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;

  if (spend_gain[0].empty()) {
    std::cout << "No treatments allocated (budget too low or no valid treatments)." << std::endl;
  } else {
    std::cout << "Allocation Path:" << std::endl;
    std::cout << std::setw(6) << "Step"
              << std::setw(10) << "Patient"
              << std::setw(12) << "Treatment"
              << std::setw(15) << "Total Spend"
              << std::setw(15) << "Total Gain" << std::endl;
    std::cout << std::string(58, '-') << std::endl;

    for (size_t i = 0; i < spend_gain[0].size(); ++i) {
      std::cout << std::setw(6) << i
                << std::setw(10) << i_k_path[0][i]
                << std::setw(12) << i_k_path[1][i]
                << std::setw(15) << std::fixed << std::setprecision(2)
                << spend_gain[0][i]
                << std::setw(15) << spend_gain[1][i] << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Final Results:" << std::endl;
    std::cout << "  - Total Spend: $" << spend_gain[0].back() << std::endl;
    std::cout << "  - Total Reward: " << spend_gain[1].back() << std::endl;
    std::cout << "  - Budget Remaining: $"
              << (budget - spend_gain[0].back()) << std::endl;

    // Check if path is complete
    if (i_k_path[2].size() > 0 && i_k_path[2][0] == 1) {
      std::cout << "  - Status: Complete path (all beneficial treatments allocated)" << std::endl;
    } else {
      std::cout << "  - Status: Budget-constrained (more treatments available)" << std::endl;
    }

    std::cout << std::endl;

    // Summary of final patient assignments
    std::cout << "Final Patient Assignments:" << std::endl;
    std::vector<std::string> final_assignments(treatment_ids.size(), 0);
    for (size_t i = 0; i < i_k_path[0].size(); ++i) {
      final_assignments[i_k_path[0][i]] = i_k_path[1][i];
    }

    for (size_t i = 0; i < final_assignments.size(); ++i) {
      std::cout << "  Patient " << i << ": Treatment " << final_assignments[i];
      if (final_assignments[i] != "0") {
        // Find the treatment index in the original arrays
        for (size_t j = 0; j < treatment_ids[i].size(); ++j) {
          if (treatment_ids[i][j] == final_assignments[i]) {
            std::cout << " (Cost=$" << costs[i][j]
                      << ", Reward=" << rewards[i][j] << ")";
            break;
          }
        }
      } else {
        std::cout << " (No treatment)";
      }
      std::cout << std::endl;
    }
  }

  std::cout << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Demo completed successfully!" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
