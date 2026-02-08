#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../src/pipeline.h"

#include <vector>
#include <iostream>

using namespace sparse_maq;

TEST_CASE("E2E test matching Python test_mckp.py") {
  /*
  Python test data:
  patients = ['a', 'b', 'c', 'd', 'e']
  treatments = [['A','B','C','D','E'], ['A','B','C'], ['A','B','C'], ['A','B','C'], ['A','B','C']]
  rewards = [[0,15,22,30], [0,18,32], [0,10,19], [0,17,28], [0,8,18]]
  costs = [[0,10,20,21], [0,15,25], [0,8,16], [0,12,22], [0,7,14]]
  budget = 50
  Expected: spend[-2] == 47.0, gain[-2] == 65.0
  */

  // Set up test data matching Python test
  std::vector<std::vector<std::string>> treatment_ids = {
    {"0", "1", "2", "3"},
    {"0", "1", "2"},
    {"0", "1", "2"},
    {"0", "1", "2"},
    {"0", "1", "2"} 
  };

  std::vector<std::vector<double>> rewards = {
    {0.0, 15.0, 22.0, 30.0},
    {0.0, 18.0, 32.0},
    {0.0, 10.0, 19.0},
    {0.0, 17.0, 28.0},
    {0.0, 8.0, 18.0}
  };

  std::vector<std::vector<double>> costs = {
    {0.0, 10.0, 20.0, 21.0},
    {0.0, 15.0, 25.0},
    {0.0, 8.0, 16.0},
    {0.0, 12.0, 22.0},
    {0.0, 7.0, 14.0}
  };

  double budget = 50.0;

  // Run the full pipeline
  solution_path result = run(treatment_ids, rewards, costs, budget);

  auto& spend_gain = result.first;
  auto& i_k_path = result.second;

  // Print results for debugging
  std::cout << "\n=== E2E Test Results ===" << std::endl;
  std::cout << "Path length: " << spend_gain[0].size() << std::endl;

  for (size_t i = 0; i < spend_gain[0].size(); ++i) {
    std::cout << "Step " << i << ": patient=" << i_k_path[0][i]
              << ", treatment=" << i_k_path[1][i]
              << ", spend=" << spend_gain[0][i]
              << ", gain=" << spend_gain[1][i] << std::endl;
  }

  // Check that we have at least 2 elements in the path
  REQUIRE(spend_gain[0].size() >= 2);
  REQUIRE(spend_gain[1].size() >= 2);

  // Get the second-to-last element (index -2 in Python)
  size_t idx = spend_gain[0].size() - 2;
  double spend_second_last = spend_gain[0][idx];
  double gain_second_last = spend_gain[1][idx];

  std::cout << "\nSecond-to-last: spend=" << spend_second_last
            << ", gain=" << gain_second_last << std::endl;
  std::cout << "Expected: spend=47.0, gain=65.0" << std::endl;

  // Verify against Python test expectations
  CHECK(spend_second_last == doctest::Approx(47.0).epsilon(0.01));
  CHECK(gain_second_last == doctest::Approx(65.0).epsilon(0.01));

  // Additional sanity checks
  // Note: Algorithm may exceed budget for "rounded up" solution
  CHECK(i_k_path[2].size() == 1);         // Should have complete_path flag
}

TEST_CASE("E2E test with simple two-patient case") {
  std::vector<std::vector<std::string>> treatment_ids = {
    {"1", "2"},
    {"3", "4"}
  };

  std::vector<std::vector<double>> rewards = {
    {10.0, 20.0},
    {8.0, 16.0}
  };

  std::vector<std::vector<double>> costs = {
    {5.0, 10.0},
    {4.0, 8.0}
  };

  double budget = 15.0;

  solution_path result = run(treatment_ids, rewards, costs, budget);

  auto& spend_gain = result.first;
  auto& i_k_path = result.second;

  std::cout << "\n=== Simple E2E Test Results ===" << std::endl;
  for (size_t i = 0; i < spend_gain[0].size(); ++i) {
    std::cout << "Step " << i << ": patient=" << i_k_path[0][i]
              << ", treatment=" << i_k_path[1][i]
              << ", spend=" << spend_gain[0][i]
              << ", gain=" << spend_gain[1][i] << std::endl;
  }

  // Basic sanity checks
  CHECK(spend_gain[0].size() > 0);
  CHECK(spend_gain[1].size() > 0);

  // Check monotonicity
  for (size_t i = 1; i < spend_gain[0].size(); ++i) {
    CHECK(spend_gain[0][i] >= spend_gain[0][i-1]);
    CHECK(spend_gain[1][i] >= spend_gain[1][i-1]);
  }
}

TEST_CASE("E2E test with dominated treatments removed") {
  // Patient 0 has a dominated treatment (id=2)
  std::vector<std::vector<std::string>> treatment_ids = {
    {"1", "2", "3"}
  };

  std::vector<std::vector<double>> rewards = {
    {10.0, 12.0, 30.0}  // 12.0 is dominated by linear interpolation
  };

  std::vector<std::vector<double>> costs = {
    {5.0, 10.0, 15.0}
  };

  double budget = 20.0;

  solution_path result = run(treatment_ids, rewards, costs, budget);

  auto& spend_gain = result.first;
  auto& i_k_path = result.second;

  std::cout << "\n=== Dominated Treatment E2E Test Results ===" << std::endl;
  for (size_t i = 0; i < spend_gain[0].size(); ++i) {
    std::cout << "Step " << i << ": patient=" << i_k_path[0][i]
              << ", treatment=" << i_k_path[1][i]
              << ", spend=" << spend_gain[0][i]
              << ", gain=" << spend_gain[1][i] << std::endl;
  }

  // Treatment 2 should not appear in the path (it's dominated)
  bool treatment_2_appears = false;
  for (size_t treatment_id : i_k_path[1]) {
    if (treatment_id == 2) {
      treatment_2_appears = true;
    }
  }
  CHECK(treatment_2_appears == false);

  // Should have treatments 1 and 3
  CHECK(spend_gain[0].size() > 0);
}
