#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../src/pipeline.h"

#include <vector>
#include <iostream>

using namespace sparse_maq;

TEST_CASE("E2E test matching Python test_mckp.py") {
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

  auto [path, treatment_num_to_id] = run_from_cpp(
    std::move(treatment_ids), 
    std::move(rewards), 
    std::move(costs), 
    std::move(budget)
  );
  
  std::cout << "\n=== E2E Test Results ===" << std::endl;
  std::cout << "Path length: " << path.cost_path.size() << std::endl;

  for (size_t i = 0; i < path.cost_path.size(); ++i) {
    std::cout << "Step " << i << ": patient=" << path.i_path[i]
              << ", treatment=" << path.k_path[i]
              << ", cost=" << path.cost_path[i]
              << ", reward=" << path.reward_path[i] << std::endl;
  }

  REQUIRE(path.cost_path.size() >= 2);
  REQUIRE(path.reward_path.size() >= 2);

  size_t idx = path.cost_path.size() - 2;
  double cost_second_last = path.cost_path[idx];
  double reward_second_last = path.reward_path[idx];

  std::cout << "\nSecond-to-last: cost=" << cost_second_last
            << ", reward=" << reward_second_last << std::endl;
  std::cout << "Expected: cost=47.0, reward=65.0" << std::endl;

  CHECK(cost_second_last == doctest::Approx(47.0).epsilon(0.01));
  CHECK(reward_second_last == doctest::Approx(65.0).epsilon(0.01));
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

  auto [result, treatment_num_to_id] = run_from_cpp(
    std::move(treatment_ids), 
    std::move(rewards), 
    std::move(costs), 
    std::move(budget)
  );

  std::cout << "\n=== Simple E2E Test Results ===" << std::endl;
  for (size_t i = 0; i < result.cost_path.size(); ++i) {
    std::cout << "Step " << i << ": patient=" << result.i_path[i]
              << ", treatment=" << result.k_path[i]
              << ", cost=" << result.cost_path[i]
              << ", reward=" << result.reward_path[i] << std::endl;
  }

  // Basic sanity checks
  CHECK(result.cost_path.size() > 0);
  CHECK(result.reward_path.size() > 0);

  // Check monotonicity
  for (size_t i = 1; i < result.cost_path.size(); ++i) {
    CHECK(result.cost_path[i] >= result.cost_path[i-1]);
    CHECK(result.reward_path[i] >= result.reward_path[i-1]);
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

  auto [result, treatment_num_to_id] = run_from_cpp(
    std::move(treatment_ids), 
    std::move(rewards), 
    std::move(costs), 
    std::move(budget)
  );

  std::cout << "\n=== Dominated Treatment E2E Test Results ===" << std::endl;
  for (size_t i = 0; i < result.cost_path.size(); ++i) {
    std::cout << "Step " << i << ": patient=" << result.i_path[i]
              << ", treatment=" << result.k_path[i]
              << ", cost=" << result.cost_path[i]
              << ", reward=" << result.reward_path[i] << std::endl;
  }

  // Treatment 2 should not appear in the path (it's dominated)
  bool treatment_1_appears = false;
  for (size_t treatment_id : result.k_path) {
    if (treatment_id == 1) {
      treatment_1_appears = true;
    }
  }
  CHECK(treatment_1_appears == false);

  // Should have treatments 1 and 3
  CHECK(result.cost_path.size() > 0);
}
