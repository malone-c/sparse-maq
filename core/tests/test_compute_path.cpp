#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../src/compute_path.hpp"
#include "../src/preprocess_data.hpp"
#include "../src/convex_hull.hpp"

#include <vector>

using namespace sparse_maq;

TEST_CASE("QueueElement priority comparison works correctly") {
  uint32_t id1 = 1, id2 = 2, id3 = 3;
  double reward1 = 10.0, reward2 = 20.0, reward3 = 30.0;
  double cost1 = 5.0, cost2 = 10.0, cost3 = 15.0;

  Treatment t1(id1, reward1, cost1);
  Treatment t2(id2, reward2, cost2);
  Treatment t3(id3, reward3, cost3);

  QueueElement e1(0, &t1, 2.0);  // priority 2.0
  QueueElement e2(1, &t2, 5.0);  // priority 5.0
  QueueElement e3(2, &t3, 1.0);  // priority 1.0

  // operator< should return true if lhs.priority < rhs.priority
  // (for max-heap, this gives us highest priority first)
  CHECK((e1 < e2) == true);   // 2.0 < 5.0
  CHECK((e2 < e1) == false);  // 5.0 < 2.0
  CHECK((e3 < e1) == true);   // 1.0 < 2.0
}

TEST_CASE("compute_path returns valid solution_path structure") {
  std::vector<std::vector<std::string>> ids = {{"1", "2"}};
  std::vector<std::vector<double>> rewards = {{10.0, 20.0}};
  std::vector<std::vector<double>> costs = {{5.0, 10.0}};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 10.0);

  CHECK(result.cost_path == std::vector<double>{5.0, 10.0});
  CHECK(result.reward_path == std::vector<double>{10.0, 20.0});
  CHECK(result.i_path == std::vector<size_t>{0, 0});
  CHECK(result.k_path == std::vector<size_t>{0, 1});
  CHECK(result.complete == true);
}

TEST_CASE("compute_path with small dataset") {
  std::vector<std::vector<std::string>> ids = {
    {"1", "2"}, 
    {"3", "4"}
  };
  std::vector<std::vector<double>> rewards = {
    {10.0, 18.0},
    {8.0, 15.0}
  };
  std::vector<std::vector<double>> costs = {
    {5.0, 10.0},
    {4.0, 8.0}
  };

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 20.0);

  // Check that we have some path
  CHECK(result.cost_path.size() > 0);
  CHECK(result.reward_path.size() > 0);
  CHECK(result.i_path.size() > 0);
  CHECK(result.k_path.size() > 0);

  // Check that spend doesn't exceed budget
  for (double spend : result.cost_path) {
    CHECK(spend <= 20.0);
  }

  // Check that gains are positive
  for (double gain : result.reward_path) {
    CHECK(gain > 0);
  }

  // Check that patient indices are valid
  for (size_t patient_idx : result.i_path) {
    CHECK(patient_idx < 2);
  }
}

TEST_CASE("compute_path respects budget constraint") {
  std::vector<std::vector<std::string>> ids = {{"1", "2", "3"}};
  std::vector<std::vector<double>> rewards = {{10.0, 20.0, 30.0}};
  std::vector<std::vector<double>> costs = {{5.0, 15.0, 25.0}};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 10.0);

  // Algorithm may exceed budget slightly for "rounded up" solution
  // Check that at least one allocation stays within budget
  if (!result.cost_path.empty()) {
    bool has_within_budget = false;
    for (double spend : result.cost_path) {
      if (spend <= 10.0) {
        has_within_budget = true;
        break;
      }
    }
    CHECK(has_within_budget == true);
  }
}

TEST_CASE("compute_path with zero budget returns empty path") {
  std::vector<std::vector<std::string>> ids = {{"1", "2"}};
  std::vector<std::vector<double>> rewards = {{10.0, 20.0}};
  std::vector<std::vector<double>> costs = {{5.0, 10.0}};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 0.0);

  CHECK(result.cost_path.size() == 0);
  CHECK(result.reward_path.size() == 0);
  CHECK(result.i_path.size() == 0);
  CHECK(result.k_path.size() == 0);
}

TEST_CASE("compute_path with large budget covers all treatments") {
  std::vector<std::vector<std::string>> ids = {
    {"1", "2"},
    {"3", "4"}
  };
  std::vector<std::vector<double>> rewards = {
    {10.0, 20.0},
    {15.0, 25.0}
  };
  std::vector<std::vector<double>> costs = {
    {5.0, 10.0},
    {7.0, 14.0}
  };

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );
  convex_hull(treatment_arrays);

  // Budget large enough for all treatments
  solution_path result = compute_path(treatment_arrays, 100.0);

  // Check complete_path flag
  CHECK(result.complete);
}

TEST_CASE("compute_path with single patient multiple treatments") {
  std::vector<std::vector<std::string>> ids = {{"1", "2", "3"}};
  std::vector<std::vector<double>> rewards = {{10.0, 25.0, 35.0}};
  std::vector<std::vector<double>> costs = {{5.0, 15.0, 25.0}};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 20.0);

  // Should allocate treatments up to budget
  CHECK(result.cost_path.size() > 0);
  CHECK(result.i_path.size() > 0);

  // All allocations should be for patient 0
  for (size_t patient_idx : result.i_path) {
    CHECK(patient_idx == 0);
  }
}

TEST_CASE("compute_path with all patients one treatment each") {
  std::vector<std::vector<std::string>> ids = {
    {"1"},
    {"2"},
    {"3"}
  };
  std::vector<std::vector<double>> rewards = {
    {10.0},
    {15.0},
    {20.0}
  };
  std::vector<std::vector<double>> costs = {
    {5.0},
    {7.0},
    {10.0}
  };

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 15.0);

  // Should allocate some treatments
  CHECK(result.cost_path.size() > 0);

  // Each patient can only appear once (since each has only one treatment)
  std::vector<bool> patient_seen(3, false);
  for (size_t patient_idx : result.i_path) {
    CHECK(patient_seen[patient_idx] == false);
    patient_seen[patient_idx] = true;
  }
}

TEST_CASE("compute_path accumulates spend and gain correctly") {
  std::vector<std::vector<std::string>> ids = {{"1"}, {"2"}};
  std::vector<std::vector<double>> rewards = {{10.0}, {8.0}};
  std::vector<std::vector<double>> costs = {{5.0}, {4.0}};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 10.0);

  // Spend should be monotonically increasing
  for (size_t i = 1; i < result.cost_path.size(); ++i) {
    CHECK(result.cost_path[i] >= result.cost_path[i-1]);
  }

  // Gain should be monotonically increasing
  for (size_t i = 1; i < result.reward_path.size(); ++i) {
    CHECK(result.reward_path[i] >= result.reward_path[i-1]);
  }
}

TEST_CASE("compute_path handles patient upgrades") {
  // Patient 0 has two treatments with increasing cost/reward
  std::vector<std::vector<std::string>> ids = {{"1", "2"}};
  std::vector<std::vector<double>> rewards = {{10.0, 22.0}};
  std::vector<std::vector<double>> costs = {{5.0, 12.0}};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 15.0);

  // Should see the same patient getting upgraded
  CHECK(result.i_path.size() > 0);

  // print out the i_path for debugging
  for (size_t i = 0; i < result.i_path.size(); ++i) {
    std::cout << "Step " << i << ": Patient " << result.i_path[i] << ", Treatment " << result.k_path[i] << std::endl;
  }

  // Check that patient 0 appears (possibly multiple times for upgrades)
  bool patient_0_appears = false;
  for (size_t patient_idx : result.i_path) {
    if (patient_idx == 0) {
      patient_0_appears = true;
    }
  }
  CHECK(patient_0_appears == true);
}
