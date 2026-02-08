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

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 10.0);

  // Check structure: first element is spend_gain with 3 vectors
  REQUIRE(result.first.size() == 3);
  // Second element is i_k_path with 3 vectors
  REQUIRE(result.second.size() == 3);
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

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 20.0);

  auto& spend_gain = result.first;
  auto& i_k_path = result.second;

  // Check that we have some path
  CHECK(spend_gain[0].size() > 0);
  CHECK(spend_gain[1].size() > 0);
  CHECK(i_k_path[0].size() > 0);
  CHECK(i_k_path[1].size() > 0);

  // Check that spend doesn't exceed budget
  for (double spend : spend_gain[0]) {
    CHECK(spend <= 20.0);
  }

  // Check that gains are positive
  for (double gain : spend_gain[1]) {
    CHECK(gain > 0);
  }

  // Check that patient indices are valid
  for (size_t patient_idx : i_k_path[0]) {
    CHECK(patient_idx < 2);
  }
}

TEST_CASE("compute_path respects budget constraint") {
  std::vector<std::vector<std::string>> ids = {{"1", "2", "3"}};
  std::vector<std::vector<double>> rewards = {{10.0, 20.0, 30.0}};
  std::vector<std::vector<double>> costs = {{5.0, 15.0, 25.0}};

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 10.0);

  auto& spend_gain = result.first;

  // Algorithm may exceed budget slightly for "rounded up" solution
  // Check that at least one allocation stays within budget
  if (!spend_gain[0].empty()) {
    bool has_within_budget = false;
    for (double spend : spend_gain[0]) {
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

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 0.0);

  auto& spend_gain = result.first;
  auto& i_k_path = result.second;

  CHECK(spend_gain[0].size() == 0);
  CHECK(spend_gain[1].size() == 0);
  CHECK(i_k_path[0].size() == 0);
  CHECK(i_k_path[1].size() == 0);
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

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  // Budget large enough for all treatments
  solution_path result = compute_path(treatment_arrays, 100.0);

  auto& i_k_path = result.second;

  // Check complete_path flag (should be 1 for complete)
  REQUIRE(i_k_path[2].size() == 1);
  CHECK(i_k_path[2][0] == 1);
}

TEST_CASE("compute_path with single patient multiple treatments") {
  std::vector<std::vector<std::string>> ids = {{"1", "2", "3"}};
  std::vector<std::vector<double>> rewards = {{10.0, 25.0, 35.0}};
  std::vector<std::vector<double>> costs = {{5.0, 15.0, 25.0}};

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 20.0);

  auto& spend_gain = result.first;
  auto& i_k_path = result.second;

  // Should allocate treatments up to budget
  CHECK(spend_gain[0].size() > 0);
  CHECK(i_k_path[0].size() > 0);

  // All allocations should be for patient 0
  for (size_t patient_idx : i_k_path[0]) {
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

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 15.0);

  auto& spend_gain = result.first;
  auto& i_k_path = result.second;

  // Should allocate some treatments
  CHECK(spend_gain[0].size() > 0);

  // Each patient can only appear once (since each has only one treatment)
  std::vector<bool> patient_seen(3, false);
  for (size_t patient_idx : i_k_path[0]) {
    CHECK(patient_seen[patient_idx] == false);
    patient_seen[patient_idx] = true;
  }
}

TEST_CASE("compute_path accumulates spend and gain correctly") {
  std::vector<std::vector<std::string>> ids = {{"1"}, {"2"}};
  std::vector<std::vector<double>> rewards = {{10.0}, {8.0}};
  std::vector<std::vector<double>> costs = {{5.0}, {4.0}};

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 10.0);

  auto& spend_gain = result.first;

  // Spend should be monotonically increasing
  for (size_t i = 1; i < spend_gain[0].size(); ++i) {
    CHECK(spend_gain[0][i] >= spend_gain[0][i-1]);
  }

  // Gain should be monotonically increasing
  for (size_t i = 1; i < spend_gain[1].size(); ++i) {
    CHECK(spend_gain[1][i] >= spend_gain[1][i-1]);
  }
}

TEST_CASE("compute_path handles patient upgrades") {
  // Patient 0 has two treatments with increasing cost/reward
  std::vector<std::vector<std::string>> ids = {{"1", "2"}};
  std::vector<std::vector<double>> rewards = {{10.0, 25.0}};
  std::vector<std::vector<double>> costs = {{5.0, 12.0}};

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  solution_path result = compute_path(treatment_arrays, 15.0);

  auto& i_k_path = result.second;

  // Should see the same patient getting upgraded
  CHECK(i_k_path[0].size() > 0);

  // Check that patient 0 appears (possibly multiple times for upgrades)
  bool patient_0_appears = false;
  for (size_t patient_idx : i_k_path[0]) {
    if (patient_idx == 0) {
      patient_0_appears = true;
    }
  }
  CHECK(patient_0_appears == true);
}
