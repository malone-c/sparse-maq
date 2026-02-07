#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../src/preprocess_data.hpp"

#include <vector>

using namespace sparse_maq;

TEST_CASE("TreatmentView getters return correct values") {
  uint32_t id = 42;
  double reward = 100.5;
  double cost = 50.25;

  TreatmentView treatment(id, reward, cost);

  CHECK(treatment.get_id() == 42);
  CHECK(treatment.get_reward() == 100.5);
  CHECK(treatment.get_cost() == 50.25);
}

TEST_CASE("TreatmentView stores references correctly") {
  uint32_t id = 1;
  double reward = 10.0;
  double cost = 5.0;

  TreatmentView treatment(id, reward, cost);

  // Modify original values
  id = 999;
  reward = 999.0;
  cost = 999.0;

  // TreatmentView should reflect the changes
  CHECK(treatment.get_id() == 999);
  CHECK(treatment.get_reward() == 999.0);
  CHECK(treatment.get_cost() == 999.0);
}

TEST_CASE("process_data creates correct TreatmentView arrays") {
  std::vector<std::vector<uint32_t>> ids = {
    {1, 2, 3},
    {4, 5}
  };
  std::vector<std::vector<double>> rewards = {
    {10.0, 20.0, 30.0},
    {40.0, 50.0}
  };
  std::vector<std::vector<double>> costs = {
    {5.0, 10.0, 15.0},
    {20.0, 25.0}
  };

  auto treatment_arrays = process_data(ids, rewards, costs);

  REQUIRE(treatment_arrays.size() == 2);

  // Check first patient
  REQUIRE(treatment_arrays[0].size() == 3);
  CHECK(treatment_arrays[0][0].get_id() == 1);
  CHECK(treatment_arrays[0][0].get_reward() == 10.0);
  CHECK(treatment_arrays[0][0].get_cost() == 5.0);

  CHECK(treatment_arrays[0][1].get_id() == 2);
  CHECK(treatment_arrays[0][1].get_reward() == 20.0);
  CHECK(treatment_arrays[0][1].get_cost() == 10.0);

  CHECK(treatment_arrays[0][2].get_id() == 3);
  CHECK(treatment_arrays[0][2].get_reward() == 30.0);
  CHECK(treatment_arrays[0][2].get_cost() == 15.0);

  // Check second patient
  REQUIRE(treatment_arrays[1].size() == 2);
  CHECK(treatment_arrays[1][0].get_id() == 4);
  CHECK(treatment_arrays[1][0].get_reward() == 40.0);
  CHECK(treatment_arrays[1][0].get_cost() == 20.0);

  CHECK(treatment_arrays[1][1].get_id() == 5);
  CHECK(treatment_arrays[1][1].get_reward() == 50.0);
  CHECK(treatment_arrays[1][1].get_cost() == 25.0);
}

TEST_CASE("process_data handles empty input") {
  std::vector<std::vector<uint32_t>> ids = {};
  std::vector<std::vector<double>> rewards = {};
  std::vector<std::vector<double>> costs = {};

  auto treatment_arrays = process_data(ids, rewards, costs);

  CHECK(treatment_arrays.size() == 0);
}

TEST_CASE("process_data handles patient with no treatments") {
  std::vector<std::vector<uint32_t>> ids = {{}};
  std::vector<std::vector<double>> rewards = {{}};
  std::vector<std::vector<double>> costs = {{}};

  auto treatment_arrays = process_data(ids, rewards, costs);

  REQUIRE(treatment_arrays.size() == 1);
  CHECK(treatment_arrays[0].size() == 0);
}

TEST_CASE("process_data handles single patient single treatment") {
  std::vector<std::vector<uint32_t>> ids = {{1}};
  std::vector<std::vector<double>> rewards = {{100.0}};
  std::vector<std::vector<double>> costs = {{50.0}};

  auto treatment_arrays = process_data(ids, rewards, costs);

  REQUIRE(treatment_arrays.size() == 1);
  REQUIRE(treatment_arrays[0].size() == 1);

  CHECK(treatment_arrays[0][0].get_id() == 1);
  CHECK(treatment_arrays[0][0].get_reward() == 100.0);
  CHECK(treatment_arrays[0][0].get_cost() == 50.0);
}

TEST_CASE("process_data references remain valid") {
  std::vector<std::vector<uint32_t>> ids = {{1, 2}};
  std::vector<std::vector<double>> rewards = {{10.0, 20.0}};
  std::vector<std::vector<double>> costs = {{5.0, 10.0}};

  auto treatment_arrays = process_data(ids, rewards, costs);

  // Modify original data
  ids[0][0] = 999;
  rewards[0][0] = 999.0;
  costs[0][0] = 999.0;

  // TreatmentView should reflect changes (verifying references work)
  CHECK(treatment_arrays[0][0].get_id() == 999);
  CHECK(treatment_arrays[0][0].get_reward() == 999.0);
  CHECK(treatment_arrays[0][0].get_cost() == 999.0);

  // Second treatment should be unchanged
  CHECK(treatment_arrays[0][1].get_id() == 2);
  CHECK(treatment_arrays[0][1].get_reward() == 20.0);
  CHECK(treatment_arrays[0][1].get_cost() == 10.0);
}

TEST_CASE("process_data handles multiple patients with varying treatment counts") {
  std::vector<std::vector<uint32_t>> ids = {
    {1},
    {2, 3, 4},
    {},
    {5, 6}
  };
  std::vector<std::vector<double>> rewards = {
    {10.0},
    {20.0, 30.0, 40.0},
    {},
    {50.0, 60.0}
  };
  std::vector<std::vector<double>> costs = {
    {5.0},
    {10.0, 15.0, 20.0},
    {},
    {25.0, 30.0}
  };

  auto treatment_arrays = process_data(ids, rewards, costs);

  REQUIRE(treatment_arrays.size() == 4);
  CHECK(treatment_arrays[0].size() == 1);
  CHECK(treatment_arrays[1].size() == 3);
  CHECK(treatment_arrays[2].size() == 0);
  CHECK(treatment_arrays[3].size() == 2);
}
