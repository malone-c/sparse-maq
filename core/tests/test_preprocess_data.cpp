#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../src/preprocess_data.hpp"

#include <vector>

using namespace sparse_maq;

TEST_CASE("Treatment getters return correct values") {
  uint32_t id = 42;
  double reward = 100.5;
  double cost = 50.25;

  Treatment treatment(id, reward, cost);

  CHECK(treatment.id == 42);
  CHECK(treatment.reward == 100.5);
  CHECK(treatment.cost == 50.25);
}

TEST_CASE("Treatment stores references correctly") {
  uint32_t id = 1;
  double reward = 10.0;
  double cost = 5.0;

  Treatment treatment(id, reward, cost);

  // Modify original values
  id = 999;
  reward = 999.0;
  cost = 999.0;

  // Treatment should reflect the changes
  CHECK(treatment.id == 999);
  CHECK(treatment.reward == 999.0);
  CHECK(treatment.cost == 999.0);
}

TEST_CASE("process_data creates correct Treatment arrays") {
  std::vector<std::vector<std::string>> ids = {
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

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);

  REQUIRE(treatment_arrays.size() == 2);

  // Check first patient
  REQUIRE(treatment_arrays[0].size() == 3);
  CHECK(treatment_arrays[0][0].id == 1);
  CHECK(treatment_arrays[0][0].reward == 10.0);
  CHECK(treatment_arrays[0][0].cost == 5.0);

  CHECK(treatment_arrays[0][1].id == 2);
  CHECK(treatment_arrays[0][1].reward == 20.0);
  CHECK(treatment_arrays[0][1].cost == 10.0);

  CHECK(treatment_arrays[0][2].id == 3);
  CHECK(treatment_arrays[0][2].reward == 30.0);
  CHECK(treatment_arrays[0][2].cost == 15.0);

  // Check second patient
  REQUIRE(treatment_arrays[1].size() == 2);
  CHECK(treatment_arrays[1][0].id == 4);
  CHECK(treatment_arrays[1][0].reward == 40.0);
  CHECK(treatment_arrays[1][0].cost == 20.0);

  CHECK(treatment_arrays[1][1].id == 5);
  CHECK(treatment_arrays[1][1].reward == 50.0);
  CHECK(treatment_arrays[1][1].cost == 25.0);
}

TEST_CASE("process_data handles empty input") {
  std::vector<std::vector<std::string>> ids = {};
  std::vector<std::vector<double>> rewards = {};
  std::vector<std::vector<double>> costs = {};

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);

  CHECK(treatment_arrays.size() == 0);
}

TEST_CASE("process_data handles patient with no treatments") {
  std::vector<std::vector<std::string>> ids = {{}};
  std::vector<std::vector<double>> rewards = {{}};
  std::vector<std::vector<double>> costs = {{}};

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);

  REQUIRE(treatment_arrays.size() == 1);
  CHECK(treatment_arrays[0].size() == 0);
}

TEST_CASE("process_data handles single patient single treatment") {
  std::vector<std::vector<std::string>> ids = {{1}};
  std::vector<std::vector<double>> rewards = {{100.0}};
  std::vector<std::vector<double>> costs = {{50.0}};

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);

  REQUIRE(treatment_arrays.size() == 1);
  REQUIRE(treatment_arrays[0].size() == 1);

  CHECK(treatment_arrays[0][0].id == 1);
  CHECK(treatment_arrays[0][0].reward == 100.0);
  CHECK(treatment_arrays[0][0].cost == 50.0);
}

TEST_CASE("process_data references remain valid") {
  std::vector<std::vector<std::string>> ids = {{1, 2}};
  std::vector<std::vector<double>> rewards = {{10.0, 20.0}};
  std::vector<std::vector<double>> costs = {{5.0, 10.0}};

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);

  // Modify original data
  ids[0][0] = 999;
  rewards[0][0] = 999.0;
  costs[0][0] = 999.0;

  // Treatment should reflect changes (verifying references work)
  CHECK(treatment_arrays[0][0].id == 999);
  CHECK(treatment_arrays[0][0].reward == 999.0);
  CHECK(treatment_arrays[0][0].cost == 999.0);

  // Second treatment should be unchanged
  CHECK(treatment_arrays[0][1].id == 2);
  CHECK(treatment_arrays[0][1].reward == 20.0);
  CHECK(treatment_arrays[0][1].cost == 10.0);
}

TEST_CASE("process_data handles multiple patients with varying treatment counts") {
  std::vector<std::vector<std::string>> ids = {
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

  auto [treatment_arrays, treatment_id_mapping] = process_data(ids, rewards, costs);

  REQUIRE(treatment_arrays.size() == 4);
  CHECK(treatment_arrays[0].size() == 1);
  CHECK(treatment_arrays[1].size() == 3);
  CHECK(treatment_arrays[2].size() == 0);
  CHECK(treatment_arrays[3].size() == 2);
}
