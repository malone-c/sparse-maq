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

TEST_CASE("process_data creates correct Treatment arrays") {
  std::vector<std::vector<std::string>> ids = {
    {"1", "2", "3"},
    {"4", "5"}
  };
  std::vector<std::vector<double>> rewards = {
    {10.0, 20.0, 30.0},
    {40.0, 50.0}
  };
  std::vector<std::vector<double>> costs = {
    {5.0, 10.0, 15.0},
    {20.0, 25.0}
  };

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids), 
    std::move(rewards), 
    std::move(costs)
  );

  REQUIRE(treatment_arrays.size() == 2);

  // Check first patient
  REQUIRE(treatment_arrays[0].size() == 3);
  CHECK(treatment_arrays[0][0].id == 0);
  CHECK(treatment_arrays[0][0].reward == 10.0);
  CHECK(treatment_arrays[0][0].cost == 5.0);

  CHECK(treatment_arrays[0][1].id == 1);
  CHECK(treatment_arrays[0][1].reward == 20.0);
  CHECK(treatment_arrays[0][1].cost == 10.0);

  CHECK(treatment_arrays[0][2].id == 2);
  CHECK(treatment_arrays[0][2].reward == 30.0);
  CHECK(treatment_arrays[0][2].cost == 15.0);

  // Check second patient
  REQUIRE(treatment_arrays[1].size() == 2);
  CHECK(treatment_arrays[1][0].id == 3);
  CHECK(treatment_arrays[1][0].reward == 40.0);
  CHECK(treatment_arrays[1][0].cost == 20.0);

  CHECK(treatment_arrays[1][1].id == 4);
  CHECK(treatment_arrays[1][1].reward == 50.0);
  CHECK(treatment_arrays[1][1].cost == 25.0);
}

TEST_CASE("process_data handles empty input") {
  std::vector<std::vector<std::string>> ids = {};
  std::vector<std::vector<double>> rewards = {};
  std::vector<std::vector<double>> costs = {};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );

  CHECK(treatment_arrays.size() == 0);
}

TEST_CASE("process_data handles patient with no treatments") {
  std::vector<std::vector<std::string>> ids = {{}};
  std::vector<std::vector<double>> rewards = {{}};
  std::vector<std::vector<double>> costs = {{}};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );

  REQUIRE(treatment_arrays.size() == 1);
  CHECK(treatment_arrays[0].size() == 0);
}

TEST_CASE("process_data handles single patient single treatment") {
  std::vector<std::vector<std::string>> ids = {{"1"}};
  std::vector<std::vector<double>> rewards = {{100.0}};
  std::vector<std::vector<double>> costs = {{50.0}};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );

  REQUIRE(treatment_arrays.size() == 1);
  REQUIRE(treatment_arrays[0].size() == 1);

  CHECK(treatment_arrays[0][0].id == 0);
  CHECK(treatment_arrays[0][0].reward == 100.0);
  CHECK(treatment_arrays[0][0].cost == 50.0);
}

TEST_CASE("process_data Treatment values are correct copies of input") {
  std::vector<std::vector<std::string>> ids = {{"1", "2"}};
  std::vector<std::vector<double>> rewards = {{10.0, 20.0}};
  std::vector<std::vector<double>> costs = {{5.0, 10.0}};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );

  // Treatments store copied values; both should be intact after the move
  REQUIRE(treatment_arrays[0].size() == 2);
  CHECK(treatment_arrays[0][0].id == 0);
  CHECK(treatment_arrays[0][0].reward == 10.0);
  CHECK(treatment_arrays[0][0].cost == 5.0);

  CHECK(treatment_arrays[0][1].id == 1);
  CHECK(treatment_arrays[0][1].reward == 20.0);
  CHECK(treatment_arrays[0][1].cost == 10.0);
}

TEST_CASE("treatment_id_mapping basic content") {
  std::vector<std::vector<std::string>> ids = {{"1", "2", "3"}};
  std::vector<std::vector<double>> rewards = {{10.0, 20.0, 30.0}};
  std::vector<std::vector<double>> costs = {{5.0, 10.0, 15.0}};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );

  REQUIRE(treatment_id_mapping.size() == 3);
  CHECK(treatment_id_mapping[0] == "1");
  CHECK(treatment_id_mapping[1] == "2");
  CHECK(treatment_id_mapping[2] == "3");
}

TEST_CASE("treatment_id_mapping encounter order") {
  std::vector<std::vector<std::string>> ids = {{"5", "1", "3"}};
  std::vector<std::vector<double>> rewards = {{10.0, 20.0, 30.0}};
  std::vector<std::vector<double>> costs = {{5.0, 10.0, 15.0}};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );

  REQUIRE(treatment_id_mapping.size() == 3);
  CHECK(treatment_id_mapping[0] == "5");
  CHECK(treatment_id_mapping[1] == "1");
  CHECK(treatment_id_mapping[2] == "3");
}

TEST_CASE("treatment_id_mapping deduplication across patients") {
  std::vector<std::vector<std::string>> ids = {
    {"A", "B"},
    {"B", "C"}
  };
  std::vector<std::vector<double>> rewards = {
    {1.0, 2.0},
    {2.0, 3.0}
  };
  std::vector<std::vector<double>> costs(rewards);

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );

  CHECK(treatment_id_mapping.size() == 3);
  CHECK(treatment_id_mapping[1] == "B");
  CHECK(treatment_arrays[0][1].id == 1);
}

TEST_CASE("treatment_id_mapping round-trip") {
  // Save original IDs before the move
  const std::vector<std::vector<std::string>> original_ids = {
    {"X", "Y", "Z"},
    {"Y", "W"}
  };
  std::vector<std::vector<std::string>> ids = original_ids;
  std::vector<std::vector<double>> rewards = {{1.0, 2.0, 3.0}, {4.0, 5.0}};
  std::vector<std::vector<double>> costs = {{0.1, 0.2, 0.3}, {0.4, 0.5}};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );

  // Core contract: treatment_id_mapping[t.id] must recover the original string ID
  for (size_t i = 0; i < treatment_arrays.size(); ++i) {
    for (size_t j = 0; j < treatment_arrays[i].size(); ++j) {
      const Treatment& t = treatment_arrays[i][j];
      CHECK(treatment_id_mapping[t.id] == original_ids[i][j]);
    }
  }
}

TEST_CASE("treatment_id_mapping empty input") {
  std::vector<std::vector<std::string>> ids = {};
  std::vector<std::vector<double>> rewards = {};
  std::vector<std::vector<double>> costs = {};

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );

  CHECK(treatment_id_mapping.size() == 0);
}

TEST_CASE("process_data handles multiple patients with varying treatment counts") {
  std::vector<std::vector<std::string>> ids = {
    {"1"},
    {"2", "3", "4"},
    {},
    {"5", "6"}
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

  auto [treatment_arrays, treatment_id_mapping] = preprocess_data_cpp(
    std::move(ids),
    std::move(rewards),
    std::move(costs)
  );

  REQUIRE(treatment_arrays.size() == 4);
  CHECK(treatment_arrays[0].size() == 1);
  CHECK(treatment_arrays[1].size() == 3);
  CHECK(treatment_arrays[2].size() == 0);
  CHECK(treatment_arrays[3].size() == 2);
}
