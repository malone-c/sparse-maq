#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../src/convex_hull.hpp"

#include <vector>
#include "../src/preprocess_data.hpp"

using namespace sparse_maq;

TEST_CASE("Convex hull basic test") {
  // Create underlying data that will outlive TreatmentView objects
  std::vector<std::vector<uint32_t>> ids = {
    {1, 2, 3},
    {1, 2, 3},
    {1, 2, 3}
  };
  std::vector<std::vector<double>> rewards = {
    {10.0, 15.0, 20.0},
    {5.0, 10.0, 15.0},
    {8.0, 12.0, 18.0}
  };
  std::vector<std::vector<double>> costs = {
    {1.0, 2.0, 3.0},
    {1.0, 2.0, 3.0},
    {1.0, 2.0, 3.0}
  };

  // Use process_data to create TreatmentView arrays
  std::vector<std::vector<TreatmentView>> treatment_arrays = process_data(ids, rewards, costs);

  convex_hull(treatment_arrays);

  // Patient 0: constant slope (10-0)/1 = (15-10)/1 = (20-15)/1 = 5, no dominated points
  CHECK(treatment_arrays[0].size() == 3);
  CHECK(treatment_arrays[0][0].get_id() == 1);
  CHECK(treatment_arrays[0][1].get_id() == 2);
  CHECK(treatment_arrays[0][2].get_id() == 3);

  // Patient 1: constant slope (5-0)/1 = (10-5)/1 = (15-10)/1 = 5, no dominated points
  CHECK(treatment_arrays[1].size() == 3);
  CHECK(treatment_arrays[1][0].get_id() == 1);
  CHECK(treatment_arrays[1][1].get_id() == 2);
  CHECK(treatment_arrays[1][2].get_id() == 3);

  // Patient 2: slopes are (8-0)/1=8, (12-8)/1=4, (18-12)/1=6
  // Middle point (id=2) is dominated because slope decreases then increases
  CHECK(treatment_arrays[2].size() == 2);
  CHECK(treatment_arrays[2][0].get_id() == 1);
  CHECK(treatment_arrays[2][1].get_id() == 3);
}

TEST_CASE("Convex hull removes dominated points") {
  // Treatment 2 is dominated: worse reward/cost ratio than linear interpolation between 1 and 3
  std::vector<std::vector<uint32_t>> ids = {{1, 2, 3}};
  std::vector<std::vector<double>> rewards = {{10.0, 12.0, 30.0}};  // 12 is dominated
  std::vector<std::vector<double>> costs = {{5.0, 10.0, 15.0}};

  std::vector<std::vector<TreatmentView>> treatment_arrays = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  // Should keep only treatments 1 and 3
  CHECK(treatment_arrays[0].size() == 2);
  CHECK(treatment_arrays[0][0].get_id() == 1);
  CHECK(treatment_arrays[0][1].get_id() == 3);
}

TEST_CASE("Convex hull filters negative and zero rewards") {
  std::vector<std::vector<uint32_t>> ids = {{1, 2, 3, 4}};
  std::vector<std::vector<double>> rewards = {{-5.0, 0.0, 10.0, 20.0}};
  std::vector<std::vector<double>> costs = {{1.0, 2.0, 3.0, 4.0}};

  std::vector<std::vector<TreatmentView>> treatment_arrays = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  // Should only keep positive reward treatments
  // Treatment 4 (id=4) is dominated: slope from 3 to 4 is (20-10)/(4-3)=10
  // We keep treatment 4 because it has higher reward
  CHECK(treatment_arrays[0].size() == 1);
  CHECK(treatment_arrays[0][0].get_id() == 4);
}

TEST_CASE("Convex hull handles single treatment") {
  std::vector<std::vector<uint32_t>> ids = {{1}};
  std::vector<std::vector<double>> rewards = {{10.0}};
  std::vector<std::vector<double>> costs = {{5.0}};

  std::vector<std::vector<TreatmentView>> treatment_arrays = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  CHECK(treatment_arrays[0].size() == 1);
  CHECK(treatment_arrays[0][0].get_id() == 1);
}

TEST_CASE("Convex hull handles empty patient") {
  std::vector<std::vector<uint32_t>> ids = {{}};
  std::vector<std::vector<double>> rewards = {{}};
  std::vector<std::vector<double>> costs = {{}};

  std::vector<std::vector<TreatmentView>> treatment_arrays = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  // Empty patient should remain empty
  CHECK(treatment_arrays[0].size() == 0);
}

TEST_CASE("Convex hull sorts by cost") {
  // Treatments intentionally unsorted by cost
  std::vector<std::vector<uint32_t>> ids = {{3, 1, 2}};
  std::vector<std::vector<double>> rewards = {{30.0, 10.0, 20.0}};
  std::vector<std::vector<double>> costs = {{15.0, 5.0, 10.0}};

  std::vector<std::vector<TreatmentView>> treatment_arrays = process_data(ids, rewards, costs);
  convex_hull(treatment_arrays);

  // After convex hull, should be sorted by cost
  CHECK(treatment_arrays[0][0].get_cost() < treatment_arrays[0][1].get_cost());
  CHECK(treatment_arrays[0][1].get_cost() < treatment_arrays[0][2].get_cost());

  // And IDs should reflect the sorted order
  CHECK(treatment_arrays[0][0].get_id() == 1);
  CHECK(treatment_arrays[0][1].get_id() == 2);
  CHECK(treatment_arrays[0][2].get_id() == 3);
}

TEST_CASE("candidate_dominates_last_selection works correctly") {
  std::vector<std::vector<uint32_t>> ids = {{1, 2, 3}};
  std::vector<std::vector<double>> rewards = {{10.0, 15.0, 25.0}};
  std::vector<std::vector<double>> costs = {{5.0, 10.0, 15.0}};

  std::vector<std::vector<TreatmentView>> treatment_arrays = process_data(ids, rewards, costs);

  std::vector<TreatmentView> selections;
  selections.push_back(treatment_arrays[0][0]);  // (5, 10)
  selections.push_back(treatment_arrays[0][1]);  // (10, 15)

  TreatmentView candidate = treatment_arrays[0][2];  // (15, 25)

  // Check if candidate (15, 25) dominates the middle point (10, 15)
  // Slope from (5,10) to (15,25) = 15/10 = 1.5
  // Slope from (5,10) to (10,15) = 5/5 = 1.0
  // Since 1.5 > 1.0, the candidate should dominate
  bool dominates = candidate_dominates_last_selection(selections, candidate);
  CHECK(dominates == true);
}
