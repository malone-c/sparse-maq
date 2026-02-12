#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../src/convex_hull.hpp"

#include <vector>
#include "../src/preprocess_data.hpp"
using sparse_maq::Treatment;

TEST_CASE("Convex hull test") {
  std::vector<std::vector<Treatment>> treatment_arrays = {
    {Treatment(1, 10.0, 1.0), Treatment(2, 15.0, 2.0), Treatment(3, 20.0, 3.0)},
    {Treatment(1, 5.0, 1.0), Treatment(2, 10.0, 2.0), Treatment(3, 15.0, 3.0)},
    {Treatment(1, 8.0, 1.0), Treatment(2, 12.0, 2.0), Treatment(3, 18.0, 3.0)}
  };
  
  convex_hull(treatment_arrays);
  
  CHECK(treatment_arrays[0].size() == 3);
  CHECK(treatment_arrays[1].size() == 3);
  CHECK(treatment_arrays[2].size() == 2);
  
  CHECK(treatment_arrays[0][0].id == 1);
  CHECK(treatment_arrays[0][1].id == 2);
  CHECK(treatment_arrays[0][2].id == 3);
  
  CHECK(treatment_arrays[1][0].id == 1);
  CHECK(treatment_arrays[1][1].id == 2);
  CHECK(treatment_arrays[1][2].id == 3);
  
  CHECK(treatment_arrays[2][0].id == 1);
  CHECK(treatment_arrays[2][1].id == 3);
}
