#ifndef MAQ_CONVEX_HULL_HPP
#define MAQ_CONVEX_HULL_HPP

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>
#include <deque>

#include "Data.hpp"

/*
Find the upper left convex hull on the (cost, reward) plane for each unit. This takes
O(num_rows * num_arms * log(num_arms)) time using the Graham scan for finding a convex hull,
with the the "angle calculation" replaced by a variant of the "LP dominance" criteria
(11.6) and (11.7) in Kellerer et al. (2004, Chapter 11).

Consider 3 points j, k, l ordered according to cost j < k < l.
This algorithm maintains a stack with the first two points on top, then iteratively
proceeds by checking if point k should be replaced by or augmented by point l.
*/

namespace sparse_maq {

inline bool candidate_dominates_last_selection(
  const std::vector<TreatmentView>& selections,
  TreatmentView candidate
) {
  uint32_t zero_id_value = 0;
  double zero_reward_value = 0.0;
  double zero_cost_value = 0.0;
  TreatmentView arm_j = TreatmentView(zero_id_value, zero_reward_value, zero_cost_value); // dummy point

  if (selections.size() >= 2) {
    arm_j = selections[selections.size() - 2]; // next to top
  }
  TreatmentView arm_k = selections[selections.size() - 1]; // top
  if (arm_k.get_reward() <= 0) {
    return true;
  }

  // C++: a/0 = Inf if a > 0, a/0 = -Inf if a <0, and 0/0 = NaN (all logical operators on NaN evaluate to false)
  // return (reward_l - reward_k) / (cost_l - cost_k) > (reward_k - reward_j) / (cost_k - cost_j);
  return (candidate.get_reward() - arm_k.get_reward()) * (arm_k.get_cost() - arm_j.get_cost()) > (arm_k.get_reward() - arm_j.get_reward()) * (candidate.get_cost() - arm_k.get_cost());
}
// TODO: Implement multithreading
void convex_hull(std::vector<std::vector<TreatmentView>>& treatment_arrays) {
  
  for (size_t unit = 0; unit < treatment_arrays.size(); unit++) {
    
    // Copy all the treatments to a queue and clear the original vector
    std::deque<TreatmentView> candidates(treatment_arrays[unit].begin(), treatment_arrays[unit].end());
    
    std::vector<TreatmentView>& selections = treatment_arrays[unit];
    selections.clear();
    
    // Sort by increasing cost
    std::sort(candidates.begin(), candidates.end(), [&](const TreatmentView lhs, const TreatmentView rhs) {
      return lhs.get_cost() < rhs.get_cost();
    });
    
    // Push first positive reward point onto stack
    while (candidates.size() > 0 && candidates[0].get_reward() <= 0) {
      candidates.pop_front();
    }
    
    if (candidates.size() == 0) {
      continue;
    }
    
    selections.push_back(candidates[0]);
    candidates.pop_front();
    
    while (candidates.size() > 0) {
      TreatmentView candidate = candidates[0];
      candidates.pop_front();
      
      while (selections.size() > 0 && candidate_dominates_last_selection(selections, candidate)) {
        selections.pop_back();
      }
      
      if (candidate.get_reward() > 0) {
        if (selections.empty() || candidate.get_reward() > selections.back().get_reward()) {
          selections.push_back(candidate);
        }
      }
    }
  }
}

} // namespace sparse_maq

#endif // MAQ_CONVEX_HULL_HPP
