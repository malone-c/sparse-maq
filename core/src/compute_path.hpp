#ifndef MAQ_COMPUTE_PATH_HPP
#define MAQ_COMPUTE_PATH_HPP

#include <cstddef>
#include <queue>
#include <vector>
#include "preprocess_data.hpp" // Include the header file where Data is defined

namespace sparse_maq {

typedef std::pair<std::vector<std::vector<double>>, std::vector<std::vector<size_t>>> solution_path;

struct QueueElement {
  QueueElement(
    size_t unit,
    const TreatmentView* treatment, 
    double priority
  ) : unit(unit), treatment_ptr(treatment), priority(priority) {}  // Fix: initialize treatment_ptr correctly

  size_t unit;
  const TreatmentView* treatment_ptr;  // Store const pointer instead of copy
  double priority;
};

bool operator <(const QueueElement& lhs, const QueueElement& rhs) {
  return lhs.priority < rhs.priority;
}

solution_path compute_path(
  const std::vector<std::vector<TreatmentView>>& treatment_arrays,
  double budget
) {
  std::vector<std::vector<double>> spend_gain(3); // 3rd entry: SEs
  std::vector<std::vector<size_t>> i_k_path(3); // 3rd entry: complete path
  std::vector<size_t> active_arm_indices(treatment_arrays.size(), 0); // active treatment entry offset by one

  // TODO: Initialise vector with some treatment (could be null treatment). This lets us force all units to get a treatment.

  // Initialize PQ with initial enrollment
  std::priority_queue<QueueElement> pqueue;
  for (size_t unit = 0; unit < treatment_arrays.size(); unit++) {
    if (treatment_arrays[unit].empty()) { continue; }

    const TreatmentView& treatment_ref = treatment_arrays[unit][0];
    double priority = treatment_ref.get_reward() / treatment_ref.get_cost();
    pqueue.emplace(unit, &treatment_ref, priority);
  }

  double spend = 0;
  double gain = 0;
  while (!pqueue.empty() && spend < budget) {
    QueueElement top = pqueue.top();
    pqueue.pop();

    if (active_arm_indices[top.unit] > 0) { // If assigned before...
      size_t active_arm_index = active_arm_indices[top.unit] - 1;
      TreatmentView active_arm = treatment_arrays[top.unit][active_arm_index];
      spend -= active_arm.get_cost();
      gain -= active_arm.get_reward();
    }

    // assign
    spend += top.treatment_ptr->get_cost();
    gain += top.treatment_ptr->get_reward();
    spend_gain[0].push_back(spend);
    spend_gain[1].push_back(gain);
    i_k_path[0].push_back(top.unit);
    i_k_path[1].push_back(top.treatment_ptr->get_id());
    active_arm_indices[top.unit]++;

    size_t next_entry = active_arm_indices[top.unit];
    if (treatment_arrays[top.unit].size() > next_entry) { // More treatments available for this unit?
      // To this:
      const TreatmentView& upgrade_ref = treatment_arrays[top.unit][next_entry];
      double priority = (upgrade_ref.get_reward() - top.treatment_ptr->get_reward()) / (upgrade_ref.get_cost() - top.treatment_ptr->get_cost());
      pqueue.emplace(top.unit, &upgrade_ref, priority);
    }

    // have we reached maximum spend? if so stop at nearest integer solution (rounded up)
    if (spend >= budget) {
      break;
    }
  }

  // "complete" path?
  i_k_path[2].push_back(pqueue.empty() ? 1 : 0);

  return std::make_pair(std::move(spend_gain), std::move(i_k_path));
}

} // namespace sparse_maq

#endif // MAQ_COMPUTE_PATH_HPP
