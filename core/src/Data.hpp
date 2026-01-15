#ifndef MAQ_DATA_HPP
#define MAQ_DATA_HPP

#include <cstddef>
#include <vector>
#include <string>
#include <iostream>

namespace sparse_maq {

class TreatmentView {
  public:
    TreatmentView(uint32_t& id, double& reward, double& cost) 
      : id_ptr(&id), reward_ptr(&reward), cost_ptr(&cost) {}

    uint32_t get_id() const { return *id_ptr; }
    double get_reward() const { return *reward_ptr; }
    double get_cost() const { return *cost_ptr; }

  private:
    uint32_t* id_ptr;
    double* reward_ptr;
    double* cost_ptr;
};

  std::vector<std::vector<TreatmentView>> process_data(
    std::vector<std::vector<uint32_t>>& treatment_id_arrays,
    std::vector<std::vector<double>>& reward_arrays,
    std::vector<std::vector<double>>& cost_arrays
  ) {
    size_t num_units = treatment_id_arrays.size();
    std::vector<std::vector<TreatmentView>> treatment_view_arrays;
    treatment_view_arrays.reserve(num_units);

    for (size_t i = 0; i < num_units; ++i) {
      size_t num_treatments = treatment_id_arrays[i].size();
      treatment_id_arrays[i].reserve(num_treatments);
      std::vector<TreatmentView> treatment_views;
      treatment_views.reserve(num_treatments);

      for (size_t j = 0; j < num_treatments; ++j) {
        treatment_views.emplace_back(
          treatment_id_arrays[i][j],
          reward_arrays[i][j],
          cost_arrays[i][j]
        );
      }

      treatment_view_arrays.push_back(std::move(treatment_views));
    }
    return treatment_view_arrays;
  }
} // namespace sparse_maq

#endif // MAQ_DATA_HPP
