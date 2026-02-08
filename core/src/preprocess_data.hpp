#ifndef MAQ_DATA_HPP
#define MAQ_DATA_HPP

#include <cstddef>
#include <unordered_set>
#include <vector>

namespace sparse_maq {

  struct Treatment { // TODO: Consider storing references in here instead of pointers for cache locality
    Treatment(size_t id, double reward, double cost)
      : id(id), reward(reward), cost(cost) {}

    size_t id;
    double reward;
    double cost;
  };

  std::pair<
    std::vector<std::vector<Treatment>>,
    std::vector<std::string>
  > process_data(
    std::vector<std::vector<std::string>>& treatment_id_arrays,
    std::vector<std::vector<double>>& reward_arrays,
    std::vector<std::vector<double>>& cost_arrays
  ) {
    std::unordered_map<std::string, size_t> treatment_id_to_num;
    std::vector<std::string> treatment_num_to_id;
    size_t cur_treatment_num {0};

    size_t num_patients = treatment_id_arrays.size();
    std::vector<std::vector<Treatment>> treatment_view_arrays;
    treatment_view_arrays.reserve(num_patients);

    // Construct treatment view arrays
    for (size_t i = 0; i < num_patients; ++i) {
      size_t num_treatments = treatment_id_arrays[i].size();
      std::vector<Treatment> treatments;
      treatments.reserve(num_treatments);

      for (size_t j = 0; j < num_treatments; ++j) {
        const std::string& treatment_id = treatment_id_arrays[i][j];
        if (!treatment_id_to_num.contains(treatment_id)) {
          treatment_id_to_num.emplace(treatment_id, cur_treatment_num);
          treatment_num_to_id.emplace_back(treatment_id);
          ++cur_treatment_num;
        }

        treatments.emplace_back(
          treatment_id_to_num[treatment_id],
          reward_arrays[i][j],
          cost_arrays[i][j]
        );
      }

      treatment_view_arrays.push_back(std::move(treatments));
    }

    return {
      treatment_view_arrays,
      treatment_num_to_id
    };
  }
} // namespace sparse_maq

#endif // MAQ_DATA_HPP
