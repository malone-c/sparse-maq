#ifndef MAQ_DATA_HPP
#define MAQ_DATA_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sparse_maq {

  struct Treatment { 
    Treatment(size_t id, double reward, double cost)
      : id(id), reward(reward), cost(cost) {}

    size_t id;
    double reward;
    double cost;
  };

  std::pair<
    std::vector<std::vector<Treatment>>,
    std::vector<std::string>
  > preprocess_data_cpp(
    std::vector<std::vector<std::string>>&& treatment_id_arrays,
    std::vector<std::vector<double>>&& reward_arrays,
    std::vector<std::vector<double>>&& cost_arrays
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

      // Free memory
      treatment_id_arrays[i].clear();
      treatment_id_arrays[i].shrink_to_fit();
      reward_arrays[i].clear();
      reward_arrays[i].shrink_to_fit();
      cost_arrays[i].clear();
      cost_arrays[i].shrink_to_fit();
    }

    return {
      treatment_view_arrays,
      treatment_num_to_id
    };
  }

  std::pair<
    std::vector<std::vector<Treatment>>,
    std::vector<std::string>
  > preprocess_data_flat(
    int64_t num_patients,
    std::vector<int32_t>&& list_offsets,
    std::vector<double>&& rewards_flat,
    std::vector<double>&& costs_flat,
    std::vector<int32_t>&& str_offsets,
    std::vector<uint8_t>&& str_data
  ) {
    // Transparent hash: one operator()(string_view) covers string, string_view,
    // and const char* via implicit conversion — no separate overloads needed.
    struct sv_hash {
      using is_transparent = void;
      size_t operator()(std::string_view sv) const noexcept {
        return std::hash<std::string_view>{}(sv);
      }
    };

    std::unordered_map<std::string, size_t, sv_hash, std::equal_to<>> treatment_id_to_num;
    std::vector<std::string> treatment_num_to_id;
    size_t cur_treatment_num{0};

    std::vector<std::vector<Treatment>> treatment_view_arrays;
    treatment_view_arrays.reserve(num_patients);

    for (int64_t i = 0; i < num_patients; ++i) {
      int32_t start = list_offsets[i];
      int32_t end = list_offsets[i + 1];
      std::vector<Treatment> treatments;
      treatments.reserve(end - start);

      for (int32_t j = start; j < end; ++j) {
        // str_offsets[j] is the byte offset of treatment j's ID in str_data.
        // str_offsets are flat — j is already the flat treatment index, no list_offsets[i] needed.
        std::string_view sv(
          reinterpret_cast<const char*>(str_data.data()) + str_offsets[j],
          str_offsets[j + 1] - str_offsets[j]
        );

        // emplace returns {iterator, inserted_bool} — one probe for both
        // lookup and insert; no second find needed.
        auto [it, inserted] = treatment_id_to_num.emplace(std::string(sv), cur_treatment_num);
        if (inserted) {
          treatment_num_to_id.emplace_back(sv);
          ++cur_treatment_num;
        }
        treatments.emplace_back(it->second, rewards_flat[j], costs_flat[j]);
      }

      treatment_view_arrays.push_back(std::move(treatments));
    }

    // Free flat buffers before convex_hull runs
    list_offsets.clear();  list_offsets.shrink_to_fit();
    rewards_flat.clear();  rewards_flat.shrink_to_fit();
    costs_flat.clear();    costs_flat.shrink_to_fit();
    str_offsets.clear();   str_offsets.shrink_to_fit();
    str_data.clear();      str_data.shrink_to_fit();

    return {treatment_view_arrays, treatment_num_to_id};
  }

} // namespace sparse_maq

#endif // MAQ_DATA_HPP
