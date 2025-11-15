// Ground-truth comparison compatible with tests/data layout.

#include "compare.hpp"

#include <set>
#include <fstream>

namespace verifier {

CompareResult compare_overlaps_with_truth(
    std::vector<std::pair<int, int>> overlaps,
    const std::vector<int>& result_list,
    const std::filesystem::path& ground_truth_file)
{
    CompareResult cr{};

    // Load truth pairs
    std::set<std::pair<long, long>> truth;
    {
        std::ifstream in(ground_truth_file);
        if (!in.good()) {
            // File missing -> mark as not covered, totals are 0
            return cr;
        }
        const nlohmann::json j = nlohmann::json::parse(in);
        for (auto& arr : j.get<std::vector<std::array<long, 2>>>()) {
            truth.emplace(arr[0], arr[1]);
        }
        cr.truth_total = truth.size();
    }

    // Build algo set (optionally filter by result_list booleans)
    std::set<std::pair<long, long>> algo;
    algo.clear();
    if (result_list.empty()) {
        for (auto& p : overlaps) {
            algo.emplace(p.first, p.second);
        }
    } else {
        for (size_t i = 0; i < overlaps.size() && i < result_list.size(); ++i) {
            if (result_list[i]) {
                algo.emplace(overlaps[i].first, overlaps[i].second);
            }
        }
    }
    cr.algo_total = algo.size();

    // Intersection size
    std::vector<std::pair<long, long>> inter;
    inter.reserve(std::min(truth.size(), algo.size()));
    std::set_intersection(
        truth.begin(), truth.end(), algo.begin(), algo.end(),
        std::back_inserter(inter));
    cr.true_positives = inter.size();
    cr.covers_truth = (cr.true_positives == cr.truth_total);
    return cr;
}

} // namespace verifier

