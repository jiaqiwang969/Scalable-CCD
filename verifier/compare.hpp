// Compare overlaps with ground-truth "boxes" JSON; no Catch2 dependency.
#pragma once

#include <filesystem>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace verifier {

struct CompareResult {
    // True positives count (overlaps matching truth)
    size_t true_positives = 0;
    // Total ground-truth positives (should equal true_positives if all found)
    size_t truth_total = 0;
    // Algorithm-reported positives (after filtering by result_list, if any)
    size_t algo_total = 0;
    // Whether algo covered all truth positives
    bool covers_truth = false;
};

// If result_list is empty, treat all overlaps as positives.
CompareResult compare_overlaps_with_truth(
    std::vector<std::pair<int, int>> overlaps,
    const std::vector<int>& result_list,
    const std::filesystem::path& ground_truth_file);

} // namespace verifier

