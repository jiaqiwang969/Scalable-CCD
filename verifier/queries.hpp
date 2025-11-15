// Per-query narrow-phase verification against queries CSV + mma_bool JSON.
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <filesystem>
#include <vector>

namespace verifier {

struct QueryFileResult {
    std::string scene;
    int step = -1;
    std::string type; // "ee" or "vf"
    std::filesystem::path csv_path;
    std::filesystem::path truth_path;
    int total = 0;
    int truth_positives = 0;
    int algo_positives = 0;
    int mismatches = 0;
    double avg_ms = 0.0;
};

// Try to verify queries for a given scene and step.
// Return one or two results (EE and/or VF) if corresponding files exist.
// - data_dir: scene root (e.g., tests/data-full/cloth-ball)
// - step: time step number
// - max_queries: limit number of queries to check (<=0 means all)
// - on_cuda_only: if true and no CUDA build, skip and return empty
std::vector<QueryFileResult> verify_queries_for_step(
    const std::filesystem::path& scene_dir,
    const std::string& scene_name,
    int step,
    int max_queries,
    bool on_cuda_only);

} // namespace verifier

