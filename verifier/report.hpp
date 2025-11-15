// Report writers: JSON aggregation and simple HTML.
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <filesystem>

namespace verifier {

// Write JSON to path (pretty).
void write_json(const std::filesystem::path& path, const nlohmann::json& j);

// Produce a minimal HTML report given the aggregated JSON.
std::string make_html_report(const nlohmann::json& aggregate);

// Add a table for query-file verification results into HTML body (provided in aggregate["query_runs"])
std::string make_html_report_with_queries(const nlohmann::json& aggregate);

} // namespace verifier
