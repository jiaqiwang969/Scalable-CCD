// Gather environment and build info into a JSON object.
#pragma once

#include <nlohmann/json.hpp>

namespace verifier {

nlohmann::json collect_env_info();

} // namespace verifier

