#pragma once

#include <scalable_ccd/scalar.hpp>
#include <Eigen/Core>
#include <tuple>
#include <vector>

namespace scalable_ccd::metal {

/// Run broad + narrow phase on Metal backend (scaffold).
/// This mirrors the CUDA API for ease of adoption.
Scalar
run_ccd_pipeline(const Eigen::MatrixXd& vertices_t0,
                 const Eigen::MatrixXd& vertices_t1,
                 const Eigen::MatrixXi& edges,
                 const Eigen::MatrixXi& faces,
                 const Scalar minimum_separation,
                 const int max_iterations,
                 const Scalar tolerance,
                 const bool allow_zero_toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
                 std::vector<std::tuple<int, int, Scalar>>& collisions,
#endif
                 const int memory_limit_GB = 0);

} // namespace scalable_ccd::metal
 #pragma once
