#pragma once

#include <Eigen/Core>

namespace verifier::gpu {

// Return toi in [0,1] if collision exists; otherwise 1
double eval_query_ee(const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V1,
                     int max_iter, double tol, double min_sep);

double eval_query_vf(const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V1,
                     int max_iter, double tol, double min_sep);

} // namespace verifier::gpu

