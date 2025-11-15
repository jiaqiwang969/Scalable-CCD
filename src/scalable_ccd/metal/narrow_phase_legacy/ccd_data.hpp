#pragma once

#include <scalable_ccd/scalar.hpp>

namespace scalable_ccd::metal {

// 与 CUDA 端 ccd_data.cuh 对齐的命名与字段（用于接口对齐/后续重构）
struct CCDData {
    // start/end positions for four points (VF: v0/v1/v2/v3; EE: two edges)
    Eigen::Matrix<Scalar, 3, 1> v0s, v1s, v2s, v3s;
    Eigen::Matrix<Scalar, 3, 1> v0e, v1e, v2e, v3e;
    Eigen::Matrix<Scalar, 3, 1> err; // 数值误差上界
    Eigen::Matrix<Scalar, 3, 1> tol; // 域容差（用于划分维度）
    Scalar ms = 0;                   // minimum separation
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    Scalar toi = 1; // per-query toi（观察/调试用）
    int aid = -1;   // overlap a id
    int bid = -1;   // overlap b id
#endif
    int nbr_checks = 0; // 与 CUDA 对齐的计数器（可选）
};

} // namespace scalable_ccd::metal
 #pragma once
