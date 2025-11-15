#pragma once

#include <scalable_ccd/scalar.hpp>
#include <Eigen/Core>
#include <tuple>
#include <vector>

namespace scalable_ccd::metal {

// 前置声明供模板包装调用（实现见 narrow_phase.mm）
Scalar narrow_phase_vf(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& faces,
    const std::vector<std::pair<int,int>>& overlaps,
    Scalar minimum_separation,
    Scalar tolerance,
    int max_iterations,
    bool allow_zero_toi
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    , std::vector<std::tuple<int, int, Scalar>>& collisions
#endif
);

Scalar narrow_phase_ee(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& edges,
    const std::vector<std::pair<int,int>>& overlaps,
    Scalar minimum_separation,
    Scalar tolerance,
    int max_iterations,
    bool allow_zero_toi
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    , std::vector<std::tuple<int, int, Scalar>>& collisions
#endif
);

// 与 CUDA 窄相位入口命名对齐的模板包装（转调现有 VF/EE 实现）。
template <bool is_vf>
inline void narrow_phase(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& edges,
    const Eigen::MatrixXi& faces,
    const std::vector<std::pair<int,int>>& overlaps,
    int threads,
    int max_iterations,
    Scalar tolerance,
    Scalar minimum_separation,
    bool allow_zero_toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<std::tuple<int,int,Scalar>>& collisions,
#endif
    Scalar& toi)
{
    (void)threads; // Metal 端线程宽度由 runtime 自行决定；保留参数对齐
    if constexpr (is_vf) {
        toi = narrow_phase_vf(
            vertices_t0, vertices_t1, faces, overlaps,
            minimum_separation, tolerance, max_iterations, allow_zero_toi
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            , collisions
#endif
        );
    } else {
        toi = narrow_phase_ee(
            vertices_t0, vertices_t1, edges, overlaps,
            minimum_separation, tolerance, max_iterations, allow_zero_toi
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            , collisions
#endif
        );
    }
}

/// Narrow-phase (scaffold): run on overlaps and return minimum toi.
// 保留实现声明（下方已给出前置声明）

} // namespace scalable_ccd::metal
 #pragma once
