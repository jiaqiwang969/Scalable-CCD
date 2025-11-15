#pragma once

#include <vector>
#include <tuple>
#include <cstdint>
#include <Eigen/Core>

#include <scalable_ccd/metal/runtime/runtime.hpp>

// 与 CUDA 端的打包阶段对齐：提供统一的 pack 接口（占位包装）。

namespace scalable_ccd::metal {

template <bool is_vf>
inline bool pack_queries(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& elems, // faces for VF, edges for EE
    const std::vector<std::pair<int,int>>& overlaps,
    float minimum_separation,
    uint32_t& out_packed_count)
{
    const int num_vertices = static_cast<int>(vertices_t0.rows());
    std::vector<float> vertices_t0_flat;
    vertices_t0_flat.reserve(static_cast<size_t>(num_vertices) * 3);
    std::vector<float> vertices_t1_flat;
    vertices_t1_flat.reserve(static_cast<size_t>(num_vertices) * 3);
    for (int i = 0; i < num_vertices; ++i) {
        vertices_t0_flat.push_back(static_cast<float>(vertices_t0(i, 0)));
        vertices_t0_flat.push_back(static_cast<float>(vertices_t0(i, 1)));
        vertices_t0_flat.push_back(static_cast<float>(vertices_t0(i, 2)));
        vertices_t1_flat.push_back(static_cast<float>(vertices_t1(i, 0)));
        vertices_t1_flat.push_back(static_cast<float>(vertices_t1(i, 1)));
        vertices_t1_flat.push_back(static_cast<float>(vertices_t1(i, 2)));
    }
    std::vector<int32_t> elements_flat;
    elements_flat.reserve(
        static_cast<size_t>(elems.rows()) * (is_vf ? 3 : 2));
    for (int i = 0; i < elems.rows(); ++i) {
        elements_flat.push_back(static_cast<int32_t>(elems(i, 0)));
        elements_flat.push_back(static_cast<int32_t>(elems(i, 1)));
        if constexpr (is_vf) {
            elements_flat.push_back(static_cast<int32_t>(elems(i, 2)));
        }
    }
    if constexpr (is_vf) {
        return MetalRuntime::instance().narrowPackVF(
            vertices_t0_flat, vertices_t1_flat, static_cast<uint32_t>(num_vertices),
            elements_flat, overlaps, minimum_separation, out_packed_count);
    } else {
        return MetalRuntime::instance().narrowPackEE(
            vertices_t0_flat, vertices_t1_flat, static_cast<uint32_t>(num_vertices),
            elements_flat, overlaps, minimum_separation, out_packed_count);
    }
}

} // namespace scalable_ccd::metal
 #pragma once
