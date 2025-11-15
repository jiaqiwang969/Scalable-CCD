#pragma once

#include <scalable_ccd/scalar.hpp>
#include <iostream>
#include <Eigen/Core>
#include <vector>
#include <tuple>
#include <chrono>
#include <scalable_ccd/metal/runtime/runtime.hpp>
#include <scalable_ccd/utils/logger.hpp>

namespace scalable_ccd::metal {

// 将与 MetalRuntime 的直接交互放到实现文件（.mm）中，模板通过 detail 包装调用，便于结构对齐。
namespace detail {
bool run_vf_min(
    const std::vector<float>& vertices_t0_flat,
    const std::vector<float>& vertices_t1_flat,
    uint32_t num_vertices,
    const std::vector<int32_t>& faces_flat,
    const std::vector<std::pair<int,int>>& overlaps,
    float minimum_separation,
    float tolerance,
    int max_iterations,
    bool allow_zero_toi,
    float& out_min_toi);

bool run_vf_all_toi(
    const std::vector<float>& vertices_t0_flat,
    const std::vector<float>& vertices_t1_flat,
    uint32_t num_vertices,
    const std::vector<int32_t>& faces_flat,
    const std::vector<std::pair<int,int>>& overlaps,
    float minimum_separation,
    float tolerance,
    int max_iterations,
    bool allow_zero_toi,
    std::vector<float>& out_toi);

bool run_ee_min(
    const std::vector<float>& vertices_t0_flat,
    const std::vector<float>& vertices_t1_flat,
    uint32_t num_vertices,
    const std::vector<int32_t>& edges_flat,
    const std::vector<std::pair<int,int>>& overlaps,
    float minimum_separation,
    float tolerance,
    int max_iterations,
    bool allow_zero_toi,
    float& out_min_toi);

bool run_ee_all_toi(
    const std::vector<float>& vertices_t0_flat,
    const std::vector<float>& vertices_t1_flat,
    uint32_t num_vertices,
    const std::vector<int32_t>& edges_flat,
    const std::vector<std::pair<int,int>>& overlaps,
    float minimum_separation,
    float tolerance,
    int max_iterations,
    bool allow_zero_toi,
    std::vector<float>& out_toi);
} // namespace detail

// 与 CUDA 端 root_finder.cuh 的命名对齐：保留 compute_tolerance/ccd* 包装，
// Metal 实现通过 detail::run_* 间接调用 MetalRuntime 的 run* 接口。

template <bool is_vf>
inline void compute_tolerance(/*CCDData* data, int query_size*/)
{
    // Metal 路径当前在 GPU 内核中按记录计算 tol，接口保留作占位以便后续对齐。
}

template <bool is_vf>
inline bool run_narrow_phase_root_finder(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& elems, // faces for VF, edges for EE
    const std::vector<std::pair<int,int>>& overlaps,
    int max_iterations,
    Scalar tolerance,
    Scalar minimum_separation,
    bool allow_zero_toi,
#ifdef SCALABLE_CCD_TOI_PER_QUERY
    std::vector<std::tuple<int,int,Scalar>>& collisions,
#endif
    Scalar& out_min_toi)
{
    auto total_start = std::chrono::high_resolution_clock::now();
    out_min_toi = 1;
    if (overlaps.empty()) return false;

    scalable_ccd::logger().info("[Metal Root Finder {}] Starting with {} overlaps, {} vertices, {} elements", 
                                is_vf ? "VF" : "EE", overlaps.size(), vertices_t0.rows(), elems.rows());

    // 扁平化顶点
    auto flatten_vertices_start = std::chrono::high_resolution_clock::now();
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
    auto flatten_vertices_end = std::chrono::high_resolution_clock::now();
    auto flatten_vertices_time = std::chrono::duration_cast<std::chrono::microseconds>(flatten_vertices_end - flatten_vertices_start).count();
    scalable_ccd::logger().info("[Metal Root Finder {}] Flatten vertices took {} us", is_vf ? "VF" : "EE", flatten_vertices_time);

    // 扁平化元素
    auto flatten_elems_start = std::chrono::high_resolution_clock::now();
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
    auto flatten_elems_end = std::chrono::high_resolution_clock::now();
    auto flatten_elems_time = std::chrono::duration_cast<std::chrono::microseconds>(flatten_elems_end - flatten_elems_start).count();
    scalable_ccd::logger().info("[Metal Root Finder {}] Flatten elements took {} us", is_vf ? "VF" : "EE", flatten_elems_time);

    float min_toi = 1.0f;
    bool ok = false;
    auto kernel_start = std::chrono::high_resolution_clock::now();
    if constexpr (is_vf) {
        ok = detail::run_vf_min(
            vertices_t0_flat, vertices_t1_flat, static_cast<uint32_t>(num_vertices),
            elements_flat, overlaps, static_cast<float>(minimum_separation),
            static_cast<float>(tolerance), max_iterations, allow_zero_toi,
            min_toi);
    } else {
        ok = detail::run_ee_min(
            vertices_t0_flat, vertices_t1_flat, static_cast<uint32_t>(num_vertices),
            elements_flat, overlaps, static_cast<float>(minimum_separation),
            static_cast<float>(tolerance), max_iterations, allow_zero_toi,
            min_toi);
    }
    auto kernel_end = std::chrono::high_resolution_clock::now();
    auto kernel_time = std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end - kernel_start).count();
    scalable_ccd::logger().info("[Metal Root Finder {}] GPU kernel (run_{}_min) took {} ms", 
                                is_vf ? "VF" : "EE", is_vf ? "vf" : "ee", kernel_time);
    
    if (ok) {
        out_min_toi = static_cast<Scalar>(min_toi);
    }

#ifdef SCALABLE_CCD_TOI_PER_QUERY
    // 可选：填充每对 toi（调用 run*RootSkeleton）
    std::vector<float> toi_all;
    bool ok_all_toi = false;
    if constexpr (is_vf) {
        ok_all_toi = detail::run_vf_all_toi(
            vertices_t0_flat, vertices_t1_flat, static_cast<uint32_t>(num_vertices),
            elements_flat, overlaps, static_cast<float>(minimum_separation),
            static_cast<float>(tolerance), max_iterations, allow_zero_toi,
            toi_all);
    } else {
        ok_all_toi = detail::run_ee_all_toi(
            vertices_t0_flat, vertices_t1_flat, static_cast<uint32_t>(num_vertices),
            elements_flat, overlaps, static_cast<float>(minimum_separation),
            static_cast<float>(tolerance), max_iterations, allow_zero_toi,
            toi_all);
    }
    if (ok_all_toi) {
        collisions.reserve(collisions.size() + overlaps.size());
        for (size_t k = 0; k < overlaps.size(); ++k) {
            collisions.emplace_back(
                overlaps[k].first, overlaps[k].second,
                static_cast<Scalar>(toi_all[k]));
        }
    }
#endif
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    scalable_ccd::logger().info("[Metal Root Finder {}] Total time: {} ms, final toi: {}", 
                               is_vf ? "VF" : "EE", total_time, out_min_toi);
    
    return false; // Metal 端当前不报告溢出；保留与 CUDA 相同的返回语义
}

} // namespace scalable_ccd::metal
 #pragma once
