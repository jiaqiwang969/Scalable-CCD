#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "root_finder.hpp"
#include <scalable_ccd/metal/runtime/runtime.hpp>

namespace scalable_ccd::metal::detail {

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
    float& out_min_toi)
{
    return MetalRuntime::instance().runVFMin(
        vertices_t0_flat, vertices_t1_flat, num_vertices, faces_flat, overlaps,
        minimum_separation, tolerance, max_iterations, allow_zero_toi, out_min_toi);
}

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
    std::vector<float>& out_toi)
{
    return MetalRuntime::instance().runVFRootSkeleton(
        vertices_t0_flat, vertices_t1_flat, num_vertices, faces_flat, overlaps,
        minimum_separation, tolerance, max_iterations, allow_zero_toi, out_toi);
}

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
    float& out_min_toi)
{
    return MetalRuntime::instance().runEEMin(
        vertices_t0_flat, vertices_t1_flat, num_vertices, edges_flat, overlaps,
        minimum_separation, tolerance, max_iterations, allow_zero_toi, out_min_toi);
}

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
    std::vector<float>& out_toi)
{
    return MetalRuntime::instance().runEERootSkeleton(
        vertices_t0_flat, vertices_t1_flat, num_vertices, edges_flat, overlaps,
        minimum_separation, tolerance, max_iterations, allow_zero_toi, out_toi);
}

} // namespace scalable_ccd::metal::detail
 #import <Metal/Metal.h>
