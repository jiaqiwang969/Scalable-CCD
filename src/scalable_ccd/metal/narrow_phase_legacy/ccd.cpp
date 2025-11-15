#include "ccd.hpp"
#include "narrow_phase.hpp"

#include <scalable_ccd/broad_phase/aabb.hpp>
#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>
#include <scalable_ccd/metal/broad_phase/broad_phase.hpp>
#include <scalable_ccd/utils/logger.hpp>
#include <iostream>
#include <chrono>

namespace scalable_ccd::metal {


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
    const int /*memory_limit_GB*/)
{
    auto pipeline_start = std::chrono::high_resolution_clock::now();
    assert(vertices_t0.rows() == vertices_t1.rows());
    assert(vertices_t0.cols() == vertices_t1.cols());
    assert(vertices_t0.cols() == 3);
    assert(edges.cols() == 2);
    assert(faces.cols() == 3);
    
    std::cout << "[Metal CCD] Starting with " << vertices_t0.rows() << " vertices, "
              << edges.rows() << " edges, " << faces.rows() << " faces\n";

    // Build AABBs and reuse Metal BroadPhase
    auto aabb_start = std::chrono::high_resolution_clock::now();
    std::cout << "[Metal CCD] Building AABBs...\n";
    std::vector<AABB> vertex_boxes, edge_boxes, face_boxes;
    build_vertex_boxes(vertices_t0, vertices_t1, vertex_boxes, minimum_separation);
    build_edge_boxes(vertex_boxes, edges, edge_boxes);
    build_face_boxes(vertex_boxes, faces, face_boxes);
    auto aabb_end = std::chrono::high_resolution_clock::now();
    auto aabb_time = std::chrono::duration_cast<std::chrono::milliseconds>(aabb_end - aabb_start).count();
    std::cout << "[Metal CCD] Built " << vertex_boxes.size() << " vertex boxes, "
              << edge_boxes.size() << " edge boxes, " << face_boxes.size() << " face boxes\n";
    scalable_ccd::logger().info("[Metal CCD Pipeline] AABB building took {} ms", aabb_time);

    auto device_aabb_start = std::chrono::high_resolution_clock::now();
    std::cout << "[Metal CCD] Creating DeviceAABBs...\n";
    auto dV = std::make_shared<metal::DeviceAABBs>(vertex_boxes);
    auto dE = std::make_shared<metal::DeviceAABBs>(edge_boxes);
    auto dF = std::make_shared<metal::DeviceAABBs>(face_boxes);
    auto device_aabb_end = std::chrono::high_resolution_clock::now();
    auto device_aabb_time = std::chrono::duration_cast<std::chrono::milliseconds>(device_aabb_end - device_aabb_start).count();
    std::cout << "[Metal CCD] DeviceAABBs created\n";
    scalable_ccd::logger().info("[Metal CCD Pipeline] DeviceAABBs creation took {} ms", device_aabb_time);

    Scalar toi = 1;
    
    // VF narrow phase
    {
        auto vf_start = std::chrono::high_resolution_clock::now();
        std::cout << "[Metal CCD] Starting VF narrow phase...\n";
        metal::BroadPhase bp;
        
        auto vf_broad_start = std::chrono::high_resolution_clock::now();
        std::cout << "[Metal CCD] Building VF broad phase...\n";
        bp.build(dV, dF);
        auto vf_broad_end = std::chrono::high_resolution_clock::now();
        auto vf_broad_time = std::chrono::duration_cast<std::chrono::milliseconds>(vf_broad_end - vf_broad_start).count();
        std::cout << "[Metal CCD] VF broad phase built\n";
        scalable_ccd::logger().info("[Metal CCD Pipeline] VF broad phase build took {} ms", vf_broad_time);
        
        auto vf_detect_start = std::chrono::high_resolution_clock::now();
        std::cout << "[Metal CCD] Detecting VF overlaps...\n";
        const auto vf_overlaps = bp.detect_overlaps();
        auto vf_detect_end = std::chrono::high_resolution_clock::now();
        auto vf_detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(vf_detect_end - vf_detect_start).count();
        std::cout << "[Metal CCD] Found " << vf_overlaps.size() << " VF overlaps\n";
        scalable_ccd::logger().info("[Metal CCD Pipeline] VF overlap detection took {} ms, found {} overlaps", 
                                    vf_detect_time, vf_overlaps.size());
        
        scalable_ccd::logger().debug("CCD: VF overlaps from broad phase: {}", vf_overlaps.size());
        
        // Use the real narrow phase implementation
        const Scalar vf_toi = narrow_phase_vf(
            vertices_t0, vertices_t1, faces, vf_overlaps,
            minimum_separation, tolerance, max_iterations, allow_zero_toi
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            , collisions
#endif
        );
        
        auto vf_end = std::chrono::high_resolution_clock::now();
        auto vf_time = std::chrono::duration_cast<std::chrono::milliseconds>(vf_end - vf_start).count();
        scalable_ccd::logger().info("[Metal CCD Pipeline] VF narrow phase total took {} ms, toi = {}", vf_time, vf_toi);
        
        if (vf_toi < toi) {
            toi = vf_toi;
        }
        // 将 VF 的最小 toi 作为 EE 的初始全局上界，便于 EE 阶段裁剪
        {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.9f", static_cast<double>(toi));
            setenv("SCALABLE_CCD_INIT_MIN_TOI", buf, 1);
        }
    }
    
    // EE narrow phase
    {
        auto ee_start = std::chrono::high_resolution_clock::now();
        metal::BroadPhase bp;
        
        auto ee_broad_start = std::chrono::high_resolution_clock::now();
        bp.build(dE);
        auto ee_broad_end = std::chrono::high_resolution_clock::now();
        auto ee_broad_time = std::chrono::duration_cast<std::chrono::milliseconds>(ee_broad_end - ee_broad_start).count();
        scalable_ccd::logger().info("[Metal CCD Pipeline] EE broad phase build took {} ms", ee_broad_time);
        
        auto ee_detect_start = std::chrono::high_resolution_clock::now();
        const auto ee_overlaps = bp.detect_overlaps();
        auto ee_detect_end = std::chrono::high_resolution_clock::now();
        auto ee_detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(ee_detect_end - ee_detect_start).count();
        scalable_ccd::logger().info("[Metal CCD Pipeline] EE overlap detection took {} ms, found {} overlaps", 
                                    ee_detect_time, ee_overlaps.size());
        
        scalable_ccd::logger().debug("CCD: EE overlaps from broad phase: {}", ee_overlaps.size());
        
        // Use the real narrow phase implementation
        const Scalar ee_toi = narrow_phase_ee(
            vertices_t0, vertices_t1, edges, ee_overlaps,
            minimum_separation, tolerance, max_iterations, allow_zero_toi
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            , collisions
#endif
        );
        
        auto ee_end = std::chrono::high_resolution_clock::now();
        auto ee_time = std::chrono::duration_cast<std::chrono::milliseconds>(ee_end - ee_start).count();
        scalable_ccd::logger().info("[Metal CCD Pipeline] EE narrow phase total took {} ms, toi = {}", ee_time, ee_toi);
        
        scalable_ccd::logger().debug("CCD: EE toi = {}", ee_toi);
        
        if (ee_toi < toi) {
            toi = ee_toi;
        }
    }
    
    auto pipeline_end = std::chrono::high_resolution_clock::now();
    auto pipeline_time = std::chrono::duration_cast<std::chrono::milliseconds>(pipeline_end - pipeline_start).count();
    scalable_ccd::logger().info("[Metal CCD Pipeline] Total pipeline time: {} ms, final toi: {}", pipeline_time, toi);
    
    return toi;
}

} // namespace scalable_ccd::metal
 #include "ccd.hpp"
