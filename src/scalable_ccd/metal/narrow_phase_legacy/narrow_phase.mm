#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "narrow_phase.hpp"
#include "root_finder.hpp"
#include <scalable_ccd/utils/logger.hpp>
#include <chrono>

namespace scalable_ccd::metal {

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
    )
{
    auto total_start = std::chrono::high_resolution_clock::now();
    scalable_ccd::logger().info("[Metal Narrow Phase VF] Processing {} overlaps", overlaps.size());
    Scalar min_toi = 1.0;
    
    // Adaptive batch processing to avoid GPU timeout/memory issues
    auto batch_calc_start = std::chrono::high_resolution_clock::now();
    size_t batch_size = 5000;  // Start with smaller batch size
    if (overlaps.size() > 1000000) {
        batch_size = 2000;  // Very large overlap count, use even smaller batches
    } else if (overlaps.size() > 500000) {
        batch_size = 3000;  // Large overlap count
    } else if (overlaps.size() > 100000) {
        batch_size = 5000;  // Medium overlap count
    } else if (overlaps.size() > 50000) {
        batch_size = 10000;  // Small to medium overlap count
    } else {
        batch_size = overlaps.size();  // Process all at once for small counts
    }
    auto batch_calc_end = std::chrono::high_resolution_clock::now();
    auto batch_calc_time = std::chrono::duration_cast<std::chrono::microseconds>(batch_calc_end - batch_calc_start).count();
    scalable_ccd::logger().info("[Metal Narrow Phase VF] Batch size calculation took {} us", batch_calc_time);
    
    if (overlaps.size() > batch_size) {
        scalable_ccd::logger().info("[Metal Narrow Phase VF] Processing in {} batches of size {}", 
                                   (overlaps.size() + batch_size - 1) / batch_size, batch_size);
        
        size_t batch_num = 0;
        for (size_t i = 0; i < overlaps.size(); i += batch_size) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            size_t end = std::min(i + batch_size, overlaps.size());
            
            auto batch_prep_start = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<int,int>> batch(overlaps.begin() + i, overlaps.begin() + end);
            auto batch_prep_end = std::chrono::high_resolution_clock::now();
            auto batch_prep_time = std::chrono::duration_cast<std::chrono::microseconds>(batch_prep_end - batch_prep_start).count();
            
            scalable_ccd::logger().info("[Metal Narrow Phase VF] Batch {}: {}-{} overlaps, prep took {} us", 
                                       batch_num, i, end, batch_prep_time);
            
            Scalar batch_toi = 1.0;
            auto root_finder_start = std::chrono::high_resolution_clock::now();
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            (void) run_narrow_phase_root_finder<true>(
                vertices_t0, vertices_t1, faces, batch, max_iterations, tolerance,
                minimum_separation, allow_zero_toi, collisions, batch_toi);
#else
            (void) run_narrow_phase_root_finder<true>(
                vertices_t0, vertices_t1, faces, batch, max_iterations, tolerance,
                minimum_separation, allow_zero_toi, batch_toi);
#endif
            auto root_finder_end = std::chrono::high_resolution_clock::now();
            auto root_finder_time = std::chrono::duration_cast<std::chrono::milliseconds>(root_finder_end - root_finder_start).count();
            
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();
            
            scalable_ccd::logger().info("[Metal Narrow Phase VF] Batch {}: root_finder took {} ms, total batch time {} ms, toi={}", 
                                       batch_num, root_finder_time, batch_time, batch_toi);
            
            if (batch_toi < min_toi) {
                min_toi = batch_toi;
                // Early exit if we found a collision at t=0
                if (min_toi == 0.0 && !allow_zero_toi) {
                    break;
                }
            }
            batch_num++;
        }
    } else {
        // Small enough to process all at once
        auto root_finder_start = std::chrono::high_resolution_clock::now();
#ifdef SCALABLE_CCD_TOI_PER_QUERY
        (void) run_narrow_phase_root_finder<true>(
            vertices_t0, vertices_t1, faces, overlaps, max_iterations, tolerance,
            minimum_separation, allow_zero_toi, collisions, min_toi);
#else
        (void) run_narrow_phase_root_finder<true>(
            vertices_t0, vertices_t1, faces, overlaps, max_iterations, tolerance,
            minimum_separation, allow_zero_toi, min_toi);
#endif
        auto root_finder_end = std::chrono::high_resolution_clock::now();
        auto root_finder_time = std::chrono::duration_cast<std::chrono::milliseconds>(root_finder_end - root_finder_start).count();
        scalable_ccd::logger().info("[Metal Narrow Phase VF] Single batch root_finder took {} ms", root_finder_time);
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    scalable_ccd::logger().info("[Metal Narrow Phase VF] Total time: {} ms, final toi: {}", total_time, min_toi);
    
    return min_toi;
}

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
    )
{
    auto total_start = std::chrono::high_resolution_clock::now();
    scalable_ccd::logger().info("[Metal Narrow Phase EE] Processing {} overlaps", overlaps.size());
    Scalar min_toi = 1.0;
    
    // Adaptive batch processing to avoid GPU timeout/memory issues
    auto batch_calc_start = std::chrono::high_resolution_clock::now();
    size_t batch_size = 5000;  // Start with smaller batch size
    if (overlaps.size() > 1000000) {
        batch_size = 2000;  // Very large overlap count, use even smaller batches
    } else if (overlaps.size() > 500000) {
        batch_size = 3000;  // Large overlap count
    } else if (overlaps.size() > 100000) {
        batch_size = 5000;  // Medium overlap count
    } else if (overlaps.size() > 50000) {
        batch_size = 10000;  // Small to medium overlap count
    } else {
        batch_size = overlaps.size();  // Process all at once for small counts
    }
    auto batch_calc_end = std::chrono::high_resolution_clock::now();
    auto batch_calc_time = std::chrono::duration_cast<std::chrono::microseconds>(batch_calc_end - batch_calc_start).count();
    scalable_ccd::logger().info("[Metal Narrow Phase EE] Batch size calculation took {} us", batch_calc_time);
    
    if (overlaps.size() > batch_size) {
        scalable_ccd::logger().info("[Metal Narrow Phase EE] Processing in {} batches of size {}", 
                                   (overlaps.size() + batch_size - 1) / batch_size, batch_size);
        
        size_t batch_num = 0;
        for (size_t i = 0; i < overlaps.size(); i += batch_size) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            size_t end = std::min(i + batch_size, overlaps.size());
            
            auto batch_prep_start = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<int,int>> batch(overlaps.begin() + i, overlaps.begin() + end);
            auto batch_prep_end = std::chrono::high_resolution_clock::now();
            auto batch_prep_time = std::chrono::duration_cast<std::chrono::microseconds>(batch_prep_end - batch_prep_start).count();
            
            scalable_ccd::logger().info("[Metal Narrow Phase EE] Batch {}: {}-{} overlaps, prep took {} us", 
                                       batch_num, i, end, batch_prep_time);
            
            Scalar batch_toi = 1.0;
            auto root_finder_start = std::chrono::high_resolution_clock::now();
#ifdef SCALABLE_CCD_TOI_PER_QUERY
            (void) run_narrow_phase_root_finder<false>(
                vertices_t0, vertices_t1, edges, batch, max_iterations, tolerance,
                minimum_separation, allow_zero_toi, collisions, batch_toi);
#else
            (void) run_narrow_phase_root_finder<false>(
                vertices_t0, vertices_t1, edges, batch, max_iterations, tolerance,
                minimum_separation, allow_zero_toi, batch_toi);
#endif
            auto root_finder_end = std::chrono::high_resolution_clock::now();
            auto root_finder_time = std::chrono::duration_cast<std::chrono::milliseconds>(root_finder_end - root_finder_start).count();
            
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();
            
            scalable_ccd::logger().info("[Metal Narrow Phase EE] Batch {}: root_finder took {} ms, total batch time {} ms, toi={}", 
                                       batch_num, root_finder_time, batch_time, batch_toi);
            
            if (batch_toi < min_toi) {
                min_toi = batch_toi;
                // Early exit if we found a collision at t=0
                if (min_toi == 0.0 && !allow_zero_toi) {
                    break;
                }
            }
            batch_num++;
        }
    } else {
        // Small enough to process all at once
        auto root_finder_start = std::chrono::high_resolution_clock::now();
#ifdef SCALABLE_CCD_TOI_PER_QUERY
        (void) run_narrow_phase_root_finder<false>(
            vertices_t0, vertices_t1, edges, overlaps, max_iterations, tolerance,
            minimum_separation, allow_zero_toi, collisions, min_toi);
#else
        (void) run_narrow_phase_root_finder<false>(
            vertices_t0, vertices_t1, edges, overlaps, max_iterations, tolerance,
            minimum_separation, allow_zero_toi, min_toi);
#endif
        auto root_finder_end = std::chrono::high_resolution_clock::now();
        auto root_finder_time = std::chrono::duration_cast<std::chrono::milliseconds>(root_finder_end - root_finder_start).count();
        scalable_ccd::logger().info("[Metal Narrow Phase EE] Single batch root_finder took {} ms", root_finder_time);
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    scalable_ccd::logger().info("[Metal Narrow Phase EE] Total time: {} ms, final toi: {}", total_time, min_toi);
    
    return min_toi;
}

} // namespace scalable_ccd::metal
 #import <Metal/Metal.h>
