#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/config.hpp>

#include <scalable_ccd/cuda/memory_handler.hpp>
#include <scalable_ccd/cuda/broad_phase/broad_phase.cuh>
#include <scalable_ccd/cuda/broad_phase/utils.cuh>
#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/utils/pca.hpp>
#include <scalable_ccd/utils/logger.hpp>
#include <scalable_ccd/utils/profiler.hpp>
#include <scalable_ccd/utils/timer.hpp>
#include <scalable_ccd/cuda/utils/timer.cuh>

#include <igl/write_triangle_mesh.h>

#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;

TEST_CASE("Test CUDA broad phase", "[gpu][cuda][broad_phase]")
{
    using namespace scalable_ccd;
    using namespace scalable_ccd::cuda;

    const fs::path data(SCALABLE_CCD_DATA_DIR);

    fs::path file_t0, file_t1, vf_ground_truth, ee_ground_truth;

    SECTION("Armadillo-Rollers")
    {
        file_t0 = data / "armadillo-rollers" / "frames" / "326.ply";
        file_t1 = data / "armadillo-rollers" / "frames" / "327.ply";
        vf_ground_truth = data / "armadillo-rollers" / "boxes" / "326vf.json";
        ee_ground_truth = data / "armadillo-rollers" / "boxes" / "326ee.json";
    }
    SECTION("Cloth-Ball")
    {
        file_t0 = data / "cloth-ball" / "frames" / "cloth_ball92.ply";
        file_t1 = data / "cloth-ball" / "frames" / "cloth_ball93.ply";
        vf_ground_truth = data / "cloth-ball" / "boxes" / "92vf.json";
        ee_ground_truth = data / "cloth-ball" / "boxes" / "92ee.json";
    }
    SECTION("Cloth-Funnel")
    {
        file_t0 = data / "cloth-funnel" / "frames" / "227.ply";
        file_t1 = data / "cloth-funnel" / "frames" / "228.ply";
        vf_ground_truth = data / "cloth-funnel" / "boxes" / "227vf.json";
        ee_ground_truth = data / "cloth-funnel" / "boxes" / "227ee.json";
    }
    SECTION("N-Body")
    {
        file_t0 = data / "n-body-simulation" / "frames" / "balls16_18.ply";
        file_t1 = data / "n-body-simulation" / "frames" / "balls16_19.ply";
        vf_ground_truth = data / "n-body-simulation" / "boxes" / "18vf.json";
        ee_ground_truth = data / "n-body-simulation" / "boxes" / "18ee.json";
    }
    SECTION("Rod-Twist")
    {
        file_t0 = data / "rod-twist" / "frames" / "3036.ply";
        file_t1 = data / "rod-twist" / "frames" / "3037.ply";
        vf_ground_truth = data / "rod-twist" / "boxes" / "3036vf.json";
        ee_ground_truth = data / "rod-twist" / "boxes" / "3036ee.json";
    }

    CAPTURE(file_t0, file_t1, vf_ground_truth, ee_ground_truth);

#ifdef SCALABLE_CCD_WITH_PROFILER
    profiler().clear();
#endif

    // ------------------------------------------------------------------------
    // Load meshes

    Eigen::MatrixXd vertices_t0, vertices_t1;
    Eigen::MatrixXi edges, faces;
    parse_mesh(file_t0, file_t1, vertices_t0, vertices_t1, faces, edges);

    // const bool pca = GENERATE(false, true);
    const bool pca = false;
    if (pca) {
        nipals_pca(vertices_t0, vertices_t1);
    }

    // ------------------------------------------------------------------------
    // Run

    std::vector<scalable_ccd::cuda::AABB> vertex_boxes, edge_boxes, face_boxes;
    build_vertex_boxes(vertices_t0, vertices_t1, vertex_boxes);
    build_edge_boxes(vertex_boxes, edges, edge_boxes);
    build_face_boxes(vertex_boxes, faces, face_boxes);

    BroadPhase broad_phase;

    // VF：Host 与 GPU 计时
    double vf_host_ms = 0.0, vf_gpu_ms = 0.0;
    broad_phase.build(
        std::make_shared<DeviceAABBs>(vertex_boxes),
        std::make_shared<DeviceAABBs>(face_boxes));
    std::vector<std::pair<int, int>> vf_overlaps;
    {
        scalable_ccd::Timer host_t;
        scalable_ccd::cuda::Timer gpu_t;
        host_t.start();
        gpu_t.start();
        vf_overlaps = broad_phase.detect_overlaps();
        gpu_t.stop();
        host_t.stop();
        vf_host_ms = host_t.getElapsedTimeInMilliSec();
        vf_gpu_ms = gpu_t.getElapsedTimeInMilliSec();
    }

    // EE：Host 与 GPU 计时
    double ee_host_ms = 0.0, ee_gpu_ms = 0.0;
    broad_phase.build(std::make_shared<DeviceAABBs>(edge_boxes));
    std::vector<std::pair<int, int>> ee_overlaps;
    {
        scalable_ccd::Timer host_t;
        scalable_ccd::cuda::Timer gpu_t;
        host_t.start();
        gpu_t.start();
        ee_overlaps = broad_phase.detect_overlaps();
        gpu_t.stop();
        host_t.stop();
        ee_host_ms = host_t.getElapsedTimeInMilliSec();
        ee_gpu_ms = gpu_t.getElapsedTimeInMilliSec();
    }

    // 输出一致性格式，便于与 CPU/Metal 对比
    std::cout << "[CUDA-SAP] VF_MS=" << vf_host_ms << " EE_MS=" << ee_host_ms
              << std::endl;
    std::cout << "[CUDA-GPU-MS] VF_MS=" << vf_gpu_ms << " EE_MS=" << ee_gpu_ms
              << std::endl;

    // const size_t expected_overlap_size = pca ? 6'954'911 : 6'852'873;
    // CHECK(vf_overlaps.size() + ee_overlaps.size() == expected_overlap_size);

    // Offset the boxes to match the way ground truth was originally generated.
    int offset = vertex_boxes.size();
    for (auto& [a, b] : ee_overlaps) {
        a += offset;
        b += offset;
    }
    offset += edge_boxes.size();
    for (auto& [v, f] : vf_overlaps) {
        f += offset;
    }

    compare_mathematica(vf_overlaps, vf_ground_truth);
    compare_mathematica(ee_overlaps, ee_ground_truth);

#ifdef SCALABLE_CCD_WITH_PROFILER
    profiler().print();
#endif
}
