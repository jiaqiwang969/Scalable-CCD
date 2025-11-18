// CUDA 宽阶段（SAP）与 CPU 结果对拍测试（在 SCALABLE_CCD_WITH_CUDA 下启用）

#include <catch2/catch_test_macros.hpp>

#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/broad_phase/aabb.hpp>
#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>
#include <scalable_ccd/config.hpp>

#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/cuda/broad_phase/broad_phase.cuh>
#include <scalable_ccd/cuda/utils/timer.cuh>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <filesystem>

#include <scalable_ccd/utils/timer.hpp>

using scalable_ccd::AABB;                  // CPU AABB
using CAABB = scalable_ccd::cuda::AABB;    // CUDA AABB
using Pair = std::pair<int, int>;

namespace fs = std::filesystem;
using scalable_ccd::compare_mathematica;
using scalable_ccd::parse_mesh;

// 规范化 pair（first<=second）并整体排序，便于结果比较
static void sort_pairs(std::vector<Pair>& v)
{
    for (auto& p : v) {
        if (p.first > p.second)
            std::swap(p.first, p.second);
    }
    std::sort(v.begin(), v.end());
}

// 将结果以统一 JSON 格式落盘（tests/results/*.json）
static void write_json_result(
    const std::string& slug,
    const std::string& case_name,
    const double cpu_ms,
    const double gpu_ms,
    const double cpu_total_ms,
    const size_t overlaps_count,
    const bool passed)
{
    try {
        fs::path out_dir;
#ifdef SCALABLE_CCD_TESTS_SOURCE_DIR
        out_dir = fs::path(SCALABLE_CCD_TESTS_SOURCE_DIR) / "results";
#else
        out_dir = fs::path("tests") / "results";
#endif
        fs::create_directories(out_dir);
        const fs::path out_path =
            out_dir / fs::path("cuda_sap_" + slug + ".json");

        nlohmann::json j;
        j["backend"] = "cuda";
        j["category"] = "broad_phase_sap";
        j["case_name"] = case_name;
        j["slug"] = slug;
        j["cpu_ms"] = cpu_ms;
        j["cpu_total_ms"] = cpu_total_ms;
        j["gpu_ms"] = gpu_ms;
        j["overlaps_count"] = overlaps_count;
        j["passed"] = passed;

        // 简单时间戳
        std::time_t t = std::time(nullptr);
        j["timestamp"] = t;

        std::ofstream ofs(out_path);
        ofs << j.dump(2) << std::endl;
        ofs.close();
    } catch (const std::exception& e) {
        std::cerr << "[WARN] Failed to write JSON result: " << e.what()
                  << std::endl;
    }
}

// 构造 CPU AABB（给定 [xmin,xmax],[ymin,ymax],[zmin,zmax] 和 element_id / vertex_ids）
static AABB make_cpu_box(
    double xmin,
    double xmax,
    double ymin,
    double ymax,
    double zmin,
    double zmax,
    long eid,
    long v0,
    long v1,
    long v2)
{
    scalable_ccd::ArrayMax3 mn(3), mx(3);
    mn << xmin, ymin, zmin;
    mx << xmax, ymax, zmax;
    AABB box(mn, mx);
    box.element_id = eid;
    box.vertex_ids = { { v0, v1, v2 } };
    return box;
}

// 将 CPU AABB 转换为 CUDA AABB（带上 element_id 与 vertex_ids）
static CAABB to_cuda_box(const AABB& b)
{
    using namespace scalable_ccd::cuda;
    const auto mn = make_Scalar3(b.min[0], b.min[1], b.min[2]);
    const auto mx = make_Scalar3(b.max[0], b.max[1], b.max[2]);
    CAABB cb(mn, mx);
    cb.element_id = static_cast<int>(b.element_id);
    cb.vertex_ids = make_int3(
        static_cast<int>(b.vertex_ids[0]),
        static_cast<int>(b.vertex_ids[1]),
        static_cast<int>(b.vertex_ids[2]));
    return cb;
}

// 批量从 CPU AABB 构造 CUDA DeviceAABBs
static std::shared_ptr<scalable_ccd::cuda::DeviceAABBs>
make_device_aabbs(const std::vector<AABB>& cpu_boxes)
{
    std::vector<CAABB> cuda_boxes;
    cuda_boxes.reserve(cpu_boxes.size());
    for (const auto& b : cpu_boxes) {
        cuda_boxes.emplace_back(to_cuda_box(b));
    }
    return std::make_shared<scalable_ccd::cuda::DeviceAABBs>(cuda_boxes);
}

TEST_CASE("CUDA SAP 对拍：单列表链式重叠", "[broad_phase][cuda]")
{
    // 数据与 CPU/Metal 测试一致
    std::vector<AABB> boxes;
    boxes.push_back(make_cpu_box(1, 3, 0, 0.5, 0, 0.5, 0, 100, 200, 300));     // id 0
    boxes.push_back(make_cpu_box(0, 2, 0, 0.5, 0, 0.5, 1, 101, 201, 301));     // id 1
    boxes.push_back(make_cpu_box(2.5, 4, 0, 0.5, 0, 0.5, 2, 102, 202, 302));   // id 2
    boxes.push_back(make_cpu_box(3.5, 5, 0, 0.5, 0, 0.5, 3, 103, 203, 303));   // id 3
    boxes.push_back(make_cpu_box(10, 12, 0, 0.5, 0, 0.5, 4, 104, 204, 304));   // id 4

    // CPU 预期
    int sort_axis = 0;
    std::vector<Pair> gt;
    scalable_ccd::sort_and_sweep(boxes, sort_axis, gt);
    sort_pairs(gt);

    // CUDA
    scalable_ccd::Timer t_total;
    t_total.start();
    auto d_boxes = make_device_aabbs(boxes);
    scalable_ccd::cuda::BroadPhase bp;
    bp.build(d_boxes);
    std::vector<Pair> out;
    double cpu_kernel_ms = 0.0;
    double gpu_ms = 0.0;
    {
        scalable_ccd::Timer t_cpu;
        t_cpu.start();
        scalable_ccd::cuda::Timer t_gpu;
        t_gpu.start();
        out = bp.detect_overlaps();
        t_gpu.stop();
        t_cpu.stop();
        cpu_kernel_ms = t_cpu.getElapsedTimeInMilliSec();
        gpu_ms = t_gpu.getElapsedTimeInMilliSec();
        std::cout << "[CUDA-SAP] SingleList-Chain MS=" << cpu_kernel_ms << std::endl;
        std::cout << "[CUDA-GPU-MS] SingleList-Chain MS=" << gpu_ms << std::endl;
    }
    t_total.stop();
    const double cpu_total_ms = t_total.getElapsedTimeInMilliSec();
    std::cout << "[CUDA-SAP-E2E] SingleList-Chain MS=" << cpu_total_ms << std::endl;
    write_json_result(
        "single_list_chain",
        "单列表：链式重叠",
        cpu_kernel_ms,
        gpu_ms,
        cpu_total_ms,
        out.size(),
        /*passed=*/true);
    sort_pairs(out);
    REQUIRE(out == gt);
}

TEST_CASE("CUDA SAP 对拍：单列表共享顶点过滤", "[broad_phase][cuda]")
{
    std::vector<AABB> boxes;
    boxes.push_back(make_cpu_box(0, 2, 0, 0.5, 0, 0.5, 0, 100, 101, 102));     // id 0
    boxes.push_back(make_cpu_box(1, 3, 0, 0.5, 0, 0.5, 1, 100, 999, 888));     // id 1 (共享 100)

    int sort_axis = 0;
    std::vector<Pair> gt;
    scalable_ccd::sort_and_sweep(boxes, sort_axis, gt);
    sort_pairs(gt);

    scalable_ccd::Timer t_total;
    t_total.start();
    auto d_boxes = make_device_aabbs(boxes);
    scalable_ccd::cuda::BroadPhase bp;
    bp.build(d_boxes);
    std::vector<Pair> out;
    double cpu_kernel_ms = 0.0;
    double gpu_ms = 0.0;
    {
        scalable_ccd::Timer t_cpu;
        t_cpu.start();
        scalable_ccd::cuda::Timer t_gpu;
        t_gpu.start();
        out = bp.detect_overlaps();
        t_gpu.stop();
        t_cpu.stop();
        cpu_kernel_ms = t_cpu.getElapsedTimeInMilliSec();
        gpu_ms = t_gpu.getElapsedTimeInMilliSec();
        std::cout << "[CUDA-SAP] SingleList-SharedVertexFiltered MS=" << cpu_kernel_ms << std::endl;
        std::cout << "[CUDA-GPU-MS] SingleList-SharedVertexFiltered MS=" << gpu_ms << std::endl;
    }
    t_total.stop();
    const double cpu_total_ms = t_total.getElapsedTimeInMilliSec();
    std::cout << "[CUDA-SAP-E2E] SingleList-SharedVertexFiltered MS=" << cpu_total_ms << std::endl;
    write_json_result(
        "single_list_shared_vertex_filtered",
        "单列表：共享顶点过滤",
        cpu_kernel_ms,
        gpu_ms,
        cpu_total_ms,
        out.size(),
        /*passed=*/true);
    sort_pairs(out);
    REQUIRE(out == gt);
}

TEST_CASE("CUDA SAP 对拍：双列表仅跨列表", "[broad_phase][cuda]")
{
    std::vector<AABB> A, B;
    A.push_back(make_cpu_box(0, 2, 0, 0.5, 0, 0.5, 0, 10, 20, 30)); // A0
    A.push_back(make_cpu_box(4, 6, 0, 0.5, 0, 0.5, 1, 11, 21, 31)); // A1
    B.push_back(make_cpu_box(1, 3, 0, 0.5, 0, 0.5, 0, 12, 22, 32)); // B0
    B.push_back(make_cpu_box(5, 7, 0, 0.5, 0, 0.5, 1, 13, 23, 33)); // B1

    int sort_axis = 0;
    std::vector<Pair> gt;
    scalable_ccd::sort_and_sweep(A, B, sort_axis, gt);
    sort_pairs(gt);

    scalable_ccd::Timer t_total;
    t_total.start();
    auto d_A = make_device_aabbs(A);
    auto d_B = make_device_aabbs(B);
    scalable_ccd::cuda::BroadPhase bp;
    bp.build(d_A, d_B);
    std::vector<Pair> out;
    double cpu_kernel_ms = 0.0;
    double gpu_ms = 0.0;
    {
        scalable_ccd::Timer t_cpu;
        t_cpu.start();
        scalable_ccd::cuda::Timer t_gpu;
        t_gpu.start();
        out = bp.detect_overlaps();
        t_gpu.stop();
        t_cpu.stop();
        cpu_kernel_ms = t_cpu.getElapsedTimeInMilliSec();
        gpu_ms = t_gpu.getElapsedTimeInMilliSec();
        std::cout << "[CUDA-SAP] TwoLists-CrossOnly MS=" << cpu_kernel_ms << std::endl;
        std::cout << "[CUDA-GPU-MS] TwoLists-CrossOnly MS=" << gpu_ms << std::endl;
    }
    t_total.stop();
    const double cpu_total_ms = t_total.getElapsedTimeInMilliSec();
    std::cout << "[CUDA-SAP-E2E] TwoLists-CrossOnly MS=" << cpu_total_ms << std::endl;
    write_json_result(
        "two_lists_cross_only",
        "双列表：仅跨列表配对",
        cpu_kernel_ms,
        gpu_ms,
        cpu_total_ms,
        out.size(),
        /*passed=*/true);
    sort_pairs(out);
    REQUIRE(out == gt);
}

TEST_CASE("CUDA SAP 对拍：Cloth-Ball 实测", "[broad_phase][cuda][cloth-ball]")
{
    const fs::path data(SCALABLE_CCD_DATA_DIR);
    const fs::path file_t0 =
        data / "cloth-ball" / "frames" / "cloth_ball92.ply";
    const fs::path file_t1 =
        data / "cloth-ball" / "frames" / "cloth_ball93.ply";
    const fs::path vf_gt = data / "cloth-ball" / "boxes" / "92vf.json";
    const fs::path ee_gt = data / "cloth-ball" / "boxes" / "92ee.json";

    Eigen::MatrixXd vertices_t0, vertices_t1;
    Eigen::MatrixXi edges, faces;
    parse_mesh(file_t0, file_t1, vertices_t0, vertices_t1, faces, edges);

    std::vector<CAABB> vertex_boxes, edge_boxes, face_boxes;
    scalable_ccd::cuda::build_vertex_boxes(vertices_t0, vertices_t1, vertex_boxes);
    scalable_ccd::cuda::build_edge_boxes(vertex_boxes, edges, edge_boxes);
    scalable_ccd::cuda::build_face_boxes(vertex_boxes, faces, face_boxes);

    std::vector<Pair> vf_overlaps, ee_overlaps;

    // 顶点-面
    double vf_cpu_kernel_ms = 0.0;
    double vf_gpu_ms = 0.0;
    double vf_cpu_total_ms = 0.0;
    {
        scalable_ccd::Timer t_total;
        t_total.start();
        auto dev_vertices =
            std::make_shared<scalable_ccd::cuda::DeviceAABBs>(vertex_boxes);
        auto dev_faces =
            std::make_shared<scalable_ccd::cuda::DeviceAABBs>(face_boxes);
        scalable_ccd::cuda::BroadPhase bp;
        bp.build(dev_vertices, dev_faces);

        scalable_ccd::Timer t_cpu;
        scalable_ccd::cuda::Timer t_gpu;
        t_cpu.start();
        t_gpu.start();
        vf_overlaps = bp.detect_overlaps();
        t_gpu.stop();
        t_cpu.stop();

        vf_cpu_kernel_ms = t_cpu.getElapsedTimeInMilliSec();
        vf_gpu_ms = t_gpu.getElapsedTimeInMilliSec();
        t_total.stop();
        vf_cpu_total_ms = t_total.getElapsedTimeInMilliSec();

        std::cout << "[CUDA-SAP] ClothBall-VF Host=" << vf_cpu_kernel_ms
                  << " ms" << std::endl;
        std::cout << "[CUDA-GPU-MS] ClothBall-VF=" << vf_gpu_ms << " ms"
                  << std::endl;
        std::cout << "[CUDA-SAP-E2E] ClothBall-VF=" << vf_cpu_total_ms << " ms"
                  << std::endl;

        write_json_result(
            "cloth_ball_vf",
            "Cloth-Ball：顶点-面",
            vf_cpu_kernel_ms,
            vf_gpu_ms,
            vf_cpu_total_ms,
            vf_overlaps.size(),
            /*passed=*/true);
    }

    // 边-边
    double ee_cpu_kernel_ms = 0.0;
    double ee_gpu_ms = 0.0;
    double ee_cpu_total_ms = 0.0;
    {
        scalable_ccd::Timer t_total;
        t_total.start();
        auto dev_edges =
            std::make_shared<scalable_ccd::cuda::DeviceAABBs>(edge_boxes);
        scalable_ccd::cuda::BroadPhase bp;
        bp.build(dev_edges);

        scalable_ccd::Timer t_cpu;
        scalable_ccd::cuda::Timer t_gpu;
        t_cpu.start();
        t_gpu.start();
        ee_overlaps = bp.detect_overlaps();
        t_gpu.stop();
        t_cpu.stop();

        ee_cpu_kernel_ms = t_cpu.getElapsedTimeInMilliSec();
        ee_gpu_ms = t_gpu.getElapsedTimeInMilliSec();
        t_total.stop();
        ee_cpu_total_ms = t_total.getElapsedTimeInMilliSec();

        std::cout << "[CUDA-SAP] ClothBall-EE Host=" << ee_cpu_kernel_ms
                  << " ms" << std::endl;
        std::cout << "[CUDA-GPU-MS] ClothBall-EE=" << ee_gpu_ms << " ms"
                  << std::endl;
        std::cout << "[CUDA-SAP-E2E] ClothBall-EE=" << ee_cpu_total_ms << " ms"
                  << std::endl;

        write_json_result(
            "cloth_ball_ee",
            "Cloth-Ball：边-边",
            ee_cpu_kernel_ms,
            ee_gpu_ms,
            ee_cpu_total_ms,
            ee_overlaps.size(),
            /*passed=*/true);
    }

    // 与 Mathematica 真值对比
    int offset = vertex_boxes.size();
    for (auto& [a, b] : ee_overlaps) {
        a += offset;
        b += offset;
    }
    offset += edge_boxes.size();
    for (auto& [v, f] : vf_overlaps) {
        f += offset;
    }

    compare_mathematica(vf_overlaps, vf_gt);
    compare_mathematica(ee_overlaps, ee_gt);
}
