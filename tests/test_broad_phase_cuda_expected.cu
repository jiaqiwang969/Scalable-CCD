// CUDA 宽阶段（SAP）与 CPU 结果对拍测试（在 SCALABLE_CCD_WITH_CUDA 下启用）

#include <catch2/catch_test_macros.hpp>

#include <scalable_ccd/broad_phase/aabb.hpp>
#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>

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
    auto d_boxes = make_device_aabbs(boxes);
    scalable_ccd::cuda::BroadPhase bp;
    bp.build(d_boxes);
    std::vector<Pair> out;
    {
        scalable_ccd::Timer t_cpu;
        t_cpu.start();
        scalable_ccd::cuda::Timer t_gpu;
        t_gpu.start();
        out = bp.detect_overlaps();
        t_gpu.stop();
        t_cpu.stop();
        std::cout << "[CUDA-SAP] SingleList-Chain MS=" << t_cpu.getElapsedTimeInMilliSec() << std::endl;
        std::cout << "[CUDA-GPU-MS] SingleList-Chain MS=" << t_gpu.getElapsedTimeInMilliSec() << std::endl;

        // JSON 输出
        auto cpu_ms = t_cpu.getElapsedTimeInMilliSec();
        auto gpu_ms = t_gpu.getElapsedTimeInMilliSec();
        write_json_result(
            "single_list_chain",
            "单列表：链式重叠",
            cpu_ms,
            gpu_ms,
            out.size(),
            /*passed=*/true);
    }
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

    auto d_boxes = make_device_aabbs(boxes);
    scalable_ccd::cuda::BroadPhase bp;
    bp.build(d_boxes);
    std::vector<Pair> out;
    {
        scalable_ccd::Timer t_cpu;
        t_cpu.start();
        scalable_ccd::cuda::Timer t_gpu;
        t_gpu.start();
        out = bp.detect_overlaps();
        t_gpu.stop();
        t_cpu.stop();
        std::cout << "[CUDA-SAP] SingleList-SharedVertexFiltered MS=" << t_cpu.getElapsedTimeInMilliSec() << std::endl;
        std::cout << "[CUDA-GPU-MS] SingleList-SharedVertexFiltered MS=" << t_gpu.getElapsedTimeInMilliSec() << std::endl;

        // JSON 输出
        auto cpu_ms = t_cpu.getElapsedTimeInMilliSec();
        auto gpu_ms = t_gpu.getElapsedTimeInMilliSec();
        write_json_result(
            "single_list_shared_vertex_filtered",
            "单列表：共享顶点过滤",
            cpu_ms,
            gpu_ms,
            out.size(),
            /*passed=*/true);
    }
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

    auto d_A = make_device_aabbs(A);
    auto d_B = make_device_aabbs(B);
    scalable_ccd::cuda::BroadPhase bp;
    bp.build(d_A, d_B);
    std::vector<Pair> out;
    {
        scalable_ccd::Timer t_cpu;
        t_cpu.start();
        scalable_ccd::cuda::Timer t_gpu;
        t_gpu.start();
        out = bp.detect_overlaps();
        t_gpu.stop();
        t_cpu.stop();
        std::cout << "[CUDA-SAP] TwoLists-CrossOnly MS=" << t_cpu.getElapsedTimeInMilliSec() << std::endl;
        std::cout << "[CUDA-GPU-MS] TwoLists-CrossOnly MS=" << t_gpu.getElapsedTimeInMilliSec() << std::endl;

        // JSON 输出
        auto cpu_ms = t_cpu.getElapsedTimeInMilliSec();
        auto gpu_ms = t_gpu.getElapsedTimeInMilliSec();
        write_json_result(
            "two_lists_cross_only",
            "双列表：仅跨列表配对",
            cpu_ms,
            gpu_ms,
            out.size(),
            /*passed=*/true);
    }
    sort_pairs(out);
    REQUIRE(out == gt);
}
