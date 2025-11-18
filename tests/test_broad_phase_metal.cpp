// Metal 宽阶段（SAP）与 CPU 结果对拍测试（可在 macOS + SCALABLE_CCD_WITH_METAL 下启用）

#include <catch2/catch_test_macros.hpp>

#include "ground_truth.hpp"
#include "io.hpp"

#include <scalable_ccd/broad_phase/aabb.hpp>
#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>
#include <scalable_ccd/metal/broad_phase_metal.hpp>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>

#include <scalable_ccd/utils/timer.hpp>

using scalable_ccd::AABB;
using Pair = std::pair<int, int>;
namespace fs = std::filesystem;

// 写出 Metal SAP 结果 JSON（与 CUDA 侧保持一致的结构）
static void write_json_result(
    const std::string& slug,
    const std::string& case_name,
    const double cpu_kernel_ms,
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
            out_dir / fs::path("metal_sap_" + slug + ".json");

        nlohmann::json j;
        j["backend"] = "metal";
        j["category"] = "broad_phase_sap";
        j["case_name"] = case_name;
        j["slug"] = slug;
        j["cpu_ms"] = cpu_kernel_ms;
        j["cpu_total_ms"] = cpu_total_ms;
        j["gpu_ms"] = gpu_ms;
        j["overlaps_count"] = overlaps_count;
        j["passed"] = passed;
        std::time_t t = std::time(nullptr);
        j["timestamp"] = t;

        std::ofstream ofs(out_path);
        ofs << j.dump(2) << std::endl;
        ofs.close();
    } catch (const std::exception& e) {
        std::cerr << "[WARN] Failed to write Metal JSON result: " << e.what()
                  << std::endl;
    }
}

static size_t read_ground_truth_size(const fs::path& path)
{
    std::ifstream in(path);
    REQUIRE(in.good());
    const nlohmann::json j = nlohmann::json::parse(in);
    return j.size();
}

static void sort_pairs(std::vector<Pair>& v)
{
    for (auto& p : v) {
        if (p.first > p.second)
            std::swap(p.first, p.second);
    }
    std::sort(v.begin(), v.end());
}

static AABB make_box(
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

static int auto_select_axis(
    const std::vector<AABB>& boxesA,
    const std::vector<AABB>* boxesB = nullptr)
{
    double sum[3] = { 0.0, 0.0, 0.0 };
    double sumsq[3] = { 0.0, 0.0, 0.0 };
    size_t total = 0;

    auto accumulate = [&](const std::vector<AABB>& boxes) {
        for (const auto& box : boxes) {
            const auto center = (box.min + box.max) / 2.0;
            for (int i = 0; i < 3; ++i) {
                const double c = center[i];
                sum[i] += c;
                sumsq[i] += c * c;
            }
            ++total;
        }
    };

    accumulate(boxesA);
    if (boxesB) {
        accumulate(*boxesB);
    }

    int axis = 0;
    double best_var = -1.0;
    if (total == 0) {
        return 0;
    }
    for (int i = 0; i < 3; ++i) {
        const double mean = sum[i] / total;
        const double var = sumsq[i] / total - mean * mean;
        if (var > best_var) {
            best_var = var;
            axis = i;
        }
    }
    return axis;
}

#if defined(SCALABLE_CCD_WITH_METAL) && defined(__APPLE__)
TEST_CASE("Metal SAP 对拍：单列表链式重叠", "[broad_phase][metal]")
{
    std::vector<AABB> boxes;
    boxes.push_back(make_box(1, 3, 0, 0.5, 0, 0.5, 0, 100, 200, 300));     // id 0
    boxes.push_back(make_box(0, 2, 0, 0.5, 0, 0.5, 1, 101, 201, 301));     // id 1
    boxes.push_back(make_box(2.5, 4, 0, 0.5, 0, 0.5, 2, 102, 202, 302));   // id 2
    boxes.push_back(make_box(3.5, 5, 0, 0.5, 0, 0.5, 3, 103, 203, 303));   // id 3
    boxes.push_back(make_box(10, 12, 0, 0.5, 0, 0.5, 4, 104, 204, 304));   // id 4

    // CPU 预期
    int sort_axis = 0;
    std::vector<Pair> gt;
    scalable_ccd::sort_and_sweep(boxes, sort_axis, gt);
    sort_pairs(gt);

    // Metal
    scalable_ccd::Timer t_total;
    t_total.start();
    scalable_ccd::metal::MetalAABBsSoA soa;
    scalable_ccd::metal::make_soa_one_list(boxes, /*axis=*/0, soa);
    scalable_ccd::metal::BroadPhase bp;
    bp.upload(soa);
    std::vector<Pair> out;
    double cpu_ms = 0.0;
    double gpu_ms = -1.0;
    {
        scalable_ccd::Timer t;
        t.start();
        out = bp.detect_overlaps(
            /*two_lists=*/false,
            /*max_overlap_cutoff=*/static_cast<uint32_t>(soa.size()),
            /*overlaps_capacity_hint=*/1024);
        t.stop();
        cpu_ms = t.getElapsedTimeInMilliSec();
        gpu_ms = bp.last_gpu_ms();
        std::cout << "[Metal-SAP] SingleList-Chain MS=" << cpu_ms << std::endl;
        std::cout << "[Metal-GPU-MS] SingleList-Chain MS=" << gpu_ms << std::endl;
    }
    t_total.stop();
    const double total_ms = t_total.getElapsedTimeInMilliSec();
    std::cout << "[Metal-SAP-E2E] SingleList-Chain MS=" << total_ms << std::endl;
    write_json_result(
        "single_list_chain",
        "单列表：链式重叠",
        cpu_ms,
        gpu_ms,
        total_ms,
        out.size(),
        /*passed=*/true);
    sort_pairs(out);
    REQUIRE(out == gt);
}

TEST_CASE("Metal SAP 对拍：单列表共享顶点过滤", "[broad_phase][metal]")
{
    std::vector<AABB> boxes;
    boxes.push_back(make_box(0, 2, 0, 0.5, 0, 0.5, 0, 100, 101, 102));     // id 0
    boxes.push_back(make_box(1, 3, 0, 0.5, 0, 0.5, 1, 100, 999, 888));     // id 1 (共享 100)

    int sort_axis = 0;
    std::vector<Pair> gt;
    scalable_ccd::sort_and_sweep(boxes, sort_axis, gt);
    sort_pairs(gt);

    scalable_ccd::Timer t_total;
    t_total.start();
    scalable_ccd::metal::MetalAABBsSoA soa;
    scalable_ccd::metal::make_soa_one_list(boxes, /*axis=*/0, soa);
    scalable_ccd::metal::BroadPhase bp;
    bp.upload(soa);
    std::vector<Pair> out;
    double cpu_ms = 0.0;
    double gpu_ms = -1.0;
    {
        scalable_ccd::Timer t;
        t.start();
        out = bp.detect_overlaps(
            /*two_lists=*/false,
            /*max_overlap_cutoff=*/static_cast<uint32_t>(soa.size()),
            /*overlaps_capacity_hint=*/16);
        t.stop();
        cpu_ms = t.getElapsedTimeInMilliSec();
        gpu_ms = bp.last_gpu_ms();
        std::cout << "[Metal-SAP] SingleList-SharedVertexFiltered MS=" << cpu_ms << std::endl;
        std::cout << "[Metal-GPU-MS] SingleList-SharedVertexFiltered MS=" << gpu_ms << std::endl;
    }
    t_total.stop();
    const double total_ms = t_total.getElapsedTimeInMilliSec();
    std::cout << "[Metal-SAP-E2E] SingleList-SharedVertexFiltered MS=" << total_ms << std::endl;
    write_json_result(
        "single_list_shared_vertex_filtered",
        "单列表：共享顶点过滤",
        cpu_ms,
        gpu_ms,
        total_ms,
        out.size(),
        /*passed=*/true);
    sort_pairs(out);
    REQUIRE(out == gt);
}

TEST_CASE("Metal SAP 对拍：双列表仅跨列表", "[broad_phase][metal]")
{
    std::vector<AABB> A, B;
    A.push_back(make_box(0, 2, 0, 0.5, 0, 0.5, 0, 10, 20, 30)); // A0
    A.push_back(make_box(4, 6, 0, 0.5, 0, 0.5, 1, 11, 21, 31)); // A1
    B.push_back(make_box(1, 3, 0, 0.5, 0, 0.5, 0, 12, 22, 32)); // B0
    B.push_back(make_box(5, 7, 0, 0.5, 0, 0.5, 1, 13, 23, 33)); // B1

    int sort_axis = 0;
    std::vector<Pair> gt;
    scalable_ccd::sort_and_sweep(A, B, sort_axis, gt);
    sort_pairs(gt);

    scalable_ccd::Timer t_total;
    t_total.start();
    scalable_ccd::metal::MetalAABBsSoA soa;
    scalable_ccd::metal::make_soa_two_lists(A, B, /*axis=*/0, soa);
    scalable_ccd::metal::BroadPhase bp;
    bp.upload(soa);
    std::vector<Pair> out;
    double cpu_ms = 0.0;
    double gpu_ms = -1.0;
    {
        scalable_ccd::Timer t;
        t.start();
        out = bp.detect_overlaps(
            /*two_lists=*/true,
            /*max_overlap_cutoff=*/static_cast<uint32_t>(soa.size()),
            /*overlaps_capacity_hint=*/16);
        t.stop();
        cpu_ms = t.getElapsedTimeInMilliSec();
        gpu_ms = bp.last_gpu_ms();
        std::cout << "[Metal-SAP] TwoLists-CrossOnly MS=" << cpu_ms << std::endl;
        std::cout << "[Metal-GPU-MS] TwoLists-CrossOnly MS=" << gpu_ms << std::endl;
    }
    t_total.stop();
    const double total_ms = t_total.getElapsedTimeInMilliSec();
    std::cout << "[Metal-SAP-E2E] TwoLists-CrossOnly MS=" << total_ms << std::endl;
    write_json_result(
        "two_lists_cross_only",
        "双列表：仅跨列表配对",
        cpu_ms,
        gpu_ms,
        total_ms,
        out.size(),
        /*passed=*/true);
    sort_pairs(out);
    REQUIRE(out == gt);
}

TEST_CASE("Metal SAP 对拍：Cloth-Ball 实测", "[broad_phase][metal][cloth-ball]")
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
    scalable_ccd::parse_mesh(
        file_t0, file_t1, vertices_t0, vertices_t1, faces, edges);

    std::vector<AABB> vertex_boxes, edge_boxes, face_boxes;
    scalable_ccd::build_vertex_boxes(vertices_t0, vertices_t1, vertex_boxes);
    scalable_ccd::build_edge_boxes(vertex_boxes, edges, edge_boxes);
    scalable_ccd::build_face_boxes(vertex_boxes, faces, face_boxes);

    const size_t vf_expected = read_ground_truth_size(vf_gt);
    const size_t ee_expected = read_ground_truth_size(ee_gt);

    std::vector<Pair> vf_overlaps, ee_overlaps;

    // 顶点-面
    double vf_cpu_kernel_ms = 0.0;
    double vf_gpu_ms = 0.0;
    double vf_cpu_total_ms = 0.0;
    {
        scalable_ccd::Timer t_total;
        t_total.start();
        scalable_ccd::metal::MetalAABBsSoA soa;
        const int vf_axis = auto_select_axis(vertex_boxes, &face_boxes);
        scalable_ccd::metal::make_soa_two_lists(
            vertex_boxes, face_boxes, vf_axis, soa);
        scalable_ccd::metal::BroadPhase bp;
        bp.upload(soa);

        scalable_ccd::Timer t_kernel;
        uint32_t capacity = static_cast<uint32_t>(
            std::max<size_t>(vf_expected + vf_expected / 10 + 4096, 4096));
        while (true) {
            t_kernel.start();
            vf_overlaps = bp.detect_overlaps_partial(
                /*two_lists=*/true,
                /*start=*/0,
                static_cast<uint32_t>(soa.size()),
                capacity);
            t_kernel.stop();
            vf_cpu_kernel_ms = t_kernel.getElapsedTimeInMilliSec();
            vf_gpu_ms = bp.last_gpu_ms();
            if (bp.last_real_count() > capacity) {
                capacity = static_cast<uint32_t>(bp.last_real_count() * 1.2 + 4096);
                vf_overlaps.clear();
                t_kernel = scalable_ccd::Timer();
                continue;
            }
            break;
        }
        t_total.stop();
        vf_cpu_total_ms = t_total.getElapsedTimeInMilliSec();

        std::cout << "[Metal-SAP] ClothBall-VF Host=" << vf_cpu_kernel_ms
                  << " ms" << std::endl;
        std::cout << "[Metal-GPU-MS] ClothBall-VF=" << vf_gpu_ms << " ms"
                  << std::endl;
        std::cout << "[Metal-SAP-E2E] ClothBall-VF=" << vf_cpu_total_ms
                  << " ms" << std::endl;

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
        scalable_ccd::metal::MetalAABBsSoA soa;
        const int ee_axis = auto_select_axis(edge_boxes, nullptr);
        scalable_ccd::metal::make_soa_one_list(edge_boxes, ee_axis, soa);
        scalable_ccd::metal::BroadPhase bp;
        bp.upload(soa);

        scalable_ccd::Timer t_kernel;
        uint32_t capacity = static_cast<uint32_t>(
            std::max<size_t>(ee_expected + ee_expected / 10 + 4096, 4096));
        while (true) {
            t_kernel.start();
            ee_overlaps = bp.detect_overlaps_partial(
                /*two_lists=*/false,
                /*start=*/0,
                static_cast<uint32_t>(soa.size()),
                capacity);
            t_kernel.stop();
            ee_cpu_kernel_ms = t_kernel.getElapsedTimeInMilliSec();
            ee_gpu_ms = bp.last_gpu_ms();
            if (bp.last_real_count() > capacity) {
                capacity = static_cast<uint32_t>(bp.last_real_count() * 1.2 + 4096);
                ee_overlaps.clear();
                t_kernel = scalable_ccd::Timer();
                continue;
            }
            break;
        }
        t_total.stop();
        ee_cpu_total_ms = t_total.getElapsedTimeInMilliSec();

        std::cout << "[Metal-SAP] ClothBall-EE Host=" << ee_cpu_kernel_ms
                  << " ms" << std::endl;
        std::cout << "[Metal-GPU-MS] ClothBall-EE=" << ee_gpu_ms << " ms"
                  << std::endl;
        std::cout << "[Metal-SAP-E2E] ClothBall-EE=" << ee_cpu_total_ms
                  << " ms" << std::endl;

        write_json_result(
            "cloth_ball_ee",
            "Cloth-Ball：边-边",
            ee_cpu_kernel_ms,
            ee_gpu_ms,
            ee_cpu_total_ms,
            ee_overlaps.size(),
            /*passed=*/true);
    }

    // 与 Mathematica 真值对比（对齐 CUDA 偏移逻辑）
    int offset = static_cast<int>(vertex_boxes.size());
    for (auto& [a, b] : ee_overlaps) {
        a += offset;
        b += offset;
    }
    offset += static_cast<int>(edge_boxes.size());
    for (auto& [v, f] : vf_overlaps) {
        f += offset; // faces 偏移
    }

    scalable_ccd::compare_mathematica(vf_overlaps, vf_gt);
    scalable_ccd::compare_mathematica(ee_overlaps, ee_gt);
}
#else
TEST_CASE("Metal backend unavailable - skipped", "[broad_phase][metal][skip]")
{
    SUCCEED("Metal not enabled on this platform/build.");
}
#endif
