// Metal 宽阶段（SAP）与 CPU 结果对拍测试（可在 macOS + SCALABLE_CCD_WITH_METAL 下启用）

#include <catch2/catch_test_macros.hpp>

#include <scalable_ccd/broad_phase/aabb.hpp>
#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>
#include <scalable_ccd/metal/broad_phase_metal.hpp>

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include <scalable_ccd/utils/timer.hpp>

using scalable_ccd::AABB;
using Pair = std::pair<int, int>;

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
    scalable_ccd::metal::MetalAABBsSoA soa;
    scalable_ccd::metal::make_soa_one_list(boxes, /*axis=*/0, soa);
    scalable_ccd::metal::BroadPhase bp;
    bp.upload(soa);
    std::vector<Pair> out;
    {
        scalable_ccd::Timer t;
        t.start();
        out = bp.detect_overlaps_partial(
            /*two_lists=*/false, /*start=*/0,
            /*max_overlap_cutoff=*/static_cast<uint32_t>(soa.size()),
            /*overlaps_capacity=*/1024);
        t.stop();
        std::cout << "[Metal-SAP] SingleList-Chain MS=" << t.getElapsedTimeInMilliSec() << std::endl;
        std::cout << "[Metal-GPU-MS] SingleList-Chain MS=" << bp.last_gpu_ms() << std::endl;
    }
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

    scalable_ccd::metal::MetalAABBsSoA soa;
    scalable_ccd::metal::make_soa_one_list(boxes, /*axis=*/0, soa);
    scalable_ccd::metal::BroadPhase bp;
    bp.upload(soa);
    std::vector<Pair> out;
    {
        scalable_ccd::Timer t;
        t.start();
        out = bp.detect_overlaps_partial(
            /*two_lists=*/false, /*start=*/0,
            /*max_overlap_cutoff=*/static_cast<uint32_t>(soa.size()),
            /*overlaps_capacity=*/16);
        t.stop();
        std::cout << "[Metal-SAP] SingleList-SharedVertexFiltered MS=" << t.getElapsedTimeInMilliSec() << std::endl;
        std::cout << "[Metal-GPU-MS] SingleList-SharedVertexFiltered MS=" << bp.last_gpu_ms() << std::endl;
    }
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

    scalable_ccd::metal::MetalAABBsSoA soa;
    scalable_ccd::metal::make_soa_two_lists(A, B, /*axis=*/0, soa);
    scalable_ccd::metal::BroadPhase bp;
    bp.upload(soa);
    std::vector<Pair> out;
    {
        scalable_ccd::Timer t;
        t.start();
        out = bp.detect_overlaps_partial(
            /*two_lists=*/true, /*start=*/0,
            /*max_overlap_cutoff=*/static_cast<uint32_t>(soa.size()),
            /*overlaps_capacity=*/16);
        t.stop();
        std::cout << "[Metal-SAP] TwoLists-CrossOnly MS=" << t.getElapsedTimeInMilliSec() << std::endl;
        std::cout << "[Metal-GPU-MS] TwoLists-CrossOnly MS=" << bp.last_gpu_ms() << std::endl;
    }
    sort_pairs(out);
    REQUIRE(out == gt);
}
#else
TEST_CASE("Metal backend unavailable - skipped", "[broad_phase][metal][skip]")
{
    SUCCEED("Metal not enabled on this platform/build.");
}
#endif
