// 可预期结果的宽阶段（CPU实现）测试
// 使用 Catch2 测试框架，生成几何上易推断的重叠对，
// 后续 Metal 宽阶段实现可用相同数据进行对拍。

#include <catch2/catch_test_macros.hpp>

#include <scalable_ccd/broad_phase/aabb.hpp>
#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>

#include <algorithm>
#include <utility>
#include <vector>

using scalable_ccd::AABB;
using Pair = std::pair<int, int>;

// 将 pair 规范化为 (min,max) 并排序，便于比较
static void sort_pairs(std::vector<Pair>& v)
{
    for (auto& p : v) {
        if (p.first > p.second)
            std::swap(p.first, p.second);
    }
    std::sort(v.begin(), v.end());
}

// 构造 AABB（给定 [xmin,xmax],[ymin,ymax],[zmin,zmax] 和 element_id / vertex_ids）
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

TEST_CASE("单列表：链式重叠（不共享顶点）", "[broad_phase][cpu][expected]")
{
    // 盒顺序打乱，沿 x 轴排序后为:
    // 1:[0,2], 0:[1,3], 2:[2.5,4], 3:[3.5,5], 4:[10,12]
    // 期望重叠：(0,1), (0,2), (2,3)
    std::vector<AABB> boxes;
    boxes.push_back(make_box(1, 3, 0, 0.5, 0, 0.5, 0, 100, 200, 300));     // id 0
    boxes.push_back(make_box(0, 2, 0, 0.5, 0, 0.5, 1, 101, 201, 301));     // id 1
    boxes.push_back(make_box(2.5, 4, 0, 0.5, 0, 0.5, 2, 102, 202, 302));   // id 2
    boxes.push_back(make_box(3.5, 5, 0, 0.5, 0, 0.5, 3, 103, 203, 303));   // id 3
    boxes.push_back(make_box(10, 12, 0, 0.5, 0, 0.5, 4, 104, 204, 304));   // id 4

    int sort_axis = 0; // x 轴
    std::vector<Pair> overlaps;
    scalable_ccd::sort_and_sweep(boxes, sort_axis, overlaps);

    std::vector<Pair> expected = { { 0, 1 }, { 0, 2 }, { 2, 3 } };
    sort_pairs(overlaps);
    sort_pairs(expected);
    REQUIRE(overlaps == expected);
}

TEST_CASE("单列表：共享顶点过滤", "[broad_phase][cpu][expected]")
{
    // 两个几何上重叠的盒子，但共享顶点 id=100，应被过滤
    std::vector<AABB> boxes;
    boxes.push_back(make_box(0, 2, 0, 0.5, 0, 0.5, 0, 100, 101, 102));     // id 0
    boxes.push_back(make_box(1, 3, 0, 0.5, 0, 0.5, 1, 100, 999, 888));     // id 1 (共享 100)

    int sort_axis = 0;
    std::vector<Pair> overlaps;
    scalable_ccd::sort_and_sweep(boxes, sort_axis, overlaps);

    std::vector<Pair> expected; // 空
    sort_pairs(overlaps);
    sort_pairs(expected);
    REQUIRE(overlaps == expected);
}

TEST_CASE("双列表：仅跨列表配对", "[broad_phase][cpu][expected]")
{
    // A0:[0,2], A1:[4,6]; B0:[1,3], B1:[5,7]
    // 期望：(A0,B0)=(0,0), (A1,B1)=(1,1)
    std::vector<AABB> A, B;
    A.push_back(make_box(0, 2, 0, 0.5, 0, 0.5, 0, 10, 20, 30)); // A0
    A.push_back(make_box(4, 6, 0, 0.5, 0, 0.5, 1, 11, 21, 31)); // A1
    B.push_back(make_box(1, 3, 0, 0.5, 0, 0.5, 0, 12, 22, 32)); // B0
    B.push_back(make_box(5, 7, 0, 0.5, 0, 0.5, 1, 13, 23, 33)); // B1

    int sort_axis = 0;
    std::vector<Pair> overlaps;
    scalable_ccd::sort_and_sweep(A, B, sort_axis, overlaps);

    std::vector<Pair> expected = { { 0, 0 }, { 1, 1 } };
    sort_pairs(overlaps);
    sort_pairs(expected);
    REQUIRE(overlaps == expected);
}
