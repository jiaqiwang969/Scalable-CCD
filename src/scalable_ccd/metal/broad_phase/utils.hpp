#pragma once

#include <scalable_ccd/broad_phase/aabb.hpp>
#include <scalable_ccd/metal/broad_phase/aabb.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace scalable_ccd::metal::bp {

/// 构建按主轴排序后的 boxes（与 CPU 排序规则一致）
void build_sorted_boxes(
    int sort_axis,
    bool is_two_lists,
    const std::shared_ptr<DeviceAABBs>& boxesA,
    const std::shared_ptr<DeviceAABBs>& boxesB,
    std::vector<AABB>& out_boxes);

/// 基于主轴区间生成候选对（不做 Y/Z 与共享顶点过滤）
void generate_axis_candidates(
    const std::vector<AABB>& boxes,
    int sort_axis,
    bool is_two_lists,
    std::vector<std::pair<int, int>>& out_pairs);

} // namespace scalable_ccd::metal::bp


