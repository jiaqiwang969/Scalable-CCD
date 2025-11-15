#include "utils.hpp"

#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>

#include <algorithm>

namespace scalable_ccd::metal::bp {

namespace {
    struct SortBoxes {
        int axis;
        explicit SortBoxes(int a) : axis(a) {}
        bool operator()(const AABB& a, const AABB& b) const
        {
            return a.min[axis] < b.min[axis];
        }
    };
}

void build_sorted_boxes(
    int sort_axis,
    bool is_two_lists,
    const std::shared_ptr<DeviceAABBs>& boxesA,
    const std::shared_ptr<DeviceAABBs>& boxesB,
    std::vector<AABB>& out_boxes)
{
    if (is_two_lists) {
        std::vector<AABB> a = boxesA ? boxesA->h_boxes : std::vector<AABB>{};
        std::vector<AABB> b = boxesB ? boxesB->h_boxes : std::vector<AABB>{};
        if (a.empty() || b.empty()) { out_boxes.clear(); return; }
        sort_along_axis(sort_axis, a);
        sort_along_axis(sort_axis, b);
        for (AABB& box : a) {
            box.element_id = -box.element_id - 1;
        }
        out_boxes.resize(a.size() + b.size());
        std::merge(
            a.begin(), a.end(), b.begin(), b.end(),
            out_boxes.begin(), SortBoxes(sort_axis));
    } else {
        out_boxes = boxesA ? boxesA->h_boxes : std::vector<AABB>{};
        if (out_boxes.empty()) return;
        sort_along_axis(sort_axis, out_boxes);
    }
}

void generate_axis_candidates(
    const std::vector<AABB>& boxes,
    int sort_axis,
    bool is_two_lists,
    std::vector<std::pair<int, int>>& out_pairs)
{
    out_pairs.clear();
    out_pairs.reserve(boxes.size());
    auto is_valid_pair = [&](long id_a, long id_b) {
        if (is_two_lists) {
            return (id_a >= 0 && id_b < 0) || (id_a < 0 && id_b >= 0);
        } else {
            return true;
        }
    };
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
        const AABB& a = boxes[i];
        for (int j = i + 1; j < static_cast<int>(boxes.size()); ++j) {
            const AABB& b = boxes[j];
            if (a.max[sort_axis] < b.min[sort_axis]) break;
            if (is_valid_pair(a.element_id, b.element_id)) {
                out_pairs.emplace_back(i, j);
            }
        }
    }
}

} // namespace scalable_ccd::metal::bp


