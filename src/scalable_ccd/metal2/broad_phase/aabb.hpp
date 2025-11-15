#pragma once

#include <scalable_ccd/broad_phase/aabb.hpp>

#include <memory>
#include <vector>

namespace scalable_ccd::metal2 {

// 与 CUDA/Metal 旧版接口一致的占位封装；首期仅保存在 host
struct DeviceAABBs {
    DeviceAABBs() = default;
    explicit DeviceAABBs(const std::vector<AABB>& boxes) : h_boxes(boxes) {}
    void clear() { h_boxes.clear(); }
    void shrink_to_fit() { h_boxes.shrink_to_fit(); }
    size_t size() const { return h_boxes.size(); }
    std::vector<AABB> h_boxes;
};

} // namespace scalable_ccd::metal2

