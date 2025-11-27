#pragma once

#include "../../broad_phase/aabb.hpp"
#include <vector>

namespace scalable_ccd::metal2 {

struct DeviceAABBs {
    DeviceAABBs() = default;
    DeviceAABBs(const std::vector<scalable_ccd::AABB>& boxes) : h_boxes(boxes)
    {
    }
    std::vector<scalable_ccd::AABB> h_boxes;
};

} // namespace scalable_ccd::metal2
