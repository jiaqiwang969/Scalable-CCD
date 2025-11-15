#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <scalable_ccd/metal/runtime/runtime.hpp>

namespace scalable_ccd::metal::bp {

// 对齐 CUDA：提供 SAP（sweep_and_prune）同名入口（Metal 侧以 CPU/混合路径实现）
bool sweep_and_prune(
    const std::vector<float>& minX,
    const std::vector<float>& maxX,
    const std::vector<float>& minY,
    const std::vector<float>& maxY,
    const std::vector<float>& minZ,
    const std::vector<float>& maxZ,
    const std::vector<int32_t>& v0,
    const std::vector<int32_t>& v1,
    const std::vector<int32_t>& v2,
    uint32_t capacity,
    std::vector<std::pair<int, int>>& outPairs,
    const std::vector<uint32_t>* startJ = nullptr,
    const std::vector<uint8_t>* listTag = nullptr,
    bool twoLists = false);

bool sweep_and_tiniest_queue(
    const std::vector<float>& minX,
    const std::vector<float>& maxX,
    const std::vector<float>& minY,
    const std::vector<float>& maxY,
    const std::vector<float>& minZ,
    const std::vector<float>& maxZ,
    const std::vector<int32_t>& v0,
    const std::vector<int32_t>& v1,
    const std::vector<int32_t>& v2,
    uint32_t capacity,
    std::vector<std::pair<int, int>>& outPairs,
    const std::vector<uint32_t>* startJ = nullptr,
    const std::vector<uint8_t>* listTag = nullptr,
    bool twoLists = false);

bool sweep_and_tiniest_queue(
    const std::vector<float>& minX,
    const std::vector<float>& maxX,
    const std::vector<float>& minY,
    const std::vector<float>& maxY,
    const std::vector<float>& minZ,
    const std::vector<float>& maxZ,
    const std::vector<int32_t>& v0,
    const std::vector<int32_t>& v1,
    const std::vector<int32_t>& v2,
    uint32_t capacity,
    std::vector<std::pair<int, int>>& outPairs,
    const std::vector<uint32_t>* startJ,
    const std::vector<uint8_t>* listTag,
    bool twoLists,
    const STQConfig& cfg);

bool filter_yz(
    const std::vector<float>& minY,
    const std::vector<float>& maxY,
    const std::vector<float>& minZ,
    const std::vector<float>& maxZ,
    const std::vector<int32_t>& v0,
    const std::vector<int32_t>& v1,
    const std::vector<int32_t>& v2,
    const std::vector<std::pair<int, int>>& pairs,
    std::vector<uint8_t>& outMask);

} // namespace scalable_ccd::metal::bp

