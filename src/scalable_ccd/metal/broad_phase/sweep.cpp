#include "sweep.hpp"

namespace scalable_ccd::metal::bp {

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
    const std::vector<uint32_t>* startJ,
    const std::vector<uint8_t>* listTag,
    bool twoLists)
{
    (void)startJ; // 未用，保留同签名兼容
    outPairs.clear();
    const size_t n = minX.size();
    if (n == 0) return true;
    if (maxX.size()!=n || minY.size()!=n || maxY.size()!=n || minZ.size()!=n || maxZ.size()!=n) return false;
    if (v0.size()!=n || v1.size()!=n || v2.size()!=n) return false;

    // 生成主轴候选（假定已按 X 主轴排序）
    std::vector<std::pair<int,int>> candidates;
    candidates.reserve(n);
    for (int i = 0; i < static_cast<int>(n); ++i) {
        for (int j = i + 1; j < static_cast<int>(n); ++j) {
            if (maxX[i] < minX[j]) break; // 主轴分离
            if (twoLists && listTag) {
                if ((*listTag)[static_cast<size_t>(i)] == (*listTag)[static_cast<size_t>(j)]) continue;
            }
            candidates.emplace_back(i, j);
            if (capacity > 0 && outPairs.size() + candidates.size() >= capacity) break;
        }
        if (capacity > 0 && outPairs.size() + candidates.size() >= capacity) break;
    }

    // YZ + 共享顶点过滤（使用 MetalRuntime 的过滤 kernel；失败则退回 CPU 逻辑）
    std::vector<uint8_t> mask;
    const bool ok = MetalRuntime::instance().filterYZ(minY, maxY, minZ, maxZ, v0, v1, v2, candidates, mask);
    if (!ok) {
        // CPU 回退：简单按 YZ 重叠与共享顶点过滤
        mask.assign(candidates.size(), 0);
        for (size_t k = 0; k < candidates.size(); ++k) {
            int i = candidates[k].first, j = candidates[k].second;
            const bool overlapY = !(maxY[i] < minY[j] || minY[i] > maxY[j]);
            const bool overlapZ = !(maxZ[i] < minZ[j] || minZ[i] > maxZ[j]);
            const bool share = (v0[i]==v0[j]) || (v0[i]==v1[j]) || (v0[i]==v2[j]) ||
                               (v1[i]==v0[j]) || (v1[i]==v1[j]) || (v1[i]==v2[j]) ||
                               (v2[i]==v0[j]) || (v2[i]==v1[j]) || (v2[i]==v2[j]);
            mask[k] = (overlapY && overlapZ && !share) ? 1 : 0;
        }
    }

    outPairs.reserve(outPairs.size() + candidates.size());
    for (size_t k = 0; k < candidates.size(); ++k) {
        if (mask[k]) {
            outPairs.push_back(candidates[k]);
            if (capacity > 0 && outPairs.size() >= capacity) break;
        }
    }
    return true;
}

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
    bool twoLists)
{
    return MetalRuntime::instance().runSweepAndTiniestQueue(
        minX, maxX, minY, maxY, minZ, maxZ, v0, v1, v2, capacity,
        outPairs, startJ, listTag, twoLists);
}

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
    const STQConfig& cfg)
{
    return MetalRuntime::instance().runSweepAndTiniestQueue(
        minX, maxX, minY, maxY, minZ, maxZ, v0, v1, v2, capacity,
        outPairs, startJ, listTag, twoLists, cfg);
}

bool filter_yz(
    const std::vector<float>& minY,
    const std::vector<float>& maxY,
    const std::vector<float>& minZ,
    const std::vector<float>& maxZ,
    const std::vector<int32_t>& v0,
    const std::vector<int32_t>& v1,
    const std::vector<int32_t>& v2,
    const std::vector<std::pair<int, int>>& pairs,
    std::vector<uint8_t>& outMask)
{
    return MetalRuntime::instance().filterYZ(
        minY, maxY, minZ, maxZ, v0, v1, v2, pairs, outMask);
}

} // namespace scalable_ccd::metal::bp
