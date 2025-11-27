#pragma once

#include <vector>
#include <utility>
#include <cstdint>
#include <unordered_map>

// Metal v2 运行时：最小功能集（预热 + YZ 过滤）
namespace scalable_ccd::metal2 {

class Metal2Runtime {
public:
    static Metal2Runtime& instance();

    bool available() const; // 有可用 Metal 设备且初始化成功
    bool warmup();          // 编译并运行一次最小 kernel

    // Buffer Pool 管理
    void clearBufferPool();  // 清空缓存的 Buffer
    void setBufferPoolEnabled(bool enabled); // 启用/禁用 Buffer Pool

    // YZ 过滤：不做 ULP 严格化与 eps 收缩；只做闭区间重叠判断 +
    // （单列表时）共享顶点剔除
    bool filterYZ(
        const std::vector<float>& minY,
        const std::vector<float>& maxY,
        const std::vector<float>& minZ,
        const std::vector<float>& maxZ,
        const std::vector<int32_t>& v0,
        const std::vector<int32_t>& v1,
        const std::vector<int32_t>& v2,
        const std::vector<std::pair<int, int>>& pairs,
        bool two_lists,
        std::vector<uint8_t>& outMask);

    // STQ（sweep_and_tiniest_queue）候选生成（占位；当前返回 false，回退 CPU）
    // 两列表：输入为合并后的 boxes（A 列表的 element_id
    // 已翻转为负），输出索引对(i,j)
    bool stqTwoLists(
        const std::vector<double>& minX,
        const std::vector<double>& maxX,
        const std::vector<double>& minY,
        const std::vector<double>& maxY,
        const std::vector<double>& minZ,
        const std::vector<double>& maxZ,
        const std::vector<int32_t>& v0,
        const std::vector<int32_t>& v1,
        const std::vector<int32_t>& v2,
        const std::vector<uint8_t>& listTag,
        std::vector<std::pair<int, int>>& outPairs);

    // 单列表：输入为单列表 boxes，输出索引对(i,j)
    bool stqSingleList(
        const std::vector<double>& minX,
        const std::vector<double>& maxX,
        const std::vector<double>& minY,
        const std::vector<double>& maxY,
        const std::vector<double>& minZ,
        const std::vector<double>& maxZ,
        const std::vector<int32_t>& v0,
        const std::vector<int32_t>& v1,
        const std::vector<int32_t>& v2,
        std::vector<std::pair<int, int>>& outPairs);

    // GPU Prefix Sum (Scan)
    bool
    scan(const std::vector<uint8_t>& inData, std::vector<uint32_t>& outData);

    // GPU Stream Compaction (Index-Only)
    bool compact(
        const std::vector<uint8_t>& valid,
        const std::vector<uint32_t>& scanOffsets,
        std::vector<uint32_t>& outIndices);

    // GPU Atomic Append (Single Pass)
    bool yzFilterAtomic(
        const std::vector<float>& minY,
        const std::vector<float>& maxY,
        const std::vector<float>& minZ,
        const std::vector<float>& maxZ,
        const std::vector<int32_t>& v0,
        const std::vector<int32_t>& v1,
        const std::vector<int32_t>& v2,
        const std::vector<std::pair<int, int>>& pairs,
        bool two_lists,
        std::vector<uint32_t>& outIndices);

    double lastYZFilterMs() const;
    double lastSTQPairsMs() const;

    // 融合内核：STQ + YZ Filter 一次完成（减少中间数据传输）
    bool stqWithYZFilter(
        const std::vector<double>& minX,
        const std::vector<double>& maxX,
        const std::vector<double>& minY,
        const std::vector<double>& maxY,
        const std::vector<double>& minZ,
        const std::vector<double>& maxZ,
        const std::vector<int32_t>& v0,
        const std::vector<int32_t>& v1,
        const std::vector<int32_t>& v2,
        const std::vector<uint8_t>& listTag, // 空则为单列表
        std::vector<std::pair<int, int>>& outPairs);

    // Narrow Phase (Root Finder)
    // Returns true on success.
    // toi is input/output (in/out).
    bool narrowPhase(
        const std::vector<double>& vertices_t0, // Flattened (n*3)
        const std::vector<double>& vertices_t1, // Flattened (n*3)
        const std::vector<int32_t>& indices,    // Flattened (m*3 or m*2)
        const std::vector<std::pair<int, int>>& overlaps,
        bool is_vf,
        float ms,
        float tolerance,
        int max_iter,
        bool allow_zero_toi,
        double& toi);

    // 优化版 Narrow Phase (Persistent Threads + Packed Data)
    bool narrowPhaseOpt(
        const std::vector<double>& vertices_t0,
        const std::vector<double>& vertices_t1,
        const std::vector<int32_t>& indices,
        const std::vector<std::pair<int, int>>& overlaps,
        bool is_vf,
        float ms,
        float tolerance,
        int max_iter,
        bool allow_zero_toi,
        double& toi);

    // 优化版 V2 (SIMD 展开 + 本地队列 + 数据预取)
    bool narrowPhaseOptV2(
        const std::vector<double>& vertices_t0,
        const std::vector<double>& vertices_t1,
        const std::vector<int32_t>& indices,
        const std::vector<std::pair<int, int>>& overlaps,
        bool is_vf,
        float ms,
        float tolerance,
        int max_iter,
        bool allow_zero_toi,
        double& toi);

private:
    Metal2Runtime();
    ~Metal2Runtime();
    Metal2Runtime(const Metal2Runtime&) = delete;
    Metal2Runtime& operator=(const Metal2Runtime&) = delete;

    struct Impl;
    Impl* impl_;
};

} // namespace scalable_ccd::metal2
