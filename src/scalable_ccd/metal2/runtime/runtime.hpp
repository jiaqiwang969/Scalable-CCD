#pragma once

#include <vector>
#include <utility>
#include <cstdint>

// Metal v2 运行时：最小功能集（预热 + YZ 过滤）
namespace scalable_ccd::metal2 {

class Metal2Runtime {
public:
    static Metal2Runtime& instance();

    bool available() const; // 有可用 Metal 设备且初始化成功
    bool warmup();          // 编译并运行一次最小 kernel

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

    // 最近一次 GPU 调用的计时（毫秒），若无则返回 <0
    double lastYZFilterMs() const;
    double lastSTQPairsMs() const;

private:
    Metal2Runtime();
    ~Metal2Runtime();
    Metal2Runtime(const Metal2Runtime&) = delete;
    Metal2Runtime& operator=(const Metal2Runtime&) = delete;

    struct Impl;
    Impl* impl_;
};

} // namespace scalable_ccd::metal2
