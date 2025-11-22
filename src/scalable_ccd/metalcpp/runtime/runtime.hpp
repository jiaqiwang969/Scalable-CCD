#pragma once

#include <vector>
#include <utility>
#include <cstdint>

// Metal C++ (metal-cpp) 运行时：最小功能（预热 + YZ 过滤）
namespace scalable_ccd::metalcpp {

class MetalCppRuntime {
public:
    static MetalCppRuntime& instance();

    bool available() const;  // 有可用 Metal 设备且初始化成功
    bool warmup();           // 编译并运行一次最小 kernel

    // YZ 过滤：闭区间重叠 + （单/双列表）共享顶点剔除
    bool filterYZ(
        const std::vector<float>& minY,
        const std::vector<float>& maxY,
        const std::vector<float>& minZ,
        const std::vector<float>& maxZ,
        const std::vector<int32_t>& v0,
        const std::vector<int32_t>& v1,
        const std::vector<int32_t>& v2,
        const std::vector<std::pair<int,int>>& pairs,
        bool two_lists,
        std::vector<uint8_t>& outMask);

    // STQ：双列表候选生成（沿 X 轴），输入为合并后的 boxes（A 列表 element_id 已翻负，listTag=1 表示 A）
    bool stqTwoLists(
        const std::vector<double>& minX,
        const std::vector<double>& maxX,
        const std::vector<uint8_t>& listTag,
        std::vector<std::pair<int,int>>& outPairs);

    double lastYZFilterMs() const;
    double lastSTQPairsMs() const;

private:
    MetalCppRuntime();
    ~MetalCppRuntime();
    MetalCppRuntime(const MetalCppRuntime&) = delete;
    MetalCppRuntime& operator=(const MetalCppRuntime&) = delete;

    struct Impl;
    Impl* impl_;
};

} // namespace scalable_ccd::metalcpp
