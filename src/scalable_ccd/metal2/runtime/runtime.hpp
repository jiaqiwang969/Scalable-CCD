#pragma once

#include <vector>
#include <utility>
#include <cstdint>

// Metal v2 运行时：最小功能集（预热 + YZ 过滤）
namespace scalable_ccd::metal2 {

class Metal2Runtime {
public:
    static Metal2Runtime& instance();

    bool available() const;  // 有可用 Metal 设备且初始化成功
    bool warmup();           // 编译并运行一次最小 kernel

    // YZ 过滤：不做 ULP 严格化与 eps 收缩；只做闭区间重叠判断 + （单列表时）共享顶点剔除
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

private:
    Metal2Runtime();
    ~Metal2Runtime();
    Metal2Runtime(const Metal2Runtime&) = delete;
    Metal2Runtime& operator=(const Metal2Runtime&) = delete;

    struct Impl;
    Impl* impl_;
};

} // namespace scalable_ccd::metal2
