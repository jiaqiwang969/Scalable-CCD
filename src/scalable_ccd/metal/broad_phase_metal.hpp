// Metal 宽阶段（Broad Phase）Host 封装（metal-cpp）
// 仅负责上传已排序 SoA 数据、调度 SAP kernel、回读 overlaps。
// 注意：本实现仅覆盖 SAP（单/双列表），用于最小闭环与对拍验证。

#pragma once

#include <scalable_ccd/broad_phase/aabb.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace scalable_ccd {
namespace metal {

struct MetalAABBsSoA {
    // float2 按 interleaved 方式存放：大小为 2*N
    std::vector<float> sorted_major; // [min,max]
    std::vector<float> mini_min;     // [bmin,cmin]
    std::vector<float> mini_max;     // [bmax,cmax]
    // 顶点 id 按 int4 存放：大小为 4*N（最后一位为填充）
    std::vector<int> vertex_ids;     // [v0,v1,v2,pad]
    std::vector<int> element_ids;    // 每盒 1 个
    size_t size() const { return element_ids.size(); }
    void clear()
    {
        sorted_major.clear();
        mini_min.clear();
        mini_max.clear();
        vertex_ids.clear();
        element_ids.clear();
    }
};

// 从 CPU 盒子生成 SoA（单列表）
void make_soa_one_list(
    const std::vector<scalable_ccd::AABB>& boxes,
    int axis,
    MetalAABBsSoA& out);

// 从 CPU 盒子生成 SoA（双列表，A 列表 element_id 取负 -id-1 后与 B 合并再排序）
void make_soa_two_lists(
    const std::vector<scalable_ccd::AABB>& boxesA,
    const std::vector<scalable_ccd::AABB>& boxesB,
    int axis,
    MetalAABBsSoA& out);

class BroadPhase {
public:
    // metallib_path 留空时使用编译期注入的 SCALABLE_CCD_METAL_LIB_PATH
    explicit BroadPhase(const std::string& metallib_path = {});
    ~BroadPhase();

    void upload(const MetalAABBsSoA& soa);
    size_t num_boxes() const { return nboxes_; }

    // 最近一次 kernel 的 GPU 时间（毫秒）；若不可用则返回负值
    double last_gpu_ms() const { return last_gpu_ms_; }
    // 最近一次 kernel 的真实计数（未截断）
    size_t last_real_count() const { return last_real_count_; }
    void set_use_stq(bool flag) { use_stq_ = flag; }

    // 执行一批 SAP（单/双列表），返回写入范围内的 overlaps。
    // 若设备端 real_count > overlaps_capacity，返回被截断的 overlaps，
    // 调用方可据此扩容或缩批并重跑本批。
    std::vector<std::pair<int, int>> detect_overlaps_partial(
        bool two_lists,
        uint32_t start_box_id,
        uint32_t max_overlap_cutoff,
        uint32_t overlaps_capacity);

    // 循环调用 detect_overlaps_partial 直至覆盖全部 boxes，自动按 cutoff 推进，
    // 在溢出时扩容/重试，返回全量 overlaps。
    std::vector<std::pair<int, int>> detect_overlaps(
        bool two_lists,
        uint32_t max_overlap_cutoff,
        uint32_t overlaps_capacity_hint = 0);

private:
    struct Impl;
    Impl* impl_ = nullptr;
    uint32_t nboxes_ = 0;
    double last_gpu_ms_ = -1.0;
    size_t last_real_count_ = 0;
    bool use_stq_ = false;
};

} // namespace metal
} // namespace scalable_ccd
