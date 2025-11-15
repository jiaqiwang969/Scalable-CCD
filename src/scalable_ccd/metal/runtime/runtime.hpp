#pragma once

#include <memory>
#include <vector>
#include <utility>
#include <cstdint>
#include <string>

namespace scalable_ccd::metal {

/// 与 CUDA 外围选项语义接近的 Metal STQ 配置（可选传入，默认使用内置/环境值）
struct STQConfig {
    // 主轴重叠相对容差（默认 0，贴近 CUDA 语义）
    float epsScale = 0.0f;
    // YZ 相对容差（默认 0，与 CPU/hybrid 严格一致）
    float yzEpsScale = 0.0f;
    // 是否禁用 startJ（默认 false 开启 startJ）
    bool disableStartJ = false;
    // 推进行为最多跳过步数（默认 0）
    uint32_t maxSkipSteps = 0u;
    // 线程组大小建议（默认 32，与 CUDA threads_per_block 对齐）
    uint32_t threadgroupWidth = 32u;
    // 近边界 YZ tie-break 的 ULP 门限（默认 2）
    uint32_t yzTieUlps = 2u;
};

/// 轻量 Metal 运行时封装（PIMPL），用于在 C++ 侧无 ObjC 头依赖。
class MetalRuntime {
public:
    static MetalRuntime& instance();

    /// 是否可用（有 Metal 设备且预热不报错）
    bool available() const;

    /// 进行一次最小 kernel 预热（运行一个 noop 核心校验数据往返）
    /// 返回 true 表示运行成功（即 GPU 路径可用）
    bool warmup();

    /// 返回 Metal 设备名称（若不可用则返回空字符串）
    std::string device_name() const;

    /// 使用 Metal 对候选对进行 yz 平面过滤，输出每对是否相交的掩码（1/0）
    /// 返回 true 表示执行成功；失败时可回退到 CPU。
    bool filterYZ(
        const std::vector<float>& minY,
        const std::vector<float>& maxY,
        const std::vector<float>& minZ,
        const std::vector<float>& maxZ,
        const std::vector<int32_t>& v0,
        const std::vector<int32_t>& v1,
        const std::vector<int32_t>& v2,
        const std::vector<std::pair<int, int>>& pairs,
        std::vector<uint8_t>& outMask);

    /// Narrow-phase scaffolding: pack overlaps to simple records (aid,bid,ms)
    bool narrowAddData(
        const std::vector<std::pair<int,int>>& overlaps,
        float ms,
        std::vector<std::tuple<int,int,float>>& outRecords);

    /// Narrow-phase pack (VF): write v0..v3 at t0/t1 and ms for each (vi,fi)
    bool narrowPackVF(
        const std::vector<float>& vertices_t0_flat, // size = 3*nV
        const std::vector<float>& vertices_t1_flat, // size = 3*nV
        uint32_t num_vertices,
        const std::vector<int32_t>& faces_flat,      // size = 3*nF
        const std::vector<std::pair<int,int>>& overlaps_vi_fi,
        float minimum_separation,
        uint32_t& out_packed_count);

    /// Narrow-phase pack (EE): write two edges' endpoints at t0/t1 and ms for each (ea,eb)
    bool narrowPackEE(
        const std::vector<float>& vertices_t0_flat, // size = 3*nV
        const std::vector<float>& vertices_t1_flat, // size = 3*nV
        uint32_t num_vertices,
        const std::vector<int32_t>& edges_flat,     // size = 2*nE
        const std::vector<std::pair<int,int>>& overlaps_ei_ej,
        float minimum_separation,
        uint32_t& out_packed_count);

    /// Narrow-phase placeholder run (VF): future root-finder entry;
    /// currently just touches inputs and returns processed count.
    bool runVFPlaceholder(
        const std::vector<float>& vertices_t0_flat,
        const std::vector<float>& vertices_t1_flat,
        uint32_t num_vertices,
        const std::vector<int32_t>& faces_flat,
        const std::vector<std::pair<int,int>>& overlaps_vi_fi,
        float minimum_separation,
        float tolerance,
        int max_iterations,
        bool allow_zero_toi,
        uint32_t& out_processed);

    // Host-side mirror for VF CCD data (float layout matching GPU record)
    struct CCDDataMetalVF {
        float v0s[3]; float v1s[3]; float v2s[3]; float v3s[3];
        float v0e[3]; float v1e[3]; float v2e[3]; float v3e[3];
        float ms;
        float err[3];
        float tol[3];
        float toi;
        int   aid;
        int   bid;
    };
    struct CCDDataMetalEE {
        float v0s[3]; float v1s[3]; float v2s[3]; float v3s[3];
        float v0e[3]; float v1e[3]; float v2e[3]; float v3e[3];
        float ms;
        float err[3];
        float tol[3];
        float toi;
        int   aid;
        int   bid;
    };
    /// Pack VF overlaps into CCD-like records and read back to host for observation
    bool getVFCCDData(
        const std::vector<float>& vertices_t0_flat,
        const std::vector<float>& vertices_t1_flat,
        uint32_t num_vertices,
        const std::vector<int32_t>& faces_flat,
        const std::vector<std::pair<int,int>>& overlaps_vi_fi,
        float ms,
        std::vector<CCDDataMetalVF>& out);
    /// Pack EE overlaps into CCD-like records and read back to host for observation
    bool getEECCDData(
        const std::vector<float>& vertices_t0_flat,
        const std::vector<float>& vertices_t1_flat,
        uint32_t num_vertices,
        const std::vector<int32_t>& edges_flat,
        const std::vector<std::pair<int,int>>& overlaps_ei_ej,
        float ms,
        std::vector<CCDDataMetalEE>& out);

    /// VF root-finder skeleton (observation): packs to CCDData on GPU then writes toi array.
    bool runVFRootSkeleton(
        const std::vector<float>& vertices_t0_flat,
        const std::vector<float>& vertices_t1_flat,
        uint32_t numVertices,
        const std::vector<int32_t>& faces_flat,
        const std::vector<std::pair<int,int>>& overlaps_vi_fi,
        float minimum_separation,
        float tolerance,
        int max_iterations,
        bool allow_zero_toi,
        std::vector<float>& out_toi);

    /// VF: run GPU and return minimal ToI across overlaps (device-side reduction).
    bool runVFMin(
        const std::vector<float>& vertices_t0_flat,
        const std::vector<float>& vertices_t1_flat,
        uint32_t numVertices,
        const std::vector<int32_t>& faces_flat,
        const std::vector<std::pair<int,int>>& overlaps_vi_fi,
        float minimum_separation,
        float tolerance,
        int max_iterations,
        bool allow_zero_toi,
        float& out_min_toi);

    /// EE: run GPU and return minimal ToI across overlaps (device-side reduction).
    bool runEEMin(
        const std::vector<float>& vertices_t0_flat,
        const std::vector<float>& vertices_t1_flat,
        uint32_t numVertices,
        const std::vector<int32_t>& edges_flat,
        const std::vector<std::pair<int,int>>& overlaps_ei_ej,
        float minimum_separation,
        float tolerance,
        int max_iterations,
        bool allow_zero_toi,
        float& out_min_toi);

    /// STQ warmup helper: run single-list Sweep & Tiniest Queue on GPU (observation only)
    /// Returns true on successful GPU execution; outputs (i,j) in index space.
    bool sweepSTQSingleList(
        const std::vector<float>& minX,
        const std::vector<float>& maxX,
        const std::vector<float>& minY,
        const std::vector<float>& maxY,
        const std::vector<float>& minZ,
        const std::vector<float>& maxZ,
        const std::vector<int32_t>& v0,
        const std::vector<int32_t>& v1,
        const std::vector<int32_t>& v2,
        const uint32_t capacity,
        std::vector<std::pair<int, int>>& outPairs,
        const std::vector<uint32_t>* startJ = nullptr,
        const std::vector<uint8_t>* listTag = nullptr,
        bool twoLists = false);

    /// STQ warmup helper with explicit configuration
    bool sweepSTQSingleList(
        const std::vector<float>& minX,
        const std::vector<float>& maxX,
        const std::vector<float>& minY,
        const std::vector<float>& maxY,
        const std::vector<float>& minZ,
        const std::vector<float>& maxZ,
        const std::vector<int32_t>& v0,
        const std::vector<int32_t>& v1,
        const std::vector<int32_t>& v2,
        const uint32_t capacity,
        std::vector<std::pair<int, int>>& outPairs,
        const std::vector<uint32_t>* startJ,
        const std::vector<uint8_t>* listTag,
        bool twoLists,
        const STQConfig& cfg);

    

    /// EE root-finder skeleton (observation): packs to CCDData on GPU then writes toi array.
    bool runEERootSkeleton(
        const std::vector<float>& vertices_t0_flat,
        const std::vector<float>& vertices_t1_flat,
        uint32_t numVertices,
        const std::vector<int32_t>& edges_flat,
        const std::vector<std::pair<int,int>>& overlaps_ei_ej,
        float minimum_separation,
        float tolerance,
        int max_iterations,
        bool allow_zero_toi,
        std::vector<float>& out_toi);

    /// 单列表主轴 sweep（每个 i 在 GPU 上前向扫描），同时做 yz 与共享顶点过滤，
    /// 直接输出 (i,j) 对（i<j，为排序后索引）。capacity 为对数上限（可设为 CPU 估计的候选数）。
    bool sweepSingleList(
        const std::vector<double>& minX,
        const std::vector<double>& maxX,
        const std::vector<double>& minY,
        const std::vector<double>& maxY,
        const std::vector<double>& minZ,
        const std::vector<double>& maxZ,
        const std::vector<int32_t>& v0,
        const std::vector<int32_t>& v1,
        const std::vector<int32_t>& v2,
        const uint32_t capacity,
        std::vector<std::pair<int, int>>& outPairs);

    /// Metal 版 STQ：单列表主轴 STQ（仅用于观测与对照，不做最终输出）
    bool runSweepAndTiniestQueue(
        const std::vector<float>& minX,
        const std::vector<float>& maxX,
        const std::vector<float>& minY,
        const std::vector<float>& maxY,
        const std::vector<float>& minZ,
        const std::vector<float>& maxZ,
        const std::vector<int32_t>& v0,
        const std::vector<int32_t>& v1,
        const std::vector<int32_t>& v2,
        const uint32_t capacity,
        std::vector<std::pair<int, int>>& outPairs,
        const std::vector<uint32_t>* startJ = nullptr,
        const std::vector<uint8_t>* listTag = nullptr,
        bool twoLists = false);

    /// 带配置版本，便于与 CUDA 外围一致化
    bool runSweepAndTiniestQueue(
        const std::vector<float>& minX,
        const std::vector<float>& maxX,
        const std::vector<float>& minY,
        const std::vector<float>& maxY,
        const std::vector<float>& minZ,
        const std::vector<float>& maxZ,
        const std::vector<int32_t>& v0,
        const std::vector<int32_t>& v1,
        const std::vector<int32_t>& v2,
        const uint32_t capacity,
        std::vector<std::pair<int, int>>& outPairs,
        const std::vector<uint32_t>* startJ,
        const std::vector<uint8_t>* listTag,
        bool twoLists,
        const STQConfig& cfg);

private:
    MetalRuntime();
    ~MetalRuntime();
    MetalRuntime(const MetalRuntime&) = delete;
    MetalRuntime& operator=(const MetalRuntime&) = delete;

    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace scalable_ccd::metal
