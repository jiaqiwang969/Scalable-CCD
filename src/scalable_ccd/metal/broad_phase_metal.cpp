// Metal 宽阶段 Host 封装实现

#include "broad_phase_metal.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>

// 仅在 Apple 平台编译/链接 metal-cpp
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_OSX
#define SCALABLE_CCD_METAL_ENABLED 1
#endif
#endif

#if SCALABLE_CCD_METAL_ENABLED
#ifndef NS_PRIVATE_IMPLEMENTATION
#define NS_PRIVATE_IMPLEMENTATION
#endif
#ifndef MTL_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#endif
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#endif

namespace scalable_ccd {
namespace metal {

namespace {

inline uint32_t get_env_u32(const char* name, uint32_t fallback)
{
    if (const char* v = std::getenv(name)) {
        int parsed = std::atoi(v);
        if (parsed > 0) {
            return static_cast<uint32_t>(parsed);
        }
    }
    return fallback;
}

} // namespace

// ---------------- CPU 侧：从 AABB 生成 SoA ----------------

static inline void push_float2(std::vector<float>& v, float a, float b)
{
    v.push_back(a);
    v.push_back(b);
}

static inline void push_int4(std::vector<int>& v, int a, int b, int c, int d)
{
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    v.push_back(d);
}

static inline int other_axis(int axis, int rank_index)
{
    // axis=0 => others 1,2 ; axis=1 => 0,2 ; axis=2 => 0,1
    static const int OTHER[3][2] = { { 1, 2 }, { 0, 2 }, { 0, 1 } };
    return OTHER[axis][rank_index];
}

void make_soa_one_list(
    const std::vector<scalable_ccd::AABB>& boxes,
    int axis,
    MetalAABBsSoA& out)
{
    std::vector<scalable_ccd::AABB> sorted = boxes;
    scalable_ccd::sort_along_axis(axis, sorted);

    out.clear();
    out.sorted_major.reserve(sorted.size() * 2);
    out.mini_min.reserve(sorted.size() * 2);
    out.mini_max.reserve(sorted.size() * 2);
    out.vertex_ids.reserve(sorted.size() * 4);
    out.element_ids.reserve(sorted.size());

    const int b = other_axis(axis, 0);
    const int c = other_axis(axis, 1);

    for (const auto& box : sorted) {
        push_float2(out.sorted_major, float(box.min[axis]), float(box.max[axis]));
        push_float2(out.mini_min, float(box.min[b]), float(box.min[c]));
        push_float2(out.mini_max, float(box.max[b]), float(box.max[c]));
        push_int4(out.vertex_ids, int(box.vertex_ids[0]), int(box.vertex_ids[1]),
            int(box.vertex_ids[2]), 0);
        out.element_ids.push_back(int(box.element_id));
    }
}

void make_soa_two_lists(
    const std::vector<scalable_ccd::AABB>& boxesA,
    const std::vector<scalable_ccd::AABB>& boxesB,
    int axis,
    MetalAABBsSoA& out)
{
    // 合并两列表后排序，A 列表 element_id 取负以标记来源
    std::vector<scalable_ccd::AABB> combined;
    combined.reserve(boxesA.size() + boxesB.size());
    for (auto a : boxesA) {
        a.element_id = -a.element_id - 1;
        combined.push_back(a);
    }
    combined.insert(combined.end(), boxesB.begin(), boxesB.end());

    scalable_ccd::sort_along_axis(axis, combined);

    out.clear();
    out.sorted_major.reserve(combined.size() * 2);
    out.mini_min.reserve(combined.size() * 2);
    out.mini_max.reserve(combined.size() * 2);
    out.vertex_ids.reserve(combined.size() * 4);
    out.element_ids.reserve(combined.size());

    const int b = other_axis(axis, 0);
    const int c = other_axis(axis, 1);

    for (const auto& box : combined) {
        push_float2(out.sorted_major, float(box.min[axis]), float(box.max[axis]));
        push_float2(out.mini_min, float(box.min[b]), float(box.min[c]));
        push_float2(out.mini_max, float(box.max[b]), float(box.max[c]));
        push_int4(out.vertex_ids, int(box.vertex_ids[0]), int(box.vertex_ids[1]),
            int(box.vertex_ids[2]), 0);
        out.element_ids.push_back(int(box.element_id));
    }
}

// ---------------- Metal Host 封装 ----------------

struct BroadPhase::Impl {
#if SCALABLE_CCD_METAL_ENABLED
    NS::SharedPtr<MTL::Device> device;
    NS::SharedPtr<MTL::CommandQueue> queue;
    NS::SharedPtr<MTL::Library> library;
    NS::SharedPtr<MTL::ComputePipelineState> pso_one;
    NS::SharedPtr<MTL::ComputePipelineState> pso_two;
    NS::SharedPtr<MTL::ComputePipelineState> pso_one_stq;
    NS::SharedPtr<MTL::ComputePipelineState> pso_two_stq;

    NS::SharedPtr<MTL::Buffer> buf_sorted_major;
    NS::SharedPtr<MTL::Buffer> buf_mini_min;
    NS::SharedPtr<MTL::Buffer> buf_mini_max;
    NS::SharedPtr<MTL::Buffer> buf_vertex_ids;
    NS::SharedPtr<MTL::Buffer> buf_element_ids;
#endif
    std::string metallib_path;
};

BroadPhase::BroadPhase(const std::string& metallib_path)
    : impl_(new Impl())
{
    impl_->metallib_path = metallib_path;
#if SCALABLE_CCD_METAL_ENABLED
    if (impl_->metallib_path.empty()) {
#ifdef SCALABLE_CCD_METAL_LIB_PATH
        impl_->metallib_path = SCALABLE_CCD_METAL_LIB_PATH;
#endif
    }

    impl_->device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!impl_->device) {
        throw std::runtime_error("Metal device unavailable");
    }
    impl_->queue = NS::TransferPtr(impl_->device->newCommandQueue());
    if (!impl_->queue) {
        throw std::runtime_error("Metal command queue creation failed");
    }

    // 加载 metallib
    NS::Error* err = nullptr;
    auto path = NS::String::string(impl_->metallib_path.c_str(), NS::UTF8StringEncoding);
    auto url = NS::URL::fileURLWithPath(path);
    impl_->library = NS::TransferPtr(impl_->device->newLibrary(url, &err));
    if (!impl_->library) {
        throw std::runtime_error("Failed to load metallib: " + impl_->metallib_path);
    }

    // 创建两个 pipeline
    auto fn_one = NS::TransferPtr(
        impl_->library->newFunction(NS::String::string("sweep_and_prune_one_list", NS::UTF8StringEncoding)));
    auto fn_two = NS::TransferPtr(
        impl_->library->newFunction(NS::String::string("sweep_and_prune_two_lists", NS::UTF8StringEncoding)));
    auto fn_one_stq = NS::TransferPtr(
        impl_->library->newFunction(NS::String::string("sweep_and_tiniest_queue_one_list", NS::UTF8StringEncoding)));
    auto fn_two_stq = NS::TransferPtr(
        impl_->library->newFunction(NS::String::string("sweep_and_tiniest_queue_two_lists", NS::UTF8StringEncoding)));
    impl_->pso_one = NS::TransferPtr(impl_->device->newComputePipelineState(fn_one.get(), &err));
    impl_->pso_two = NS::TransferPtr(impl_->device->newComputePipelineState(fn_two.get(), &err));
    impl_->pso_one_stq = NS::TransferPtr(impl_->device->newComputePipelineState(fn_one_stq.get(), &err));
    impl_->pso_two_stq = NS::TransferPtr(impl_->device->newComputePipelineState(fn_two_stq.get(), &err));
    if (!impl_->pso_one || !impl_->pso_two) {
        throw std::runtime_error("Failed to create compute pipeline states");
    }
#else
    (void)impl_;
#endif
}

BroadPhase::~BroadPhase()
{
    delete impl_;
    impl_ = nullptr;
}

void BroadPhase::upload(const MetalAABBsSoA& soa)
{
    nboxes_ = static_cast<uint32_t>(soa.size());
#if SCALABLE_CCD_METAL_ENABLED
    auto dev = impl_->device.get();
    auto mk = [&](size_t bytes, const void* src) -> NS::SharedPtr<MTL::Buffer> {
        auto buf = NS::TransferPtr(dev->newBuffer(bytes, MTL::ResourceStorageModeShared));
        if (src && bytes) {
            std::memcpy(buf->contents(), src, bytes);
        }
        return buf;
    };
    impl_->buf_sorted_major = mk(sizeof(float) * 2 * soa.size(), soa.sorted_major.data());
    impl_->buf_mini_min = mk(sizeof(float) * 2 * soa.size(), soa.mini_min.data());
    impl_->buf_mini_max = mk(sizeof(float) * 2 * soa.size(), soa.mini_max.data());
    impl_->buf_vertex_ids = mk(sizeof(int) * 4 * soa.size(), soa.vertex_ids.data());
    impl_->buf_element_ids = mk(sizeof(int) * soa.size(), soa.element_ids.data());
#else
    (void)soa;
#endif
}

std::vector<std::pair<int, int>> BroadPhase::detect_overlaps_partial(
    bool two_lists,
    uint32_t start_box_id,
    uint32_t max_overlap_cutoff,
    uint32_t overlaps_capacity)
{
#if SCALABLE_CCD_METAL_ENABLED
    last_gpu_ms_ = -1.0;
    last_real_count_ = 0;
    if (nboxes_ == 0 || max_overlap_cutoff == 0 || start_box_id >= nboxes_) {
        return {};
    }
    auto dev = impl_->device.get();
    auto q = impl_->queue.get();

    // 输出缓冲与计数器
    auto buf_overlaps =
        NS::TransferPtr(dev->newBuffer(sizeof(int) * 2 * overlaps_capacity, MTL::ResourceStorageModeShared));
    auto buf_counter =
        NS::TransferPtr(dev->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared));
    *reinterpret_cast<uint32_t*>(buf_counter->contents()) = 0u;

    auto cmd = NS::TransferPtr(q->commandBuffer());
    auto enc = NS::TransferPtr(cmd->computeCommandEncoder());
    NS::SharedPtr<MTL::ComputePipelineState> pso_ptr;
    if (use_stq_) {
        pso_ptr = two_lists ? impl_->pso_two_stq : impl_->pso_one_stq;
    } else {
        pso_ptr = two_lists ? impl_->pso_two : impl_->pso_one;
    }
    if (!pso_ptr) {
        pso_ptr = two_lists ? impl_->pso_two : impl_->pso_one;
    }
    auto pso = pso_ptr.get();
    enc->setComputePipelineState(pso);

    enc->setBuffer(impl_->buf_sorted_major.get(), 0, 0);
    enc->setBuffer(impl_->buf_mini_min.get(), 0, 1);
    enc->setBuffer(impl_->buf_mini_max.get(), 0, 2);
    enc->setBuffer(impl_->buf_vertex_ids.get(), 0, 3);
    enc->setBuffer(impl_->buf_element_ids.get(), 0, 4);

    enc->setBytes(&nboxes_, sizeof(uint32_t), 5);
    enc->setBytes(&start_box_id, sizeof(uint32_t), 6);
    enc->setBytes(&max_overlap_cutoff, sizeof(uint32_t), 7);
    enc->setBuffer(buf_overlaps.get(), 0, 8);
    enc->setBuffer(buf_counter.get(), 0, 9);
    enc->setBytes(&overlaps_capacity, sizeof(uint32_t), 10);

    // 每个线程处理一个起点 i
    uint32_t nthreads = std::min(
        max_overlap_cutoff, static_cast<uint32_t>(nboxes_ > start_box_id ? (nboxes_ - start_box_id) : 0));
    if (nthreads == 0) {
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        return {};
    }
    const bool stq = use_stq_;
    uint32_t tg_limit = stq ? 32u : 64u;
    tg_limit = get_env_u32(
        stq ? "SCALABLE_CCD_METAL_STQ_TG" : "SCALABLE_CCD_METAL_SAP_TG",
        tg_limit);
    const NS::UInteger w = pso->threadExecutionWidth();
    NS::UInteger tg_w = tg_limit;
    if (w > 0) {
        tg_w = std::min<NS::UInteger>(w, tg_limit);
    }
    if (tg_w == 0) {
        tg_w = tg_limit;
    }
    tg_w = std::max<NS::UInteger>(1, tg_w);
    MTL::Size grid(nthreads, 1, 1);
    MTL::Size tg(tg_w, 1, 1);
    enc->dispatchThreads(grid, tg);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    // GPU 时间（毫秒），优先使用 GPUStart/EndTime；回退到 kernelStart/EndTime
    {
        const double t0 = cmd->GPUStartTime();
        const double t1 = cmd->GPUEndTime();
        if (t1 > t0 && t0 > 0.0) {
            last_gpu_ms_ = (t1 - t0) * 1000.0;
        } else {
            const double k0 = cmd->kernelStartTime();
            const double k1 = cmd->kernelEndTime();
            if (k1 > k0 && k0 > 0.0) {
                last_gpu_ms_ = (k1 - k0) * 1000.0;
            } else {
                last_gpu_ms_ = -1.0;
            }
        }
    }

    uint32_t real_count = *reinterpret_cast<uint32_t*>(buf_counter->contents());
    last_real_count_ = real_count;
    uint32_t written = std::min<uint32_t>(real_count, overlaps_capacity);
    std::vector<std::pair<int, int>> out;
    out.resize(written);
    auto p = reinterpret_cast<const int32_t*>(buf_overlaps->contents());
    for (uint32_t i = 0; i < written; ++i) {
        out[i] = { p[2 * i + 0], p[2 * i + 1] };
    }
    return out;
#else
    (void)two_lists;
    (void)start_box_id;
    (void)max_overlap_cutoff;
    (void)overlaps_capacity;
    throw std::runtime_error("Metal backend not available on this platform");
#endif
}

std::vector<std::pair<int, int>> BroadPhase::detect_overlaps(
    bool two_lists,
    uint32_t max_overlap_cutoff,
    uint32_t overlaps_capacity_hint)
{
#if SCALABLE_CCD_METAL_ENABLED
    std::vector<std::pair<int, int>> all;
    if (nboxes_ == 0) {
        return all;
    }
    // 默认 cutoff = 总数
    if (max_overlap_cutoff == 0) {
        max_overlap_cutoff = nboxes_;
    }
    // 默认容量估算
    uint32_t capacity = overlaps_capacity_hint;
    if (capacity == 0) {
        capacity =
            static_cast<uint32_t>(std::max<uint64_t>(nboxes_ / 2 + 4096, 4096));
    }

    uint32_t start = 0;
    while (start < nboxes_) {
        bool done = false;
        uint32_t cutoff = std::min<uint32_t>(max_overlap_cutoff, nboxes_ - start);
        while (!done) {
            auto batch = detect_overlaps_partial(
                two_lists, start, cutoff, capacity);
            if (last_real_count_ > capacity) {
                // 扩容重试
                capacity = static_cast<uint32_t>(last_real_count_ * 1.2 + 4096);
                continue;
            }
            // 成功
            all.insert(all.end(), batch.begin(), batch.end());
            done = true;
        }
        start += cutoff;
    }
    return all;
#else
    (void)two_lists;
    (void)max_overlap_cutoff;
    (void)overlaps_capacity_hint;
    throw std::runtime_error("Metal backend not available on this platform");
#endif
}

} // namespace metal
} // namespace scalable_ccd
