#include "runtime.hpp"

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <chrono>
#include <cstring>

namespace scalable_ccd::metalcpp {

namespace {
static const char* kNoopSrc = R"(
using namespace metal;
kernel void noop(device int* out [[buffer(0)]]) {
  out[0] = 1;
}
)";

static const char* kYZFilterSrc = R"(
using namespace metal;
kernel void yzFilter(
  device const float* minY [[buffer(0)]],
  device const float* maxY [[buffer(1)]],
  device const float* minZ [[buffer(2)]],
  device const float* maxZ [[buffer(3)]],
  device const int3*   vids [[buffer(4)]],   // (v0,v1,v2) per box
  device const int2* pairs [[buffer(5)]],
  constant uint& nPairs [[buffer(6)]],
  constant uint& twoLists [[buffer(7)]],
  device uchar* outMask [[buffer(8)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= nPairs) return;
  int2 pr = pairs[gid];
  int i = pr.x, j = pr.y;
  float minYi = minY[i], maxYi = maxY[i];
  float minYj = minY[j], maxYj = maxY[j];
  float minZi = minZ[i], maxZi = maxZ[i];
  float minZj = minZ[j], maxZj = maxZ[j];
  bool overlapY = !(maxYi < minYj || minYi > maxYj);
  bool overlapZ = !(maxZi < minZj || minZi > maxZj);
  if (!(overlapY && overlapZ)) { outMask[gid] = 0; return; }
  // 与 CPU 一致：无论单/双列表均剔除共享顶点
  int3 ai = vids[i];
  int3 aj = vids[j];
  bool share =
    (ai.x == aj.x) || (ai.x == aj.y) || (ai.x == aj.z) ||
    (ai.y == aj.x) || (ai.y == aj.y) || (ai.y == aj.z) ||
    (ai.z == aj.x) || (ai.z == aj.y) || (ai.z == aj.z);
  outMask[gid] = share ? 0 : 1;
}
)";
} // namespace

struct MetalCppRuntime::Impl {
    MTL::Device* dev = nullptr;
    MTL::CommandQueue* q = nullptr;
    MTL::ComputePipelineState* cpsNoop = nullptr;
    MTL::ComputePipelineState* cpsYZ = nullptr;
    double lastYZMs = -1.0;
};

MetalCppRuntime& MetalCppRuntime::instance()
{
    static MetalCppRuntime inst;
    return inst;
}

MetalCppRuntime::MetalCppRuntime()
{
    impl_ = new Impl();
    impl_->dev = MTL::CreateSystemDefaultDevice();
    if (!impl_->dev) return;
    impl_->q = impl_->dev->newCommandQueue();
}

MetalCppRuntime::~MetalCppRuntime()
{
    if (impl_) {
        if (impl_->cpsYZ) { impl_->cpsYZ->release(); impl_->cpsYZ = nullptr; }
        if (impl_->cpsNoop) { impl_->cpsNoop->release(); impl_->cpsNoop = nullptr; }
        if (impl_->q) { impl_->q->release(); impl_->q = nullptr; }
        if (impl_->dev) { impl_->dev->release(); impl_->dev = nullptr; }
    }
    delete impl_;
}

bool MetalCppRuntime::available() const
{
    return impl_ && impl_->dev && impl_->q;
}

static MTL::Library* compileLibrary(MTL::Device* dev, const char* src)
{
    NS::Error* err = nullptr;
    auto nsSrc = NS::String::string(src, NS::UTF8StringEncoding);
    auto opts = MTL::CompileOptions::alloc()->init();
    auto lib = dev->newLibrary(nsSrc, opts, &err);
    opts->release();
    return lib;
}

bool MetalCppRuntime::warmup()
{
    if (!available()) return false;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    if (!impl_->cpsNoop) {
        auto lib = compileLibrary(impl_->dev, kNoopSrc);
        if (!lib) { pool->release(); return false; }
        auto fn = lib->newFunction(NS::String::string("noop", NS::UTF8StringEncoding));
        if (!fn) { lib->release(); pool->release(); return false; }
        NS::Error* err = nullptr;
        auto cps = impl_->dev->newComputePipelineState(fn, &err);
        fn->release();
        lib->release();
        if (!cps) { pool->release(); return false; }
        impl_->cpsNoop = cps;
    }
    auto out = impl_->dev->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
    auto cb = impl_->q->commandBuffer();
    auto enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(impl_->cpsNoop);
    enc->setBuffer(out, 0, 0);
    auto tgW = impl_->cpsNoop->maxTotalThreadsPerThreadgroup();
    if (tgW == 0) tgW = 1;
    MTL::Size grid = MTL::Size(1,1,1);
    MTL::Size tg = MTL::Size(1,1,1);
    enc->dispatchThreads(grid, tg);
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    int v = *reinterpret_cast<int*>(out->contents());
    enc->release();
    cb->release();
    out->release();
    pool->release();
    return v == 1;
}

bool MetalCppRuntime::filterYZ(
    const std::vector<float>& minY,
    const std::vector<float>& maxY,
    const std::vector<float>& minZ,
    const std::vector<float>& maxZ,
    const std::vector<int32_t>& v0,
    const std::vector<int32_t>& v1,
    const std::vector<int32_t>& v2,
    const std::vector<std::pair<int,int>>& pairs,
    bool two_lists,
    std::vector<uint8_t>& outMask)
{
    if (!available()) return false;
    if (minY.size()!=maxY.size() || minZ.size()!=maxZ.size()) return false;
    if (v0.size()!=minY.size() || v1.size()!=minY.size() || v2.size()!=minY.size()) return false;
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    if (!impl_->cpsYZ) {
        auto lib = compileLibrary(impl_->dev, kYZFilterSrc);
        if (!lib) { pool->release(); return false; }
        auto fn = lib->newFunction(NS::String::string("yzFilter", NS::UTF8StringEncoding));
        if (!fn) { lib->release(); pool->release(); return false; }
        NS::Error* err = nullptr;
        auto cps = impl_->dev->newComputePipelineState(fn, &err);
        fn->release();
        lib->release();
        if (!cps) { pool->release(); return false; }
        impl_->cpsYZ = cps;
    }
    size_t n = minY.size();
    size_t m = pairs.size();
    outMask.assign(m, 0);
    if (m == 0) return true;
    auto bMinY = impl_->dev->newBuffer(sizeof(float)*n, MTL::ResourceStorageModeShared);
    auto bMaxY = impl_->dev->newBuffer(sizeof(float)*n, MTL::ResourceStorageModeShared);
    auto bMinZ = impl_->dev->newBuffer(sizeof(float)*n, MTL::ResourceStorageModeShared);
    auto bMaxZ = impl_->dev->newBuffer(sizeof(float)*n, MTL::ResourceStorageModeShared);
    std::memcpy(bMinY->contents(), minY.data(), sizeof(float)*n);
    std::memcpy(bMaxY->contents(), maxY.data(), sizeof(float)*n);
    std::memcpy(bMinZ->contents(), minZ.data(), sizeof(float)*n);
    std::memcpy(bMaxZ->contents(), maxZ.data(), sizeof(float)*n);
    struct I3 { int32_t x; int32_t y; int32_t z; };
    auto bVids = impl_->dev->newBuffer(sizeof(I3)*n, MTL::ResourceStorageModeShared);
    I3* vids = reinterpret_cast<I3*>(bVids->contents());
    for (size_t i=0;i<n;++i) vids[i] = I3{v0[i],v1[i],v2[i]};
    struct I2 { int32_t x; int32_t y; };
    auto bPairs = impl_->dev->newBuffer(sizeof(I2)*m, MTL::ResourceStorageModeShared);
    I2* pp = reinterpret_cast<I2*>(bPairs->contents());
    for (size_t i=0;i<m;++i) pp[i] = I2{pairs[i].first, pairs[i].second};
    uint32_t mU32 = static_cast<uint32_t>(m);
    auto bM = impl_->dev->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    *reinterpret_cast<uint32_t*>(bM->contents()) = mU32;
    uint32_t two = two_lists ? 1u : 0u;
    auto bTwo = impl_->dev->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    *reinterpret_cast<uint32_t*>(bTwo->contents()) = two;
    auto bMask = impl_->dev->newBuffer(sizeof(uint8_t)*m, MTL::ResourceStorageModeShared);

    auto cb = impl_->q->commandBuffer();
    auto enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(impl_->cpsYZ);
    enc->setBuffer(bMinY, 0, 0);
    enc->setBuffer(bMaxY, 0, 1);
    enc->setBuffer(bMinZ, 0, 2);
    enc->setBuffer(bMaxZ, 0, 3);
    enc->setBuffer(bVids, 0, 4);
    enc->setBuffer(bPairs, 0, 5);
    enc->setBuffer(bM, 0, 6);
    enc->setBuffer(bTwo, 0, 7);
    enc->setBuffer(bMask, 0, 8);
    auto w = impl_->cpsYZ->maxTotalThreadsPerThreadgroup();
    if (w == 0) w = 64;
    MTL::Size grid = MTL::Size(m,1,1);
    MTL::Size tg = MTL::Size(std::min<NS::UInteger>(w, m),1,1);
    enc->dispatchThreads(grid, tg);
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    std::memcpy(outMask.data(), bMask->contents(), sizeof(uint8_t)*m);
    // release resources
    enc->release();
    cb->release();
    bMask->release();
    bTwo->release();
    bM->release();
    bPairs->release();
    bVids->release();
    bMaxZ->release();
    bMinZ->release();
    bMaxY->release();
    bMinY->release();
    // 计时采集：Metal-cpp 暂不直接提供 GPUStartTime/GPUEndTime
    impl_->lastYZMs = -1.0;
    pool->release();
    return true;
}

double MetalCppRuntime::lastYZFilterMs() const { return impl_ ? impl_->lastYZMs : -1.0; }

} // namespace scalable_ccd::metalcpp

