// Metal2 runtime minimal implementation: warmup + yzFilter kernel
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "runtime.hpp"

namespace scalable_ccd::metal2 {

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

static const char* kSTQTwoListsSrc = R"(
using namespace metal;
kernel void stqTwo(
  device const float* minX [[buffer(0)]],
  device const float* maxX [[buffer(1)]],
  device const uchar*  listTag [[buffer(2)]], // 1 for A, 0 for B
  constant uint& baseI [[buffer(3)]],
  constant uint& n [[buffer(4)]],
  constant uint& maxN [[buffer(5)]],          // per-i neighbor cap
  device atomic_uint* gCount [[buffer(6)]],
  device int2* outPairs [[buffer(7)]],
  device atomic_uint* gSat [[buffer(8)]],
  uint gid [[thread_position_in_grid]]
) {
  uint i = baseI + gid;
  if (i >= n) return;
  uint emitted = 0;
  float amax = maxX[i];
  uchar tagi = listTag[i];
  for (uint j = i + 1; j < n; ++j) {
    float bmin = minX[j];
    if (amax < bmin) break; // no more overlap along X
    if (tagi == listTag[j]) continue; // only cross-list
    uint pos = atomic_fetch_add_explicit(gCount, 1u, memory_order_relaxed);
    outPairs[pos] = int2((int)i, (int)j);
    emitted++;
    if (emitted >= maxN) break;
  }
  if (emitted >= maxN) { atomic_store_explicit(gSat, 1u, memory_order_relaxed); }
}
)";

static const char* kSTQTwoPTQSrc = R"(
using namespace metal;
kernel void stqTwoPTQ(
  device const float* minX [[buffer(0)]],
  device const float* maxX [[buffer(1)]],
  device const uchar*  listTag [[buffer(2)]], // 1 for A, 0 for B
  constant uint& startI [[buffer(3)]],
  constant uint& endI [[buffer(4)]],          // exclusive
  constant uint& maxN [[buffer(5)]],          // per-i neighbor cap
  device atomic_uint* gHead [[buffer(6)]],    // global queue head
  device atomic_uint* gCount [[buffer(7)]],   // global pair counter
  device int2* outPairs [[buffer(8)]],
  device atomic_uint* gSat [[buffer(9)]],
  uint tid [[thread_position_in_grid]]
) {
  while (true) {
    uint idx = atomic_fetch_add_explicit(gHead, 1u, memory_order_relaxed);
    uint i = startI + idx;
    if (i >= endI) break;
    uint emitted = 0;
    float amax = maxX[i];
    uchar tagi = listTag[i];
    for (uint j = i + 1; j < endI; ++j) {
      float bmin = minX[j];
      if (amax < bmin) break;
      if (tagi == listTag[j]) continue;
      uint pos = atomic_fetch_add_explicit(gCount, 1u, memory_order_relaxed);
      outPairs[pos] = int2((int)i, (int)j);
      emitted++;
      if (emitted >= maxN) break;
    }
    if (emitted >= maxN) { atomic_store_explicit(gSat, 1u, memory_order_relaxed); }
  }
}
)";

static const char* kSTQSingleListSrc = R"(
using namespace metal;
kernel void stqSingle(
  device const float* minX [[buffer(0)]],
  device const float* maxX [[buffer(1)]],
  constant uint& baseI [[buffer(2)]],
  constant uint& n [[buffer(3)]],
  constant uint& maxN [[buffer(4)]],
  device atomic_uint* gCount [[buffer(5)]],
  device int2* outPairs [[buffer(6)]],
  device atomic_uint* gSat [[buffer(7)]],
  uint gid [[thread_position_in_grid]]
) {
  uint i = baseI + gid;
  if (i >= n) return;
  uint emitted = 0;
  float amax = maxX[i];
  for (uint j = i + 1; j < n; ++j) {
    float bmin = minX[j];
    if (amax < bmin) break;
    uint pos = atomic_fetch_add_explicit(gCount, 1u, memory_order_relaxed);
    outPairs[pos] = int2((int)i, (int)j);
    emitted++;
    if (emitted >= maxN) break;
  }
  if (emitted >= maxN) { atomic_store_explicit(gSat, 1u, memory_order_relaxed); }
}
)";

static const char* kSTQSinglePTQSrc = R"(
using namespace metal;
kernel void stqSinglePTQ(
  device const float* minX [[buffer(0)]],
  device const float* maxX [[buffer(1)]],
  constant uint& startI [[buffer(2)]],
  constant uint& endI [[buffer(3)]],
  constant uint& maxN [[buffer(4)]],
  device atomic_uint* gHead [[buffer(5)]],
  device atomic_uint* gCount [[buffer(6)]],
  device int2* outPairs [[buffer(7)]],
  device atomic_uint* gSat [[buffer(8)]],
  uint tid [[thread_position_in_grid]]
) {
  while (true) {
    uint idx = atomic_fetch_add_explicit(gHead, 1u, memory_order_relaxed);
    uint i = startI + idx;
    if (i >= endI) break;
    uint emitted = 0;
    float amax = maxX[i];
    for (uint j = i + 1; j < endI; ++j) {
      float bmin = minX[j];
      if (amax < bmin) break;
      uint pos = atomic_fetch_add_explicit(gCount, 1u, memory_order_relaxed);
      outPairs[pos] = int2((int)i, (int)j);
      emitted++;
      if (emitted >= maxN) break;
    }
    if (emitted >= maxN) { atomic_store_explicit(gSat, 1u, memory_order_relaxed); }
  }
}
)";
} // namespace

struct Metal2Runtime::Impl {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> q = nil;
    id<MTLLibrary> libNoop = nil;
    id<MTLComputePipelineState> cpsNoop = nil;
    id<MTLLibrary> libYZ = nil;
    id<MTLComputePipelineState> cpsYZ = nil;
    id<MTLLibrary> libSTQTwo = nil;
    id<MTLComputePipelineState> cpsSTQTwo = nil;
    id<MTLLibrary> libSTQTwoPTQ = nil;
    id<MTLComputePipelineState> cpsSTQTwoPTQ = nil;
    id<MTLLibrary> libSTQSingle = nil;
    id<MTLComputePipelineState> cpsSTQSingle = nil;
    id<MTLLibrary> libSTQSinglePTQ = nil;
    id<MTLComputePipelineState> cpsSTQSinglePTQ = nil;
    bool ok = false;
    double lastYZMs = -1.0;
    double lastPairsMs = -1.0;
};

Metal2Runtime& Metal2Runtime::instance()
{
    static Metal2Runtime inst;
    return inst;
}

Metal2Runtime::Metal2Runtime()
{
    @autoreleasepool {
        impl_ = new Impl();
        impl_->dev = MTLCreateSystemDefaultDevice();
        if (!impl_->dev) return;
        impl_->q = [impl_->dev newCommandQueue];
        if (!impl_->q) return;
        impl_->ok = true;
    }
}

Metal2Runtime::~Metal2Runtime()
{
    delete impl_;
}

bool Metal2Runtime::available() const
{
    return impl_ && impl_->ok && impl_->dev && impl_->q;
}

bool Metal2Runtime::warmup()
{
    if (!available()) return false;
    @autoreleasepool {
        NSError* err = nil;
        if (!impl_->libNoop) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            impl_->libNoop = [impl_->dev newLibraryWithSource:[NSString stringWithUTF8String:kNoopSrc] options:opts error:&err];
            if (!impl_->libNoop) return false;
            id<MTLFunction> fn = [impl_->libNoop newFunctionWithName:@"noop"];
            if (!fn) return false;
            impl_->cpsNoop = [impl_->dev newComputePipelineStateWithFunction:fn error:&err];
            if (!impl_->cpsNoop) return false;
        }
        id<MTLBuffer> out = [impl_->dev newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [impl_->q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->cpsNoop];
        [enc setBuffer:out offset:0 atIndex:0];
        MTLSize grid = MTLSizeMake(1,1,1);
        MTLSize tg = MTLSizeMake(1,1,1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        int v = *static_cast<int*>([out contents]);
        return v == 1;
    }
}

bool Metal2Runtime::filterYZ(
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
    @autoreleasepool {
        NSError* err = nil;
        if (!impl_->cpsYZ) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            impl_->libYZ = [impl_->dev newLibraryWithSource:[NSString stringWithUTF8String:kYZFilterSrc] options:opts error:&err];
            if (!impl_->libYZ) return false;
            id<MTLFunction> fn = [impl_->libYZ newFunctionWithName:@"yzFilter"];
            if (!fn) return false;
            impl_->cpsYZ = [impl_->dev newComputePipelineStateWithFunction:fn error:&err];
            if (!impl_->cpsYZ) return false;
        }
        const size_t n = minY.size();
        const size_t m = pairs.size();
        outMask.assign(m, 0);
        if (m == 0) return true;
        id<MTLBuffer> bMinY = [impl_->dev newBufferWithBytes:minY.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxY = [impl_->dev newBufferWithBytes:maxY.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMinZ = [impl_->dev newBufferWithBytes:minZ.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxZ = [impl_->dev newBufferWithBytes:maxZ.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        struct I3 { int32_t x; int32_t y; int32_t z; };
        std::vector<I3> vids(n);
        for (size_t i=0;i<n;++i){ vids[i] = I3{ v0[i], v1[i], v2[i] }; }
        id<MTLBuffer> bVids = [impl_->dev newBufferWithBytes:vids.data() length:sizeof(I3)*n options:MTLResourceStorageModeShared];
        struct I2 { int32_t x; int32_t y; };
        std::vector<I2> pp(m);
        for (size_t i=0;i<m;++i){ pp[i] = I2{ pairs[i].first, pairs[i].second }; }
        id<MTLBuffer> bPairs = [impl_->dev newBufferWithBytes:pp.data() length:sizeof(I2)*m options:MTLResourceStorageModeShared];
        uint32_t mU32 = static_cast<uint32_t>(m);
        id<MTLBuffer> bM = [impl_->dev newBufferWithBytes:&mU32 length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        uint32_t two = two_lists ? 1u : 0u;
        id<MTLBuffer> bTwo = [impl_->dev newBufferWithBytes:&two length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMask = [impl_->dev newBufferWithLength:sizeof(uint8_t)*m options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [impl_->q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->cpsYZ];
        [enc setBuffer:bMinY offset:0 atIndex:0];
        [enc setBuffer:bMaxY offset:0 atIndex:1];
        [enc setBuffer:bMinZ offset:0 atIndex:2];
        [enc setBuffer:bMaxZ offset:0 atIndex:3];
        [enc setBuffer:bVids offset:0 atIndex:4];
        [enc setBuffer:bPairs offset:0 atIndex:5];
        [enc setBuffer:bM offset:0 atIndex:6];
        [enc setBuffer:bTwo offset:0 atIndex:7];
        [enc setBuffer:bMask offset:0 atIndex:8];
        NSUInteger w = impl_->cpsYZ.maxTotalThreadsPerThreadgroup;
        if (w == 0) w = 64;
        MTLSize grid = MTLSizeMake(m,1,1);
        MTLSize tg = MTLSizeMake(std::min<NSUInteger>(w, m),1,1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(outMask.data(), [bMask contents], sizeof(uint8_t)*m);
        // timing
        CFTimeInterval t0 = cb.GPUStartTime, t1 = cb.GPUEndTime;
        if (t1 > t0 && t0 > 0) {
            impl_->lastYZMs = (t1 - t0) * 1000.0;
        } else {
            impl_->lastYZMs = -1.0;
        }
        return true;
    }
}

bool Metal2Runtime::stqTwoLists(
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
    std::vector<std::pair<int,int>>& outPairs)
{
    if (!available()) return false;
    const size_t n = minX.size();
    if (n == 0) { outPairs.clear(); return true; }
    if (maxX.size()!=n || listTag.size()!=n) return false;
    @autoreleasepool {
        NSError* err = nil;
        // 编译两种内核（per-i 与 PTQ），按环境变量选择
        auto ensurePipelines = [&]()->bool {
            if (!impl_->cpsSTQTwo) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl_->libSTQTwo = [impl_->dev newLibraryWithSource:[NSString stringWithUTF8String:kSTQTwoListsSrc] options:opts error:&err];
                if (!impl_->libSTQTwo) return false;
                id<MTLFunction> fn = [impl_->libSTQTwo newFunctionWithName:@"stqTwo"];
                if (!fn) return false;
                impl_->cpsSTQTwo = [impl_->dev newComputePipelineStateWithFunction:fn error:&err];
                if (!impl_->cpsSTQTwo) return false;
            }
            if (!impl_->cpsSTQTwoPTQ) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl_->libSTQTwoPTQ = [impl_->dev newLibraryWithSource:[NSString stringWithUTF8String:kSTQTwoPTQSrc] options:opts error:&err];
                if (!impl_->libSTQTwoPTQ) return false;
                id<MTLFunction> fn = [impl_->libSTQTwoPTQ newFunctionWithName:@"stqTwoPTQ"];
                if (!fn) return false;
                impl_->cpsSTQTwoPTQ = [impl_->dev newComputePipelineStateWithFunction:fn error:&err];
                if (!impl_->cpsSTQTwoPTQ) return false;
            }
            return true;
        };
        if (!ensurePipelines()) return false;
        // 参数：每个 i 允许最多发 maxN 个邻居，按 chunkI 分批
        auto envU = [](const char* k, uint32_t def)->uint32_t{
            const char* v = std::getenv(k); if(!v) return def;
            try { long long t = std::stoll(v); return t>0 ? (uint32_t)t : def; } catch(...) { return def; }
        };
        auto envB = [](const char* k, bool def)->bool{
            const char* v = std::getenv(k); if(!v) return def;
            std::string s(v); for(auto&c:s) c=(char)tolower((unsigned char)c);
            if (s=="0"||s=="false"||s=="off"||s=="no") return false;
            return true;
        };
        auto envD = [](const char* k, double def)->double{
            const char* v = std::getenv(k); if(!v) return def;
            try { return std::stod(v); } catch(...) { return def; }
        };
        uint32_t maxN = envU("SCALABLE_CCD_METAL2_STQ_MAX_NEIGHBORS", 64);
        uint32_t chunkI = envU("SCALABLE_CCD_METAL2_STQ_CHUNK_I", 8192);
        bool usePTQ = envB("SCALABLE_CCD_METAL2_STQ_PERSISTENT", true);
        uint32_t ptqThreads = envU("SCALABLE_CCD_METAL2_STQ_PERSIST_THREADS", 2048);
        double timeout_ms = envD("SCALABLE_CCD_METAL2_STQ_TIMEOUT_MS", 200.0);
        impl_->lastPairsMs = -1.0;
        outPairs.clear();
        // 设备缓冲（共享内存）
        std::vector<float> minXf(n), maxXf(n);
        for (size_t i=0;i<n;++i){ minXf[i]=(float)minX[i]; maxXf[i]=(float)maxX[i]; }
        id<MTLBuffer> bMinX = [impl_->dev newBufferWithBytes:minXf.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxX = [impl_->dev newBufferWithBytes:maxXf.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bTag  = [impl_->dev newBufferWithBytes:listTag.data() length:sizeof(uint8_t)*n options:MTLResourceStorageModeShared];

        double acc_ms = 0.0;
        bool aborted = false;
        for (uint32_t base = 0; base < n; base += chunkI) {
            uint32_t cur = std::min<uint32_t>(chunkI, static_cast<uint32_t>(n - base));
            uint64_t cap = static_cast<uint64_t>(cur) * static_cast<uint64_t>(maxN);
            if (cap == 0) continue;
            uint32_t startI = base;
            uint32_t endI = base + cur;
            uint32_t zero = 0;
            id<MTLBuffer> bMaxN = [impl_->dev newBufferWithBytes:&maxN length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bPairs = [impl_->dev newBufferWithLength:sizeof(int32_t)*2*cap options:MTLResourceStorageModeShared];
            id<MTLBuffer> bCnt  = [impl_->dev newBufferWithBytes:&zero length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bSat  = [impl_->dev newBufferWithBytes:&zero length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [impl_->q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (usePTQ) {
                id<MTLBuffer> bStart = [impl_->dev newBufferWithBytes:&startI length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                id<MTLBuffer> bEnd   = [impl_->dev newBufferWithBytes:&endI length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                id<MTLBuffer> bHead  = [impl_->dev newBufferWithBytes:&zero length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                [enc setComputePipelineState:impl_->cpsSTQTwoPTQ];
                [enc setBuffer:bMinX offset:0 atIndex:0];
                [enc setBuffer:bMaxX offset:0 atIndex:1];
                [enc setBuffer:bTag  offset:0 atIndex:2];
                [enc setBuffer:bStart offset:0 atIndex:3];
                [enc setBuffer:bEnd   offset:0 atIndex:4];
                [enc setBuffer:bMaxN  offset:0 atIndex:5];
                [enc setBuffer:bHead  offset:0 atIndex:6];
                [enc setBuffer:bCnt   offset:0 atIndex:7];
                [enc setBuffer:bPairs offset:0 atIndex:8];
                [enc setBuffer:bSat   offset:0 atIndex:9];
                NSUInteger w = impl_->cpsSTQTwoPTQ.maxTotalThreadsPerThreadgroup;
                if (w == 0) w = 64;
                NSUInteger tgSize = std::min<NSUInteger>(w, 256);
                NSUInteger gridSize = std::min<NSUInteger>(ptqThreads, ptqThreads - (ptqThreads % tgSize) + tgSize);
                MTLSize grid = MTLSizeMake(gridSize,1,1);
                MTLSize tg   = MTLSizeMake(tgSize,1,1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            } else {
                id<MTLBuffer> bBase = [impl_->dev newBufferWithBytes:&base length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                id<MTLBuffer> bN    = [impl_->dev newBufferWithBytes:&n length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                [enc setComputePipelineState:impl_->cpsSTQTwo];
                [enc setBuffer:bMinX offset:0 atIndex:0];
                [enc setBuffer:bMaxX offset:0 atIndex:1];
                [enc setBuffer:bTag  offset:0 atIndex:2];
                [enc setBuffer:bBase offset:0 atIndex:3];
                [enc setBuffer:bN    offset:0 atIndex:4];
                [enc setBuffer:bMaxN offset:0 atIndex:5];
                [enc setBuffer:bCnt  offset:0 atIndex:6];
                [enc setBuffer:bPairs offset:0 atIndex:7];
                [enc setBuffer:bSat   offset:0 atIndex:8];
                NSUInteger w = impl_->cpsSTQTwo.maxTotalThreadsPerThreadgroup;
                if (w == 0) w = 64;
                MTLSize grid = MTLSizeMake(cur,1,1);
                MTLSize tg   = MTLSizeMake(std::min<NSUInteger>(w, cur),1,1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            CFTimeInterval t0 = cb.GPUStartTime, t1 = cb.GPUEndTime;
            double ms = (t1 > t0 && t0 > 0) ? (t1 - t0) * 1000.0 : 0.0;
            if (impl_->lastPairsMs < 0) impl_->lastPairsMs = 0.0;
            impl_->lastPairsMs += ms;
            acc_ms += ms;
            // 读回 pairs
            uint32_t cnt = *static_cast<uint32_t*>([bCnt contents]);
            if (cnt > cap) cnt = (uint32_t)cap; // 防御
            uint32_t sat = *static_cast<uint32_t*>([bSat contents]);
            struct I2 { int32_t x; int32_t y; };
            I2* pp = static_cast<I2*>([bPairs contents]);
            outPairs.reserve(outPairs.size() + cnt);
            for (uint32_t i = 0; i < cnt; ++i) outPairs.emplace_back(pp[i].x, pp[i].y);
            if (cnt >= cap || sat) { aborted = true; break; } // 溢出或饱和，回退 CPU
            if (timeout_ms > 0.0 && acc_ms > timeout_ms) { aborted = true; break; } // 超时
        }
        if (aborted) { outPairs.clear(); return false; }
        return true;
    }
}

bool Metal2Runtime::stqSingleList(
    const std::vector<double>& minX,
    const std::vector<double>& maxX,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<double>&,
    const std::vector<int32_t>&,
    const std::vector<int32_t>&,
    const std::vector<int32_t>&,
    std::vector<std::pair<int,int>>& outPairs)
{
    if (!available()) return false;
    const size_t n = minX.size();
    if (n == 0) { outPairs.clear(); return true; }
    if (maxX.size()!=n) return false;
    @autoreleasepool {
        NSError* err = nil;
        auto ensurePipelines = [&]()->bool{
            if (!impl_->cpsSTQSingle) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl_->libSTQSingle = [impl_->dev newLibraryWithSource:[NSString stringWithUTF8String:kSTQSingleListSrc] options:opts error:&err];
                if (!impl_->libSTQSingle) return false;
                id<MTLFunction> fn = [impl_->libSTQSingle newFunctionWithName:@"stqSingle"];
                if (!fn) return false;
                impl_->cpsSTQSingle = [impl_->dev newComputePipelineStateWithFunction:fn error:&err];
                if (!impl_->cpsSTQSingle) return false;
            }
            if (!impl_->cpsSTQSinglePTQ) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl_->libSTQSinglePTQ = [impl_->dev newLibraryWithSource:[NSString stringWithUTF8String:kSTQSinglePTQSrc] options:opts error:&err];
                if (!impl_->libSTQSinglePTQ) return false;
                id<MTLFunction> fn = [impl_->libSTQSinglePTQ newFunctionWithName:@"stqSinglePTQ"];
                if (!fn) return false;
                impl_->cpsSTQSinglePTQ = [impl_->dev newComputePipelineStateWithFunction:fn error:&err];
                if (!impl_->cpsSTQSinglePTQ) return false;
            }
            return true;
        };
        if (!ensurePipelines()) return false;
        auto envU = [](const char* k, uint32_t def)->uint32_t{
            const char* v = std::getenv(k); if(!v) return def;
            try { long long t = std::stoll(v); return t>0 ? (uint32_t)t : def; } catch(...) { return def; }
        };
        auto envB = [](const char* k, bool def)->bool{
            const char* v = std::getenv(k); if(!v) return def;
            std::string s(v); for(auto&c:s) c=(char)tolower((unsigned char)c);
            if (s=="0"||s=="false"||s=="off"||s=="no") return false;
            return true;
        };
        auto envD = [](const char* k, double def)->double{
            const char* v = std::getenv(k); if(!v) return def;
            try { return std::stod(v); } catch(...) { return def; }
        };
        uint32_t maxN = envU("SCALABLE_CCD_METAL2_STQ_MAX_NEIGHBORS", 64);
        uint32_t chunkI = envU("SCALABLE_CCD_METAL2_STQ_CHUNK_I", 8192);
        bool usePTQ = envB("SCALABLE_CCD_METAL2_STQ_PERSISTENT", true);
        uint32_t ptqThreads = envU("SCALABLE_CCD_METAL2_STQ_PERSIST_THREADS", 2048);
        double timeout_ms = envD("SCALABLE_CCD_METAL2_STQ_TIMEOUT_MS", 200.0);
        impl_->lastPairsMs = -1.0;
        outPairs.clear();
        std::vector<float> minXf(n), maxXf(n);
        for (size_t i=0;i<n;++i){ minXf[i]=(float)minX[i]; maxXf[i]=(float)maxX[i]; }
        id<MTLBuffer> bMinX = [impl_->dev newBufferWithBytes:minXf.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxX = [impl_->dev newBufferWithBytes:maxXf.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        double acc_ms = 0.0;
        bool aborted = false;
        for (uint32_t base = 0; base < n; base += chunkI) {
            uint32_t cur = std::min<uint32_t>(chunkI, static_cast<uint32_t>(n - base));
            uint64_t cap = static_cast<uint64_t>(cur) * static_cast<uint64_t>(maxN);
            if (cap == 0) continue;
            uint32_t startI = base;
            uint32_t endI = base + cur;
            uint32_t zero = 0;
            id<MTLBuffer> bMaxN = [impl_->dev newBufferWithBytes:&maxN length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> bPairs = [impl_->dev newBufferWithLength:sizeof(int32_t)*2*cap options:MTLResourceStorageModeShared];
            id<MTLBuffer> bCnt  = [impl_->dev newBufferWithBytes:&zero length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [impl_->q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (usePTQ) {
                id<MTLBuffer> bStart = [impl_->dev newBufferWithBytes:&startI length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                id<MTLBuffer> bEnd   = [impl_->dev newBufferWithBytes:&endI length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                id<MTLBuffer> bHead  = [impl_->dev newBufferWithBytes:&zero length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                [enc setComputePipelineState:impl_->cpsSTQSinglePTQ];
                [enc setBuffer:bMinX offset:0 atIndex:0];
                [enc setBuffer:bMaxX offset:0 atIndex:1];
                [enc setBuffer:bStart offset:0 atIndex:2];
                [enc setBuffer:bEnd   offset:0 atIndex:3];
                [enc setBuffer:bMaxN  offset:0 atIndex:4];
                [enc setBuffer:bHead  offset:0 atIndex:5];
                [enc setBuffer:bCnt   offset:0 atIndex:6];
                [enc setBuffer:bPairs offset:0 atIndex:7];
                [enc setBuffer:bCnt   offset:0 atIndex:8]; // reuse bCnt as gSat (init zero)
                NSUInteger w = impl_->cpsSTQSinglePTQ.maxTotalThreadsPerThreadgroup;
                if (w == 0) w = 64;
                NSUInteger tgSize = std::min<NSUInteger>(w, 256);
                NSUInteger gridSize = std::min<NSUInteger>(ptqThreads, ptqThreads - (ptqThreads % tgSize) + tgSize);
                MTLSize grid = MTLSizeMake(gridSize,1,1);
                MTLSize tg   = MTLSizeMake(tgSize,1,1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            } else {
                id<MTLBuffer> bBase = [impl_->dev newBufferWithBytes:&base length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                id<MTLBuffer> bN    = [impl_->dev newBufferWithBytes:&n length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                [enc setComputePipelineState:impl_->cpsSTQSingle];
                [enc setBuffer:bMinX offset:0 atIndex:0];
                [enc setBuffer:bMaxX offset:0 atIndex:1];
                [enc setBuffer:bBase offset:0 atIndex:2];
                [enc setBuffer:bN    offset:0 atIndex:3];
                [enc setBuffer:bMaxN offset:0 atIndex:4];
                [enc setBuffer:bCnt  offset:0 atIndex:5];
                [enc setBuffer:bPairs offset:0 atIndex:6];
                [enc setBuffer:bCnt  offset:0 atIndex:7]; // reuse as gSat
                NSUInteger w = impl_->cpsSTQSingle.maxTotalThreadsPerThreadgroup;
                if (w == 0) w = 64;
                MTLSize grid = MTLSizeMake(cur,1,1);
                MTLSize tg   = MTLSizeMake(std::min<NSUInteger>(w, cur),1,1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            CFTimeInterval t0 = cb.GPUStartTime, t1 = cb.GPUEndTime;
            double ms = (t1 > t0 && t0 > 0) ? (t1 - t0) * 1000.0 : 0.0;
            if (impl_->lastPairsMs < 0) impl_->lastPairsMs = 0.0;
            impl_->lastPairsMs += ms;
            acc_ms += ms;
            uint32_t cnt = *static_cast<uint32_t*>([bCnt contents]);
            if (cnt > cap) cnt = (uint32_t)cap;
            uint32_t sat = *static_cast<uint32_t*>([bCnt contents]); // reused as gSat
            struct I2 { int32_t x; int32_t y; };
            I2* pp = static_cast<I2*>([bPairs contents]);
            outPairs.reserve(outPairs.size() + cnt);
            for (uint32_t i = 0; i < cnt; ++i) outPairs.emplace_back(pp[i].x, pp[i].y);
            if (cnt >= cap || sat) { aborted = true; break; }
            if (timeout_ms > 0.0 && acc_ms > timeout_ms) { aborted = true; break; }
        }
        if (aborted) { outPairs.clear(); return false; }
        return true;
    }
}

double Metal2Runtime::lastYZFilterMs() const { return impl_ ? impl_->lastYZMs : -1.0; }
double Metal2Runtime::lastSTQPairsMs() const { return impl_ ? impl_->lastPairsMs : -1.0; }

} // namespace scalable_ccd::metal2
