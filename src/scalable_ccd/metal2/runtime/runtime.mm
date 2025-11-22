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
  float eps = 1e-5f;
  bool overlapY = !(maxYi < minYj - eps || minYi > maxYj + eps);
  bool overlapZ = !(maxZi < minZj - eps || minZi > maxZj + eps);
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

    static const char* kScanSrc = R"(
    #include <metal_stdlib>
    using namespace metal;

    // Hillis-Steele Scan (Inclusive)
    // Supports up to 1024 elements per threadgroup (typical max)
    // For larger arrays, we need a multi-block approach or multiple passes.
    // Here we implement a simple single-block scan for small arrays, 
    // and a multi-block scan structure for larger ones.
    
    // Single block scan (up to 1024 elements) - EXCLUSIVE
    // Input: uint8_t (bool mask), Output: uint32_t
    kernel void scanSingleBlock(
        device const uchar* inData [[buffer(0)]],
        device uint* outData [[buffer(1)]],
        uint tid [[thread_index_in_threadgroup]],
        uint gid [[thread_position_in_grid]],
        uint n [[threads_per_threadgroup]])
    {
        threadgroup uint temp[1024];
        uint myIn = (gid < n) ? (uint)inData[gid] : 0;
        temp[tid] = myIn;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint offset = 1; offset < n; offset <<= 1) {
            uint val = 0;
            if (tid >= offset) val = temp[tid - offset];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid >= offset) temp[tid] += val;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Write Exclusive Scan result
        if (gid < n) outData[gid] = temp[tid] - myIn;
    }
    
    // Blelloch Scan (Work-efficient) - Phase 1: Reduce (Upsweep)
    // Input: uint8_t
    kernel void scanBlellocReduce(
        device const uchar* inData [[buffer(0)]],
        device uint* outData [[buffer(1)]],
        device uint* blockSums [[buffer(2)]],
        device const uint& n [[buffer(3)]],
        uint tid [[thread_index_in_threadgroup]],
        uint gid [[thread_position_in_grid]],
        uint bid [[threadgroup_position_in_grid]],
        uint tpg [[threads_per_threadgroup]])
    {
        uint i = gid * 2;
        uint i1 = i;
        uint i2 = i + 1;
        
        threadgroup uint temp[1024]; // 2*512 threads -> 1024 elements
        
        // Load input
        temp[2*tid] = (i1 < n) ? (uint)inData[i1] : 0;
        temp[2*tid+1] = (i2 < n) ? (uint)inData[i2] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Upsweep
        uint offset = 1;
        for (uint d = tpg; d > 0; d >>= 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                temp[bi] += temp[ai];
            }
            offset <<= 1;
        }
        
        // Save block sum
        if (tid == 0) {
            if (blockSums) blockSums[bid] = temp[2*tpg - 1];
            temp[2*tpg - 1] = 0; // Clear last element for exclusive scan
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Downsweep
        for (uint d = 1; d <= tpg; d <<= 1) {
            offset >>= 1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < d) {
                uint ai = offset * (2 * tid + 1) - 1;
                uint bi = offset * (2 * tid + 2) - 1;
                uint t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Write output (Exclusive Scan result for this block)
        if (i1 < n) outData[i1] = temp[2*tid];
        if (i2 < n) outData[i2] = temp[2*tid+1];
    }
    
    // Add block sums to results (Uniform Add)
    // Must match block size of scanBlellocReduce (1024 items per block)
    kernel void scanUniformAdd(
        device uint* outData [[buffer(0)]],
        device const uint* blockSums [[buffer(1)]],
        device const uint& n [[buffer(2)]],
        uint gid [[thread_position_in_grid]],
        uint bid [[threadgroup_position_in_grid]])
    {
        uint i = gid * 2; // Each thread processes 2 elements
        uint val = blockSums[bid];
        if (i < n) outData[i] += val;
        if (i + 1 < n) outData[i+1] += val;
    }
)";

} // namespace

struct Metal2Runtime::Impl {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
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

    id<MTLLibrary> libScan = nil;
    id<MTLComputePipelineState> cpsScanSingle = nil;
    id<MTLComputePipelineState> cpsScanReduce = nil;
    id<MTLComputePipelineState> cpsScanAdd = nil;

    id<MTLLibrary> libCompact = nil;
    id<MTLComputePipelineState> cpsCompact = nil;

    bool ok = false;
    double lastYZMs = -1.0;
    double lastPairsMs = -1.0;
};

Metal2Runtime& Metal2Runtime::instance()
{
    static Metal2Runtime s_inst;
    return s_inst;
}

Metal2Runtime::Metal2Runtime() : impl_(new Impl)
{
    impl_->dev = MTLCreateSystemDefaultDevice();
    if (impl_->dev) {
        impl_->q = [impl_->dev newCommandQueue];
        impl_->ok = true;
    }
}

Metal2Runtime::~Metal2Runtime() { delete impl_; }

bool Metal2Runtime::available() const { return impl_ && impl_->ok; }

bool Metal2Runtime::warmup()
{
    if (!available())
        return false;
    // TODO: Compile shaders here to avoid lag on first use
    return true;
}

bool Metal2Runtime::filterYZ(
    const std::vector<float>& minY,
    const std::vector<float>& maxY,
    const std::vector<float>& minZ,
    const std::vector<float>& maxZ,
    const std::vector<int32_t>& v0,
    const std::vector<int32_t>& v1,
    const std::vector<int32_t>& v2,
    const std::vector<std::pair<int, int>>& pairs,
    bool two_lists,
    std::vector<uint8_t>& outMask)
{
    if (!available())
        return false;
    if (minY.size() != maxY.size() || minZ.size() != maxZ.size())
        return false;
    if (v0.size() != minY.size() || v1.size() != minY.size()
        || v2.size() != minY.size())
        return false;
    @autoreleasepool {
        NSError* err = nil;
        if (!impl_->cpsYZ) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            impl_->libYZ = [impl_->dev
                newLibraryWithSource:[NSString
                                         stringWithUTF8String:kYZFilterSrc]
                             options:opts
                               error:&err];
            if (!impl_->libYZ)
                return false;
            id<MTLFunction> fn = [impl_->libYZ newFunctionWithName:@"yzFilter"];
            if (!fn)
                return false;
            impl_->cpsYZ =
                [impl_->dev newComputePipelineStateWithFunction:fn error:&err];
            if (!impl_->cpsYZ)
                return false;
        }
        const size_t n = minY.size();
        const size_t m = pairs.size();
        outMask.assign(m, 0);
        if (m == 0)
            return true;
        id<MTLBuffer> bMinY =
            [impl_->dev newBufferWithBytes:minY.data()
                                    length:sizeof(float) * n
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxY =
            [impl_->dev newBufferWithBytes:maxY.data()
                                    length:sizeof(float) * n
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMinZ =
            [impl_->dev newBufferWithBytes:minZ.data()
                                    length:sizeof(float) * n
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxZ =
            [impl_->dev newBufferWithBytes:maxZ.data()
                                    length:sizeof(float) * n
                                   options:MTLResourceStorageModeShared];
        struct I3 {
            int32_t x;
            int32_t y;
            int32_t z;
            int32_t w;
        };
        std::vector<I3> vids(n);
        for (size_t i = 0; i < n; ++i) {
            vids[i] = I3 { v0[i], v1[i], v2[i], 0 };
        }
        id<MTLBuffer> bVids =
            [impl_->dev newBufferWithBytes:vids.data()
                                    length:sizeof(I3) * n
                                   options:MTLResourceStorageModeShared];
        struct I2 {
            int32_t x;
            int32_t y;
        };
        std::vector<I2> pp(m);
        for (size_t i = 0; i < m; ++i) {
            pp[i] = I2 { pairs[i].first, pairs[i].second };
        }
        id<MTLBuffer> bPairs =
            [impl_->dev newBufferWithBytes:pp.data()
                                    length:sizeof(I2) * m
                                   options:MTLResourceStorageModeShared];
        uint32_t mU32 = static_cast<uint32_t>(m);
        id<MTLBuffer> bM =
            [impl_->dev newBufferWithBytes:&mU32
                                    length:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];
        uint32_t two = two_lists ? 1u : 0u;
        id<MTLBuffer> bTwo =
            [impl_->dev newBufferWithBytes:&two
                                    length:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMask =
            [impl_->dev newBufferWithLength:sizeof(uint8_t) * m
                                    options:MTLResourceStorageModeShared];

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
        if (w == 0)
            w = 64;
        MTLSize grid = MTLSizeMake(m, 1, 1);
        MTLSize tg = MTLSizeMake(std::min<NSUInteger>(w, m), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(outMask.data(), [bMask contents], sizeof(uint8_t) * m);
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
    std::vector<std::pair<int, int>>& outPairs)
{
    if (!available())
        return false;
    const size_t n = minX.size();
    if (n == 0) {
        outPairs.clear();
        return true;
    }
    if (maxX.size() != n || listTag.size() != n)
        return false;
    @autoreleasepool {
        NSError* err = nil;
        // 编译两种内核（per-i 与 PTQ），按环境变量选择
        auto ensurePipelines = [&]() -> bool {
            if (!impl_->cpsSTQTwo) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl_->libSTQTwo = [impl_->dev
                    newLibraryWithSource:
                        [NSString stringWithUTF8String:kSTQTwoListsSrc]
                                 options:opts
                                   error:&err];
                if (!impl_->libSTQTwo)
                    return false;
                id<MTLFunction> fn =
                    [impl_->libSTQTwo newFunctionWithName:@"stqTwo"];
                if (!fn)
                    return false;
                impl_->cpsSTQTwo =
                    [impl_->dev newComputePipelineStateWithFunction:fn
                                                              error:&err];
                if (!impl_->cpsSTQTwo)
                    return false;
            }
            if (!impl_->cpsSTQTwoPTQ) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl_->libSTQTwoPTQ = [impl_->dev
                    newLibraryWithSource:[NSString
                                             stringWithUTF8String:kSTQTwoPTQSrc]
                                 options:opts
                                   error:&err];
                if (!impl_->libSTQTwoPTQ)
                    return false;
                id<MTLFunction> fn =
                    [impl_->libSTQTwoPTQ newFunctionWithName:@"stqTwoPTQ"];
                if (!fn)
                    return false;
                impl_->cpsSTQTwoPTQ =
                    [impl_->dev newComputePipelineStateWithFunction:fn
                                                              error:&err];
                if (!impl_->cpsSTQTwoPTQ)
                    return false;
            }
            return true;
        };
        if (!ensurePipelines())
            return false;
        // 参数：每个 i 允许最多发 maxN 个邻居，按 chunkI 分批
        auto envU = [](const char* k, uint32_t def) -> uint32_t {
            const char* v = std::getenv(k);
            if (!v)
                return def;
            try {
                long long t = std::stoll(v);
                return t > 0 ? (uint32_t)t : def;
            } catch (...) {
                return def;
            }
        };
        auto envB = [](const char* k, bool def) -> bool {
            const char* v = std::getenv(k);
            if (!v)
                return def;
            std::string s(v);
            for (auto& c : s)
                c = (char)tolower((unsigned char)c);
            if (s == "0" || s == "false" || s == "off" || s == "no")
                return false;
            return true;
        };
        auto envD = [](const char* k, double def) -> double {
            const char* v = std::getenv(k);
            if (!v)
                return def;
            try {
                return std::stod(v);
            } catch (...) {
                return def;
            }
        };
        uint32_t maxN = envU("SCALABLE_CCD_METAL2_STQ_MAX_NEIGHBORS", 64);
        uint32_t chunkI = envU("SCALABLE_CCD_METAL2_STQ_CHUNK_I", 8192);
        bool usePTQ = envB("SCALABLE_CCD_METAL2_STQ_PERSISTENT", true);
        uint32_t ptqThreads =
            envU("SCALABLE_CCD_METAL2_STQ_PERSIST_THREADS", 2048);
        double timeout_ms = envD("SCALABLE_CCD_METAL2_STQ_TIMEOUT_MS", 200.0);
        impl_->lastPairsMs = -1.0;
        outPairs.clear();
        // 设备缓冲（共享内存）
        std::vector<float> minXf(n), maxXf(n);
        for (size_t i = 0; i < n; ++i) {
            minXf[i] = (float)minX[i];
            maxXf[i] = (float)maxX[i];
        }
        id<MTLBuffer> bMinX =
            [impl_->dev newBufferWithBytes:minXf.data()
                                    length:sizeof(float) * n
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxX =
            [impl_->dev newBufferWithBytes:maxXf.data()
                                    length:sizeof(float) * n
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bTag =
            [impl_->dev newBufferWithBytes:listTag.data()
                                    length:sizeof(uint8_t) * n
                                   options:MTLResourceStorageModeShared];

        double acc_ms = 0.0;
        bool aborted = false;

        for (uint32_t base = 0; base < n; base += chunkI) {
            uint32_t cur =
                std::min<uint32_t>(chunkI, static_cast<uint32_t>(n - base));
            uint64_t cap = (uint64_t)cur * (uint64_t)maxN;
            if (cap < 1024 * 1024 * 8)
                cap = 1024 * 1024 * 8;

            uint32_t startI = base;
            uint32_t endI = base + cur;
            uint32_t zero = 0;
            id<MTLBuffer> bMaxN =
                [impl_->dev newBufferWithBytes:&maxN
                                        length:sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
            id<MTLBuffer> bPairs =
                [impl_->dev newBufferWithLength:sizeof(int32_t) * 2 * cap
                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> bCnt =
                [impl_->dev newBufferWithBytes:&zero
                                        length:sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
            id<MTLBuffer> bSat =
                [impl_->dev newBufferWithBytes:&zero
                                        length:sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [impl_->q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (usePTQ) {
                id<MTLBuffer> bStart = [impl_->dev
                    newBufferWithBytes:&startI
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> bEnd = [impl_->dev
                    newBufferWithBytes:&endI
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> bHead = [impl_->dev
                    newBufferWithBytes:&zero
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
                [enc setComputePipelineState:impl_->cpsSTQTwoPTQ];
                [enc setBuffer:bMinX offset:0 atIndex:0];
                [enc setBuffer:bMaxX offset:0 atIndex:1];
                [enc setBuffer:bTag offset:0 atIndex:2];
                [enc setBuffer:bStart offset:0 atIndex:3];
                [enc setBuffer:bEnd offset:0 atIndex:4];
                [enc setBuffer:bMaxN offset:0 atIndex:5];
                [enc setBuffer:bHead offset:0 atIndex:6];
                [enc setBuffer:bCnt offset:0 atIndex:7];
                [enc setBuffer:bPairs offset:0 atIndex:8];
                [enc setBuffer:bSat offset:0 atIndex:9];
                NSUInteger w =
                    impl_->cpsSTQTwoPTQ.maxTotalThreadsPerThreadgroup;
                if (w == 0)
                    w = 64;
                NSUInteger tgSize = std::min<NSUInteger>(w, 256);
                NSUInteger gridSize = std::min<NSUInteger>(
                    ptqThreads, ptqThreads - (ptqThreads % tgSize) + tgSize);
                MTLSize grid = MTLSizeMake(gridSize, 1, 1);
                MTLSize tg = MTLSizeMake(tgSize, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            } else {
                id<MTLBuffer> bBase = [impl_->dev
                    newBufferWithBytes:&base
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> bN = [impl_->dev
                    newBufferWithBytes:&n
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
                [enc setComputePipelineState:impl_->cpsSTQTwo];
                [enc setBuffer:bMinX offset:0 atIndex:0];
                [enc setBuffer:bMaxX offset:0 atIndex:1];
                [enc setBuffer:bTag offset:0 atIndex:2];
                [enc setBuffer:bBase offset:0 atIndex:3];
                [enc setBuffer:bN offset:0 atIndex:4];
                [enc setBuffer:bMaxN offset:0 atIndex:5];
                [enc setBuffer:bCnt offset:0 atIndex:6];
                [enc setBuffer:bPairs offset:0 atIndex:7];
                [enc setBuffer:bSat offset:0 atIndex:8];
                NSUInteger w = impl_->cpsSTQTwo.maxTotalThreadsPerThreadgroup;
                if (w == 0)
                    w = 64;
                MTLSize grid = MTLSizeMake(cur, 1, 1);
                MTLSize tg = MTLSizeMake(std::min<NSUInteger>(w, cur), 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            CFTimeInterval t0 = cb.GPUStartTime, t1 = cb.GPUEndTime;
            double ms = (t1 > t0 && t0 > 0) ? (t1 - t0) * 1000.0 : 0.0;
            if (impl_->lastPairsMs < 0)
                impl_->lastPairsMs = 0.0;
            impl_->lastPairsMs += ms;
            acc_ms += ms;
            // 读回 pairs
            uint32_t cnt = *static_cast<uint32_t*>([bCnt contents]);
            if (cnt > cap)
                cnt = (uint32_t)cap; // 防御
            uint32_t sat = *static_cast<uint32_t*>([bSat contents]);
            struct I2 {
                int32_t x;
                int32_t y;
            };
            I2* pp = static_cast<I2*>([bPairs contents]);
            outPairs.reserve(outPairs.size() + cnt);
            for (uint32_t i = 0; i < cnt; ++i)
                outPairs.emplace_back(pp[i].x, pp[i].y);
            if (cnt >= cap || sat) {
                printf(
                    "Metal2 STQ(two) ABORT: Overflow (cnt=%u, cap=%llu, sat=%u)\n",
                    cnt, cap, sat);
                aborted = true;
                break;
            } // 溢出或饱和，回退 CPU
            if (timeout_ms > 0.0 && acc_ms > timeout_ms) {
                printf(
                    "Metal2 STQ(two) ABORT: Timeout (acc=%.2f ms, limit=%.2f ms)\n",
                    acc_ms, timeout_ms);
                aborted = true;
                break;
            } // 超时
        }
        if (aborted) {
            outPairs.clear();
            return false;
        }
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
    std::vector<std::pair<int, int>>& outPairs)
{
    if (!available())
        return false;
    const size_t n = minX.size();
    if (n == 0) {
        outPairs.clear();
        return true;
    }
    if (maxX.size() != n)
        return false;
    @autoreleasepool {
        NSError* err = nil;
        auto ensurePipelines = [&]() -> bool {
            if (!impl_->cpsSTQSingle) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl_->libSTQSingle = [impl_->dev
                    newLibraryWithSource:
                        [NSString stringWithUTF8String:kSTQSingleListSrc]
                                 options:opts
                                   error:&err];
                if (!impl_->libSTQSingle)
                    return false;
                id<MTLFunction> fn =
                    [impl_->libSTQSingle newFunctionWithName:@"stqSingle"];
                if (!fn)
                    return false;
                impl_->cpsSTQSingle =
                    [impl_->dev newComputePipelineStateWithFunction:fn
                                                              error:&err];
                if (!impl_->cpsSTQSingle)
                    return false;
            }
            if (!impl_->cpsSTQSinglePTQ) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl_->libSTQSinglePTQ = [impl_->dev
                    newLibraryWithSource:
                        [NSString stringWithUTF8String:kSTQSinglePTQSrc]
                                 options:opts
                                   error:&err];
                if (!impl_->libSTQSinglePTQ)
                    return false;
                id<MTLFunction> fn = [impl_->libSTQSinglePTQ
                    newFunctionWithName:@"stqSinglePTQ"];
                if (!fn)
                    return false;
                impl_->cpsSTQSinglePTQ =
                    [impl_->dev newComputePipelineStateWithFunction:fn
                                                              error:&err];
                if (!impl_->cpsSTQSinglePTQ)
                    return false;
            }
            return true;
        };
        if (!ensurePipelines())
            return false;
        auto envU = [](const char* k, uint32_t def) -> uint32_t {
            const char* v = std::getenv(k);
            if (!v)
                return def;
            try {
                long long t = std::stoll(v);
                return t > 0 ? (uint32_t)t : def;
            } catch (...) {
                return def;
            }
        };
        auto envB = [](const char* k, bool def) -> bool {
            const char* v = std::getenv(k);
            if (!v)
                return def;
            std::string s(v);
            for (auto& c : s)
                c = (char)tolower((unsigned char)c);
            if (s == "0" || s == "false" || s == "off" || s == "no")
                return false;
            return true;
        };
        auto envD = [](const char* k, double def) -> double {
            const char* v = std::getenv(k);
            if (!v)
                return def;
            try {
                return std::stod(v);
            } catch (...) {
                return def;
            }
        };
        uint32_t maxN = envU("SCALABLE_CCD_METAL2_STQ_MAX_NEIGHBORS", 64);
        uint32_t chunkI = envU("SCALABLE_CCD_METAL2_STQ_CHUNK_I", 8192);
        bool usePTQ = envB("SCALABLE_CCD_METAL2_STQ_PERSISTENT", true);
        uint32_t ptqThreads =
            envU("SCALABLE_CCD_METAL2_STQ_PERSIST_THREADS", 2048);
        double timeout_ms = envD("SCALABLE_CCD_METAL2_STQ_TIMEOUT_MS", 200.0);
        impl_->lastPairsMs = -1.0;
        outPairs.clear();
        std::vector<float> minXf(n), maxXf(n);
        for (size_t i = 0; i < n; ++i) {
            minXf[i] = (float)minX[i];
            maxXf[i] = (float)maxX[i];
        }
        id<MTLBuffer> bMinX =
            [impl_->dev newBufferWithBytes:minXf.data()
                                    length:sizeof(float) * n
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxX =
            [impl_->dev newBufferWithBytes:maxXf.data()
                                    length:sizeof(float) * n
                                   options:MTLResourceStorageModeShared];
        double acc_ms = 0.0;
        bool aborted = false;

        for (uint32_t base = 0; base < n; base += chunkI) {
            uint32_t cur =
                std::min<uint32_t>(chunkI, static_cast<uint32_t>(n - base));
            uint64_t cap = (uint64_t)cur * (uint64_t)maxN;
            if (cap < 1024 * 1024 * 8)
                cap = 1024 * 1024 * 8;

            uint32_t startI = base;
            uint32_t endI = base + cur;
            uint32_t zero = 0;
            id<MTLBuffer> bMaxN =
                [impl_->dev newBufferWithBytes:&maxN
                                        length:sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
            id<MTLBuffer> bPairs =
                [impl_->dev newBufferWithLength:sizeof(int32_t) * 2 * cap
                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> bCnt =
                [impl_->dev newBufferWithBytes:&zero
                                        length:sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
            id<MTLBuffer> bSat =
                [impl_->dev newBufferWithBytes:&zero
                                        length:sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [impl_->q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (usePTQ) {
                id<MTLBuffer> bStart = [impl_->dev
                    newBufferWithBytes:&startI
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> bEnd = [impl_->dev
                    newBufferWithBytes:&endI
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> bHead = [impl_->dev
                    newBufferWithBytes:&zero
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
                [enc setComputePipelineState:impl_->cpsSTQSinglePTQ];
                [enc setBuffer:bMinX offset:0 atIndex:0];
                [enc setBuffer:bMaxX offset:0 atIndex:1];
                [enc setBuffer:bStart offset:0 atIndex:2];
                [enc setBuffer:bEnd offset:0 atIndex:3];
                [enc setBuffer:bMaxN offset:0 atIndex:4];
                [enc setBuffer:bHead offset:0 atIndex:5];
                [enc setBuffer:bCnt offset:0 atIndex:6];
                [enc setBuffer:bPairs offset:0 atIndex:7];
                [enc setBuffer:bSat offset:0 atIndex:8];
                NSUInteger w =
                    impl_->cpsSTQSinglePTQ.maxTotalThreadsPerThreadgroup;
                if (w == 0)
                    w = 64;
                NSUInteger tgSize = std::min<NSUInteger>(w, 256);
                NSUInteger gridSize = std::min<NSUInteger>(
                    ptqThreads, ptqThreads - (ptqThreads % tgSize) + tgSize);
                MTLSize grid = MTLSizeMake(gridSize, 1, 1);
                MTLSize tg = MTLSizeMake(tgSize, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            } else {
                id<MTLBuffer> bBase = [impl_->dev
                    newBufferWithBytes:&base
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
                id<MTLBuffer> bN = [impl_->dev
                    newBufferWithBytes:&n
                                length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
                [enc setComputePipelineState:impl_->cpsSTQSingle];
                [enc setBuffer:bMinX offset:0 atIndex:0];
                [enc setBuffer:bMaxX offset:0 atIndex:1];
                [enc setBuffer:bBase offset:0 atIndex:2];
                [enc setBuffer:bN offset:0 atIndex:3];
                [enc setBuffer:bMaxN offset:0 atIndex:4];
                [enc setBuffer:bCnt offset:0 atIndex:5];
                [enc setBuffer:bPairs offset:0 atIndex:6];
                [enc setBuffer:bSat offset:0 atIndex:7];
                NSUInteger w =
                    impl_->cpsSTQSingle.maxTotalThreadsPerThreadgroup;
                if (w == 0)
                    w = 64;
                MTLSize grid = MTLSizeMake(cur, 1, 1);
                MTLSize tg = MTLSizeMake(std::min<NSUInteger>(w, cur), 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            CFTimeInterval t0 = cb.GPUStartTime, t1 = cb.GPUEndTime;
            double ms = (t1 > t0 && t0 > 0) ? (t1 - t0) * 1000.0 : 0.0;
            if (impl_->lastPairsMs < 0)
                impl_->lastPairsMs = 0.0;
            impl_->lastPairsMs += ms;
            acc_ms += ms;
            uint32_t cnt = *static_cast<uint32_t*>([bCnt contents]);
            if (cnt > cap)
                cnt = (uint32_t)cap;
            uint32_t sat = *static_cast<uint32_t*>([bSat contents]);
            struct I2 {
                int32_t x;
                int32_t y;
            };
            I2* pp = static_cast<I2*>([bPairs contents]);
            outPairs.reserve(outPairs.size() + cnt);
            for (uint32_t i = 0; i < cnt; ++i)
                outPairs.emplace_back(pp[i].x, pp[i].y);
            if (cnt >= cap || sat) {
                printf(
                    "Metal2 STQ(single) ABORT: Overflow (cnt=%u, cap=%llu, sat=%u)\n",
                    cnt, cap, sat);
                aborted = true;
                break;
            }
            if (timeout_ms > 0.0 && acc_ms > timeout_ms) {
                printf(
                    "Metal2 STQ(single) ABORT: Timeout (acc=%.2f ms, limit=%.2f ms)\n",
                    acc_ms, timeout_ms);
                aborted = true;
                break;
            }
        }
        if (aborted) {
            outPairs.clear();
            return false;
        }
        return true;
    }
}

double Metal2Runtime::lastYZFilterMs() const
{
    return impl_ ? impl_->lastYZMs : -1.0;
}
double Metal2Runtime::lastSTQPairsMs() const
{
    return impl_ ? impl_->lastPairsMs : -1.0;
}

} // namespace scalable_ccd::metal2

namespace {
const char* kCompactSrc = R"(
    #include <metal_stdlib>
    using namespace metal;

    kernel void compactIndices(
        device const uchar* valid [[buffer(0)]],
        device const uint* scanOffsets [[buffer(1)]],
        device uint* outIndices [[buffer(2)]],
        device const uint& n [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        if (gid >= n) return;
        
        if (valid[gid]) {
            uint idx = scanOffsets[gid]; // Exclusive scan gives the index
            outIndices[idx] = gid; // Store the source index
        }
    }
)";
} // namespace

// ... (Other methods unchanged) ...

bool scalable_ccd::metal2::Metal2Runtime::scan(
    const std::vector<uint8_t>& inData, std::vector<uint32_t>& outData)
{
    if (!available())
        return false;
    size_t n = inData.size();
    if (n == 0) {
        outData.clear();
        return true;
    }
    outData.resize(n);

    @autoreleasepool {
        NSError* err = nil;
        if (!impl_->cpsScanSingle) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            impl_->libScan = [impl_->dev
                newLibraryWithSource:[NSString stringWithUTF8String:kScanSrc]
                             options:opts
                               error:&err];
            if (!impl_->libScan)
                return false;

            id<MTLFunction> fnSingle =
                [impl_->libScan newFunctionWithName:@"scanSingleBlock"];
            impl_->cpsScanSingle =
                [impl_->dev newComputePipelineStateWithFunction:fnSingle
                                                          error:&err];

            id<MTLFunction> fnReduce =
                [impl_->libScan newFunctionWithName:@"scanBlellocReduce"];
            impl_->cpsScanReduce =
                [impl_->dev newComputePipelineStateWithFunction:fnReduce
                                                          error:&err];

            id<MTLFunction> fnAdd =
                [impl_->libScan newFunctionWithName:@"scanUniformAdd"];
            impl_->cpsScanAdd =
                [impl_->dev newComputePipelineStateWithFunction:fnAdd
                                                          error:&err];

            if (!impl_->cpsScanSingle || !impl_->cpsScanReduce
                || !impl_->cpsScanAdd)
                return false;
        }

        id<MTLBuffer> bIn =
            [impl_->dev newBufferWithBytes:inData.data()
                                    length:sizeof(uint8_t) * n
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bOut =
            [impl_->dev newBufferWithLength:sizeof(uint32_t) * n
                                    options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [impl_->q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        uint32_t blockSize = 1024;
        uint32_t numBlocks = (n + blockSize - 1) / blockSize;

        if (numBlocks == 1) {
            [enc setComputePipelineState:impl_->cpsScanSingle];
            [enc setBuffer:bIn offset:0 atIndex:0];
            [enc setBuffer:bOut offset:0 atIndex:1];
            uint32_t pot = 1;
            while (pot < n)
                pot <<= 1;
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(pot, 1, 1)];
        } else {
            id<MTLBuffer> bBlockSums =
                [impl_->dev newBufferWithLength:sizeof(uint32_t) * numBlocks
                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> bN =
                [impl_->dev newBufferWithBytes:&n
                                        length:sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];

            [enc setComputePipelineState:impl_->cpsScanReduce];
            [enc setBuffer:bIn offset:0 atIndex:0];
            [enc setBuffer:bOut offset:0 atIndex:1];
            [enc setBuffer:bBlockSums offset:0 atIndex:2];
            [enc setBuffer:bN offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(numBlocks, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            uint32_t* sums = (uint32_t*)[bBlockSums contents];
            uint32_t* scannedSums =
                (uint32_t*)malloc(sizeof(uint32_t) * numBlocks);
            scannedSums[0] = 0;
            for (uint32_t i = 1; i < numBlocks; ++i)
                scannedSums[i] = scannedSums[i - 1] + sums[i - 1];
            memcpy(sums, scannedSums, sizeof(uint32_t) * numBlocks);
            free(scannedSums);

            cb = [impl_->q commandBuffer];
            enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:impl_->cpsScanAdd];
            [enc setBuffer:bOut offset:0 atIndex:0];
            [enc setBuffer:bBlockSums offset:0 atIndex:1];
            [enc setBuffer:bN offset:0 atIndex:2];
            uint32_t numThreads = (n + 1) / 2;
            uint32_t tpg = 512;
            uint32_t groups = (numThreads + tpg - 1) / tpg;
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        memcpy(outData.data(), [bOut contents], sizeof(uint32_t) * n);
    }
    return true;
}

bool scalable_ccd::metal2::Metal2Runtime::compact(
    const std::vector<uint8_t>& valid,
    const std::vector<uint32_t>& scanOffsets,
    std::vector<uint32_t>& outIndices)
{
    if (!available())
        return false;
    size_t n = valid.size();
    if (n == 0) {
        outIndices.clear();
        return true;
    }

    uint32_t total = scanOffsets.back() + valid.back();
    outIndices.resize(total);
    if (total == 0)
        return true;

    @autoreleasepool {
        NSError* err = nil;
        if (!impl_->cpsCompact) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            impl_->libCompact = [impl_->dev
                newLibraryWithSource:[NSString stringWithUTF8String:kCompactSrc]
                             options:opts
                               error:&err];
            if (!impl_->libCompact)
                return false;
            id<MTLFunction> fn =
                [impl_->libCompact newFunctionWithName:@"compactIndices"];
            impl_->cpsCompact =
                [impl_->dev newComputePipelineStateWithFunction:fn error:&err];
            if (!impl_->cpsCompact)
                return false;
        }

        id<MTLBuffer> bValid =
            [impl_->dev newBufferWithBytes:valid.data()
                                    length:sizeof(uint8_t) * n
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bScan =
            [impl_->dev newBufferWithBytes:scanOffsets.data()
                                    length:sizeof(uint32_t) * n
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bOutIndices =
            [impl_->dev newBufferWithLength:sizeof(uint32_t) * total
                                    options:MTLResourceStorageModeShared];

        uint32_t n32 = (uint32_t)n;
        id<MTLBuffer> bN =
            [impl_->dev newBufferWithBytes:&n32
                                    length:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [impl_->q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->cpsCompact];
        [enc setBuffer:bValid offset:0 atIndex:0];
        [enc setBuffer:bScan offset:0 atIndex:1];
        [enc setBuffer:bOutIndices offset:0 atIndex:2];
        [enc setBuffer:bN offset:0 atIndex:3];

        uint32_t tpg = 256;
        uint32_t groups = (n + tpg - 1) / tpg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        memcpy(
            outIndices.data(), [bOutIndices contents],
            sizeof(uint32_t) * total);
    }
    return true;
}
