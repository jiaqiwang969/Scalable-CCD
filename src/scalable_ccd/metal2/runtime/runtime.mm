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
  if (twoLists == 0u) {
    int3 ai = vids[i];
    int3 aj = vids[j];
    bool share =
      (ai.x == aj.x) || (ai.x == aj.y) || (ai.x == aj.z) ||
      (ai.y == aj.x) || (ai.y == aj.y) || (ai.y == aj.z) ||
      (ai.z == aj.x) || (ai.z == aj.y) || (ai.z == aj.z);
    outMask[gid] = share ? 0 : 1;
  } else {
    outMask[gid] = 1;
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
    bool ok = false;
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
        return true;
    }
}

} // namespace scalable_ccd::metal2
