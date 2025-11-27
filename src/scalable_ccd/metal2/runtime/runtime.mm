// Metal2 runtime minimal implementation: warmup + yzFilter kernel
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "runtime.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <mutex>

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
    static const char* kYZFilterAtomicSrc = R"(
using namespace metal;
kernel void yzFilterAtomic(
  device const float* minY [[buffer(0)]],
  device const float* maxY [[buffer(1)]],
  device const float* minZ [[buffer(2)]],
  device const float* maxZ [[buffer(3)]],
  device const int3*   vids [[buffer(4)]],
  device const int2* pairs [[buffer(5)]],
  constant uint& nPairs [[buffer(6)]],
  constant uint& twoLists [[buffer(7)]],
  device atomic_uint* gCount [[buffer(8)]],
  device uint* outIndices [[buffer(9)]],
  constant uint& maxCapacity [[buffer(10)]],
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
  if (!(overlapY && overlapZ)) return;
  
  int3 ai = vids[i];
  int3 aj = vids[j];
  bool share =
    (ai.x == aj.x) || (ai.x == aj.y) || (ai.x == aj.z) ||
    (ai.y == aj.x) || (ai.y == aj.y) || (ai.y == aj.z) ||
    (ai.z == aj.x) || (ai.z == aj.y) || (ai.z == aj.z);
  
  if (!share) {
    uint pos = atomic_fetch_add_explicit(gCount, 1u, memory_order_relaxed);
    if (pos < maxCapacity) {
        outIndices[pos] = gid;
    }
  }
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

    // 融合内核：STQ + YZ Filter（单遍完成候选生成与过滤）
    // 优化版本：使用打包结构体 + SIMD 聚合
    static const char* kSTQFusedTwoSrc = R"(
#include <metal_stdlib>
using namespace metal;

// 打包的 AABB 结构体（24 字节，缓存友好）
struct PackedAABB {
    float minX, maxX, minY, maxY, minZ, maxZ;
};

// 共享顶点检测
inline bool share_vertex(int3 a, int3 b) {
    return (a.x == b.x) || (a.x == b.y) || (a.x == b.z) ||
           (a.y == b.x) || (a.y == b.y) || (a.y == b.z) ||
           (a.z == b.x) || (a.z == b.y) || (a.z == b.z);
}

// 闭区间重叠检测（与 CPU 一致）
inline bool overlaps_closed(float aMin, float aMax, float bMin, float bMax) {
    return !(aMax < bMin || aMin > bMax);
}

kernel void stqFusedTwo(
  device const PackedAABB* boxes [[buffer(0)]],
  device const int3* vids [[buffer(1)]],
  device const uchar* listTag [[buffer(2)]],
  constant uint& startI [[buffer(3)]],
  constant uint& endI [[buffer(4)]],
  constant uint& maxN [[buffer(5)]],           // 保留参数但不再强制截断
  device atomic_uint* gHead [[buffer(6)]],
  device atomic_uint* gCount [[buffer(7)]],
  device int2* outPairs [[buffer(8)]],
  device atomic_uint* gSat [[buffer(9)]],
  constant uint& bufferCapacity [[buffer(10)]], // 新参数：缓冲区容量
  uint tid [[thread_position_in_grid]],
  uint simd_lane [[thread_index_in_simdgroup]],
  uint simd_size [[threads_per_simdgroup]]
) {
  // 本地缓存用于 SIMD 聚合
  int2 local_pair;
  bool has_pair = false;

  while (true) {
    uint idx = atomic_fetch_add_explicit(gHead, 1u, memory_order_relaxed);
    uint i = startI + idx;
    if (i >= endI) break;

    PackedAABB boxI = boxes[i];
    uchar tagI = listTag[i];
    int3 vidI = vids[i];

    for (uint j = i + 1; j < endI; ++j) {
      PackedAABB boxJ = boxes[j];

      // X 轴检查（sweep 终止条件）
      if (boxI.maxX < boxJ.minX) break;

      // 跨列表检查
      if (tagI == listTag[j]) continue;

      // Y 轴重叠检查
      if (!overlaps_closed(boxI.minY, boxI.maxY, boxJ.minY, boxJ.maxY)) continue;

      // Z 轴重叠检查
      if (!overlaps_closed(boxI.minZ, boxI.maxZ, boxJ.minZ, boxJ.maxZ)) continue;

      // 共享顶点检查
      if (share_vertex(vidI, vids[j])) continue;

      // CUDA 风格：不限制每个 i 的输出数量
      // 使用原子操作获取位置并检查是否溢出
      uint pos = atomic_fetch_add_explicit(gCount, 1u, memory_order_relaxed);
      if (pos < bufferCapacity) {
        outPairs[pos] = int2((int)i, (int)j);
      } else {
        // 缓冲区溢出，设置饱和标志
        atomic_store_explicit(gSat, 1u, memory_order_relaxed);
      }
    }
  }
}
)";

    // 融合内核：STQ + YZ Filter（单列表版本）
    // 优化版本：CUDA 风格，不限制每个 i 的输出数量
    static const char* kSTQFusedSingleSrc = R"(
#include <metal_stdlib>
using namespace metal;

struct PackedAABB {
    float minX, maxX, minY, maxY, minZ, maxZ;
};

inline bool share_vertex(int3 a, int3 b) {
    return (a.x == b.x) || (a.x == b.y) || (a.x == b.z) ||
           (a.y == b.x) || (a.y == b.y) || (a.y == b.z) ||
           (a.z == b.x) || (a.z == b.y) || (a.z == b.z);
}

inline bool overlaps_closed(float aMin, float aMax, float bMin, float bMax) {
    return !(aMax < bMin || aMin > bMax);
}

kernel void stqFusedSingle(
  device const PackedAABB* boxes [[buffer(0)]],
  device const int3* vids [[buffer(1)]],
  constant uint& startI [[buffer(2)]],
  constant uint& endI [[buffer(3)]],
  constant uint& maxN [[buffer(4)]],           // 保留参数但不再强制截断
  device atomic_uint* gHead [[buffer(5)]],
  device atomic_uint* gCount [[buffer(6)]],
  device int2* outPairs [[buffer(7)]],
  device atomic_uint* gSat [[buffer(8)]],
  constant uint& bufferCapacity [[buffer(9)]], // 新参数：缓冲区容量
  uint tid [[thread_position_in_grid]]
) {
  while (true) {
    uint idx = atomic_fetch_add_explicit(gHead, 1u, memory_order_relaxed);
    uint i = startI + idx;
    if (i >= endI) break;

    PackedAABB boxI = boxes[i];
    int3 vidI = vids[i];

    for (uint j = i + 1; j < endI; ++j) {
      PackedAABB boxJ = boxes[j];
      if (boxI.maxX < boxJ.minX) break;

      if (!overlaps_closed(boxI.minY, boxI.maxY, boxJ.minY, boxJ.maxY)) continue;
      if (!overlaps_closed(boxI.minZ, boxI.maxZ, boxJ.minZ, boxJ.maxZ)) continue;
      if (share_vertex(vidI, vids[j])) continue;

      // CUDA 风格：不限制每个 i 的输出数量
      // 使用原子操作获取位置并检查是否溢出
      uint pos = atomic_fetch_add_explicit(gCount, 1u, memory_order_relaxed);
      if (pos < bufferCapacity) {
        outPairs[pos] = int2((int)i, (int)j);
      } else {
        // 缓冲区溢出，设置饱和标志
        atomic_store_explicit(gSat, 1u, memory_order_relaxed);
      }
    }
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

    // 优化版 Narrow Phase Shader V2 - SIMD 展开 + 数据预取 + 本地队列
    static const char* kNarrowPhaseOptV2Src = R"(
#include <metal_stdlib>
using namespace metal;

struct CCDConfig {
    float co_domain_tolerance;
    int max_iter;
    bool use_ms;
    bool allow_zero_toi;
};

struct Interval {
    float lower;
    float upper;
};

struct CCDDomain {
    Interval tuv[3];
    int query_id;
};

struct Vector3Host {
    float x, y, z, _pad;
};

struct CCDData {
    Vector3Host v0s, v1s, v2s, v3s;
    Vector3Host v0e, v1e, v2e, v3e;
    float ms;
    float tol[3];
    Vector3Host err;
    int nbr_checks;
};

inline float3 toFloat3(Vector3Host v) {
    return float3(v.x, v.y, v.z);
}

inline void atomic_min_float(device atomic_uint* addr, float val) {
    uint old_val = atomic_load_explicit(addr, memory_order_relaxed);
    uint new_val = as_type<uint>(val);
    while (as_type<float>(old_val) > val) {
        if (atomic_compare_exchange_weak_explicit(addr, &old_val, new_val,
                                                   memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

inline bool sum_less_than_one(float num1, float num2) {
    return num1 + num2 <= 1.0f / (1.0f - 1.19209e-07f);
}

// 预计算的顶点数据（缓存到寄存器）
struct CachedVertices {
    float3 v0s, v1s, v2s, v3s;
    float3 v0e, v1e, v2e, v3e;
    float3 err;
    float ms;
    float tol0, tol1, tol2;
};

inline CachedVertices cache_vertices(device const CCDData& d) {
    CachedVertices c;
    c.v0s = toFloat3(d.v0s); c.v1s = toFloat3(d.v1s);
    c.v2s = toFloat3(d.v2s); c.v3s = toFloat3(d.v3s);
    c.v0e = toFloat3(d.v0e); c.v1e = toFloat3(d.v1e);
    c.v2e = toFloat3(d.v2e); c.v3e = toFloat3(d.v3e);
    c.err = toFloat3(d.err);
    c.ms = d.ms;
    c.tol0 = d.tol[0]; c.tol1 = d.tol[1]; c.tol2 = d.tol[2];
    return c;
}

// 展开的 VF 角点计算 - 避免循环
inline void compute_codomain_vf_unrolled(
    thread const CachedVertices& c,
    float t0, float t1, float u0, float u1, float v0, float v1,
    thread float3& cmin, thread float3& cmax)
{
    // 预计算插值后的顶点
    float3 p0_t0 = mix(c.v0s, c.v0e, t0);
    float3 p0_t1 = mix(c.v0s, c.v0e, t1);
    float3 p1_t0 = mix(c.v1s, c.v1e, t0);
    float3 p1_t1 = mix(c.v1s, c.v1e, t1);
    float3 p2_t0 = mix(c.v2s, c.v2e, t0);
    float3 p2_t1 = mix(c.v2s, c.v2e, t1);
    float3 p3_t0 = mix(c.v3s, c.v3e, t0);
    float3 p3_t1 = mix(c.v3s, c.v3e, t1);

    // 计算 8 个角点 (t,u,v) ∈ {t0,t1} × {u0,u1} × {v0,v1}
    // corner 0: (t0, u0, v0)
    float3 c0 = p0_t0 - (p2_t0 - p1_t0) * u0 - (p3_t0 - p1_t0) * v0 - p1_t0;
    // corner 1: (t1, u0, v0)
    float3 c1 = p0_t1 - (p2_t1 - p1_t1) * u0 - (p3_t1 - p1_t1) * v0 - p1_t1;
    // corner 2: (t0, u1, v0)
    float3 c2 = p0_t0 - (p2_t0 - p1_t0) * u1 - (p3_t0 - p1_t0) * v0 - p1_t0;
    // corner 3: (t1, u1, v0)
    float3 c3 = p0_t1 - (p2_t1 - p1_t1) * u1 - (p3_t1 - p1_t1) * v0 - p1_t1;
    // corner 4: (t0, u0, v1)
    float3 c4 = p0_t0 - (p2_t0 - p1_t0) * u0 - (p3_t0 - p1_t0) * v1 - p1_t0;
    // corner 5: (t1, u0, v1)
    float3 c5 = p0_t1 - (p2_t1 - p1_t1) * u0 - (p3_t1 - p1_t1) * v1 - p1_t1;
    // corner 6: (t0, u1, v1)
    float3 c6 = p0_t0 - (p2_t0 - p1_t0) * u1 - (p3_t0 - p1_t0) * v1 - p1_t0;
    // corner 7: (t1, u1, v1)
    float3 c7 = p0_t1 - (p2_t1 - p1_t1) * u1 - (p3_t1 - p1_t1) * v1 - p1_t1;

    // 并行归约求 min/max
    cmin = min(min(min(c0, c1), min(c2, c3)), min(min(c4, c5), min(c6, c7)));
    cmax = max(max(max(c0, c1), max(c2, c3)), max(max(c4, c5), max(c6, c7)));
}

inline void compute_codomain_ee_unrolled(
    thread const CachedVertices& c,
    float t0, float t1, float u0, float u1, float v0, float v1,
    thread float3& cmin, thread float3& cmax)
{
    float3 ea0_t0 = mix(c.v0s, c.v0e, t0);
    float3 ea0_t1 = mix(c.v0s, c.v0e, t1);
    float3 ea1_t0 = mix(c.v1s, c.v1e, t0);
    float3 ea1_t1 = mix(c.v1s, c.v1e, t1);
    float3 eb0_t0 = mix(c.v2s, c.v2e, t0);
    float3 eb0_t1 = mix(c.v2s, c.v2e, t1);
    float3 eb1_t0 = mix(c.v3s, c.v3e, t0);
    float3 eb1_t1 = mix(c.v3s, c.v3e, t1);

    float3 c0 = mix(ea0_t0, ea1_t0, u0) - mix(eb0_t0, eb1_t0, v0);
    float3 c1 = mix(ea0_t1, ea1_t1, u0) - mix(eb0_t1, eb1_t1, v0);
    float3 c2 = mix(ea0_t0, ea1_t0, u1) - mix(eb0_t0, eb1_t0, v0);
    float3 c3 = mix(ea0_t1, ea1_t1, u1) - mix(eb0_t1, eb1_t1, v0);
    float3 c4 = mix(ea0_t0, ea1_t0, u0) - mix(eb0_t0, eb1_t0, v1);
    float3 c5 = mix(ea0_t1, ea1_t1, u0) - mix(eb0_t1, eb1_t1, v1);
    float3 c6 = mix(ea0_t0, ea1_t0, u1) - mix(eb0_t0, eb1_t0, v1);
    float3 c7 = mix(ea0_t1, ea1_t1, u1) - mix(eb0_t1, eb1_t1, v1);

    cmin = min(min(min(c0, c1), min(c2, c3)), min(min(c4, c5), min(c6, c7)));
    cmax = max(max(max(c0, c1), max(c2, c3)), max(max(c4, c5), max(c6, c7)));
}

kernel void compute_tolerance_v2(
    device CCDData* data [[buffer(0)]],
    constant CCDConfig& config [[buffer(1)]],
    constant uint& is_vf [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;

    device CCDData& d = data[id];
    float3 v0s = toFloat3(d.v0s), v1s = toFloat3(d.v1s), v2s = toFloat3(d.v2s), v3s = toFloat3(d.v3s);
    float3 v0e = toFloat3(d.v0e), v1e = toFloat3(d.v1e), v2e = toFloat3(d.v2e), v3e = toFloat3(d.v3e);

    float3 p000, p001, p010, p011, p100, p101, p110, p111;

    if (is_vf) {
        p000 = v0s - v1s; p001 = v0s - v3s;
        p011 = v0s - (v2s + v3s - v1s); p010 = v0s - v2s;
        p100 = v0e - v1e; p101 = v0e - v3e;
        p111 = v0e - (v2e + v3e - v1e); p110 = v0e - v2e;
    } else {
        p000 = v0s - v2s; p001 = v0s - v3s;
        p010 = v1s - v2s; p011 = v1s - v3s;
        p100 = v0e - v2e; p101 = v0e - v3e;
        p110 = v1e - v2e; p111 = v1e - v3e;
    }

    // 内联 max_linf_4
    float3 d1 = abs(p100 - p000), d2 = abs(p101 - p001);
    float3 d3 = abs(p111 - p011), d4 = abs(p110 - p010);
    float m1 = max(max(d1.x, max(d1.y, d1.z)), max(d2.x, max(d2.y, d2.z)));
    float m2 = max(max(d3.x, max(d3.y, d3.z)), max(d4.x, max(d4.y, d4.z)));
    float div = 3.0f * max(m1, m2);

    d.tol[0] = config.co_domain_tolerance / div;

    // tol[1]
    float3 e1 = abs(p100 - p000), e2 = abs(p101 - p001);
    float3 e3 = abs(p111 - p011), e4 = abs(p110 - p010);
    // 对于 VF，使用不同的点组合
    if (is_vf) {
        e1 = abs(p100 - p000); e2 = abs(p101 - p001);
        e3 = abs(p111 - p011); e4 = abs(p110 - p010);
        float n1 = max(max(e1.x, max(e1.y, e1.z)), max(e2.x, max(e2.y, e2.z)));
        float n2 = max(max(e3.x, max(e3.y, e3.z)), max(e4.x, max(e4.y, e4.z)));
        d.tol[1] = config.co_domain_tolerance / (3.0f * max(n1, n2));
    } else {
        d.tol[1] = d.tol[0];
    }

    // tol[2] - 简化计算
    d.tol[2] = d.tol[0];

    float filter = config.use_ms ? (is_vf ? 4.053116e-06f : 3.814698e-06f)
                                 : (is_vf ? 3.576279e-06f : 3.337861e-06f);
    float3 max_val = max(abs(v0s), max(abs(v1s), max(abs(v2s), abs(v3s))));
    max_val = max(max_val, max(abs(v0e), max(abs(v1e), max(abs(v2e), abs(v3e)))));
    max_val = max(max_val, float3(1.0f));
    float3 err_val = max_val * max_val * max_val * filter;
    d.err = {err_val.x, err_val.y, err_val.z, 0.0f};
    d.nbr_checks = 0;
}

kernel void init_buffer_v2(
    device CCDDomain* buffer [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;
    buffer[id].tuv[0] = {0.0f, 1.0f};
    buffer[id].tuv[1] = {0.0f, 1.0f};
    buffer[id].tuv[2] = {0.0f, 1.0f};
    buffer[id].query_id = int(id);
}

// 简化版 V2：保留 SIMD 展开和数据预取，移除 threadgroup 本地队列
// 本地队列在 Metal 上增加了额外开销，反而降低性能
kernel void ccd_persistent_v2(
    device CCDDomain* bufferData [[buffer(0)]],
    device atomic_uint* head [[buffer(1)]],
    device atomic_uint* tail [[buffer(2)]],
    device atomic_int* overflow [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    device CCDData* data [[buffer(5)]],
    device atomic_uint* toi_atomic [[buffer(6)]],
    constant CCDConfig& config [[buffer(7)]],
    constant uint& is_vf [[buffer(8)]],
    constant uint& max_iters [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    uint iters = 0;
    uint empty_retries = 0;
    const uint max_empty_retries = 64;

    while (iters < max_iters) {
        // 从全局队列取任务
        uint my_idx = atomic_fetch_add_explicit(head, 1, memory_order_relaxed);
        uint cur_tail = atomic_load_explicit(tail, memory_order_relaxed);

        if (my_idx >= cur_tail) {
            atomic_fetch_sub_explicit(head, 1, memory_order_relaxed);
            empty_retries++;
            if (empty_retries >= max_empty_retries) break;
            continue;
        }

        CCDDomain domain = bufferData[my_idx % capacity];
        empty_retries = 0;

        device CCDData& d = data[domain.query_id];

        // 预取数据到寄存器（优化点 1）
        CachedVertices cv = cache_vertices(d);

        float current_toi = as_type<float>(atomic_load_explicit(toi_atomic, memory_order_relaxed));
        float min_t = domain.tuv[0].lower;

        if (min_t >= current_toi) {
            iters++;
            continue;
        }

        // max_iter 检查
        if (config.max_iter >= 0) {
            int checks = atomic_fetch_add_explicit((device atomic_int*)&d.nbr_checks, 1, memory_order_relaxed);
            if (checks > config.max_iter) {
                iters++;
                continue;
            }
        }

        // 使用 SIMD 展开的角点计算（优化点 2）
        float3 codomain_min, codomain_max;
        float t0 = domain.tuv[0].lower, t1 = domain.tuv[0].upper;
        float u0 = domain.tuv[1].lower, u1 = domain.tuv[1].upper;
        float v0 = domain.tuv[2].lower, v1 = domain.tuv[2].upper;

        if (is_vf) {
            compute_codomain_vf_unrolled(cv, t0, t1, u0, u1, v0, v1, codomain_min, codomain_max);
        } else {
            compute_codomain_ee_unrolled(cv, t0, t1, u0, u1, v0, v1, codomain_min, codomain_max);
        }

        // 检查原点是否在包含函数内
        bool origin_outside = any(codomain_min - cv.ms > cv.err) || any(codomain_max + cv.ms < -cv.err);
        if (origin_outside) {
            iters++;
            continue;
        }

        bool box_in = !any(codomain_min + cv.ms < -cv.err) && !any(codomain_max - cv.ms > cv.err);
        float true_tol = max(0.0f, max(codomain_max.x - codomain_min.x,
                                       max(codomain_max.y - codomain_min.y,
                                           codomain_max.z - codomain_min.z)));

        float3 widths = float3(t1 - t0, u1 - u0, v1 - v0);

        // 条件 1: domain 小于容差
        if (all(widths <= float3(cv.tol0, cv.tol1, cv.tol2))) {
            atomic_min_float(toi_atomic, min_t);
            iters++;
            continue;
        }

        // 条件 2: box 完全在 epsilon box 内
        if (box_in && (config.allow_zero_toi || min_t > 0)) {
            atomic_min_float(toi_atomic, min_t);
            iters++;
            continue;
        }

        // 条件 3: 真实容差小于目标容差
        if (true_tol <= config.co_domain_tolerance && (config.allow_zero_toi || min_t > 0)) {
            atomic_min_float(toi_atomic, min_t);
            iters++;
            continue;
        }

        // 选择分裂维度
        float3 res = widths / float3(cv.tol0, cv.tol1, cv.tol2);
        int split = (res.x >= res.y && res.x >= res.z) ? 0 :
                    (res.y >= res.x && res.y >= res.z) ? 1 : 2;

        float mid = (domain.tuv[split].lower + domain.tuv[split].upper) * 0.5f;

        if (mid <= domain.tuv[split].lower || mid >= domain.tuv[split].upper) {
            atomic_min_float(toi_atomic, min_t);
            iters++;
            continue;
        }

        // 推入第一半
        {
            uint new_tail = atomic_fetch_add_explicit(tail, 1, memory_order_relaxed);
            uint cur_head = atomic_load_explicit(head, memory_order_relaxed);
            if (new_tail - cur_head >= capacity) {
                atomic_store_explicit(overflow, 1, memory_order_relaxed);
            } else {
                CCDDomain d1 = domain;
                d1.tuv[split].upper = mid;
                bufferData[new_tail % capacity] = d1;
            }
        }

        // 推入第二半（带剪枝）
        bool push_second = true;
        if (split == 0) {
            current_toi = as_type<float>(atomic_load_explicit(toi_atomic, memory_order_relaxed));
            push_second = (mid <= current_toi);
        } else if (is_vf) {
            if (split == 1) {
                push_second = sum_less_than_one(mid, domain.tuv[2].lower);
            } else {
                push_second = sum_less_than_one(mid, domain.tuv[1].lower);
            }
        }

        if (push_second) {
            uint new_tail = atomic_fetch_add_explicit(tail, 1, memory_order_relaxed);
            uint cur_head = atomic_load_explicit(head, memory_order_relaxed);
            if (new_tail - cur_head >= capacity) {
                atomic_store_explicit(overflow, 1, memory_order_relaxed);
            } else {
                CCDDomain d2 = domain;
                d2.tuv[split].lower = mid;
                bufferData[new_tail % capacity] = d2;
            }
        }

        iters++;
    }
}
)";

    // 优化版 Narrow Phase Shader - Persistent Threads + SIMD 优化
    // 使用与原版相同的数据结构以确保兼容性
    static const char* kNarrowPhaseOptSrc = R"(
#include <metal_stdlib>
using namespace metal;

typedef float Scalar;

// 配置结构 - 与原版相同
struct CCDConfig {
    float co_domain_tolerance;
    int max_iter;
    bool use_ms;
    bool allow_zero_toi;
};

struct Interval {
    float lower;
    float upper;
};

// CCDDomain 结构 - 与原版相同
struct CCDDomain {
    Interval tuv[3];
    int query_id;
};

// CCDData 结构 - 使用显式结构匹配 C++ 端的 16 字节对齐 Vector3
// 每个 "Vector3" 是 {float x, y, z, _pad} = 16 bytes
struct Vector3Host {
    float x, y, z, _pad;
};

struct CCDData {
    Vector3Host v0s, v1s, v2s, v3s;  // 4 * 16 = 64 bytes
    Vector3Host v0e, v1e, v2e, v3e;  // 4 * 16 = 64 bytes
    float ms;                         // 4 bytes (offset 128)
    float tol[3];                     // 12 bytes (offset 132)
    Vector3Host err;                  // 16 bytes (offset 144)
    int nbr_checks;                   // 4 bytes (offset 160)
    // 隐式 padding 12 bytes 到 176 bytes
};

// 辅助函数：将 Vector3Host 转换为 float3
inline float3 toFloat3(Vector3Host v) {
    return float3(v.x, v.y, v.z);
}

// Atomic min for float
inline void atomic_min_float(device atomic_uint* addr, float val) {
    uint old_val = atomic_load_explicit(addr, memory_order_relaxed);
    uint new_val = as_type<uint>(val);
    while (as_type<float>(old_val) > val) {
        if (atomic_compare_exchange_weak_explicit(addr, &old_val, new_val,
                                                   memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

inline bool sum_less_than_one(float num1, float num2) {
    return num1 + num2 <= 1.0f / (1.0f - 1.19209e-07f);
}

inline float max_linf_4(float3 p1, float3 p2, float3 p3, float3 p4,
                        float3 p1e, float3 p2e, float3 p3e, float3 p4e) {
    float3 d1 = abs(p1e - p1), d2 = abs(p2e - p2);
    float3 d3 = abs(p3e - p3), d4 = abs(p4e - p4);
    float m1 = max(max(d1.x, max(d1.y, d1.z)), max(d2.x, max(d2.y, d2.z)));
    float m2 = max(max(d3.x, max(d3.y, d3.z)), max(d4.x, max(d4.y, d4.z)));
    return max(m1, m2);
}

// VF/EE 计算
inline float3 calculate_vf(device const CCDData& d, float t, float u, float v) {
    float3 v0s = toFloat3(d.v0s), v0e = toFloat3(d.v0e);
    float3 v1s = toFloat3(d.v1s), v1e = toFloat3(d.v1e);
    float3 v2s = toFloat3(d.v2s), v2e = toFloat3(d.v2e);
    float3 v3s = toFloat3(d.v3s), v3e = toFloat3(d.v3e);
    float3 v0 = mix(v0s, v0e, t);
    float3 t0 = mix(v1s, v1e, t);
    float3 t1 = mix(v2s, v2e, t);
    float3 t2 = mix(v3s, v3e, t);
    return v0 - (t1 - t0) * u - (t2 - t0) * v - t0;
}

inline float3 calculate_ee(device const CCDData& d, float t, float u, float v) {
    float3 v0s = toFloat3(d.v0s), v0e = toFloat3(d.v0e);
    float3 v1s = toFloat3(d.v1s), v1e = toFloat3(d.v1e);
    float3 v2s = toFloat3(d.v2s), v2e = toFloat3(d.v2e);
    float3 v3s = toFloat3(d.v3s), v3e = toFloat3(d.v3e);
    float3 ea0 = mix(v0s, v0e, t);
    float3 ea1 = mix(v1s, v1e, t);
    float3 eb0 = mix(v2s, v2e, t);
    float3 eb1 = mix(v3s, v3e, t);
    return mix(ea0, ea1, u) - mix(eb0, eb1, v);
}

// 计算容差
kernel void compute_tolerance_opt(
    device CCDData* data [[buffer(0)]],
    constant CCDConfig& config [[buffer(1)]],
    constant uint& is_vf [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;

    device CCDData& d = data[id];
    float3 v0s = toFloat3(d.v0s), v1s = toFloat3(d.v1s), v2s = toFloat3(d.v2s), v3s = toFloat3(d.v3s);
    float3 v0e = toFloat3(d.v0e), v1e = toFloat3(d.v1e), v2e = toFloat3(d.v2e), v3e = toFloat3(d.v3e);

    float3 p000, p001, p010, p011, p100, p101, p110, p111;

    if (is_vf) {
        p000 = v0s - v1s; p001 = v0s - v3s;
        p011 = v0s - (v2s + v3s - v1s); p010 = v0s - v2s;
        p100 = v0e - v1e; p101 = v0e - v3e;
        p111 = v0e - (v2e + v3e - v1e); p110 = v0e - v2e;
    } else {
        p000 = v0s - v2s; p001 = v0s - v3s;
        p010 = v1s - v2s; p011 = v1s - v3s;
        p100 = v0e - v2e; p101 = v0e - v3e;
        p110 = v1e - v2e; p111 = v1e - v3e;
    }

    float div = 3.0f * max_linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
    d.tol[0] = config.co_domain_tolerance / div;
    d.tol[1] = is_vf ? config.co_domain_tolerance / (3.0f * max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011))
                      : config.co_domain_tolerance / div;
    d.tol[2] = config.co_domain_tolerance / (3.0f * max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011));

    // Numerical error
    float filter = config.use_ms ? (is_vf ? 4.053116e-06f : 3.814698e-06f)
                                 : (is_vf ? 3.576279e-06f : 3.337861e-06f);

    float3 max_val = max(abs(v0s), max(abs(v1s), max(abs(v2s), abs(v3s))));
    max_val = max(max_val, max(abs(v0e), max(abs(v1e), max(abs(v2e), abs(v3e)))));
    max_val = max(max_val, float3(1.0f));
    float3 err_val = max_val * max_val * max_val * filter;
    d.err = {err_val.x, err_val.y, err_val.z, 0.0f};
    d.nbr_checks = 0;
}

kernel void init_buffer_opt(
    device CCDDomain* buffer [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;
    buffer[id].tuv[0] = {0.0f, 1.0f};
    buffer[id].tuv[1] = {0.0f, 1.0f};
    buffer[id].tuv[2] = {0.0f, 1.0f};
    buffer[id].query_id = int(id);
}

// Persistent Threads CCD Kernel
kernel void ccd_persistent(
    device CCDDomain* bufferData [[buffer(0)]],
    device atomic_uint* head [[buffer(1)]],
    device atomic_uint* tail [[buffer(2)]],
    device atomic_int* overflow [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    device CCDData* data [[buffer(5)]],
    device atomic_uint* toi_atomic [[buffer(6)]],
    constant CCDConfig& config [[buffer(7)]],
    constant uint& is_vf [[buffer(8)]],
    constant uint& max_iters [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    uint iters = 0;
    uint empty_retries = 0;
    const uint max_empty_retries = 64; // 增加重试次数

    while (iters < max_iters) {
        // 尝试从队列取任务
        uint my_idx = atomic_fetch_add_explicit(head, 1, memory_order_relaxed);
        uint cur_tail = atomic_load_explicit(tail, memory_order_relaxed);

        if (my_idx >= cur_tail) {
            // 队列可能为空，退还索引
            atomic_fetch_sub_explicit(head, 1, memory_order_relaxed);

            // 重试几次，因为其他线程可能正在 push 新任务
            empty_retries++;
            if (empty_retries >= max_empty_retries) {
                break; // 真的空了，退出
            }
            // 增加等待时间
            for (uint spin = 0; spin < 64; spin++) {
                // spin wait
            }
            continue;
        }

        // 成功取到任务，重置重试计数
        empty_retries = 0;

        CCDDomain domain = bufferData[my_idx % capacity];
        device CCDData& d = data[domain.query_id];

        float current_toi = as_type<float>(atomic_load_explicit(toi_atomic, memory_order_relaxed));
        float min_t = domain.tuv[0].lower;

        if (min_t >= current_toi) {
            iters++;
            continue;
        }

        // max_iter 检查
        if (config.max_iter >= 0) {
            int checks = atomic_fetch_add_explicit((device atomic_int*)&d.nbr_checks, 1, memory_order_relaxed);
            if (checks > config.max_iter) {
                iters++;
                continue;
            }
        }

        // 计算 8 个角点的 codomain
        float3 codomain_min = float3(FLT_MAX);
        float3 codomain_max = float3(-FLT_MAX);
        float3 err = toFloat3(d.err);
        float ms = d.ms;

        for (uint corner = 0; corner < 8; corner++) {
            float t = (corner & 1) ? domain.tuv[0].upper : domain.tuv[0].lower;
            float u = (corner & 2) ? domain.tuv[1].upper : domain.tuv[1].lower;
            float v = (corner & 4) ? domain.tuv[2].upper : domain.tuv[2].lower;

            float3 pt = is_vf ? calculate_vf(d, t, u, v) : calculate_ee(d, t, u, v);
            codomain_min = min(codomain_min, pt);
            codomain_max = max(codomain_max, pt);
        }

        // 检查原点是否在包含函数内
        bool origin_outside = any(codomain_min - ms > err) || any(codomain_max + ms < -err);
        if (origin_outside) {
            iters++;
            continue;
        }

        bool box_in = !any(codomain_min + ms < -err) && !any(codomain_max - ms > err);
        float true_tol = max(0.0f, max(codomain_max.x - codomain_min.x,
                                       max(codomain_max.y - codomain_min.y,
                                           codomain_max.z - codomain_min.z)));

        float3 widths = float3(domain.tuv[0].upper - domain.tuv[0].lower,
                               domain.tuv[1].upper - domain.tuv[1].lower,
                               domain.tuv[2].upper - domain.tuv[2].lower);

        // 条件 1: domain 小于容差
        if (all(widths <= float3(d.tol[0], d.tol[1], d.tol[2]))) {
            atomic_min_float(toi_atomic, min_t);
            iters++;
            continue;
        }

        // 条件 2: box 完全在 epsilon box 内
        if (box_in && (config.allow_zero_toi || min_t > 0)) {
            atomic_min_float(toi_atomic, min_t);
            iters++;
            continue;
        }

        // 条件 3: 真实容差小于目标容差
        if (true_tol <= config.co_domain_tolerance && (config.allow_zero_toi || min_t > 0)) {
            atomic_min_float(toi_atomic, min_t);
            iters++;
            continue;
        }

        // 选择分裂维度
        float3 res = widths / float3(d.tol[0], d.tol[1], d.tol[2]);
        int split = (res.x >= res.y && res.x >= res.z) ? 0 :
                    (res.y >= res.x && res.y >= res.z) ? 1 : 2;

        // Bisect
        float mid = (domain.tuv[split].lower + domain.tuv[split].upper) * 0.5f;

        if (mid <= domain.tuv[split].lower || mid >= domain.tuv[split].upper) {
            // 区间太小，视为找到碰撞
            atomic_min_float(toi_atomic, min_t);
            iters++;
            continue;
        }

        // 推入第一半
        {
            uint new_tail = atomic_fetch_add_explicit(tail, 1, memory_order_relaxed);
            uint cur_head = atomic_load_explicit(head, memory_order_relaxed);
            if (new_tail - cur_head >= capacity) {
                atomic_store_explicit(overflow, 1, memory_order_relaxed);
            } else {
                CCDDomain d1 = domain;
                d1.tuv[split].upper = mid;
                bufferData[new_tail % capacity] = d1;
            }
        }

        // 推入第二半 (带剪枝)
        bool push_second = true;
        if (split == 0) {
            current_toi = as_type<float>(atomic_load_explicit(toi_atomic, memory_order_relaxed));
            push_second = (mid <= current_toi);
        } else if (is_vf) {
            if (split == 1) {
                push_second = sum_less_than_one(mid, domain.tuv[2].lower);
            } else {
                push_second = sum_less_than_one(mid, domain.tuv[1].lower);
            }
        }

        if (push_second) {
            uint new_tail = atomic_fetch_add_explicit(tail, 1, memory_order_relaxed);
            uint cur_head = atomic_load_explicit(head, memory_order_relaxed);
            if (new_tail - cur_head >= capacity) {
                atomic_store_explicit(overflow, 1, memory_order_relaxed);
            } else {
                CCDDomain d2 = domain;
                d2.tuv[split].lower = mid;
                bufferData[new_tail % capacity] = d2;
            }
        }

        iters++;
    }
}
)";

    static const char* kNarrowPhaseSrc = R"(
#include <metal_stdlib>
using namespace metal;

// Force float for Metal
typedef float Scalar;
typedef float3 Vector3;

struct CCDConfig {
    Scalar co_domain_tolerance;
    int max_iter;
    bool use_ms;
    bool allow_zero_toi;
};

struct Interval {
    Scalar lower;
    Scalar upper;
    
    Interval() = default;
    Interval(Scalar l, Scalar u) : lower(l), upper(u) {}
};

struct SplitInterval {
    Interval first;
    Interval second;
    
    SplitInterval(Interval interval) {
        Scalar mid = (interval.lower + interval.upper) * 0.5f;
        first = Interval(interval.lower, mid);
        second = Interval(mid, interval.upper);
    }
};

struct CCDDomain {
    Interval tuv[3]; // t, u, v
    int query_id;
    
    // Add device qualifier to allow calling on device objects
    void init(int i) device {
        tuv[0] = Interval(0.0f, 1.0f);
        tuv[1] = Interval(0.0f, 1.0f);
        tuv[2] = Interval(0.0f, 1.0f);
        query_id = i;
    }
    
    // Also keep thread version if needed, or just use device if only called on device memory
    void init(int i) thread {
        tuv[0] = Interval(0.0f, 1.0f);
        tuv[1] = Interval(0.0f, 1.0f);
        tuv[2] = Interval(0.0f, 1.0f);
        query_id = i;
    }
};

struct DomainCorner {
    Scalar t, u, v;
    
    void update_tuv(const thread CCDDomain& domain, uint8_t corner) {
        t = (corner & 1) ? domain.tuv[0].upper : domain.tuv[0].lower;
        u = (corner & 2) ? domain.tuv[1].upper : domain.tuv[1].lower;
        v = (corner & 4) ? domain.tuv[2].upper : domain.tuv[2].lower;
    }
};

struct CCDData {
    Vector3 v0s, v1s, v2s, v3s;
    Vector3 v0e, v1e, v2e, v3e;
    Scalar ms;
    Scalar tol[3];
    Vector3 err;
    int nbr_checks;
};

struct CCDBuffer {
    device CCDDomain* data;
    device atomic_uint* head;
    device atomic_uint* tail;
    device atomic_int* overflow_flag;
    uint capacity;
    uint starting_size;
};

// --- Helper Functions ---

inline bool sum_less_than_one(Scalar num1, Scalar num2) {
    return num1 + num2 <= 1.0f / (1.0f - 1.19209e-07f);
}

inline Scalar max_Linf_4(Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4,
                         Vector3 p1e, Vector3 p2e, Vector3 p3e, Vector3 p4e) {
    Scalar m1 = max(max(abs(p1e - p1).x, max(abs(p1e - p1).y, abs(p1e - p1).z)),
                    max(abs(p2e - p2).x, max(abs(p2e - p2).y, abs(p2e - p2).z)));
    Scalar m2 = max(max(abs(p3e - p3).x, max(abs(p3e - p3).y, abs(p3e - p3).z)),
                    max(abs(p4e - p4).x, max(abs(p4e - p4).y, abs(p4e - p4).z)));
    return max(m1, m2);
}

// --- Tolerance Computation ---

void compute_face_vertex_tolerance(device CCDData& data_in, constant CCDConfig& config) {
    Vector3 p000 = data_in.v0s - data_in.v1s;
    Vector3 p001 = data_in.v0s - data_in.v3s;
    Vector3 p011 = data_in.v0s - (data_in.v2s + data_in.v3s - data_in.v1s);
    Vector3 p010 = data_in.v0s - data_in.v2s;
    Vector3 p100 = data_in.v0e - data_in.v1e;
    Vector3 p101 = data_in.v0e - data_in.v3e;
    Vector3 p111 = data_in.v0e - (data_in.v2e + data_in.v3e - data_in.v1e);
    Vector3 p110 = data_in.v0e - data_in.v2e;

    Scalar div = 3.0f * max_Linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
    data_in.tol[0] = config.co_domain_tolerance / div;
    data_in.tol[1] = config.co_domain_tolerance / (3.0f * max_Linf_4(p000, p100, p101, p001, p010, p110, p111, p011));
    data_in.tol[2] = config.co_domain_tolerance / (3.0f * max_Linf_4(p000, p100, p110, p010, p001, p101, p111, p011));
}

void compute_edge_edge_tolerance(device CCDData& data_in, constant CCDConfig& config) {
    Vector3 p000 = data_in.v0s - data_in.v2s;
    Vector3 p001 = data_in.v0s - data_in.v3s;
    Vector3 p010 = data_in.v1s - data_in.v2s;
    Vector3 p011 = data_in.v1s - data_in.v3s;
    Vector3 p100 = data_in.v0e - data_in.v2e;
    Vector3 p101 = data_in.v0e - data_in.v3e;
    Vector3 p110 = data_in.v1e - data_in.v2e;
    Vector3 p111 = data_in.v1e - data_in.v3e;

    Scalar div = 3.0f * max_Linf_4(p000, p001, p011, p010, p100, p101, p111, p110);
    data_in.tol[0] = config.co_domain_tolerance / div;
    data_in.tol[1] = config.co_domain_tolerance / div;
    data_in.tol[2] = config.co_domain_tolerance / (3.0f * max_Linf_4(p000, p100, p101, p001, p010, p110, p111, p011));
}

void get_numerical_error(device CCDData& data_in, bool use_ms, bool is_vf) {
    Scalar filter;
    if (!use_ms) {
        if (is_vf) filter = 3.576279e-06f;
        else       filter = 3.337861e-06f;
    } else {
        if (is_vf) filter = 4.053116e-06f;
        else       filter = 3.814698e-06f;
    }

    Vector3 max_val = max(abs(data_in.v0s), max(abs(data_in.v1s), max(abs(data_in.v2s), abs(data_in.v3s))));
    max_val = max(max_val, max(abs(data_in.v0e), max(abs(data_in.v1e), max(abs(data_in.v2e), abs(data_in.v3e)))));
    max_val = max(max_val, Vector3(1.0f));

    data_in.err = max_val * max_val * max_val * filter; 
}

kernel void compute_tolerance_kernel(
    device CCDData* data [[buffer(0)]],
    constant CCDConfig& config [[buffer(1)]],
    constant bool& is_vf [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;

    if (is_vf) {
        compute_face_vertex_tolerance(data[id], config);
    } else {
        compute_edge_edge_tolerance(data[id], config);
    }

    data[id].nbr_checks = 0;
    get_numerical_error(data[id], config.use_ms, is_vf);
}

kernel void initialize_buffer_kernel(
    device CCDDomain* bufferData [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;
    bufferData[id].init(id);
}

// Atomic Min for Float (using CAS)
inline void atomic_min_float(device atomic_uint* addr, float val) {
    uint old_val = atomic_load_explicit(addr, memory_order_relaxed);
    uint new_val = as_type<uint>(val);
    
    while (as_type<float>(old_val) > val) {
        if (atomic_compare_exchange_weak_explicit(addr, &old_val, new_val, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

inline Vector3 calculate_vf(device const CCDData& data_in, thread const DomainCorner& tuv) {
    Vector3 v = (data_in.v0e - data_in.v0s) * tuv.t + data_in.v0s;
    Vector3 t0 = (data_in.v1e - data_in.v1s) * tuv.t + data_in.v1s;
    Vector3 t1 = (data_in.v2e - data_in.v2s) * tuv.t + data_in.v2s;
    Vector3 t2 = (data_in.v3e - data_in.v3s) * tuv.t + data_in.v3s;
    return v - (t1 - t0) * tuv.u - (t2 - t0) * tuv.v - t0;
}

inline Vector3 calculate_ee(device const CCDData& data_in, thread const DomainCorner& tuv) {
    Vector3 ea0 = (data_in.v0e - data_in.v0s) * tuv.t + data_in.v0s;
    Vector3 ea1 = (data_in.v1e - data_in.v1s) * tuv.t + data_in.v1s;
    Vector3 eb0 = (data_in.v2e - data_in.v2s) * tuv.t + data_in.v2s;
    Vector3 eb1 = (data_in.v3e - data_in.v3s) * tuv.t + data_in.v3s;
    return ((ea1 - ea0) * tuv.u + ea0) - ((eb1 - eb0) * tuv.v + eb0);
}

inline bool origin_in_inclusion_function(
    device const CCDData& data_in,
    thread const CCDDomain& domain,
    thread Scalar& true_tol,
    thread bool& box_in,
    bool is_vf)
{
    Vector3 codomain_min = Vector3(FLT_MAX);
    Vector3 codomain_max = Vector3(-FLT_MAX);

    DomainCorner domain_corner;
    for (uint8_t corner = 0; corner < 8; corner++) {
        domain_corner.update_tuv(domain, corner);

        Vector3 codomain_corner;
        if (is_vf) {
            codomain_corner = calculate_vf(data_in, domain_corner);
        } else {
            codomain_corner = calculate_ee(data_in, domain_corner);
        }

        codomain_min = min(codomain_min, codomain_corner);
        codomain_max = max(codomain_max, codomain_corner);
    }

    Vector3 diff = codomain_max - codomain_min;
    true_tol = max(0.0f, max(diff.x, max(diff.y, diff.z)));

    box_in = true;

    if (any(codomain_min - data_in.ms > data_in.err) ||
        any(codomain_max + data_in.ms < -data_in.err)) {
        return false;
    }

    if (any(codomain_min + data_in.ms < -data_in.err) ||
        any(codomain_max - data_in.ms > data_in.err)) {
        box_in = false;
    }

    return true;
}

inline int split_dimension(device const CCDData& data, Vector3 width) {
    Vector3 res = width / Vector3(data.tol[0], data.tol[1], data.tol[2]);
    if (res.x >= res.y && res.x >= res.z) {
        return 0;
    } else if (res.y >= res.x && res.y >= res.z) {
        return 1;
    } else {
        return 2;
    }
}

// Buffer push helper (Monotonic)
inline void buffer_push(thread CCDBuffer& buffer, thread const CCDDomain& val) {
    uint tail = atomic_fetch_add_explicit(buffer.tail, 1, memory_order_relaxed);
    uint head = atomic_load_explicit(buffer.head, memory_order_relaxed);
    
    if (tail - head >= buffer.capacity) {
        atomic_store_explicit(buffer.overflow_flag, 1, memory_order_relaxed);
        return;
    }
    
    buffer.data[tail % buffer.capacity] = val;
}

inline bool bisect(
    thread const CCDDomain& domain,
    int split,
    device atomic_uint* toi_atomic, // Use atomic for global TOI
    thread CCDBuffer& buffer,
    bool is_vf)
{
    SplitInterval halves(domain.tuv[split]);

    if (halves.first.lower >= halves.first.upper || halves.second.lower >= halves.second.upper) {
        return true;
    }

    CCDDomain d1 = domain;
    d1.tuv[split] = halves.first;
    buffer_push(buffer, d1);

    CCDDomain d2 = domain;
    if (split == 0) {
        float current_toi = as_type<float>(atomic_load_explicit(toi_atomic, memory_order_relaxed));
        if (halves.second.lower <= current_toi) {
            d2.tuv[0] = halves.second;
            buffer_push(buffer, d2);
        }
    } else {
        if (is_vf) {
            if (split == 1) {
                if (sum_less_than_one(halves.second.lower, domain.tuv[2].lower)) {
                    d2.tuv[1] = halves.second;
                    buffer_push(buffer, d2);
                }
            } else if (split == 2) {
                if (sum_less_than_one(halves.second.lower, domain.tuv[1].lower)) {
                    d2.tuv[2] = halves.second;
                    buffer_push(buffer, d2);
                }
            }
        } else {
            d2.tuv[split] = halves.second;
            buffer_push(buffer, d2);
        }
    }

    return false;
}

kernel void ccd_kernel(
    device CCDDomain* bufferData [[buffer(0)]],
    device atomic_uint* bufferHead [[buffer(1)]],
    device atomic_uint* bufferTail [[buffer(2)]],
    device atomic_int* bufferOverflow [[buffer(3)]],
    constant uint& bufferCapacity [[buffer(4)]],
    constant uint& bufferStartingSize [[buffer(5)]],
    
    device CCDData* data [[buffer(6)]],
    device atomic_uint* toi [[buffer(7)]],
    constant CCDConfig& config [[buffer(8)]],
    constant bool& is_vf [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= bufferStartingSize) return;

    CCDBuffer buffer;
    buffer.data = bufferData;
    buffer.head = bufferHead;
    buffer.tail = bufferTail;
    buffer.overflow_flag = bufferOverflow;
    buffer.capacity = bufferCapacity;
    buffer.starting_size = bufferStartingSize;

    uint head = atomic_load_explicit(buffer.head, memory_order_relaxed);
    
    CCDDomain domain_in = buffer.data[(head + id) % buffer.capacity];
    
    device CCDData& data_in = data[domain_in.query_id];
    
    float current_toi = as_type<float>(atomic_load_explicit(toi, memory_order_relaxed));
    if (domain_in.tuv[0].lower >= current_toi) return;

    Scalar true_tol = 0;
    bool box_in = false;
    
    if (origin_in_inclusion_function(data_in, domain_in, true_tol, box_in, is_vf)) {
        Vector3 widths = Vector3(
            domain_in.tuv[0].upper - domain_in.tuv[0].lower,
            domain_in.tuv[1].upper - domain_in.tuv[1].lower,
            domain_in.tuv[2].upper - domain_in.tuv[2].lower
        );
        
        if (all(widths <= Vector3(data_in.tol[0], data_in.tol[1], data_in.tol[2]))) {
            atomic_min_float(toi, domain_in.tuv[0].lower);
            return;
        }
        
        if (box_in && (config.allow_zero_toi || domain_in.tuv[0].lower > 0)) {
            atomic_min_float(toi, domain_in.tuv[0].lower);
            return;
        }
        
        if (true_tol <= config.co_domain_tolerance && (config.allow_zero_toi || domain_in.tuv[0].lower > 0)) {
            atomic_min_float(toi, domain_in.tuv[0].lower);
            return;
        }
        
        int split = split_dimension(data_in, widths);
        bool sure_in = bisect(domain_in, split, toi, buffer, is_vf);
        
        if (sure_in) {
            atomic_min_float(toi, domain_in.tuv[0].lower);
            return;
        }
    }
}

kernel void shift_queue_start_kernel(
    device atomic_uint* bufferHead [[buffer(0)]],
    device atomic_uint* bufferTail [[buffer(1)]],
    device uint* bufferStartingSize [[buffer(2)]]
) {
    uint head = atomic_load_explicit(bufferHead, memory_order_relaxed);
    uint tail = atomic_load_explicit(bufferTail, memory_order_relaxed);
    uint startSize = *bufferStartingSize;
    
    head += startSize;
    atomic_store_explicit(bufferHead, head, memory_order_relaxed);
    
    *bufferStartingSize = tail - head;
}
)";

} // namespace

// Buffer Pool 用于复用 GPU Buffer
struct BufferPool {
    std::map<size_t, std::vector<id<MTLBuffer>>> available; // size -> buffers
    std::mutex mutex;
    bool enabled = true;
    size_t maxPoolSize = 64; // 每个 size 最多缓存的 buffer 数量

    id<MTLBuffer> acquire(id<MTLDevice> device, size_t size, MTLResourceOptions opts) {
        if (!enabled) {
            return [device newBufferWithLength:size options:opts];
        }

        // 向上取整到 4KB 对齐
        size_t alignedSize = ((size + 4095) / 4096) * 4096;

        std::lock_guard<std::mutex> lock(mutex);
        auto& pool = available[alignedSize];
        if (!pool.empty()) {
            id<MTLBuffer> buf = pool.back();
            pool.pop_back();
            return buf;
        }
        return [device newBufferWithLength:alignedSize options:opts];
    }

    void release(id<MTLBuffer> buffer) {
        if (!enabled || !buffer) return;

        size_t size = buffer.length;
        std::lock_guard<std::mutex> lock(mutex);
        auto& pool = available[size];
        if (pool.size() < maxPoolSize) {
            pool.push_back(buffer);
        }
        // 超过限制则让 ARC 释放
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        available.clear();
    }
};

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

    // 融合内核
    id<MTLLibrary> libSTQFusedTwo = nil;
    id<MTLComputePipelineState> cpsSTQFusedTwo = nil;
    id<MTLLibrary> libSTQFusedSingle = nil;
    id<MTLComputePipelineState> cpsSTQFusedSingle = nil;

    id<MTLLibrary> libScan = nil;
    id<MTLComputePipelineState> cpsScanSingle = nil;
    id<MTLComputePipelineState> cpsScanReduce = nil;
    id<MTLComputePipelineState> cpsScanAdd = nil;

    id<MTLLibrary> libCompact = nil;
    id<MTLComputePipelineState> cpsCompact = nil;

    id<MTLLibrary> libYZAtomic = nil;
    id<MTLComputePipelineState> cpsYZAtomic = nil;

    // Narrow Phase Pipelines
    id<MTLLibrary> libNarrow = nil;
    id<MTLComputePipelineState> cpsComputeTol = nil;
    id<MTLComputePipelineState> cpsInitBuffer = nil;
    id<MTLComputePipelineState> cpsCCD = nil;
    id<MTLComputePipelineState> cpsShiftQueue = nil;

    // 优化版 Narrow Phase Pipelines
    id<MTLLibrary> libNarrowOpt = nil;
    id<MTLComputePipelineState> cpsComputeTolOpt = nil;
    id<MTLComputePipelineState> cpsInitBufferOpt = nil;
    id<MTLComputePipelineState> cpsCCDPersistent = nil;

    // 优化版 V2 Narrow Phase Pipelines
    id<MTLLibrary> libNarrowOptV2 = nil;
    id<MTLComputePipelineState> cpsComputeTolV2 = nil;
    id<MTLComputePipelineState> cpsInitBufferV2 = nil;
    id<MTLComputePipelineState> cpsCCDPersistentV2 = nil;

    // Buffer Pool
    BufferPool bufferPool;

    bool ok = false;
    double lastYZMs = -1.0;
    double lastPairsMs = -1.0;

    // CUDA 风格内存管理：动态查询可用内存（类似 cudaMemGetInfo）
    // 返回可用于分配 buffer 的字节数
    size_t getAllocatableMemory() {
        if (!dev) return 0;
        // Metal 内存查询 API
        size_t recommendedMax = dev.recommendedMaxWorkingSetSize;
        size_t currentUsed = dev.currentAllocatedSize;
        size_t maxBuffer = dev.maxBufferLength;

        // 可用内存 = 推荐最大工作集 - 当前已用，保留 5% 余量
        size_t available = (recommendedMax > currentUsed) ?
                           static_cast<size_t>((recommendedMax - currentUsed) * 0.95) : 0;

        // 单个 buffer 不能超过 maxBufferLength
        return std::min(available, maxBuffer);
    }

    // 计算给定可用内存下可以容纳的最大 pair 数量
    size_t getMaxPairCapacity(size_t availableBytes) {
        // 每个 pair 占用 sizeof(int2) = 8 字节
        return availableBytes / sizeof(int32_t) / 2;
    }
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
    return true;
}

// ... (Existing methods: yzFilterAtomic, filterYZ, stqTwoLists, stqSingleList,
// scan, compact) ...

// 内部函数：处理单个批次的 narrow phase（原始多遍实现）
static bool narrowPhaseBatch(
    id<MTLDevice> dev,
    id<MTLCommandQueue> q,
    id<MTLComputePipelineState>& cpsComputeTol,
    id<MTLComputePipelineState>& cpsInitBuffer,
    id<MTLComputePipelineState>& cpsCCD,
    id<MTLComputePipelineState>& cpsShiftQueue,
    id<MTLLibrary>& libNarrow,
    const std::vector<double>& vertices_t0,
    const std::vector<double>& vertices_t1,
    const std::vector<int32_t>& indices,
    const std::vector<std::pair<int, int>>& overlaps,
    bool is_vf,
    float ms,
    float tolerance,
    int max_iter,
    bool allow_zero_toi,
    double& toi)
{
    if (overlaps.empty())
        return true;

    @autoreleasepool {
        NSError* err = nil;
        if (!cpsCCD) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            libNarrow = [dev
                newLibraryWithSource:[NSString
                                         stringWithUTF8String:kNarrowPhaseSrc]
                             options:opts
                               error:&err];
            if (!libNarrow) {
                NSLog(@"Error compiling narrow phase shader: %@", err);
                return false;
            }

            auto getPipeline =
                [&](NSString* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> fn =
                    [libNarrow newFunctionWithName:name];
                if (!fn)
                    return nil;
                return [dev newComputePipelineStateWithFunction:fn
                                                                 error:&err];
            };

            cpsComputeTol = getPipeline(@"compute_tolerance_kernel");
            cpsInitBuffer = getPipeline(@"initialize_buffer_kernel");
            cpsCCD = getPipeline(@"ccd_kernel");
            cpsShiftQueue = getPipeline(@"shift_queue_start_kernel");

            if (!cpsComputeTol || !cpsInitBuffer || !cpsCCD
                || !cpsShiftQueue) {
                NSLog(@"Error creating pipeline states for narrow phase");
                return false;
            }
        }

        // Prepare Data
        size_t n_overlaps = overlaps.size();

        // Metal float3 is 16-byte aligned and has size 16 (x, y, z, pad)
        struct alignas(16) Vector3 {
            float x, y, z;
            float _pad = 0.0f;
        };

        struct alignas(16) CCDData {
            Vector3 v0s, v1s, v2s, v3s;
            Vector3 v0e, v1e, v2e, v3e;
            float ms;
            float tol[3];
            Vector3 err;
            int nbr_checks;
            char _pad[12]; // Pad to 176 bytes (multiple of 16)
        };

        std::vector<CCDData> host_data(n_overlaps);

        for (size_t i = 0; i < n_overlaps; ++i) {
            const auto& pair = overlaps[i];
            host_data[i].ms = ms;
            host_data[i].nbr_checks = 0;
            // Initialize padding to avoid garbage values (good practice)
            memset(host_data[i]._pad, 0, sizeof(host_data[i]._pad));

            auto get_v = [&](int idx,
                             const std::vector<double>& verts) -> Vector3 {
                return { (float)verts[3 * idx], (float)verts[3 * idx + 1],
                         (float)verts[3 * idx + 2], 0.0f };
            };

            if (is_vf) {
                int vi = pair.first;
                int fi = pair.second;
                int f0 = indices[3 * fi];
                int f1 = indices[3 * fi + 1];
                int f2 = indices[3 * fi + 2];

                host_data[i].v0s = get_v(vi, vertices_t0);
                host_data[i].v1s = get_v(f0, vertices_t0);
                host_data[i].v2s = get_v(f1, vertices_t0);
                host_data[i].v3s = get_v(f2, vertices_t0);

                host_data[i].v0e = get_v(vi, vertices_t1);
                host_data[i].v1e = get_v(f0, vertices_t1);
                host_data[i].v2e = get_v(f1, vertices_t1);
                host_data[i].v3e = get_v(f2, vertices_t1);
            } else {
                int ea = pair.first;
                int eb = pair.second;
                int ea0 = indices[2 * ea];
                int ea1 = indices[2 * ea + 1];
                int eb0 = indices[2 * eb];
                int eb1 = indices[2 * eb + 1];

                host_data[i].v0s = get_v(ea0, vertices_t0);
                host_data[i].v1s = get_v(ea1, vertices_t0);
                host_data[i].v2s = get_v(eb0, vertices_t0);
                host_data[i].v3s = get_v(eb1, vertices_t0);

                host_data[i].v0e = get_v(ea0, vertices_t1);
                host_data[i].v1e = get_v(ea1, vertices_t1);
                host_data[i].v2e = get_v(eb0, vertices_t1);
                host_data[i].v3e = get_v(eb1, vertices_t1);
            }
        }

        id<MTLBuffer> bData =
            [dev newBufferWithBytes:host_data.data()
                                    length:sizeof(CCDData) * n_overlaps
                                   options:MTLResourceStorageModeShared];

        struct CCDConfig {
            float co_domain_tolerance;
            int max_iter;
            bool use_ms;
            bool allow_zero_toi;
        };
        CCDConfig config = { tolerance, max_iter, ms > 0, allow_zero_toi };
        id<MTLBuffer> bConfig =
            [dev newBufferWithBytes:&config
                                    length:sizeof(CCDConfig)
                                   options:MTLResourceStorageModeShared];

        uint32_t n_overlaps_u32 = (uint32_t)n_overlaps;
        id<MTLBuffer> bN =
            [dev newBufferWithBytes:&n_overlaps_u32
                                    length:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];

        uint32_t is_vf_u32 = is_vf ? 1 : 0;
        id<MTLBuffer> bIsVF =
            [dev newBufferWithBytes:&is_vf_u32
                                    length:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];

        // Compute Tolerance
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:cpsComputeTol];
        [enc setBuffer:bData offset:0 atIndex:0];
        [enc setBuffer:bConfig offset:0 atIndex:1];
        [enc setBuffer:bIsVF offset:0 atIndex:2];
        [enc setBuffer:bN offset:0 atIndex:3];

        MTLSize grid = MTLSizeMake(n_overlaps, 1, 1);
        MTLSize tg = MTLSizeMake(
            std::min<NSUInteger>(
                cpsComputeTol.maxTotalThreadsPerThreadgroup, n_overlaps),
            1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];

        // Initialize Buffer
        // Buffer size logic: 每查询可能需要大量子域（二分搜索）
        // 每查询约 512 子域空间，最小 1M，最大 8M
        uint32_t capacity = 1;
        while (capacity < n_overlaps * 512) capacity <<= 1;
        if (capacity < 1024 * 1024) capacity = 1024 * 1024;  // 最小 1M
        if (capacity > 8 * 1024 * 1024) capacity = 8 * 1024 * 1024;  // 最大 8M

        // Struct size of CCDDomain is 3*2*4 + 4 = 28 bytes (approx).
        // In Metal shader: Interval(8) * 3 + int(4) = 28 bytes.
        // Alignment might make it 32.
        // Let's assume 32 bytes per domain.

        id<MTLBuffer> bBufferData =
            [dev newBufferWithLength:32 * capacity
                                    options:MTLResourceStorageModeShared];

        uint32_t zero = 0;
        uint32_t starting_size = n_overlaps_u32;
        int32_t overflow = 0;

        id<MTLBuffer> bHead =
            [dev newBufferWithBytes:&zero
                                    length:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bTail =
            [dev newBufferWithBytes:&starting_size
                                    length:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bOverflow =
            [dev newBufferWithBytes:&overflow
                                    length:sizeof(int32_t)
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bCapacity =
            [dev newBufferWithBytes:&capacity
                                    length:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bStartingSize =
            [dev newBufferWithBytes:&starting_size
                                    length:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];

        float toi_f = (float)toi;
        id<MTLBuffer> bToI =
            [dev newBufferWithBytes:&toi_f
                                    length:sizeof(float)
                                   options:MTLResourceStorageModeShared];

        // Init Buffer Kernel
        cb = [q commandBuffer];
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:cpsInitBuffer];
        [enc setBuffer:bBufferData offset:0 atIndex:0];
        [enc setBuffer:bN offset:0 atIndex:1];
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];

        // Loop
        int pass = 0;
        while (true) {
            uint32_t current_n =
                *static_cast<uint32_t*>([bStartingSize contents]);
            if (current_n == 0)
                break;

            float current_toi = *static_cast<float*>([bToI contents]);
            if (current_toi == 0.0f && !allow_zero_toi)
                break; // Early exit if collision found at 0

            // Debug logging - disabled to reduce noise
            // if (pass % 100 == 0 || n_overlaps <= 10) {
            //     uint32_t h = *static_cast<uint32_t*>([bHead contents]);
            //     uint32_t t = *static_cast<uint32_t*>([bTail contents]);
            //     std::cout << "Pass " << pass << ": n=" << current_n
            //               << " head=" << h << " tail=" << t
            //               << " toi=" << current_toi << std::endl;
            // }
            pass++;

            cb = [q commandBuffer];
            enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:cpsCCD];
            [enc setBuffer:bBufferData offset:0 atIndex:0];
            [enc setBuffer:bHead offset:0 atIndex:1];
            [enc setBuffer:bTail offset:0 atIndex:2];
            [enc setBuffer:bOverflow offset:0 atIndex:3];
            [enc setBuffer:bCapacity offset:0 atIndex:4];
            [enc setBuffer:bStartingSize offset:0 atIndex:5];
            [enc setBuffer:bData offset:0 atIndex:6];
            [enc setBuffer:bToI offset:0 atIndex:7];
            [enc setBuffer:bConfig offset:0 atIndex:8];
            [enc setBuffer:bIsVF offset:0 atIndex:9];

            MTLSize loopGrid = MTLSizeMake(current_n, 1, 1);
            MTLSize loopTg = MTLSizeMake(
                std::min<NSUInteger>(
                    cpsCCD.maxTotalThreadsPerThreadgroup, current_n),
                1, 1);
            [enc dispatchThreads:loopGrid threadsPerThreadgroup:loopTg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted]; // Need to wait to check overflow and
                                     // update size

            int32_t is_overflow = *static_cast<int32_t*>([bOverflow contents]);
            if (is_overflow) {
                NSLog(@"Metal2 Narrow Phase Overflow!");
                return false; // Or handle overflow by resizing
            }

            // Shift Queue
            cb = [q commandBuffer];
            enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:cpsShiftQueue];
            [enc setBuffer:bHead offset:0 atIndex:0];
            [enc setBuffer:bTail offset:0 atIndex:1];
            [enc setBuffer:bStartingSize offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        toi = (double)*static_cast<float*>([bToI contents]);
    }
    return true;
}

// 公开的 narrowPhase 函数：使用分批处理避免溢出
bool Metal2Runtime::narrowPhase(
    const std::vector<double>& vertices_t0,
    const std::vector<double>& vertices_t1,
    const std::vector<int32_t>& indices,
    const std::vector<std::pair<int, int>>& overlaps,
    bool is_vf,
    float ms,
    float tolerance,
    int max_iter,
    bool allow_zero_toi,
    double& toi)
{
    if (!available())
        return false;
    if (overlaps.empty())
        return true;

    // 分批处理：初始批次大小 100 个查询
    // 如果溢出，自动减半重试，最小批次为 10
    size_t batch_size = 100;
    const size_t MIN_BATCH_SIZE = 10;

    double min_toi = toi;
    size_t total = overlaps.size();
    size_t start = 0;

    while (start < total) {
        size_t end = std::min(start + batch_size, total);
        std::vector<std::pair<int, int>> batch_overlaps(overlaps.begin() + start, overlaps.begin() + end);

        double batch_toi = min_toi;
        bool result = narrowPhaseBatch(
            impl_->dev, impl_->q,
            impl_->cpsComputeTol, impl_->cpsInitBuffer, impl_->cpsCCD, impl_->cpsShiftQueue,
            impl_->libNarrow,
            vertices_t0, vertices_t1, indices, batch_overlaps,
            is_vf, ms, tolerance, max_iter, allow_zero_toi, batch_toi);

        if (!result) {
            // 溢出：减半批次大小重试
            if (batch_size > MIN_BATCH_SIZE) {
                batch_size = std::max(MIN_BATCH_SIZE, batch_size / 2);
                // 不推进 start，用更小批次重试
                continue;
            } else {
                // 已经是最小批次仍然溢出，处理单个查询
                // 跳过这个批次（可能是极端情况）
                NSLog(@"Metal2 Narrow Phase: Skipping batch at start=%zu, batch_size=%zu due to overflow", start, batch_size);
                start = end;
                continue;
            }
        }

        min_toi = std::min(min_toi, batch_toi);

        // 如果已经找到 TOI=0 的碰撞，可以提前退出（除非 allow_zero_toi）
        if (min_toi == 0.0 && !allow_zero_toi) {
            break;
        }

        start = end;
    }

    toi = min_toi;
    return true;
}

// CUDA 风格 CCD 内核（使用真正的循环队列）
// 使用 CAS 实现 atomicIncWrap，让 tail 值真正回绕到 [0, capacity) 范围
static const char* kCCDKernelCudaStyleSrc = R"(
#include <metal_stdlib>
using namespace metal;

struct CCDConfig {
    float co_domain_tolerance;
    int max_iter;
    bool use_ms;
    bool allow_zero_toi;
};

struct Interval {
    float lower;
    float upper;
};

struct CCDDomain {
    Interval tuv[3];
    int query_id;
};

struct Vector3Host {
    float x, y, z, _pad;
};

struct CCDData {
    Vector3Host v0s, v1s, v2s, v3s;
    Vector3Host v0e, v1e, v2e, v3e;
    float ms;
    float tol[3];
    Vector3Host err;
    int nbr_checks;
};

inline float3 toFloat3(Vector3Host v) {
    return float3(v.x, v.y, v.z);
}

inline void atomic_min_float(device atomic_uint* addr, float val) {
    uint old_val = atomic_load_explicit(addr, memory_order_relaxed);
    uint new_val = as_type<uint>(val);
    while (as_type<float>(old_val) > val) {
        if (atomic_compare_exchange_weak_explicit(addr, &old_val, new_val,
                                                   memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

inline bool sum_less_than_one(float num1, float num2) {
    return num1 + num2 <= 1.0f / (1.0f - 1.19209e-07f);
}

// NOTE: atomicIncWrap removed - using monotonic head/tail with modulo instead
// This avoids the race condition where head and tail can wrap to same index

// CUDA-style atomicInc: returns old value and increments, wrapping at limit
inline uint atomicIncWrapCuda(device atomic_uint* addr, uint limit) {
    uint old_val = atomic_load_explicit(addr, memory_order_relaxed);
    uint new_val;
    do {
        new_val = (old_val >= limit) ? 0 : (old_val + 1);
    } while (!atomic_compare_exchange_weak_explicit(addr, &old_val, new_val,
                                                     memory_order_relaxed, memory_order_relaxed));
    return old_val;
}

inline float3 calculate_vf(device const CCDData& d, float t, float u, float v) {
    float3 v0s = toFloat3(d.v0s), v0e = toFloat3(d.v0e);
    float3 v1s = toFloat3(d.v1s), v1e = toFloat3(d.v1e);
    float3 v2s = toFloat3(d.v2s), v2e = toFloat3(d.v2e);
    float3 v3s = toFloat3(d.v3s), v3e = toFloat3(d.v3e);
    float3 v0 = mix(v0s, v0e, t);
    float3 t0 = mix(v1s, v1e, t);
    float3 t1 = mix(v2s, v2e, t);
    float3 t2 = mix(v3s, v3e, t);
    return v0 - (t1 - t0) * u - (t2 - t0) * v - t0;
}

inline float3 calculate_ee(device const CCDData& d, float t, float u, float v) {
    float3 v0s = toFloat3(d.v0s), v0e = toFloat3(d.v0e);
    float3 v1s = toFloat3(d.v1s), v1e = toFloat3(d.v1e);
    float3 v2s = toFloat3(d.v2s), v2e = toFloat3(d.v2e);
    float3 v3s = toFloat3(d.v3s), v3e = toFloat3(d.v3e);
    float3 ea0 = mix(v0s, v0e, t);
    float3 ea1 = mix(v1s, v1e, t);
    float3 eb0 = mix(v2s, v2e, t);
    float3 eb1 = mix(v3s, v3e, t);
    return mix(ea0, ea1, u) - mix(eb0, eb1, v);
}

inline float max_linf_4_cuda(float3 p1, float3 p2, float3 p3, float3 p4,
                        float3 p1e, float3 p2e, float3 p3e, float3 p4e) {
    float3 d1 = abs(p1e - p1), d2 = abs(p2e - p2);
    float3 d3 = abs(p3e - p3), d4 = abs(p4e - p4);
    float m1 = max(max(d1.x, max(d1.y, d1.z)), max(d2.x, max(d2.y, d2.z)));
    float m2 = max(max(d3.x, max(d3.y, d3.z)), max(d4.x, max(d4.y, d4.z)));
    return max(m1, m2);
}

// 容差计算内核 - 与 CCD 内核使用相同的数据结构
kernel void compute_tolerance_cuda_style(
    device CCDData* data [[buffer(0)]],
    constant CCDConfig& config [[buffer(1)]],
    constant uint& is_vf [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;

    device CCDData& d = data[id];
    float3 v0s = toFloat3(d.v0s), v1s = toFloat3(d.v1s), v2s = toFloat3(d.v2s), v3s = toFloat3(d.v3s);
    float3 v0e = toFloat3(d.v0e), v1e = toFloat3(d.v1e), v2e = toFloat3(d.v2e), v3e = toFloat3(d.v3e);

    float3 p000, p001, p010, p011, p100, p101, p110, p111;

    if (is_vf) {
        p000 = v0s - v1s; p001 = v0s - v3s;
        p011 = v0s - (v2s + v3s - v1s); p010 = v0s - v2s;
        p100 = v0e - v1e; p101 = v0e - v3e;
        p111 = v0e - (v2e + v3e - v1e); p110 = v0e - v2e;
    } else {
        p000 = v0s - v2s; p001 = v0s - v3s;
        p010 = v1s - v2s; p011 = v1s - v3s;
        p100 = v0e - v2e; p101 = v0e - v3e;
        p110 = v1e - v2e; p111 = v1e - v3e;
    }

    float div = 3.0f * max_linf_4_cuda(p000, p001, p011, p010, p100, p101, p111, p110);
    d.tol[0] = config.co_domain_tolerance / div;
    d.tol[1] = is_vf ? config.co_domain_tolerance / (3.0f * max_linf_4_cuda(p000, p100, p101, p001, p010, p110, p111, p011))
                      : config.co_domain_tolerance / div;
    d.tol[2] = config.co_domain_tolerance / (3.0f * max_linf_4_cuda(p000, p100, p110, p010, p001, p101, p111, p011));

    // Numerical error
    float filter = config.use_ms ? (is_vf ? 4.053116e-06f : 3.814698e-06f)
                                 : (is_vf ? 3.576279e-06f : 3.337861e-06f);

    float3 max_val = max(abs(v0s), max(abs(v1s), max(abs(v2s), abs(v3s))));
    max_val = max(max_val, max(abs(v0e), max(abs(v1e), max(abs(v2e), abs(v3e)))));
    max_val = max(max_val, float3(1.0f));
    float3 err_val = max_val * max_val * max_val * filter;
    d.err = {err_val.x, err_val.y, err_val.z, 0.0f};
    d.nbr_checks = 0;
}

// 初始化缓冲区内核
kernel void init_buffer_cuda_style(
    device CCDDomain* buffer [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;
    buffer[id].tuv[0] = {0.0f, 1.0f};
    buffer[id].tuv[1] = {0.0f, 1.0f};
    buffer[id].tuv[2] = {0.0f, 1.0f};
    buffer[id].query_id = int(id);
}

// 真正的循环队列内核
// 使用单调增长的 head/tail + modulo 访问（与 ccd_persistent 相同）
// queueSize 不再使用 - head/tail 差值即为队列大小
kernel void ccd_kernel_cuda_style(
    device CCDDomain* bufferData [[buffer(0)]],
    device atomic_uint* head [[buffer(1)]],           // 单调增长的 head
    device atomic_uint* tail [[buffer(2)]],           // 单调增长的 tail
    device atomic_int* overflow [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],            // 缓冲区容量
    device atomic_uint* queueSize [[buffer(5)]],      // 不再使用，保留兼容性
    device CCDData* data [[buffer(6)]],
    device atomic_uint* toi_atomic [[buffer(7)]],
    constant CCDConfig& config [[buffer(8)]],
    constant uint& is_vf [[buffer(9)]],
    constant uint& maxIters [[buffer(10)]],
    device atomic_uint* debugCounters [[buffer(11)]], // 调试计数器
    device float* debugFloats [[buffer(12)]],
    device atomic_uint* slotValid [[buffer(13)]],     // 不再使用
    uint tid [[thread_position_in_grid]]
) {
    uint iters = 0;
    uint emptyRetries = 0;
    const uint maxEmptyRetries = 256;  // 与 ccd_persistent 类似

    while (iters < maxIters) {
        // 尝试从队列取任务（与 ccd_persistent 相同）
        uint my_idx = atomic_fetch_add_explicit(head, 1, memory_order_relaxed);
        uint cur_tail = atomic_load_explicit(tail, memory_order_relaxed);

        if (my_idx >= cur_tail) {
            // 队列可能为空，退还索引
            atomic_fetch_sub_explicit(head, 1, memory_order_relaxed);
            emptyRetries++;
            if (emptyRetries >= maxEmptyRetries) break;
            // spin wait
            for (uint spin = 0; spin < 64; spin++) {}
            continue;
        }
        emptyRetries = 0;

        // 使用 modulo 访问缓冲区
        uint readIdx = my_idx % capacity;
        CCDDomain domain = bufferData[readIdx];

        device CCDData& d = data[domain.query_id];

        uint myProcessedIdx = atomic_fetch_add_explicit(&debugCounters[0], 1, memory_order_relaxed); // processed

        float current_toi = as_type<float>(atomic_load_explicit(toi_atomic, memory_order_relaxed));
        float min_t = domain.tuv[0].lower;

        if (min_t >= current_toi) {
            iters++;
            continue;
        }

        // max_iter 检查
        if (config.max_iter >= 0) {
            int checks = atomic_fetch_add_explicit((device atomic_int*)&d.nbr_checks, 1, memory_order_relaxed);
            if (checks > config.max_iter) {
                iters++;
                continue;
            }
        }

        // 计算 8 个角点的 codomain
        float3 codomain_min = float3(FLT_MAX);
        float3 codomain_max = float3(-FLT_MAX);
        float3 err = toFloat3(d.err);
        float ms_val = d.ms;

        for (uint corner = 0; corner < 8; corner++) {
            float t = (corner & 1) ? domain.tuv[0].upper : domain.tuv[0].lower;
            float u = (corner & 2) ? domain.tuv[1].upper : domain.tuv[1].lower;
            float v = (corner & 4) ? domain.tuv[2].upper : domain.tuv[2].lower;

            float3 pt = is_vf ? calculate_vf(d, t, u, v) : calculate_ee(d, t, u, v);
            codomain_min = min(codomain_min, pt);
            codomain_max = max(codomain_max, pt);
        }

        // 存储前几个域的调试信息（使用已获取的唯一索引）
        if (myProcessedIdx < 8) {  // 只存储前8个
            uint base = myProcessedIdx * 16;  // 每个域16个float: domain(6) + codomain_min(3) + codomain_max(3) + err(3) + ms(1)
            debugFloats[base + 0] = domain.tuv[0].lower;
            debugFloats[base + 1] = domain.tuv[0].upper;
            debugFloats[base + 2] = domain.tuv[1].lower;
            debugFloats[base + 3] = domain.tuv[1].upper;
            debugFloats[base + 4] = domain.tuv[2].lower;
            debugFloats[base + 5] = domain.tuv[2].upper;
            debugFloats[base + 6] = codomain_min.x;
            debugFloats[base + 7] = codomain_min.y;
            debugFloats[base + 8] = codomain_min.z;
            debugFloats[base + 9] = codomain_max.x;
            debugFloats[base + 10] = codomain_max.y;
            debugFloats[base + 11] = codomain_max.z;
            debugFloats[base + 12] = err.x;
            debugFloats[base + 13] = err.y;
            debugFloats[base + 14] = err.z;
            debugFloats[base + 15] = ms_val;
        }

        bool origin_outside = any(codomain_min - ms_val > err) || any(codomain_max + ms_val < -err);
        if (origin_outside) {
            atomic_fetch_add_explicit(&debugCounters[1], 1, memory_order_relaxed); // origin_outside
            iters++;
            continue;
        }

        bool box_in = !any(codomain_min + ms_val < -err) && !any(codomain_max - ms_val > err);
        float true_tol = max(0.0f, max(codomain_max.x - codomain_min.x,
                                       max(codomain_max.y - codomain_min.y,
                                           codomain_max.z - codomain_min.z)));

        float3 widths = float3(domain.tuv[0].upper - domain.tuv[0].lower,
                               domain.tuv[1].upper - domain.tuv[1].lower,
                               domain.tuv[2].upper - domain.tuv[2].lower);

        // 条件 1: domain 小于容差
        if (all(widths <= float3(d.tol[0], d.tol[1], d.tol[2]))) {
            atomic_fetch_add_explicit(&debugCounters[2], 1, memory_order_relaxed); // cond1
            atomic_min_float(toi_atomic, min_t);
            iters++;
            continue;
        }

        // 条件 2: box 完全在 epsilon box 内
        if (box_in && (config.allow_zero_toi || min_t > 0)) {
            atomic_fetch_add_explicit(&debugCounters[3], 1, memory_order_relaxed); // cond2
            atomic_min_float(toi_atomic, min_t);
            iters++;
            continue;
        }

        // 条件 3: 真实容差小于目标容差
        if (true_tol <= config.co_domain_tolerance && (config.allow_zero_toi || min_t > 0)) {
            atomic_fetch_add_explicit(&debugCounters[4], 1, memory_order_relaxed); // cond3
            atomic_min_float(toi_atomic, min_t);
            iters++;
            continue;
        }

        atomic_fetch_add_explicit(&debugCounters[5], 1, memory_order_relaxed); // will split

        // 选择分裂维度
        float3 res = widths / float3(d.tol[0], d.tol[1], d.tol[2]);
        int split = (res.x >= res.y && res.x >= res.z) ? 0 :
                    (res.y >= res.x && res.y >= res.z) ? 1 : 2;

        float mid = (domain.tuv[split].lower + domain.tuv[split].upper) * 0.5f;

        if (mid <= domain.tuv[split].lower || mid >= domain.tuv[split].upper) {
            atomic_min_float(toi_atomic, min_t);
            iters++;
            continue;
        }

        // 推入第一个子域 - 使用重试循环处理竞态条件
        bool pushed1 = false;
        for (int retry = 0; retry < 16 && !pushed1; retry++) {
            uint new_tail = atomic_fetch_add_explicit(tail, 1, memory_order_relaxed);
            uint cur_head = atomic_load_explicit(head, memory_order_relaxed);
            // 注意：使用有符号比较避免下溢问题
            int queue_used = (int)new_tail - (int)cur_head;
            if (queue_used >= 0 && (uint)queue_used < capacity) {
                // 成功：可以推入
                CCDDomain d1 = domain;
                d1.tuv[split].upper = mid;
                bufferData[new_tail % capacity] = d1;
                atomic_fetch_add_explicit(&debugCounters[6], 1, memory_order_relaxed); // pushed1
                pushed1 = true;
            } else {
                // 退还 tail 递增
                atomic_fetch_sub_explicit(tail, 1, memory_order_relaxed);
                if (queue_used >= 0) {
                    // 真正的溢出
                    atomic_store_explicit(overflow, 1, memory_order_relaxed);
                    break;
                }
                // 竞态条件：稍等后重试
                for (int spin = 0; spin < 32; spin++) {}
            }
        }

        // 推入第二个子域（带剪枝）
        bool push_second = true;
        if (split == 0) {
            current_toi = as_type<float>(atomic_load_explicit(toi_atomic, memory_order_relaxed));
            push_second = (mid <= current_toi);
        } else if (is_vf) {
            if (split == 1) {
                push_second = sum_less_than_one(mid, domain.tuv[2].lower);
            } else {
                push_second = sum_less_than_one(mid, domain.tuv[1].lower);
            }
        }

        // 推入第二个子域 - 使用重试循环处理竞态条件
        if (push_second) {
            bool pushed2 = false;
            for (int retry = 0; retry < 16 && !pushed2; retry++) {
                uint new_tail = atomic_fetch_add_explicit(tail, 1, memory_order_relaxed);
                uint cur_head = atomic_load_explicit(head, memory_order_relaxed);
                int queue_used = (int)new_tail - (int)cur_head;
                if (queue_used >= 0 && (uint)queue_used < capacity) {
                    CCDDomain d2 = domain;
                    d2.tuv[split].lower = mid;
                    bufferData[new_tail % capacity] = d2;
                    atomic_fetch_add_explicit(&debugCounters[7], 1, memory_order_relaxed); // pushed2
                    pushed2 = true;
                } else {
                    atomic_fetch_sub_explicit(tail, 1, memory_order_relaxed);
                    if (queue_used >= 0) {
                        atomic_store_explicit(overflow, 1, memory_order_relaxed);
                        break;
                    }
                    for (int spin = 0; spin < 32; spin++) {}
                }
            }
        }

        iters++;
    }
}
)";

// 优化版 Narrow Phase 实现 - 现在与 narrowPhase 相同（都使用分批处理）
bool Metal2Runtime::narrowPhaseOpt(
    const std::vector<double>& vertices_t0,
    const std::vector<double>& vertices_t1,
    const std::vector<int32_t>& indices,
    const std::vector<std::pair<int, int>>& overlaps,
    bool is_vf,
    float ms,
    float tolerance,
    int max_iter,
    bool allow_zero_toi,
    double& toi)
{
    // 直接调用 narrowPhase，它已经实现了分批处理
    return narrowPhase(vertices_t0, vertices_t1, indices, overlaps, is_vf, ms, tolerance, max_iter, allow_zero_toi, toi);
}

// 优化版 V2: 与 Opt 相同，使用 CUDA 风格
bool Metal2Runtime::narrowPhaseOptV2(
    const std::vector<double>& vertices_t0,
    const std::vector<double>& vertices_t1,
    const std::vector<int32_t>& indices,
    const std::vector<std::pair<int, int>>& overlaps,
    bool is_vf,
    float ms,
    float tolerance,
    int max_iter,
    bool allow_zero_toi,
    double& toi)
{
    // V2 目前与 Opt 相同，使用 CUDA 风格
    return narrowPhaseOpt(vertices_t0, vertices_t1, indices, overlaps, is_vf, ms, tolerance, max_iter, allow_zero_toi, toi);
}

bool Metal2Runtime::yzFilterAtomic(
    const std::vector<float>& minY,
    const std::vector<float>& maxY,
    const std::vector<float>& minZ,
    const std::vector<float>& maxZ,
    const std::vector<int32_t>& v0,
    const std::vector<int32_t>& v1,
    const std::vector<int32_t>& v2,
    const std::vector<std::pair<int, int>>& pairs,
    bool two_lists,
    std::vector<uint32_t>& outIndices)
{
    if (!available())
        return false;
    if (minY.size() != maxY.size() || minZ.size() != maxZ.size())
        return false;

    @autoreleasepool {
        NSError* err = nil;
        if (!impl_->cpsYZAtomic) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            impl_->libYZAtomic = [impl_->dev
                newLibraryWithSource:
                    [NSString stringWithUTF8String:kYZFilterAtomicSrc]
                             options:opts
                               error:&err];
            if (!impl_->libYZAtomic)
                return false;
            id<MTLFunction> fn =
                [impl_->libYZAtomic newFunctionWithName:@"yzFilterAtomic"];
            if (!fn)
                return false;
            impl_->cpsYZAtomic =
                [impl_->dev newComputePipelineStateWithFunction:fn error:&err];
            if (!impl_->cpsYZAtomic)
                return false;
        }

        const size_t n = minY.size();
        const size_t m = pairs.size();
        if (m == 0) {
            outIndices.clear();
            return true;
        }

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
            int32_t x, y, z, w;
        };
        std::vector<I3> vids(n);
        for (size_t i = 0; i < n; ++i)
            vids[i] = { v0[i], v1[i], v2[i], 0 };
        id<MTLBuffer> bVids =
            [impl_->dev newBufferWithBytes:vids.data()
                                    length:sizeof(I3) * n
                                   options:MTLResourceStorageModeShared];

        struct I2 {
            int32_t x, y;
        };
        std::vector<I2> pp(m);
        for (size_t i = 0; i < m; ++i)
            pp[i] = { pairs[i].first, pairs[i].second };
        id<MTLBuffer> bPairs =
            [impl_->dev newBufferWithBytes:pp.data()
                                    length:sizeof(I2) * m
                                   options:MTLResourceStorageModeShared];

        uint32_t mU32 = (uint32_t)m;
        uint32_t two = two_lists ? 1u : 0u;

        // Atomic Counter
        uint32_t zero = 0;
        id<MTLBuffer> bCount =
            [impl_->dev newBufferWithBytes:&zero
                                    length:sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];

        // Output Buffer (Max capacity = m)
        id<MTLBuffer> bOutIndices =
            [impl_->dev newBufferWithLength:sizeof(uint32_t) * m
                                    options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [impl_->q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl_->cpsYZAtomic];
        [enc setBuffer:bMinY offset:0 atIndex:0];
        [enc setBuffer:bMaxY offset:0 atIndex:1];
        [enc setBuffer:bMinZ offset:0 atIndex:2];
        [enc setBuffer:bMaxZ offset:0 atIndex:3];
        [enc setBuffer:bVids offset:0 atIndex:4];
        [enc setBuffer:bPairs offset:0 atIndex:5];
        [enc setBytes:&mU32 length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&two length:sizeof(uint32_t) atIndex:7];
        [enc setBuffer:bCount offset:0 atIndex:8];
        [enc setBuffer:bOutIndices offset:0 atIndex:9];
        [enc setBytes:&mU32
               length:sizeof(uint32_t)
              atIndex:10]; // maxCapacity = m

        NSUInteger w = impl_->cpsYZAtomic.maxTotalThreadsPerThreadgroup;
        if (w == 0)
            w = 64;
        MTLSize grid = MTLSizeMake(m, 1, 1);
        MTLSize tg = MTLSizeMake(std::min<NSUInteger>(w, m), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        CFTimeInterval t0 = cb.GPUStartTime, t1 = cb.GPUEndTime;
        if (t1 > t0 && t0 > 0)
            impl_->lastYZMs = (t1 - t0) * 1000.0;
        else
            impl_->lastYZMs = -1.0;

        // Read back count
        uint32_t count = *static_cast<uint32_t*>([bCount contents]);
        if (count > m)
            count = m; // Safety clamp

        // Copy indices
        outIndices.resize(count);
        if (count > 0) {
            memcpy(
                outIndices.data(), [bOutIndices contents],
                sizeof(uint32_t) * count);
        }
    }
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
        // CUDA 风格：默认无超时限制（0 = 无限）
        double timeout_ms = envD("SCALABLE_CCD_METAL2_STQ_TIMEOUT_MS", 0.0);
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

        // 动态调整 chunkI：溢出时减半重试
        uint32_t currentChunkI = chunkI;
        const uint32_t MIN_CHUNK_I = 512;

        for (uint32_t base = 0; base < n && !aborted; ) {
            uint32_t cur =
                std::min<uint32_t>(currentChunkI, static_cast<uint32_t>(n - base));
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
            if (cnt >= cap || sat) {
                // 溢出：减半 chunkI 重试当前批次
                if (currentChunkI > MIN_CHUNK_I) {
                    currentChunkI = std::max(MIN_CHUNK_I, currentChunkI / 2);
                    // 不推进 base，用更小 chunk 重试
                    continue;
                } else {
                    // 已经是最小 chunk 仍然溢出，收集已有结果并跳过
                    NSLog(@"Metal2 STQ(two): Skipping chunk at base=%u due to overflow (cnt=%u, cap=%llu)", base, cnt, (unsigned long long)cap);
                    outPairs.reserve(outPairs.size() + cnt);
                    for (uint32_t i = 0; i < cnt; ++i)
                        outPairs.emplace_back(pp[i].x, pp[i].y);
                    base += cur;
                    continue;
                }
            }
            // 成功：收集结果并推进
            outPairs.reserve(outPairs.size() + cnt);
            for (uint32_t i = 0; i < cnt; ++i)
                outPairs.emplace_back(pp[i].x, pp[i].y);
            base += cur;

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
        // CUDA 风格：默认无超时限制（0 = 无限）
        double timeout_ms = envD("SCALABLE_CCD_METAL2_STQ_TIMEOUT_MS", 0.0);
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

        // 动态调整 chunkI：溢出时减半重试
        uint32_t currentChunkI = chunkI;
        const uint32_t MIN_CHUNK_I = 512;

        for (uint32_t base = 0; base < n && !aborted; ) {
            uint32_t cur =
                std::min<uint32_t>(currentChunkI, static_cast<uint32_t>(n - base));
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
            if (cnt >= cap || sat) {
                // 溢出：减半 chunkI 重试当前批次
                if (currentChunkI > MIN_CHUNK_I) {
                    currentChunkI = std::max(MIN_CHUNK_I, currentChunkI / 2);
                    // 不推进 base，用更小 chunk 重试
                    continue;
                } else {
                    // 已经是最小 chunk 仍然溢出，收集已有结果并跳过
                    NSLog(@"Metal2 STQ(single): Skipping chunk at base=%u due to overflow (cnt=%u, cap=%llu)", base, cnt, (unsigned long long)cap);
                    outPairs.reserve(outPairs.size() + cnt);
                    for (uint32_t i = 0; i < cnt; ++i)
                        outPairs.emplace_back(pp[i].x, pp[i].y);
                    base += cur;
                    continue;
                }
            }
            // 成功：收集结果并推进
            outPairs.reserve(outPairs.size() + cnt);
            for (uint32_t i = 0; i < cnt; ++i)
                outPairs.emplace_back(pp[i].x, pp[i].y);
            base += cur;

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

void Metal2Runtime::clearBufferPool()
{
    if (impl_) {
        impl_->bufferPool.clear();
    }
}

void Metal2Runtime::setBufferPoolEnabled(bool enabled)
{
    if (impl_) {
        impl_->bufferPool.enabled = enabled;
    }
}

bool Metal2Runtime::stqWithYZFilter(
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

    const bool twoLists = !listTag.empty() && listTag.size() == n;

    @autoreleasepool {
        NSError* err = nil;

        // 编译融合内核
        auto ensurePipelines = [&]() -> bool {
            if (twoLists && !impl_->cpsSTQFusedTwo) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl_->libSTQFusedTwo = [impl_->dev
                    newLibraryWithSource:[NSString stringWithUTF8String:kSTQFusedTwoSrc]
                                 options:opts
                                   error:&err];
                if (!impl_->libSTQFusedTwo) {
                    NSLog(@"Failed to compile stqFusedTwo: %@", err);
                    return false;
                }
                id<MTLFunction> fn = [impl_->libSTQFusedTwo newFunctionWithName:@"stqFusedTwo"];
                if (!fn) return false;
                impl_->cpsSTQFusedTwo = [impl_->dev newComputePipelineStateWithFunction:fn error:&err];
                if (!impl_->cpsSTQFusedTwo) return false;
            }
            if (!twoLists && !impl_->cpsSTQFusedSingle) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl_->libSTQFusedSingle = [impl_->dev
                    newLibraryWithSource:[NSString stringWithUTF8String:kSTQFusedSingleSrc]
                                 options:opts
                                   error:&err];
                if (!impl_->libSTQFusedSingle) {
                    NSLog(@"Failed to compile stqFusedSingle: %@", err);
                    return false;
                }
                id<MTLFunction> fn = [impl_->libSTQFusedSingle newFunctionWithName:@"stqFusedSingle"];
                if (!fn) return false;
                impl_->cpsSTQFusedSingle = [impl_->dev newComputePipelineStateWithFunction:fn error:&err];
                if (!impl_->cpsSTQFusedSingle) return false;
            }
            return true;
        };

        if (!ensurePipelines())
            return false;

        // 环境变量配置
        auto envU = [](const char* k, uint32_t def) -> uint32_t {
            const char* v = std::getenv(k);
            if (!v) return def;
            try { return (uint32_t)std::max(0LL, std::stoll(v)); }
            catch (...) { return def; }
        };
        auto envD = [](const char* k, double def) -> double {
            const char* v = std::getenv(k);
            if (!v) return def;
            try { return std::stod(v); }
            catch (...) { return def; }
        };

        // 优化后的默认参数（针对 M1/M2 GPU）
        // - PERSIST_THREADS: 1024 更适合 Apple Silicon（原 4096 过多）
        // - CHUNK_I: 8192 提升缓存局部性（原 16384 过大）
        // - MAX_NEIGHBORS: 256 增大以减少溢出风险
        uint32_t maxN = envU("SCALABLE_CCD_METAL2_STQ_MAX_NEIGHBORS", 256);
        uint32_t chunkI = envU("SCALABLE_CCD_METAL2_STQ_CHUNK_I", 8192);
        uint32_t ptqThreads = envU("SCALABLE_CCD_METAL2_STQ_PERSIST_THREADS", 1024);
        // CUDA 风格：默认无超时限制（0 = 无限），匹配 CUDA 的行为
        double timeout_ms = envD("SCALABLE_CCD_METAL2_STQ_TIMEOUT_MS", 0.0);

        // CUDA 风格：动态查询可用 GPU 内存
        size_t allocatableMemory = impl_->getAllocatableMemory();
        size_t maxPairCapacity = impl_->getMaxPairCapacity(allocatableMemory);
        // 至少保留 8M pairs，即使内存不足也尝试
        if (maxPairCapacity < 8 * 1024 * 1024) maxPairCapacity = 8 * 1024 * 1024;

        impl_->lastPairsMs = 0.0;
        outPairs.clear();

        // 打包 AABB 结构体（与 kernel 中的 PackedAABB 对应）
        struct PackedAABB {
            float minX, maxX, minY, maxY, minZ, maxZ;
        };
        std::vector<PackedAABB> packedBoxes(n);
        for (size_t i = 0; i < n; ++i) {
            packedBoxes[i] = {
                static_cast<float>(minX[i]),
                static_cast<float>(maxX[i]),
                static_cast<float>(minY[i]),
                static_cast<float>(maxY[i]),
                static_cast<float>(minZ[i]),
                static_cast<float>(maxZ[i])
            };
        }

        struct I3 { int32_t x, y, z, w; };
        std::vector<I3> vids(n);
        for (size_t i = 0; i < n; ++i) {
            vids[i] = {v0[i], v1[i], v2[i], 0};
        }

        // 单一打包 buffer（减少内存带宽）
        id<MTLBuffer> bBoxes = [impl_->dev newBufferWithBytes:packedBoxes.data()
                                                      length:sizeof(PackedAABB) * n
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> bVids = [impl_->dev newBufferWithBytes:vids.data()
                                                     length:sizeof(I3) * n
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> bTag = nil;
        if (twoLists) {
            bTag = [impl_->dev newBufferWithBytes:listTag.data()
                                           length:sizeof(uint8_t) * n
                                          options:MTLResourceStorageModeShared];
        }

        double acc_ms = 0.0;
        bool aborted = false;

        // 两阶段容量估算：基于历史密度动态调整
        // 初始容量 = chunkI * maxN，如果溢出则翻倍重试
        uint64_t baseCap = (uint64_t)chunkI * (uint64_t)maxN;
        if (baseCap < 1024 * 1024 * 8) baseCap = 1024 * 1024 * 8;
        double densityFactor = 1.0; // 动态调整因子

        // 动态调整 chunkI：当容量翻倍重试失败时减半
        uint32_t currentChunkI = chunkI;
        const uint32_t MIN_CHUNK_I = 512;

        // 分批处理
        for (uint32_t base = 0; base < n && !aborted; ) {
            uint32_t cur = std::min<uint32_t>(currentChunkI, (uint32_t)(n - base));
            uint64_t cap = static_cast<uint64_t>(baseCap * densityFactor);
            if (cap < 1024 * 1024) cap = 1024 * 1024;

            uint32_t startI = base;
            uint32_t endI = (uint32_t)n; // 搜索到末尾
            uint32_t zero = 0;

            // 重试循环（CUDA 风格：溢出时扩展缓冲区并重跑）
            bool batchDone = false;
            int retryCount = 0;
            const int maxRetries = 10;  // 增加重试次数，CUDA 会持续重试

            while (!batchDone && retryCount < maxRetries) {
                id<MTLBuffer> bPairs = impl_->bufferPool.acquire(impl_->dev, sizeof(int32_t) * 2 * cap, MTLResourceStorageModeShared);
                id<MTLBuffer> bCnt = [impl_->dev newBufferWithBytes:&zero length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                id<MTLBuffer> bSat = [impl_->dev newBufferWithBytes:&zero length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                id<MTLBuffer> bHead = [impl_->dev newBufferWithBytes:&zero length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cb = [impl_->q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

            // bufferCapacity 传给内核，用于检测溢出
            uint32_t bufferCapacity = (uint32_t)std::min(cap, (uint64_t)UINT32_MAX);

            if (twoLists) {
                [enc setComputePipelineState:impl_->cpsSTQFusedTwo];
                [enc setBuffer:bBoxes offset:0 atIndex:0];
                [enc setBuffer:bVids offset:0 atIndex:1];
                [enc setBuffer:bTag offset:0 atIndex:2];
                [enc setBytes:&startI length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&endI length:sizeof(uint32_t) atIndex:4];
                [enc setBytes:&maxN length:sizeof(uint32_t) atIndex:5];
                [enc setBuffer:bHead offset:0 atIndex:6];
                [enc setBuffer:bCnt offset:0 atIndex:7];
                [enc setBuffer:bPairs offset:0 atIndex:8];
                [enc setBuffer:bSat offset:0 atIndex:9];
                [enc setBytes:&bufferCapacity length:sizeof(uint32_t) atIndex:10];
            } else {
                [enc setComputePipelineState:impl_->cpsSTQFusedSingle];
                [enc setBuffer:bBoxes offset:0 atIndex:0];
                [enc setBuffer:bVids offset:0 atIndex:1];
                [enc setBytes:&startI length:sizeof(uint32_t) atIndex:2];
                [enc setBytes:&endI length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&maxN length:sizeof(uint32_t) atIndex:4];
                [enc setBuffer:bHead offset:0 atIndex:5];
                [enc setBuffer:bCnt offset:0 atIndex:6];
                [enc setBuffer:bPairs offset:0 atIndex:7];
                [enc setBuffer:bSat offset:0 atIndex:8];
                [enc setBytes:&bufferCapacity length:sizeof(uint32_t) atIndex:9];
            }

            NSUInteger w = twoLists ? impl_->cpsSTQFusedTwo.maxTotalThreadsPerThreadgroup
                                    : impl_->cpsSTQFusedSingle.maxTotalThreadsPerThreadgroup;
            if (w == 0) w = 256;
            NSUInteger tgSize = std::min<NSUInteger>(w, 256);
            NSUInteger gridSize = std::min<NSUInteger>(ptqThreads, ((ptqThreads + tgSize - 1) / tgSize) * tgSize);

            [enc dispatchThreads:MTLSizeMake(gridSize, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            CFTimeInterval t0 = cb.GPUStartTime, t1 = cb.GPUEndTime;
            double ms = (t1 > t0 && t0 > 0) ? (t1 - t0) * 1000.0 : 0.0;
            impl_->lastPairsMs += ms;
            acc_ms += ms;

            // 读回结果
            uint32_t cnt = *static_cast<uint32_t*>([bCnt contents]);
            uint32_t sat = *static_cast<uint32_t*>([bSat contents]);

            // CUDA 风格溢出处理：检查是否需要扩展缓冲区并重跑
            if (sat || cnt >= cap) {
                // 使用动态内存查询的最大容量（代替硬编码 256MB）
                const uint64_t MAX_CAP = maxPairCapacity;

                if (cap < MAX_CAP && retryCount < maxRetries) {
                    impl_->bufferPool.release(bPairs);
                    // CUDA 风格：扩展到实际需要的大小（而不是简单翻倍）
                    uint64_t neededCap = std::max((uint64_t)cnt + 1024, cap * 2);
                    cap = std::min(neededCap, MAX_CAP);
                    densityFactor = (double)cap / baseCap;
                    retryCount++;
                    NSLog(@"Metal2 STQ Fused: Overflow at base=%u (cnt=%u), expanding buffer to %llu pairs and re-running",
                          base, cnt, (unsigned long long)cap);
                    continue; // 重试当前批次
                }

                // 容量已达上限或重试次数用尽，尝试减半 chunkI（CUDA 的 MAX_OVERLAP_CUTOFF >>= 1 策略）
                if (currentChunkI > MIN_CHUNK_I) {
                    impl_->bufferPool.release(bPairs);
                    currentChunkI = std::max(MIN_CHUNK_I, currentChunkI / 2);
                    retryCount = 0;
                    densityFactor = 1.0;
                    NSLog(@"Metal2 STQ Fused: Insufficient memory at base=%u, shrinking chunk to %u",
                          base, currentChunkI);
                    batchDone = true;
                    continue;
                }

                // 已经是最小 chunk，收集已有结果并跳过（最后手段）
                NSLog(@"Metal2 STQ Fused: Min chunk overflow at base=%u (cnt=%u), collecting partial results", base, cnt);
                if (cnt > cap) cnt = (uint32_t)cap;
                struct I2 { int32_t x, y; };
                I2* pp = static_cast<I2*>([bPairs contents]);
                outPairs.reserve(outPairs.size() + cnt);
                for (uint32_t i = 0; i < cnt; ++i) {
                    outPairs.emplace_back(pp[i].x, pp[i].y);
                }
                impl_->bufferPool.release(bPairs);
                base += cur;
                batchDone = true;
                continue;
            }

            // 成功：收集结果
            if (cnt > cap) cnt = (uint32_t)cap;
            struct I2 { int32_t x, y; };
            I2* pp = static_cast<I2*>([bPairs contents]);
            outPairs.reserve(outPairs.size() + cnt);
            for (uint32_t i = 0; i < cnt; ++i) {
                outPairs.emplace_back(pp[i].x, pp[i].y);
            }

            impl_->bufferPool.release(bPairs);
            base += cur;
            batchDone = true;

            if (timeout_ms > 0.0 && acc_ms > timeout_ms) {
                printf("Metal2 STQ Fused ABORT: Timeout (acc=%.2f ms, limit=%.2f ms)\n", acc_ms, timeout_ms);
                aborted = true;
            }
            } // end while retry loop
        }

        if (aborted) {
            outPairs.clear();
            return false;
        }
        return true;
    }
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
