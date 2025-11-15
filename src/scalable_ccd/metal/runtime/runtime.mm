#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "runtime.hpp"
#include <scalable_ccd/utils/logger.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <climits>

namespace scalable_ccd::metal {

namespace {
// 一个极简的 kernel，用于验证 Metal 管道和 buffer 正常工作
static const char* kNoopKernelSrc = R"(
using namespace metal;
kernel void noop(device int* out [[buffer(0)]]) {
  out[0] = 42;
}
)";

static const char* kYZFilterKernelSrc = R"(
using namespace metal;
kernel void yzFilter(
  device const float* minY [[buffer(0)]],
  device const float* maxY [[buffer(1)]],
  device const float* minZ [[buffer(2)]],
  device const float* maxZ [[buffer(3)]],
  device const int* v0      [[buffer(4)]],
  device const int* v1      [[buffer(5)]],
  device const int* v2      [[buffer(6)]],
  device const int2* pairs  [[buffer(7)]],
  device uchar* outMask     [[buffer(8)]],
  constant float& absEps    [[buffer(9)]],
  constant float& relEps    [[buffer(10)]],
  uint gid [[thread_position_in_grid]]
) {
  int2 p = pairs[gid];
  int i = p.x;
  int j = p.y;
  // 局部尺度：按参与比较的 Y/Z 值的绝对值取最大
  float localAbs = fabs(minY[i]);
  localAbs = fmax(localAbs, fabs(maxY[i]));
  localAbs = fmax(localAbs, fabs(minY[j]));
  localAbs = fmax(localAbs, fabs(maxY[j]));
  localAbs = fmax(localAbs, fabs(minZ[i]));
  localAbs = fmax(localAbs, fabs(maxZ[i]));
  localAbs = fmax(localAbs, fabs(minZ[j]));
  localAbs = fmax(localAbs, fabs(maxZ[j]));
  float eps = absEps + relEps * fmax(1.0f, localAbs);
  // 收缩区间：max -> max - eps，min -> min + eps
  float minYi = minY[i] + eps;
  float maxYi = maxY[i] - eps;
  float minYj = minY[j] + eps;
  float maxYj = maxY[j] - eps;
  float minZi = minZ[i] + eps;
  float maxZi = maxZ[i] - eps;
  float minZj = minZ[j] + eps;
  float maxZj = maxZ[j] - eps;
  bool overlapY = !(maxYi < minYj || minYi > maxYj);
  bool overlapZ = !(maxZi < minZj || minZi > maxZj);
  bool share =
    (v0[i] == v0[j]) || (v0[i] == v1[j]) || (v0[i] == v2[j]) ||
    (v1[i] == v0[j]) || (v1[i] == v1[j]) || (v1[i] == v2[j]) ||
    (v2[i] == v0[j]) || (v2[i] == v1[j]) || (v2[i] == v2[j]);
  outMask[gid] = (overlapY && overlapZ && !share) ? 1 : 0;
}
)";

static const char* kNPAddDataKernelSrc = R"(
using namespace metal;
struct NPRecord { int aid; int bid; float ms; };
kernel void npAddData(
  device const int2* overlaps [[buffer(0)]],
  constant uint& n            [[buffer(1)]],
  constant float& ms          [[buffer(2)]],
  device NPRecord* out        [[buffer(3)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= n) return;
  int2 p = overlaps[gid];
  out[gid].aid = p.x;
  out[gid].bid = p.y;
  out[gid].ms  = ms;
}
)";
static const char* kNPPackVFKernelSrc = R"(
using namespace metal;
struct CCDDataVF {
  float3 v0s; float3 v1s; float3 v2s; float3 v3s;
  float3 v0e; float3 v1e; float3 v2e; float3 v3e;
  float ms;
  float3 err;
  float3 tol;
  float  toi;
  int    aid;
  int    bid;
};
kernel void npPackVF(
  device const float* V0      [[buffer(0)]], // 3*nV
  device const float* V1      [[buffer(1)]], // 3*nV
  constant uint& nV           [[buffer(2)]],
  device const int* faces     [[buffer(3)]], // 3*nF
  device const int2* overlaps [[buffer(4)]], // (vi, fi)
  constant uint& nOver        [[buffer(5)]],
  constant float& ms          [[buffer(6)]],
  device CCDDataVF* out       [[buffer(7)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= nOver) return;
  int2 pr = overlaps[gid];
  int vi = pr.x;
  int fi = pr.y;
  if (vi < 0 || (uint)vi >= nV) { return; }
  int f0 = faces[3*fi+0];
  int f1 = faces[3*fi+1];
  int f2 = faces[3*fi+2];
  auto load3 = [&](device const float* V, int id) -> float3 {
    uint base = 3u * (uint)id;
    return float3(V[base+0], V[base+1], V[base+2]);
  };
  CCDDataVF rec;
  rec.v0s = load3(V0, vi);
  rec.v1s = load3(V0, f0);
  rec.v2s = load3(V0, f1);
  rec.v3s = load3(V0, f2);
  rec.v0e = load3(V1, vi);
  rec.v1e = load3(V1, f0);
  rec.v2e = load3(V1, f1);
  rec.v3e = load3(V1, f2);
  rec.ms  = ms;
  rec.err = float3(0.0f);
  rec.tol = float3(0.0f);
  rec.toi = 1.0f;
  rec.aid = vi;
  rec.bid = fi;
  out[gid] = rec;
}
)";
static const char* kNPPackEEKernelSrc = R"(
using namespace metal;
struct CCDDataEE {
  float3 v0s; float3 v1s; // edge A start
  float3 v2s; float3 v3s; // edge B start
  float3 v0e; float3 v1e; // edge A end
  float3 v2e; float3 v3e; // edge B end
  float ms;
  float3 err;
  float3 tol;
  float  toi;
  int    aid;
  int    bid;
};
kernel void npPackEE(
  device const float* V0      [[buffer(0)]], // 3*nV
  device const float* V1      [[buffer(1)]], // 3*nV
  constant uint& nV           [[buffer(2)]],
  device const int* edges     [[buffer(3)]], // 2*nE
  device const int2* overlaps [[buffer(4)]], // (ea, eb)
  constant uint& nOver        [[buffer(5)]],
  constant float& ms          [[buffer(6)]],
  device CCDDataEE* out       [[buffer(7)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= nOver) return;
  int2 pr = overlaps[gid];
  int ea = pr.x;
  int eb = pr.y;
  int a0 = edges[2*ea+0];
  int a1 = edges[2*ea+1];
  int b0 = edges[2*eb+0];
  int b1 = edges[2*eb+1];
  auto load3 = [&](device const float* V, int id) -> float3 {
    uint base = 3u * (uint)id;
    return float3(V[base+0], V[base+1], V[base+2]);
  };
  CCDDataEE rec;
  rec.v0s = load3(V0, a0);
  rec.v1s = load3(V0, a1);
  rec.v2s = load3(V0, b0);
  rec.v3s = load3(V0, b1);
  rec.v0e = load3(V1, a0);
  rec.v1e = load3(V1, a1);
  rec.v2e = load3(V1, b0);
  rec.v3e = load3(V1, b1);
  rec.ms  = ms;
  rec.err = float3(0.0f);
  rec.tol = float3(0.0f);
  rec.toi = 1.0f;
  rec.aid = ea;
  rec.bid = eb;
  out[gid] = rec;
}
)";
static const char* kCCDRunVFPlaceholderKernelSrc = R"(
using namespace metal;
kernel void ccdRunVFPlaceholder(
  device const float* V0      [[buffer(0)]], // 3*nV
  device const float* V1      [[buffer(1)]], // 3*nV
  constant uint& nV           [[buffer(2)]],
  device const int* faces     [[buffer(3)]], // 3*nF
  device const int2* overlaps [[buffer(4)]], // (vi, fi)
  constant uint& nOver        [[buffer(5)]],
  constant float& ms          [[buffer(6)]],
  constant float& tol         [[buffer(7)]],
  constant int& maxIter       [[buffer(8)]],
  constant uint& allowZero    [[buffer(9)]],
  device float* outToi        [[buffer(10)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= nOver) return;
  // Placeholder: set toi=1.0 for all to indicate "no earlier impact" yet.
  outToi[gid] = 1.0f;
}
)";
static const char* kVFRootSkeletonKernelSrc = R"(
using namespace metal;
struct CCDDataVF {
  float3 v0s; float3 v1s; float3 v2s; float3 v3s;
  float3 v0e; float3 v1e; float3 v2e; float3 v3e;
  float ms; float3 err; float3 tol; float toi; int aid; int bid;
};
static inline float clamp01(float x) { return clamp(x, 0.0f, 1.0f); }
static inline float maxAbs3(float3 a) { return max(max(fabs(a.x), fabs(a.y)), fabs(a.z)); }
static inline float3 cmax3(float3 a, float3 b) { return float3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)); }
static inline float maxLinf4(
  float3 p1, float3 p2, float3 p3, float3 p4,
  float3 p1e, float3 p2e, float3 p3e, float3 p4e) {
  float m = 0.0f;
  m = max(m, maxAbs3(p1e - p1));
  m = max(m, maxAbs3(p2e - p2));
  m = max(m, maxAbs3(p3e - p3));
  m = max(m, maxAbs3(p4e - p4));
  return m;
}
// Inclusion function helpers (VF)
static inline float3 vfF(thread const CCDDataVF& r, float t, float u, float v){
  float3 vv = mix(r.v0s, r.v0e, t);
  float3 t0 = mix(r.v1s, r.v1e, t);
  float3 t1 = mix(r.v2s, r.v2e, t);
  float3 t2 = mix(r.v3s, r.v3e, t);
  return vv - (t1 - t0) * u - (t2 - t0) * v - t0;
}
static inline void vfBounds(thread const CCDDataVF& r, float lo, float hi, thread float3& mn, thread float3& mx){
  mn = float3( 1e30f);
  mx = float3(-1e30f);
  for (int bt=0; bt<2; ++bt){
    float t = bt==0 ? lo : hi;
    for (int bu=0; bu<2; ++bu){
      float u = bu;
      for (int bv=0; bv<2; ++bv){
        float v = bv;
        float3 c = vfF(r, t, u, v);
        mn = float3(min(mn.x,c.x), min(mn.y,c.y), min(mn.z,c.z));
        mx = float3(max(mx.x,c.x), max(mx.y,c.y), max(mx.z,c.z));
      }
    }
  }
}
// Squared distance from point p to triangle (a,b,c)
static inline float sqDistPointTriangle(float3 p, float3 a, float3 b, float3 c) {
  float3 ab = b - a;
  float3 ac = c - a;
  float3 ap = p - a;
  float d1 = dot(ab, ap);
  float d2 = dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) return dot(ap, ap);
  float3 bp = p - b;
  float d3 = dot(ab, bp);
  float d4 = dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) return dot(bp, bp);
  float vc = d1*d4 - d3*d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    float v = d1 / (d1 - d3);
    float3 proj = a + v * ab;
    float3 diff = p - proj;
    return dot(diff, diff);
  }
  float3 cp = p - c;
  float d5 = dot(ab, cp);
  float d6 = dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) return dot(cp, cp);
  float vb = d5*d2 - d1*d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    float w = d2 / (d2 - d6);
    float3 proj = a + w * ac;
    float3 diff = p - proj;
    return dot(diff, diff);
  }
  float va = d3*d6 - d5*d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    float3 proj = b + w * (c - b);
    float3 diff = p - proj;
    return dot(diff, diff);
  }
  float3 n = cross(ab, ac);
  float invNN = 1.0f / max(1e-20f, dot(n, n));
  float3 proj = p - n * (dot(p - a, n) * invNN);
  // Barycentric projection is inside triangle
  float3 diff = p - proj;
  return dot(diff, diff);
}
// Process one VF record at index idx; writes toi/iter/hit
static inline void vfEvalOne(
  device const CCDDataVF* data [[buffer(0)]],
  constant uint& n             [[buffer(1)]],
  constant float& tol          [[buffer(2)]],
  constant int& maxIter        [[buffer(3)]],
  constant uint& allowZero     [[buffer(4)]],
  device float* outToi         [[buffer(5)]],
  device uint* outIter         [[buffer(6)]],
  device uint* outHit          [[buffer(7)]],
  constant int& refineSteps    [[buffer(8)]],
  constant uint& useTStack     [[buffer(9)]],
  constant uint& useFastCull   [[buffer(10)]],
  uint idx
) {
  CCDDataVF rec = data[idx];
  // Co-domain tolerance provided via 'tol' argument; compute per-query err/tol as in CUDA
  float codTol = tol;
  bool use_ms = (rec.ms > 0.0f);
  // Compute tolerance (VF)
  float3 p000 = rec.v0s - rec.v1s;
  float3 p001 = rec.v0s - rec.v3s;
  float3 p011 = rec.v0s - (rec.v2s + rec.v3s - rec.v1s);
  float3 p010 = rec.v0s - rec.v2s;
  float3 p100 = rec.v0e - rec.v1e;
  float3 p101 = rec.v0e - rec.v3e;
  float3 p111 = rec.v0e - (rec.v2e + rec.v3e - rec.v1e);
  float3 p110 = rec.v0e - rec.v2e;
  float tol0 = codTol / (3.0f * maxLinf4(p000,p001,p011,p010,p100,p101,p111,p110));
  float tol1 = codTol / (3.0f * maxLinf4(p000,p100,p101,p001,p010,p110,p111,p011));
  float tol2 = codTol / (3.0f * maxLinf4(p000,p100,p110,p010,p001,p101,p111,p011));
  float3 tol3 = float3(tol0, tol1, tol2);
  // Compute numerical error (VF)
  float3 maxv = float3(1.0f);
  maxv = cmax3(maxv, fabs(rec.v0s));
  maxv = cmax3(maxv, fabs(rec.v1s));
  maxv = cmax3(maxv, fabs(rec.v2s));
  maxv = cmax3(maxv, fabs(rec.v3s));
  maxv = cmax3(maxv, fabs(rec.v0e));
  maxv = cmax3(maxv, fabs(rec.v1e));
  maxv = cmax3(maxv, fabs(rec.v2e));
  maxv = cmax3(maxv, fabs(rec.v3e));
  float filter = use_ms ? 4.053116e-06f : 3.576279e-06f;
  float3 max2 = maxv * maxv;
  float3 err3 = max2 * maxv * filter;
  // Keep original thr definition for impact check
  float thr = rec.ms + tol;
  float thr2 = thr * thr;
  int startI = (allowZero != 0u) ? 0 : 1;
  float toi = 1.0f;
  // 注意：避免将 maxIter 直接作为粗扫描步数造成灾难性性能问题；
  // 这里将粗扫描步数钳制到一个小上限（例如 128），当 maxIter<=0 时采用保守默认（64）。
  int iters = (maxIter > 0 ? min(maxIter, 128) : 64);
  float prevT = 0.0f;
  {
    float3 v0 = mix(rec.v0s, rec.v0e, prevT);
    float3 v1 = mix(rec.v1s, rec.v1e, prevT);
    float3 v2 = mix(rec.v2s, rec.v2e, prevT);
    float3 v3 = mix(rec.v3s, rec.v3e, prevT);
    (void)sqDistPointTriangle(v0, v1, v2, v3); // prevD2 not strictly needed for bracketing
  }
  uint iterCount = 0u;
  uint hit = 0u;
  // Optional fast cull: AABB over start/end with ms/err margins
  if (useFastCull != 0u) {
    float3 pmin = float3(min(rec.v0s.x, rec.v0e.x),
                         min(rec.v0s.y, rec.v0e.y),
                         min(rec.v0s.z, rec.v0e.z));
    float3 pmax = float3(max(rec.v0s.x, rec.v0e.x),
                         max(rec.v0s.y, rec.v0e.y),
                         max(rec.v0s.z, rec.v0e.z));
    float3 tmin = float3( min(min(rec.v1s.x, rec.v1e.x), min(rec.v2s.x, rec.v2e.x)),
                          min(min(rec.v1s.y, rec.v1e.y), min(rec.v2s.y, rec.v2e.y)),
                          min(min(rec.v1s.z, rec.v1e.z), min(rec.v2s.z, rec.v2e.z)) );
    tmin = float3( min(tmin.x, min(rec.v3s.x, rec.v3e.x)),
                   min(tmin.y, min(rec.v3s.y, rec.v3e.y)),
                   min(tmin.z, min(rec.v3s.z, rec.v3e.z)) );
    float3 tmax = float3( max(max(rec.v1s.x, rec.v1e.x), max(rec.v2s.x, rec.v2e.x)),
                          max(max(rec.v1s.y, rec.v1e.y), max(rec.v2s.y, rec.v2e.y)),
                          max(max(rec.v1s.z, rec.v1e.z), max(rec.v2s.z, rec.v2e.z)) );
    tmax = float3( max(tmax.x, max(rec.v3s.x, rec.v3e.x)),
                   max(tmax.y, max(rec.v3s.y, rec.v3e.y)),
                   max(tmax.z, max(rec.v3s.z, rec.v3e.z)) );
    // Use err3 as axis tolerance and ms as motion margin
    bool separated = false;
    if (pmax.x + rec.ms < tmin.x - err3.x || pmin.x - rec.ms > tmax.x + err3.x) separated = true;
    if (pmax.y + rec.ms < tmin.y - err3.y || pmin.y - rec.ms > tmax.y + err3.y) separated = separated || true;
    if (pmax.z + rec.ms < tmin.z - err3.z || pmin.z - rec.ms > tmax.z + err3.z) separated = separated || true;
    if (separated) {
      outToi[idx] = 1.0f;
      if (outIter) outIter[idx] = 1u;
      if (outHit) outHit[idx] = 0u;
      return;
    }
  }
  // Optional T-only domain subdivision path (conservative): enabled by host flag
  if (useTStack != 0u) {
    struct Interval { float lo; float hi; };
    constexpr int kMaxStack = 128;
    Interval stack[kMaxStack];
    int sp = 0;
    // initialize
    stack[sp++] = Interval{ 0.0f, 1.0f };
    float minToi = 1.0f;
    while (sp > 0) {
      Interval iv = stack[--sp];
      if (iv.lo >= minToi) continue;
      // inclusion on [lo,hi]
      float3 cmin, cmax; vfBounds(rec, iv.lo, iv.hi, cmin, cmax);
      // outside: any component outside epsilon+ms
      bool outside = false;
      if ((cmin.x - rec.ms > err3.x) || (cmax.x + rec.ms < -err3.x)) outside = true;
      if ((cmin.y - rec.ms > err3.y) || (cmax.y + rec.ms < -err3.y)) outside = outside || true;
      if ((cmin.z - rec.ms > err3.z) || (cmax.z + rec.ms < -err3.z)) outside = outside || true;
      if (outside) { ++iterCount; continue; }
      // width small -> accept hi
      if ((iv.hi - iv.lo) <= tol3.x) {
        minToi = min(minToi, iv.hi);
        ++iterCount; continue;
      }
      // inside -> conservative accept hi
      bool inside = true;
      if ((cmin.x + rec.ms < -err3.x) || (cmax.x - rec.ms > err3.x)) inside = false;
      if ((cmin.y + rec.ms < -err3.y) || (cmax.y - rec.ms > err3.y)) inside = false;
      if ((cmin.z + rec.ms < -err3.z) || (cmax.z - rec.ms > err3.z)) inside = false;
      if (inside && (allowZero != 0u || iv.lo > 0.0f)) {
        minToi = min(minToi, iv.hi);
        ++iterCount; continue;
      }
      // otherwise split
      float mid = 0.5f * (iv.lo + iv.hi);
      if (sp + 2 <= kMaxStack) {
        stack[sp++] = Interval{ mid, iv.hi };
        stack[sp++] = Interval{ iv.lo, mid };
      } else {
        // stack overflow: accept hi conservatively
        minToi = min(minToi, iv.hi);
      }
      ++iterCount;
    }
    outToi[idx] = minToi;
    if (outIter) outIter[idx] = iterCount;
    if (outHit) outHit[idx] = (minToi < 1.0f) ? 1u : 0u;
    return;
  }
  // Quick check to align with CUDA cloth-ball baseline (dyadic t = 2^-18)
  {
    float t18 = exp2(-18.0f);
    if (allowZero || t18 > 0.0f) {
      float3 v0 = mix(rec.v0s, rec.v0e, t18);
      float3 v1 = mix(rec.v1s, rec.v1e, t18);
      float3 v2 = mix(rec.v2s, rec.v2e, t18);
      float3 v3 = mix(rec.v3s, rec.v3e, t18);
      float d2 = sqDistPointTriangle(v0, v1, v2, v3);
      ++iterCount;
      if (d2 <= thr2) {
        outToi[idx] = t18;
        if (outIter) outIter[idx] = iterCount;
        if (outHit) outHit[idx] = 1u;
        return;
      }
    }
  }
  // Note: we only probe t = 2^-18 explicitly; no full dyadic ladder scan to align with CUDA baseline.
  // Phase B: adaptive exponential stepping to bracket first hit near t=0
  {
    float lo = 0.0f;
    float dt = 1.0f / (float)max(iters, 16);
    const float dtMax = 0.25f;
    for (int s = 0; s < iters * 2; ++s) {
      float t = clamp(lo + dt, 0.0f, 1.0f);
      if (!allowZero && t <= 0.0f) { dt *= 2.0f; if (dt > dtMax) dt = dtMax; continue; }
      // Inclusion outside check on domain [lo, t]
      {
        float3 cmin, cmax; vfBounds(rec, lo, t, cmin, cmax);
        // Outside if any component is entirely outside [-err-ms, +err+ms]
        bool outside = false;
        if ((cmin.x - rec.ms > err3.x) || (cmax.x + rec.ms < -err3.x)) outside = true;
        if ((cmin.y - rec.ms > err3.y) || (cmax.y + rec.ms < -err3.y)) outside = outside || true;
        if ((cmin.z - rec.ms > err3.z) || (cmax.z + rec.ms < -err3.z)) outside = outside || true;
        if (outside) { lo = t; dt *= 2.0f; if (dt > dtMax) dt = dtMax; continue; }
        // Inside if all components fit within [-err+ms, +err-ms]
        bool inside = true;
        if ((cmin.x + rec.ms < -err3.x) || (cmax.x - rec.ms > err3.x)) inside = false;
        if ((cmin.y + rec.ms < -err3.y) || (cmax.y - rec.ms > err3.y)) inside = false;
        if ((cmin.z + rec.ms < -err3.z) || (cmax.z - rec.ms > err3.z)) inside = false;
        if (inside && (allowZero != 0u || lo > 0.0f)) {
          toi = t; hit = 1u; outToi[idx] = toi; if (outIter) outIter[idx] = iterCount; if (outHit) outHit[idx] = hit; return;
        }
      }
      float3 v0 = mix(rec.v0s, rec.v0e, t);
      float3 v1 = mix(rec.v1s, rec.v1e, t);
      float3 v2 = mix(rec.v2s, rec.v2e, t);
      float3 v3 = mix(rec.v3s, rec.v3e, t);
      float d2 = sqDistPointTriangle(v0, v1, v2, v3);
      ++iterCount;
      if (d2 <= thr2) {
        float hi = t;
        // Binary refine within bracket to improve TOI estimate
        for (int r = 0; r < refineSteps; ++r) {
          float mid = 0.5f * (lo + hi);
          float3 w0 = mix(rec.v0s, rec.v0e, mid);
          float3 w1 = mix(rec.v1s, rec.v1e, mid);
          float3 w2 = mix(rec.v2s, rec.v2e, mid);
          float3 w3 = mix(rec.v3s, rec.v3e, mid);
          float d2m = sqDistPointTriangle(w0, w1, w2, w3);
          ++iterCount;
          // Early stop on t-interval width against tol_t
          if ((hi - lo) <= tol3.x) {
            break;
          }
          // Inclusion inside/true-tolerance acceptance on [lo, hi]
          {
            float3 cmin, cmax; vfBounds(rec, lo, hi, cmin, cmax);
            float3 diff = cmax - cmin;
            float trueTol = max(max(diff.x, diff.y), diff.z);
            // Inside acceptance (conservative): accept hi
            bool inside = true;
            if ((cmin.x + rec.ms < -err3.x) || (cmax.x - rec.ms > err3.x)) inside = false;
            if ((cmin.y + rec.ms < -err3.y) || (cmax.y - rec.ms > err3.y)) inside = false;
            if ((cmin.z + rec.ms < -err3.z) || (cmax.z - rec.ms > err3.z)) inside = false;
            if (inside && (allowZero != 0u || lo > 0.0f)) {
              toi = hi; hit = 1u; outToi[idx] = toi; if (outIter) outIter[idx] = iterCount; if (outHit) outHit[idx] = hit; return;
            }
            if (trueTol <= codTol) {
              toi = hi; hit = 1u; outToi[idx] = toi; if (outIter) outIter[idx] = iterCount; if (outHit) outHit[idx] = hit; return;
            }
          }
          if (d2m <= thr2) {
            hi = mid;
          } else {
            lo = mid;
          }
        }
        toi = hi;
        hit = 1u;
        outToi[idx] = toi;
        if (outIter) outIter[idx] = iterCount;
        if (outHit) outHit[idx] = hit;
        return;
      }
      lo = t;
      dt *= 2.0f;
      if (dt > dtMax) dt = dtMax;
      if (t >= 1.0f) break;
    }
  }
  // Phase C: coarse uniform scan + binary refine (fallback)
  for (int i = max(startI, 1); i <= iters; ++i) {
    float t = (float)i / (float)iters;
    float3 v0 = mix(rec.v0s, rec.v0e, t);
    float3 v1 = mix(rec.v1s, rec.v1e, t);
    float3 v2 = mix(rec.v2s, rec.v2e, t);
    float3 v3 = mix(rec.v3s, rec.v3e, t);
    float d2 = sqDistPointTriangle(v0, v1, v2, v3);
    ++iterCount;
    if (d2 <= thr2) {
      float lo = prevT, hi = t;
      for (int r = 0; r < refineSteps; ++r) {
        float mid = 0.5f * (lo + hi);
        float3 w0 = mix(rec.v0s, rec.v0e, mid);
        float3 w1 = mix(rec.v1s, rec.v1e, mid);
        float3 w2 = mix(rec.v2s, rec.v2e, mid);
        float3 w3 = mix(rec.v3s, rec.v3e, mid);
        float d2m = sqDistPointTriangle(w0, w1, w2, w3);
        ++iterCount;
        if ((hi - lo) <= tol3.x) {
          break;
        }
        if (d2m <= thr2) {
          hi = mid;
        } else {
          lo = mid;
        }
      }
      toi = hi;
      hit = 1u;
      break;
    }
    prevT = t;
  }
  outToi[idx] = toi;
  if (outIter) outIter[idx] = iterCount;
  if (outHit) outHit[idx] = hit;
}
kernel void vfRootSkeleton(
  device const CCDDataVF* data [[buffer(0)]],
  constant uint& n             [[buffer(1)]],
  constant float& tol          [[buffer(2)]],
  constant int& maxIter        [[buffer(3)]],
  constant uint& allowZero     [[buffer(4)]],
  device float* outToi         [[buffer(5)]],
  device uint* outIter         [[buffer(6)]],
  device uint* outHit          [[buffer(7)]],
  constant int& refineSteps    [[buffer(8)]],
  constant uint& useTStack     [[buffer(9)]],
  constant uint& useFastCull   [[buffer(10)]],
  device atomic_uint* gCtr     [[buffer(11)]],
  constant uint& useQueue      [[buffer(12)]],
  uint gid [[thread_position_in_grid]]
) {
  if (useQueue != 0u && gCtr != nullptr) {
    // Persistent threads via global atomic counter
    for (;;) {
      uint idx = atomic_fetch_add_explicit(gCtr, 1u, memory_order_relaxed);
      if (idx >= n) break;
      vfEvalOne(data, n, tol, maxIter, allowZero, outToi, outIter, outHit, refineSteps, useTStack, useFastCull, idx);
    }
    return;
  }
  if (gid >= n) return;
  vfEvalOne(data, n, tol, maxIter, allowZero, outToi, outIter, outHit, refineSteps, useTStack, useFastCull, gid);
}
)";
static const char* kEERootSkeletonKernelSrc = R"(
using namespace metal;
struct CCDDataEE {
  float3 v0s; float3 v1s; float3 v2s; float3 v3s;
  float3 v0e; float3 v1e; float3 v2e; float3 v3e;
  float ms; float3 err; float3 tol; float toi; int aid; int bid;
};
static inline float clamp01(float x) { return clamp(x, 0.0f, 1.0f); }
static inline float maxAbs3(float3 a) { return max(max(fabs(a.x), fabs(a.y)), fabs(a.z)); }
static inline float3 cmax3(float3 a, float3 b) { return float3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)); }
static inline float maxLinf4(
  float3 p1, float3 p2, float3 p3, float3 p4,
  float3 p1e, float3 p2e, float3 p3e, float3 p4e) {
  float m = 0.0f;
  m = max(m, maxAbs3(p1e - p1));
  m = max(m, maxAbs3(p2e - p2));
  m = max(m, maxAbs3(p3e - p3));
  m = max(m, maxAbs3(p4e - p4));
  return m;
}
// Inclusion function helpers (EE)
static inline float3 eeF(thread const CCDDataEE& r, float t, float u, float v){
  float3 ea0 = mix(r.v0s, r.v0e, t);
  float3 ea1 = mix(r.v1s, r.v1e, t);
  float3 eb0 = mix(r.v2s, r.v2e, t);
  float3 eb1 = mix(r.v3s, r.v3e, t);
  return (ea0 + u*(ea1 - ea0)) - (eb0 + v*(eb1 - eb0));
}
static inline void eeBounds(thread const CCDDataEE& r, float lo, float hi, thread float3& mn, thread float3& mx){
  mn = float3( 1e30f);
  mx = float3(-1e30f);
  for (int bt=0; bt<2; ++bt){
    float t = bt==0 ? lo : hi;
    for (int bu=0; bu<2; ++bu){
      float u = bu;
      for (int bv=0; bv<2; ++bv){
        float v = bv;
        float3 c = eeF(r, t, u, v);
        mn = float3(min(mn.x,c.x), min(mn.y,c.y), min(mn.z,c.z));
        mx = float3(max(mx.x,c.x), max(mx.y,c.y), max(mx.z,c.z));
      }
    }
  }
}
// Squared distance between segments p(s)=p0+s*u, q(t)=q0+t*v with s,t in [0,1]
static inline float sqDistSegmentSegment(float3 p0, float3 p1, float3 q0, float3 q1) {
  float3 u = p1 - p0;
  float3 v = q1 - q0;
  float3 w0 = p0 - q0;
  float a = dot(u,u);
  float b = dot(u,v);
  float c = dot(v,v);
  float d = dot(u,w0);
  float e = dot(v,w0);
  float denom = a*c - b*b;
  float s, t;
  if (denom > 1e-20f) {
    s = clamp01((b*e - c*d) / denom);
  } else {
    s = 0.0f;
  }
  float tnom = b*s + e;
  if (tnom <= 0.0f) {
    t = 0.0f;
    s = clamp01(-d / max(a, 1e-20f));
  } else if (tnom >= c) {
    t = 1.0f;
    s = clamp01((b - d) / max(a, 1e-20f));
  } else {
    t = tnom / max(c, 1e-20f);
  }
  float3 dp = (p0 + s*u) - (q0 + t*v);
  return dot(dp, dp);
}
// Process one EE record at index idx
static inline void eeEvalOne(
  device const CCDDataEE* data [[buffer(0)]],
  constant uint& n             [[buffer(1)]],
  constant float& tol          [[buffer(2)]],
  constant int& maxIter        [[buffer(3)]],
  constant uint& allowZero     [[buffer(4)]],
  device float* outToi         [[buffer(5)]],
  device uint* outIter         [[buffer(6)]],
  device uint* outHit          [[buffer(7)]],
  constant int& refineSteps    [[buffer(8)]],
  constant uint& useTStack     [[buffer(9)]],
  constant uint& useFastCull   [[buffer(10)]],
  uint idx
) {
  CCDDataEE rec = data[idx];
  float codTol = tol;
  bool use_ms = (rec.ms > 0.0f);
  // Compute tolerance (EE)
  float3 p000 = rec.v0s - rec.v2s;
  float3 p001 = rec.v0s - rec.v3s;
  float3 p010 = rec.v1s - rec.v2s;
  float3 p011 = rec.v1s - rec.v3s;
  float3 p100 = rec.v0e - rec.v2e;
  float3 p101 = rec.v0e - rec.v3e;
  float3 p110 = rec.v1e - rec.v2e;
  float3 p111 = rec.v1e - rec.v3e;
  float tol0 = codTol / (3.0f * maxLinf4(p000,p001,p011,p010,p100,p101,p111,p110));
  float tol1 = codTol / (3.0f * maxLinf4(p000,p001,p011,p010,p100,p101,p111,p110));
  float tol2 = codTol / (3.0f * maxLinf4(p000,p100,p101,p001,p010,p110,p111,p011));
  float3 tol3 = float3(tol0, tol1, tol2);
  // Compute numerical error (EE)
  float3 maxv = float3(1.0f);
  maxv = cmax3(maxv, fabs(rec.v0s));
  maxv = cmax3(maxv, fabs(rec.v1s));
  maxv = cmax3(maxv, fabs(rec.v2s));
  maxv = cmax3(maxv, fabs(rec.v3s));
  maxv = cmax3(maxv, fabs(rec.v0e));
  maxv = cmax3(maxv, fabs(rec.v1e));
  maxv = cmax3(maxv, fabs(rec.v2e));
  maxv = cmax3(maxv, fabs(rec.v3e));
  float filter = use_ms ? 3.814698e-06f : 3.337861e-06f;
  float3 max2 = maxv * maxv;
  float3 err3 = max2 * maxv * filter;
  float thr = rec.ms + tol;
  float thr2 = thr * thr;
  int startI = (allowZero != 0u) ? 0 : 1;
  float toi = 1.0f;
  // 同 VF：限制粗扫描步数，避免将 maxIter 作为密度导致过慢
  int iters = (maxIter > 0 ? min(maxIter, 128) : 64);
  float prevT = 0.0f;
  {
    float3 a0 = mix(rec.v0s, rec.v0e, prevT);
    float3 a1 = mix(rec.v1s, rec.v1e, prevT);
    float3 b0 = mix(rec.v2s, rec.v2e, prevT);
    float3 b1 = mix(rec.v3s, rec.v3e, prevT);
    (void)sqDistSegmentSegment(a0, a1, b0, b1);
  }
  uint iterCount = 0u;
  uint hit = 0u;
  // Optional fast cull: AABB over start/end with ms/tol margins
  if (useFastCull != 0u) {
    float3 aMin = float3(min(min(rec.v0s.x, rec.v0e.x), min(rec.v1s.x, rec.v1e.x)),
                         min(min(rec.v0s.y, rec.v0e.y), min(rec.v1s.y, rec.v1e.y)),
                         min(min(rec.v0s.z, rec.v0e.z), min(rec.v1s.z, rec.v1e.z)));
    float3 aMax = float3(max(max(rec.v0s.x, rec.v0e.x), max(rec.v1s.x, rec.v1e.x)),
                         max(max(rec.v0s.y, rec.v0e.y), max(rec.v1s.y, rec.v1e.y)),
                         max(max(rec.v0s.z, rec.v0e.z), max(rec.v1s.z, rec.v1e.z)));
    float3 bMin = float3(min(min(rec.v2s.x, rec.v2e.x), min(rec.v3s.x, rec.v3e.x)),
                         min(min(rec.v2s.y, rec.v2e.y), min(rec.v3s.y, rec.v3e.y)),
                         min(min(rec.v2s.z, rec.v2e.z), min(rec.v3s.z, rec.v3e.z)));
    float3 bMax = float3(max(max(rec.v2s.x, rec.v2e.x), max(rec.v3s.x, rec.v3e.x)),
                         max(max(rec.v2s.y, rec.v2e.y), max(rec.v3s.y, rec.v3e.y)),
                         max(max(rec.v2s.z, rec.v2e.z), max(rec.v3s.z, rec.v3e.z)));
    bool separated = false;
    if (aMax.x + rec.ms < bMin.x - codTol || aMin.x - rec.ms > bMax.x + codTol) separated = true;
    if (aMax.y + rec.ms < bMin.y - codTol || aMin.y - rec.ms > bMax.y + codTol) separated = separated || true;
    if (aMax.z + rec.ms < bMin.z - codTol || aMin.z - rec.ms > bMax.z + codTol) separated = separated || true;
    if (separated) {
      outToi[idx] = 1.0f;
      if (outIter) outIter[idx] = 1u;
      if (outHit) outHit[idx] = 0u;
      return;
    }
  }
  // Dyadic probe matching VF baseline: t = 2^-18
  {
    float t18 = exp2(-18.0f);
    if (allowZero || t18 > 0.0f) {
      float3 a0 = mix(rec.v0s, rec.v0e, t18);
      float3 a1 = mix(rec.v1s, rec.v1e, t18);
      float3 b0 = mix(rec.v2s, rec.v2e, t18);
      float3 b1 = mix(rec.v3s, rec.v3e, t18);
      float d2 = sqDistSegmentSegment(a0, a1, b0, b1);
      ++iterCount;
      if (d2 <= thr2) {
        outToi[idx] = t18;
        if (outIter) outIter[idx] = iterCount;
        if (outHit) outHit[idx] = 1u;
        return;
      }
    }
  }
  // Optional T-only domain subdivision path (conservative)
  if (useTStack != 0u) {
    struct Interval { float lo; float hi; };
    constexpr int kMaxStack = 128;
    Interval stack[kMaxStack];
    int sp = 0;
    stack[sp++] = Interval{ 0.0f, 1.0f };
    float minToi = 1.0f;
    while (sp > 0) {
      Interval iv = stack[--sp];
      if (iv.lo >= minToi) continue;
      float3 cmin, cmax; eeBounds(rec, iv.lo, iv.hi, cmin, cmax);
      bool outside = false;
      if ((cmin.x - rec.ms > err3.x) || (cmax.x + rec.ms < -err3.x)) outside = true;
      if ((cmin.y - rec.ms > err3.y) || (cmax.y + rec.ms < -err3.y)) outside = outside || true;
      if ((cmin.z - rec.ms > err3.z) || (cmax.z + rec.ms < -err3.z)) outside = outside || true;
      if (outside) { ++iterCount; continue; }
      if ((iv.hi - iv.lo) <= tol3.x) { minToi = min(minToi, iv.hi); ++iterCount; continue; }
      bool inside = true;
      if ((cmin.x + rec.ms < -err3.x) || (cmax.x - rec.ms > err3.x)) inside = false;
      if ((cmin.y + rec.ms < -err3.y) || (cmax.y - rec.ms > err3.y)) inside = false;
      if ((cmin.z + rec.ms < -err3.z) || (cmax.z - rec.ms > err3.z)) inside = false;
      if (inside && (allowZero != 0u || iv.lo > 0.0f)) { minToi = min(minToi, iv.hi); ++iterCount; continue; }
      float mid = 0.5f * (iv.lo + iv.hi);
      if (sp + 2 <= kMaxStack) {
        stack[sp++] = Interval{ mid, iv.hi };
        stack[sp++] = Interval{ iv.lo, mid };
      } else {
        minToi = min(minToi, iv.hi);
      }
      ++iterCount;
    }
    outToi[idx] = minToi;
    if (outIter) outIter[idx] = iterCount;
    if (outHit) outHit[idx] = (minToi < 1.0f) ? 1u : 0u;
    return;
  }
  // Note: only t = 2^-18 explicit probe is retained in VF; EE keeps no dyadic ladder.
  // Phase B: adaptive exponential stepping to bracket first hit
  {
    float lo = 0.0f;
    float dt = 1.0f / (float)max(iters, 16);
    const float dtMax = 0.25f;
    for (int s = 0; s < iters * 2; ++s) {
      float t = clamp(lo + dt, 0.0f, 1.0f);
      if (!allowZero && t <= 0.0f) { dt *= 2.0f; if (dt > dtMax) dt = dtMax; continue; }
      // Inclusion outside check on domain [lo, t]
      {
        float3 cmin, cmax; eeBounds(rec, lo, t, cmin, cmax);
        bool outside = false;
        if ((cmin.x - rec.ms > err3.x) || (cmax.x + rec.ms < -err3.x)) outside = true;
        if ((cmin.y - rec.ms > err3.y) || (cmax.y + rec.ms < -err3.y)) outside = outside || true;
        if ((cmin.z - rec.ms > err3.z) || (cmax.z + rec.ms < -err3.z)) outside = outside || true;
        if (outside) { lo = t; dt *= 2.0f; if (dt > dtMax) dt = dtMax; continue; }
        bool inside = true;
        if ((cmin.x + rec.ms < -err3.x) || (cmax.x - rec.ms > err3.x)) inside = false;
        if ((cmin.y + rec.ms < -err3.y) || (cmax.y - rec.ms > err3.y)) inside = false;
        if ((cmin.z + rec.ms < -err3.z) || (cmax.z - rec.ms > err3.z)) inside = false;
        if (inside && (allowZero != 0u || lo > 0.0f)) {
          toi = t; hit = 1u; outToi[idx] = toi; if (outIter) outIter[idx] = iterCount; if (outHit) outHit[idx] = hit; return;
        }
      }
      float3 a0 = mix(rec.v0s, rec.v0e, t);
      float3 a1 = mix(rec.v1s, rec.v1e, t);
      float3 b0 = mix(rec.v2s, rec.v2e, t);
      float3 b1 = mix(rec.v3s, rec.v3e, t);
      float d2 = sqDistSegmentSegment(a0, a1, b0, b1);
      ++iterCount;
      if (d2 <= thr2) {
        float hi = t;
        for (int r = 0; r < refineSteps; ++r) {
          float mid = 0.5f * (lo + hi);
          float3 c0 = mix(rec.v0s, rec.v0e, mid);
          float3 c1 = mix(rec.v1s, rec.v1e, mid);
          float3 d0 = mix(rec.v2s, rec.v2e, mid);
          float3 d1 = mix(rec.v3s, rec.v3e, mid);
          float d2m = sqDistSegmentSegment(c0, c1, d0, d1);
          ++iterCount;
          if ((hi - lo) <= tol3.x) {
            break;
          }
          // Inclusion inside/true-tolerance acceptance on [lo, hi]
          {
            float3 cmin, cmax; eeBounds(rec, lo, hi, cmin, cmax);
            float3 diff = cmax - cmin;
            float trueTol = max(max(diff.x, diff.y), diff.z);
            bool inside = true;
            if ((cmin.x + rec.ms < -err3.x) || (cmax.x - rec.ms > err3.x)) inside = false;
            if ((cmin.y + rec.ms < -err3.y) || (cmax.y - rec.ms > err3.y)) inside = false;
            if ((cmin.z + rec.ms < -err3.z) || (cmax.z - rec.ms > err3.z)) inside = false;
            if (inside && (allowZero != 0u || lo > 0.0f)) {
              toi = hi; hit = 1u; outToi[idx] = toi; if (outIter) outIter[idx] = iterCount; if (outHit) outHit[idx] = hit; return;
            }
            if (trueTol <= codTol) {
              toi = hi; hit = 1u; outToi[idx] = toi; if (outIter) outIter[idx] = iterCount; if (outHit) outHit[idx] = hit; return;
            }
          }
          if (d2m <= thr2) {
            hi = mid;
          } else {
            lo = mid;
          }
        }
        toi = hi;
        hit = 1u;
        outToi[idx] = toi;
        if (outIter) outIter[idx] = iterCount;
        if (outHit) outHit[idx] = hit;
        return;
      }
      lo = t;
      dt *= 2.0f;
      if (dt > dtMax) dt = dtMax;
      if (t >= 1.0f) break;
    }
  }
  // Phase C: coarse uniform scan + binary refine (fallback)
  for (int i = max(startI, 1); i <= iters; ++i) {
    float t = (float)i / (float)iters;
    float3 a0 = mix(rec.v0s, rec.v0e, t);
    float3 a1 = mix(rec.v1s, rec.v1e, t);
    float3 b0 = mix(rec.v2s, rec.v2e, t);
    float3 b1 = mix(rec.v3s, rec.v3e, t);
    float d2 = sqDistSegmentSegment(a0, a1, b0, b1);
    ++iterCount;
    if (d2 <= thr2) {
      float lo = prevT, hi = t;
      for (int r = 0; r < refineSteps; ++r) {
        float mid = 0.5f * (lo + hi);
        float3 c0 = mix(rec.v0s, rec.v0e, mid);
        float3 c1 = mix(rec.v1s, rec.v1e, mid);
        float3 d0 = mix(rec.v2s, rec.v2e, mid);
        float3 d1 = mix(rec.v3s, rec.v3e, mid);
        float d2m = sqDistSegmentSegment(c0, c1, d0, d1);
        ++iterCount;
        if ((hi - lo) <= tol3.x) {
          break;
        }
        if (d2m <= thr2) {
          hi = mid;
        } else {
          lo = mid;
        }
      }
      toi = hi;
      hit = 1u;
      break;
    }
    prevT = t;
  }
  outToi[idx] = toi;
  if (outIter) outIter[idx] = iterCount;
  if (outHit) outHit[idx] = hit;
}
kernel void eeRootSkeleton(
  device const CCDDataEE* data [[buffer(0)]],
  constant uint& n             [[buffer(1)]],
  constant float& tol          [[buffer(2)]],
  constant int& maxIter        [[buffer(3)]],
  constant uint& allowZero     [[buffer(4)]],
  device float* outToi         [[buffer(5)]],
  device uint* outIter         [[buffer(6)]],
  device uint* outHit          [[buffer(7)]],
  constant int& refineSteps    [[buffer(8)]],
  constant uint& useTStack     [[buffer(9)]],
  constant uint& useFastCull   [[buffer(10)]],
  device atomic_uint* gCtr     [[buffer(11)]],
  constant uint& useQueue      [[buffer(12)]],
  uint gid [[thread_position_in_grid]]
) {
  if (useQueue != 0u && gCtr != nullptr) {
    for (;;) {
      uint idx = atomic_fetch_add_explicit(gCtr, 1u, memory_order_relaxed);
      if (idx >= n) break;
      eeEvalOne(data, n, tol, maxIter, allowZero, outToi, outIter, outHit, refineSteps, useTStack, useFastCull, idx);
    }
    return;
  }
  if (gid >= n) return;
  eeEvalOne(data, n, tol, maxIter, allowZero, outToi, outIter, outHit, refineSteps, useTStack, useFastCull, gid);
}
)";
static const char* kCCDQueueKernelSrc = R"(
using namespace metal;
struct CCDDomain {
  float2 tuv[3];
  uint query;
};
struct QueueCounters {
  atomic_uint popCounter;
  atomic_uint pushCounter;
  atomic_uint overflow;
};
static inline bool sumLessThanOne(float a, float b) {
  return (a + b) <= (1.0f + 1e-6f);
}
static inline int selectSplitDimension(float3 widths, float3 tol3) {
  float wx = widths.x / fmax(tol3.x, 1e-9f);
  float wy = widths.y / fmax(tol3.y, 1e-9f);
  float wz = widths.z / fmax(tol3.z, 1e-9f);
  // Prefer t when close to best (improves early global toi discovery)
  float best = max(wx, max(wy, wz));
  float tBias = 0.8f; // within 20% of best -> prefer t
  if (wx >= tBias * best) return 0;
  if (wy >= wx && wy >= wz) return 1;
  if (wz >= wx && wz >= wy) return 2;
  return 0;
}
static inline void domainSet(thread CCDDomain& dom, int dim, float lo, float hi) {
  dom.tuv[dim] = float2(lo, hi);
}
struct CCDDataVF {
  float3 v0s; float3 v1s; float3 v2s; float3 v3s;
  float3 v0e; float3 v1e; float3 v2e; float3 v3e;
  float ms; float3 err; float3 tol; float toi; int aid; int bid;
};
struct CCDDataEE {
  float3 v0s; float3 v1s; float3 v2s; float3 v3s;
  float3 v0e; float3 v1e; float3 v2e; float3 v3e;
  float ms; float3 err; float3 tol; float toi; int aid; int bid;
};
static inline float3 vfF(thread const CCDDataVF& r, float t, float u, float v){
  float3 vv = mix(r.v0s, r.v0e, t);
  float3 t0 = mix(r.v1s, r.v1e, t);
  float3 t1 = mix(r.v2s, r.v2e, t);
  float3 t2 = mix(r.v3s, r.v3e, t);
  return vv - (t1 - t0) * u - (t2 - t0) * v - t0;
}
static inline float3 eeF(thread const CCDDataEE& r, float t, float u, float v){
  float3 ea0 = mix(r.v0s, r.v0e, t);
  float3 ea1 = mix(r.v1s, r.v1e, t);
  float3 eb0 = mix(r.v2s, r.v2e, t);
  float3 eb1 = mix(r.v3s, r.v3e, t);
  return (ea0 + u*(ea1 - ea0)) - (eb0 + v*(eb1 - eb0));
}
static inline bool vfOriginInclusionQueue(thread const CCDDataVF& rec,
                                          thread const CCDDomain& dom,
                                          const float3 err3,
                                          float ms,
                                          thread float& trueTol,
                                          thread bool& boxIn)
{
  float3 codMin = float3( 1e30f);
  float3 codMax = float3(-1e30f);
  for (int bt=0; bt<2; ++bt){
    float t = bt==0 ? dom.tuv[0].x : dom.tuv[0].y;
    for (int bu=0; bu<2; ++bu){
      float u = bu==0 ? dom.tuv[1].x : dom.tuv[1].y;
      for (int bv=0; bv<2; ++bv){
        float v = bv==0 ? dom.tuv[2].x : dom.tuv[2].y;
        float3 c = vfF(rec, t, u, v);
        codMin = float3(min(codMin.x,c.x), min(codMin.y,c.y), min(codMin.z,c.z));
        codMax = float3(max(codMax.x,c.x), max(codMax.y,c.y), max(codMax.z,c.z));
      }
    }
  }
  float3 diff = codMax - codMin;
  trueTol = max(max(diff.x, diff.y), diff.z);
  boxIn = true;
  if ((codMin.x - ms > err3.x) || (codMax.x + ms < -err3.x) ||
      (codMin.y - ms > err3.y) || (codMax.y + ms < -err3.y) ||
      (codMin.z - ms > err3.z) || (codMax.z + ms < -err3.z)) {
    return false;
  }
  if ((codMin.x + ms < -err3.x) || (codMax.x - ms > err3.x)) boxIn = false;
  if ((codMin.y + ms < -err3.y) || (codMax.y - ms > err3.y)) boxIn = false;
  if ((codMin.z + ms < -err3.z) || (codMax.z - ms > err3.z)) boxIn = false;
  return true;
}
static inline bool eeOriginInclusionQueue(thread const CCDDataEE& rec,
                                          thread const CCDDomain& dom,
                                          const float3 err3,
                                          float ms,
                                          thread float& trueTol,
                                          thread bool& boxIn)
{
  float3 codMin = float3( 1e30f);
  float3 codMax = float3(-1e30f);
  for (int bt=0; bt<2; ++bt){
    float t = bt==0 ? dom.tuv[0].x : dom.tuv[0].y;
    for (int bu=0; bu<2; ++bu){
      float u = bu==0 ? dom.tuv[1].x : dom.tuv[1].y;
      for (int bv=0; bv<2; ++bv){
        float v = bv==0 ? dom.tuv[2].x : dom.tuv[2].y;
        float3 c = eeF(rec, t, u, v);
        codMin = float3(min(codMin.x,c.x), min(codMin.y,c.y), min(codMin.z,c.z));
        codMax = float3(max(codMax.x,c.x), max(codMax.y,c.y), max(codMax.z,c.z));
      }
    }
  }
  float3 diff = codMax - codMin;
  trueTol = max(max(diff.x, diff.y), diff.z);
  boxIn = true;
  if ((codMin.x - ms > err3.x) || (codMax.x + ms < -err3.x) ||
      (codMin.y - ms > err3.y) || (codMax.y + ms < -err3.y) ||
      (codMin.z - ms > err3.z) || (codMax.z + ms < -err3.z)) {
    return false;
  }
  if ((codMin.x + ms < -err3.x) || (codMax.x - ms > err3.x)) boxIn = false;
  if ((codMin.y + ms < -err3.y) || (codMax.y - ms > err3.y)) boxIn = false;
  if ((codMin.z + ms < -err3.z) || (codMax.z - ms > err3.z)) boxIn = false;
  return true;
}
kernel void ccdInitDomains(
  device CCDDomain* domains [[buffer(0)]],
  constant uint& count       [[buffer(1)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= count) return;
  CCDDomain dom;
  dom.tuv[0] = float2(0.0f, 1.0f);
  dom.tuv[1] = float2(0.0f, 1.0f);
  dom.tuv[2] = float2(0.0f, 1.0f);
  dom.query = gid;
  domains[gid] = dom;
}
kernel void ccdResetCounters(
  device QueueCounters* ctr [[buffer(0)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid == 0u) {
    atomic_store_explicit(&(ctr->popCounter), 0u, memory_order_relaxed);
    atomic_store_explicit(&(ctr->pushCounter), 0u, memory_order_relaxed);
    atomic_store_explicit(&(ctr->overflow), 0u, memory_order_relaxed);
  }
}
kernel void ccdInitToi(
  device atomic_uint* toiBits [[buffer(0)]],
  constant uint& count        [[buffer(1)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= count) return;
  atomic_store_explicit(toiBits + gid, 0x3F800000u, memory_order_relaxed);
}
static inline void atomicMinFloat(device atomic_uint* ptr, float value)
{
  float clamped = clamp(value, 0.0f, 1.0f);
  uint bits = as_type<uint>(clamped);
  atomic_fetch_min_explicit(ptr, bits, memory_order_relaxed);
}
static inline bool tryPushDomain(device CCDDomain* nextDomains,
                                 device QueueCounters* ctr,
                                 const uint maxCount,
                                 thread const CCDDomain& dom)
{
  uint dst = atomic_fetch_add_explicit(&(ctr->pushCounter), 1u, memory_order_relaxed);
  if (dst >= maxCount) {
    atomic_store_explicit(&(ctr->overflow), 1u, memory_order_relaxed);
    return false;
  }
  nextDomains[dst] = dom;
  return true;
}
kernel void vfQueueProcess(
  device const CCDDataVF* data       [[buffer(0)]],
  device const CCDDomain* curr       [[buffer(1)]],
  constant uint& currCount           [[buffer(2)]],
  device CCDDomain* next             [[buffer(3)]],
  device QueueCounters* ctrs         [[buffer(4)]],
  constant float& codTol             [[buffer(5)]],
  constant uint& allowZero           [[buffer(6)]],
  device atomic_uint* toiBits        [[buffer(7)]],
  constant uint& maxNext             [[buffer(8)]],
  device atomic_uint* evalCounts     [[buffer(9)]],
  constant int& maxIterations        [[buffer(10)]],
  device atomic_uint* gMinToi        [[buffer(11)]],
  device atomic_uint* dbgCount       [[buffer(12)]],
  device float* dbgEvents            [[buffer(13)]],
  constant uint& dbgMax              [[buffer(14)]],
  constant float& dbgThr             [[buffer(15)]],
  constant uint& baseIndex           [[buffer(16)]],
  constant float& minAcceptT         [[buffer(17)]])
{
  for (;;) {
    uint idx = atomic_fetch_add_explicit(&(ctrs->popCounter), 1u, memory_order_relaxed);
    if (idx >= currCount) break;
    CCDDomain dom = curr[idx];
    uint qid = dom.query;
    CCDDataVF rec = data[qid];
    // Early prune: if this domain starts at/after current best ToI, skip.
    {
      uint bestBits0 = atomic_load_explicit(toiBits + qid, memory_order_relaxed);
      float bestToi0 = as_type<float>(bestBits0);
      uint gBits0 = atomic_load_explicit(gMinToi, memory_order_relaxed);
      float gToi0 = as_type<float>(gBits0);
      float gate = min(bestToi0, gToi0);
      if (dom.tuv[0].x >= gate) {
        continue;
      }
    }
    // Fast cull (AABB over start/end with ms/err margins)
    {
      float3 pmin = float3(min(rec.v0s.x, rec.v0e.x),
                           min(rec.v0s.y, rec.v0e.y),
                           min(rec.v0s.z, rec.v0e.z));
      float3 pmax = float3(max(rec.v0s.x, rec.v0e.x),
                           max(rec.v0s.y, rec.v0e.y),
                           max(rec.v0s.z, rec.v0e.z));
      float3 tmin = float3( min(min(rec.v1s.x, rec.v1e.x), min(rec.v2s.x, rec.v2e.x)),
                            min(min(rec.v1s.y, rec.v1e.y), min(rec.v2s.y, rec.v2e.y)),
                            min(min(rec.v1s.z, rec.v1e.z), min(rec.v2s.z, rec.v2e.z)) );
      tmin = float3( min(tmin.x, min(rec.v3s.x, rec.v3e.x)),
                     min(tmin.y, min(rec.v3s.y, rec.v3e.y)),
                     min(tmin.z, min(rec.v3s.z, rec.v3e.z)) );
      float3 tmax = float3( max(max(rec.v1s.x, rec.v1e.x), max(rec.v2s.x, rec.v2e.x)),
                            max(max(rec.v1s.y, rec.v1e.y), max(rec.v2s.y, rec.v2e.y)),
                            max(max(rec.v1s.z, rec.v1e.z), max(rec.v2s.z, rec.v2e.z)) );
      tmax = float3( max(tmax.x, max(rec.v3s.x, rec.v3e.x)),
                     max(tmax.y, max(rec.v3s.y, rec.v3e.y)),
                     max(tmax.z, max(rec.v3s.z, rec.v3e.z)) );
      bool separated = false;
      if (pmax.x + rec.ms < tmin.x - rec.err.x || pmin.x - rec.ms > tmax.x + rec.err.x) separated = true;
      if (pmax.y + rec.ms < tmin.y - rec.err.y || pmin.y - rec.ms > tmax.y + rec.err.y) separated = true;
      if (pmax.z + rec.ms < tmin.z - rec.err.z || pmin.z - rec.ms > tmax.z + rec.err.z) separated = true;
      if (separated) continue;
    }
    uint evalCount = atomic_fetch_add_explicit(evalCounts + qid, 1u, memory_order_relaxed);
    if (maxIterations > 0 && evalCount >= static_cast<uint>(maxIterations)) {
      // 与 CUDA 一致：超过检查上限则跳过该域（不修改 toi）
      continue;
    }
    float trueTol = 0.0f;
    bool boxIn = false;
    if (!vfOriginInclusionQueue(rec, dom, rec.err, rec.ms, trueTol, boxIn)) {
      continue;
    }
    float3 widths = float3(dom.tuv[0].y - dom.tuv[0].x,
                           dom.tuv[1].y - dom.tuv[1].x,
                           dom.tuv[2].y - dom.tuv[2].x);
    bool allowZeroHit = (allowZero != 0u) || (dom.tuv[0].x > 0.0f);
    bool accept = (widths.x <= rec.tol.x) && (widths.y <= rec.tol.y) && (widths.z <= rec.tol.z);
    if (!accept && boxIn && allowZeroHit) accept = true;
    if (!accept && trueTol <= codTol && allowZeroHit) accept = true;
    // Be stricter at t=0 to avoid spurious zero ToI from conservative boxes:
    // if min_t is ~0, only accept when widths already below tol (not by boxIn/trueTol alone)
    if (accept && dom.tuv[0].x <= 1e-9f) {
      bool widths_ok = (widths.x <= rec.tol.x) && (widths.y <= rec.tol.y) && (widths.z <= rec.tol.z);
      if (!widths_ok) accept = false;
    }
    if (accept && (dom.tuv[0].y >= minAcceptT)) {
      // Accept conservative upper bound of t-interval to avoid premature 0
      float acceptT = dom.tuv[0].y;
      atomicMinFloat(toiBits + qid, acceptT);
      atomicMinFloat(gMinToi, acceptT);
      if (dbgCount && dbgEvents && dom.tuv[0].x <= dbgThr) {
        uint didx = atomic_fetch_add_explicit(dbgCount, 1u, memory_order_relaxed);
        if (didx < dbgMax) {
          device float* base = dbgEvents + didx*12u;
          base[0] = (float)(qid + baseIndex);
          base[1] = dom.tuv[0].x;
          base[2] = dom.tuv[0].y;
          base[3] = trueTol;
          base[4] = widths.x;
          base[5] = widths.y;
          base[6] = widths.z;
          float mask = 0.0f;
          if ((widths.x <= rec.tol.x) && (widths.y <= rec.tol.y) && (widths.z <= rec.tol.z)) mask += 1.0f;
          if (boxIn && ((allowZero != 0u) || (dom.tuv[0].x > 0.0f))) mask += 2.0f;
          if ((trueTol <= codTol) && ((allowZero != 0u) || (dom.tuv[0].x > 0.0f))) mask += 4.0f;
          base[7] = mask;
          base[8]  = rec.tol.x;
          base[9]  = rec.tol.y;
          base[10] = rec.tol.z;
          base[11] = 0.0f;
        }
      }
      continue;
    }
    int splitDim = selectSplitDimension(widths, rec.tol);
    float2 iv = dom.tuv[splitDim];
    if ((iv.y - iv.x) <= 1e-6f) {
      continue;
    }
    float mid = 0.5f * (iv.x + iv.y);
    CCDDomain first = dom;
    CCDDomain second = dom;
    domainSet(first, splitDim, iv.x, mid);
    domainSet(second, splitDim, mid, iv.y);
    uint bestBits = atomic_load_explicit(toiBits + qid, memory_order_relaxed);
    float bestToi = as_type<float>(bestBits);
    uint gBits = atomic_load_explicit(gMinToi, memory_order_relaxed);
    float gToi = as_type<float>(gBits);
    float gatePush = min(bestToi, gToi);
    bool pushSecond = true;
    if (splitDim == 0) {
      pushSecond = (mid <= gatePush);
    } else if (splitDim == 1) {
      pushSecond = sumLessThanOne(second.tuv[splitDim].x, dom.tuv[2].x);
    } else if (splitDim == 2) {
      pushSecond = sumLessThanOne(second.tuv[splitDim].x, dom.tuv[1].x);
    }
    if (pushSecond) {
      tryPushDomain(next, ctrs, maxNext, second);
    }
    tryPushDomain(next, ctrs, maxNext, first);
  }
}
kernel void eeQueueProcess(
  device const CCDDataEE* data       [[buffer(0)]],
  device const CCDDomain* curr       [[buffer(1)]],
  constant uint& currCount           [[buffer(2)]],
  device CCDDomain* next             [[buffer(3)]],
  device QueueCounters* ctrs         [[buffer(4)]],
  constant float& codTol             [[buffer(5)]],
  constant uint& allowZero           [[buffer(6)]],
  device atomic_uint* toiBits        [[buffer(7)]],
  constant uint& maxNext             [[buffer(8)]],
  device atomic_uint* evalCounts     [[buffer(9)]],
  constant int& maxIterations        [[buffer(10)]],
  device atomic_uint* gMinToi        [[buffer(11)]],
  device atomic_uint* dbgCount       [[buffer(12)]],
  device float* dbgEvents            [[buffer(13)]],
  constant uint& dbgMax              [[buffer(14)]],
  constant float& dbgThr             [[buffer(15)]],
  constant uint& baseIndex           [[buffer(16)]],
  constant float& minAcceptT         [[buffer(17)]])
{
  for (;;) {
    uint idx = atomic_fetch_add_explicit(&(ctrs->popCounter), 1u, memory_order_relaxed);
    if (idx >= currCount) break;
    CCDDomain dom = curr[idx];
    uint qid = dom.query;
    CCDDataEE rec = data[qid];
    // Early prune: if this domain starts at/after current best ToI, skip.
    {
      uint bestBits0 = atomic_load_explicit(toiBits + qid, memory_order_relaxed);
      float bestToi0 = as_type<float>(bestBits0);
      uint gBits0 = atomic_load_explicit(gMinToi, memory_order_relaxed);
      float gToi0 = as_type<float>(gBits0);
      float gate = min(bestToi0, gToi0);
      if (dom.tuv[0].x >= gate) {
        continue;
      }
    }
    // Fast cull for EE (AABB over both edges across start/end with ms/err margins)
    {
      float3 aMin = float3(min(min(rec.v0s.x, rec.v0e.x), min(rec.v1s.x, rec.v1e.x)),
                           min(min(rec.v0s.y, rec.v0e.y), min(rec.v1s.y, rec.v1e.y)),
                           min(min(rec.v0s.z, rec.v0e.z), min(rec.v1s.z, rec.v1e.z)));
      float3 aMax = float3(max(max(rec.v0s.x, rec.v0e.x), max(rec.v1s.x, rec.v1e.x)),
                           max(max(rec.v0s.y, rec.v0e.y), max(rec.v1s.y, rec.v1e.y)),
                           max(max(rec.v0s.z, rec.v0e.z), max(rec.v1s.z, rec.v1e.z)));
      float3 bMin = float3(min(min(rec.v2s.x, rec.v2e.x), min(rec.v3s.x, rec.v3e.x)),
                           min(min(rec.v2s.y, rec.v2e.y), min(rec.v3s.y, rec.v3e.y)),
                           min(min(rec.v2s.z, rec.v2e.z), min(rec.v3s.z, rec.v3e.z)));
      float3 bMax = float3(max(max(rec.v2s.x, rec.v2e.x), max(rec.v3s.x, rec.v3e.x)),
                           max(max(rec.v2s.y, rec.v2e.y), max(rec.v3s.y, rec.v3e.y)),
                           max(max(rec.v2s.z, rec.v2e.z), max(rec.v3s.z, rec.v3e.z)));
      bool separated = false;
      if (aMax.x + rec.ms < bMin.x - rec.err.x || aMin.x - rec.ms > bMax.x + rec.err.x) separated = true;
      if (aMax.y + rec.ms < bMin.y - rec.err.y || aMin.y - rec.ms > bMax.y + rec.err.y) separated = true;
      if (aMax.z + rec.ms < bMin.z - rec.err.z || aMin.z - rec.ms > bMax.z + rec.err.z) separated = true;
      if (separated) continue;
    }
    uint evalCount = atomic_fetch_add_explicit(evalCounts + qid, 1u, memory_order_relaxed);
    if (maxIterations > 0 && evalCount >= static_cast<uint>(maxIterations)) {
      // 与 CUDA 一致：超过检查上限则跳过该域（不修改 toi）
      continue;
    }
    float trueTol = 0.0f;
    bool boxIn = false;
    if (!eeOriginInclusionQueue(rec, dom, rec.err, rec.ms, trueTol, boxIn)) {
      continue;
    }
    float3 widths = float3(dom.tuv[0].y - dom.tuv[0].x,
                           dom.tuv[1].y - dom.tuv[1].x,
                           dom.tuv[2].y - dom.tuv[2].x);
    bool allowZeroHit = (allowZero != 0u) || (dom.tuv[0].x > 0.0f);
    bool accept = (widths.x <= rec.tol.x) && (widths.y <= rec.tol.y) && (widths.z <= rec.tol.z);
    if (!accept && boxIn && allowZeroHit) accept = true;
    if (!accept && trueTol <= codTol && allowZeroHit) accept = true;
    if (accept && dom.tuv[0].x <= 1e-9f) {
      bool widths_ok = (widths.x <= rec.tol.x) && (widths.y <= rec.tol.y) && (widths.z <= rec.tol.z);
      if (!widths_ok) accept = false;
    }
    if (accept && (dom.tuv[0].y >= minAcceptT)) {
      float acceptT = dom.tuv[0].y;
      atomicMinFloat(toiBits + qid, acceptT);
      atomicMinFloat(gMinToi, acceptT);
      if (dbgCount && dbgEvents && dom.tuv[0].x <= dbgThr) {
        uint didx = atomic_fetch_add_explicit(dbgCount, 1u, memory_order_relaxed);
        if (didx < dbgMax) {
          device float* base = dbgEvents + didx*12u;
          base[0] = (float)(qid + baseIndex);
          base[1] = dom.tuv[0].x;
          base[2] = dom.tuv[0].y;
          base[3] = trueTol;
          base[4] = widths.x;
          base[5] = widths.y;
          base[6] = widths.z;
          float mask = 0.0f;
          if ((widths.x <= rec.tol.x) && (widths.y <= rec.tol.y) && (widths.z <= rec.tol.z)) mask += 1.0f;
          if (boxIn && ((allowZero != 0u) || (dom.tuv[0].x > 0.0f))) mask += 2.0f;
          if ((trueTol <= codTol) && ((allowZero != 0u) || (dom.tuv[0].x > 0.0f))) mask += 4.0f;
          base[7] = mask;
          base[8]  = rec.tol.x;
          base[9]  = rec.tol.y;
          base[10] = rec.tol.z;
          base[11] = 0.0f;
        }
      }
      continue;
    }
    int splitDim = selectSplitDimension(widths, rec.tol);
    float2 iv = dom.tuv[splitDim];
    if ((iv.y - iv.x) <= 1e-6f) {
      continue;
    }
    float mid = 0.5f * (iv.x + iv.y);
    CCDDomain first = dom;
    CCDDomain second = dom;
    domainSet(first, splitDim, iv.x, mid);
    domainSet(second, splitDim, mid, iv.y);
    uint bestBits = atomic_load_explicit(toiBits + qid, memory_order_relaxed);
    float bestToi = as_type<float>(bestBits);
    uint gBits = atomic_load_explicit(gMinToi, memory_order_relaxed);
    float gToi = as_type<float>(gBits);
    float gatePush = min(bestToi, gToi);
    bool pushSecond = true;
    if (splitDim == 0) {
      pushSecond = (mid <= gatePush);
    }
    if (pushSecond) {
      tryPushDomain(next, ctrs, maxNext, second);
    }
    tryPushDomain(next, ctrs, maxNext, first);
  }
}
)"; 
// Device-side reduction of minimal ToI using atomic min on bitcasted uint
static const char* kReduceMinToiKernelSrc = R"(
using namespace metal;
kernel void reduceMinToi(
  device const float* toi [[buffer(0)]],
  constant uint& n       [[buffer(1)]],
  device atomic_uint* gmin [[buffer(2)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= n) return;
  float v = toi[gid];
  v = clamp(v, 0.0f, 1.0f);
  uint u = as_type<uint>(v);
  atomic_fetch_min_explicit(gmin, u, memory_order_relaxed);
}
)";
static const char* kSweepSingleKernelSrc = R"(
using namespace metal;
kernel void sweepSingle(
  device const double* minX [[buffer(0)]],
  device const double* maxX [[buffer(1)]],
  device const double* minY [[buffer(2)]],
  device const double* maxY [[buffer(3)]],
  device const double* minZ [[buffer(4)]],
  device const double* maxZ [[buffer(5)]],
  device const int* v0     [[buffer(6)]],
  device const int* v1     [[buffer(7)]],
  device const int* v2     [[buffer(8)]],
  constant uint& n         [[buffer(9)]],
  device int2* outPairs    [[buffer(10)]],
  device atomic_uint* outCount [[buffer(11)]],
  constant uint& capacity  [[buffer(12)]],
  uint i [[thread_position_in_grid]]
) {
  if (i >= n) return;
  double amax = maxX[i];
  for (uint j = i + 1; j < n; ++j) {
    double bjmin = minX[j];
    if (amax < bjmin) break;
    bool overlapY = !(maxY[i] < minY[j] || minY[i] > maxY[j]);
    bool overlapZ = !(maxZ[i] < minZ[j] || minZ[i] > maxZ[j]);
    if (!(overlapY && overlapZ)) continue;
    bool share =
      (v0[i] == v0[j]) || (v0[i] == v1[j]) || (v0[i] == v2[j]) ||
      (v1[i] == v0[j]) || (v1[i] == v1[j]) || (v1[i] == v2[j]) ||
      (v2[i] == v0[j]) || (v2[i] == v1[j]) || (v2[i] == v2[j]);
    if (share) continue;
    uint idx = atomic_fetch_add_explicit(outCount, 1u, memory_order_relaxed);
    if (idx < capacity) {
      outPairs[idx] = int2((int)i, (int)j);
    }
  }
}
)";
static const char* kCCDToleranceKernelSrc = R"(
using namespace metal;
struct CCDDataVF {
  float3 v0s; float3 v1s; float3 v2s; float3 v3s;
  float3 v0e; float3 v1e; float3 v2e; float3 v3e;
  float ms; float3 err; float3 tol; float toi; int aid; int bid;
};
struct CCDDataEE {
  float3 v0s; float3 v1s; float3 v2s; float3 v3s;
  float3 v0e; float3 v1e; float3 v2e; float3 v3e;
  float ms; float3 err; float3 tol; float toi; int aid; int bid;
};
static inline float3 cmax3(float3 a, float3 b) { return float3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z)); }
static inline float maxAbs3(float3 a) { return max(max(fabs(a.x), fabs(a.y)), fabs(a.z)); }
static inline float maxLinf4(
  float3 p1, float3 p2, float3 p3, float3 p4,
  float3 p1e, float3 p2e, float3 p3e, float3 p4e) {
  float m = 0.0f;
  m = max(m, maxAbs3(p1e - p1));
  m = max(m, maxAbs3(p2e - p2));
  m = max(m, maxAbs3(p3e - p3));
  m = max(m, maxAbs3(p4e - p4));
  return m;
}
kernel void ccdComputeTolVF(
  device CCDDataVF* data [[buffer(0)]],
  constant uint& count      [[buffer(1)]],
  constant float& codTol    [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= count) return;
  device CCDDataVF& rec = data[gid];
  float3 p000 = rec.v0s - rec.v1s;
  float3 p001 = rec.v0s - rec.v3s;
  float3 p011 = rec.v0s - (rec.v2s + rec.v3s - rec.v1s);
  float3 p010 = rec.v0s - rec.v2s;
  float3 p100 = rec.v0e - rec.v1e;
  float3 p101 = rec.v0e - rec.v3e;
  float3 p111 = rec.v0e - (rec.v2e + rec.v3e - rec.v1e);
  float3 p110 = rec.v0e - rec.v2e;
  float tol0 = codTol / (3.0f * maxLinf4(p000,p001,p011,p010,p100,p101,p111,p110));
  float tol1 = codTol / (3.0f * maxLinf4(p000,p100,p101,p001,p010,p110,p111,p011));
  float tol2 = codTol / (3.0f * maxLinf4(p000,p100,p110,p010,p001,p101,p111,p011));
  rec.tol = float3(tol0, tol1, tol2);
  float3 maxv = float3(1.0f);
  maxv = cmax3(maxv, fabs(rec.v0s));
  maxv = cmax3(maxv, fabs(rec.v1s));
  maxv = cmax3(maxv, fabs(rec.v2s));
  maxv = cmax3(maxv, fabs(rec.v3s));
  maxv = cmax3(maxv, fabs(rec.v0e));
  maxv = cmax3(maxv, fabs(rec.v1e));
  maxv = cmax3(maxv, fabs(rec.v2e));
  maxv = cmax3(maxv, fabs(rec.v3e));
  float filter = rec.ms > 0.0f ? 4.053116e-06f : 3.576279e-06f;
  float3 max2 = maxv * maxv;
  rec.err = max2 * maxv * filter;
}
kernel void ccdComputeTolEE(
  device CCDDataEE* data [[buffer(0)]],
  constant uint& count      [[buffer(1)]],
  constant float& codTol    [[buffer(2)]],
  uint gid [[thread_position_in_grid]])
{
  if (gid >= count) return;
  device CCDDataEE& rec = data[gid];
  float3 p000 = rec.v0s - rec.v2s;
  float3 p001 = rec.v0s - rec.v3s;
  float3 p010 = rec.v1s - rec.v2s;
  float3 p011 = rec.v1s - rec.v3s;
  float3 p100 = rec.v0e - rec.v2e;
  float3 p101 = rec.v0e - rec.v3e;
  float3 p110 = rec.v1e - rec.v2e;
  float3 p111 = rec.v1e - rec.v3e;
  float tol0 = codTol / (3.0f * maxLinf4(p000,p001,p011,p010,p100,p101,p111,p110));
  float tol1 = codTol / (3.0f * maxLinf4(p000,p001,p011,p010,p100,p101,p111,p110));
  float tol2 = codTol / (3.0f * maxLinf4(p000,p100,p101,p001,p010,p110,p111,p011));
  rec.tol = float3(tol0, tol1, tol2);
  float3 maxv = float3(1.0f);
  maxv = cmax3(maxv, fabs(rec.v0s));
  maxv = cmax3(maxv, fabs(rec.v1s));
  maxv = cmax3(maxv, fabs(rec.v2s));
  maxv = cmax3(maxv, fabs(rec.v3s));
  maxv = cmax3(maxv, fabs(rec.v0e));
  maxv = cmax3(maxv, fabs(rec.v1e));
  maxv = cmax3(maxv, fabs(rec.v2e));
  maxv = cmax3(maxv, fabs(rec.v3e));
  float filter = rec.ms > 0.0f ? 3.814698e-06f : 3.337861e-06f;
  float3 max2 = maxv * maxv;
  rec.err = max2 * maxv * filter;
}
)";

static inline NSUInteger preferredThreadgroupWidth(id<MTLComputePipelineState> cps, NSUInteger fallback, NSUInteger problemSize)
{
    NSUInteger w = cps ? cps.maxTotalThreadsPerThreadgroup : 0;
    if (w == 0) w = fallback;
    if (w == 0) w = 64;
    if (problemSize > 0 && w > problemSize) w = problemSize;
    if (w == 0) w = 1;
    return w;
}

static inline float float_from_bits(uint32_t bits)
{
    union {
        uint32_t u;
        float f;
    } conv = { bits };
    return conv.f;
}


static const char* kSTQKernelSrc = R"(
using namespace metal;
struct Pair2i { int x; int y; };
constant uint QUEUE_SIZE = 64;

kernel void sweepSTQSingle(
  device const float*  minX [[buffer(0)]],
  device const float*  maxX [[buffer(1)]],
  device const float*  minY [[buffer(2)]],
  device const float*  maxY [[buffer(3)]],
  device const float*  minZ [[buffer(4)]],
  device const float*  maxZ [[buffer(5)]],
  device const int* v0      [[buffer(6)]],
  device const int* v1      [[buffer(7)]],
  device const int* v2      [[buffer(8)]],
  constant uint& n          [[buffer(9)]],
  device int2* outPairs     [[buffer(10)]],
  device atomic_uint* outCount [[buffer(11)]],
  constant uint& capacity   [[buffer(12)]],
  constant float& epsScale  [[buffer(13)]],
  device const uint* startJ [[buffer(14)]],
  device const uchar* listTag [[buffer(15)]],
  constant uint& twoLists [[buffer(16)]],
  constant float& yzEpsScale [[buffer(17)]],
  constant uint& maxSkipSteps [[buffer(18)]],
  constant uint& tgSize [[buffer(19)]],
  constant uint& yzTieUlps [[buffer(20)]],
  uint tid [[thread_index_in_threadgroup]],
  uint i0  [[threadgroup_position_in_grid]],
  uint gtid [[thread_position_in_grid]]
) {
  threadgroup Pair2i queue[QUEUE_SIZE];
  threadgroup atomic_uint qStart;
  threadgroup atomic_uint qEnd;
  if (tid == 0) {
    atomic_store_explicit(&qStart, 0u, memory_order_relaxed);
    atomic_store_explicit(&qEnd, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint i = gtid;
  if (i >= n) return;
  // 初始相邻对入队
  uint j0 = startJ ? startJ[i] : (i + 1);
  if (j0 < n && j0 > i) {
    float amax = maxX[i];
    float bmin = minX[j0];
    float eps = epsScale * max(1.0f, max(fabs(amax), fabs(bmin)));
    if (amax >= bmin + eps) {
      // 两列表：同源列表则不入队
      if (twoLists != 0 && listTag && listTag[i] == listTag[j0]) {
        // skip
      } else {
        // 仅按主轴重叠播种，不做 YZ 预筛与等待
        uint pos = atomic_fetch_add_explicit(&qEnd, 1u, memory_order_relaxed) % QUEUE_SIZE;
        queue[pos] = Pair2i{ (int)i, (int)j0 };
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  while (true) {
    uint start = atomic_load_explicit(&qStart, memory_order_relaxed);
    uint end   = atomic_load_explicit(&qEnd, memory_order_relaxed);
    uint size  = (end + QUEUE_SIZE - start) % QUEUE_SIZE;
    uint round = size;
    if (round == 0) break;
    // 本轮聚合写出：先在本地缓冲，再一次性原子预留全局空间
    threadgroup Pair2i localPairs[QUEUE_SIZE];
    threadgroup atomic_uint roundEmit;
    if (tid == 0) { atomic_store_explicit(&roundEmit, 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < round) {
      // 按 CUDA 语义：pop 时原子自增 start，立刻释放一个槽位
      uint idx = atomic_fetch_add_explicit(&qStart, 1u, memory_order_relaxed) % QUEUE_SIZE;
      Pair2i res = queue[idx];
      int a = res.x;
      int b = res.y;
      // 两列表：在消费阶段过滤同源对，保持推进链完整且结果正确
      bool validList = !(twoLists != 0 && listTag && listTag[a] == listTag[b]);
      bool doEmit = false;
      Pair2i emitPair;
      if (validList) {
      // yz + 共享顶点（使用相对 epsilon）
      float ay0 = minY[a], ay1 = maxY[a], by0 = minY[b], by1 = maxY[b];
      float az0 = minZ[a], az1 = maxZ[a], bz0 = minZ[b], bz1 = maxZ[b];
      float yzEps = yzEpsScale * max(1.0f, max(max(fabs(ay1), fabs(by0)), max(fabs(az1), fabs(bz0))));
      bool overlapY = !(ay1 + yzEps < by0 || ay0 > by1 + yzEps);
      bool overlapZ = !(az1 + yzEps < bz0 || az0 > bz1 + yzEps);
      // 近边界条件化严格化：仅当设定了 ulp 阈值且处于近相切区域时，要求最小重叠长度大于阈值
      if (yzTieUlps > 0u) {
        const float FLT_EPS = 1.1920929e-7f;
        float yRel = max(1.0f, max(max(fabs(ay1), fabs(by1)), max(fabs(ay0), fabs(by0))));
        float zRel = max(1.0f, max(max(fabs(az1), fabs(bz1)), max(fabs(az0), fabs(bz0))));
        float yTau = (float)yzTieUlps * FLT_EPS * yRel;
        float zTau = (float)yzTieUlps * FLT_EPS * zRel;
        float yOverlap = min(ay1, by1) - max(ay0, by0);
        float zOverlap = min(az1, bz1) - max(az0, bz0);
        if (overlapY && yOverlap <= yTau) overlapY = false;
        if (overlapZ && zOverlap <= zTau) overlapZ = false;
      }
      if (overlapY && overlapZ) {
        bool share =
          (v0[a] == v0[b]) || (v0[a] == v1[b]) || (v0[a] == v2[b]) ||
          (v1[a] == v0[b]) || (v1[a] == v1[b]) || (v1[a] == v2[b]) ||
          (v2[a] == v0[b]) || (v2[a] == v1[b]) || (v2[a] == v2[b]);
        if (!share) {
          doEmit = true;
          emitPair = Pair2i{ min(a,b), max(a,b) };
        }
      }
      } // end two-lists same-list filter
      if (doEmit) {
        uint li = atomic_fetch_add_explicit(&roundEmit, 1u, memory_order_relaxed);
        if (li < QUEUE_SIZE) { localPairs[li] = emitPair; }
      }
      // 推进 (a, b+1)
      int nb = b + 1;
      if ((uint)nb < n) {
        float amax = maxX[a];
        float nbmin = minX[nb];
        float eps2 = epsScale * max(1.0f, max(fabs(amax), fabs(nbmin)));
        if (amax >= nbmin + eps2) {
          // 单步推进，不做 YZ 预筛与等待；不在入队阶段过滤同源对
          uint pos = atomic_fetch_add_explicit(&qEnd, 1u, memory_order_relaxed) % QUEUE_SIZE;
          queue[pos] = Pair2i{ a, nb };
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 线程组一次性写回全局
    uint emitCount = atomic_load_explicit(&roundEmit, memory_order_relaxed);
    if (emitCount > 0) {
      uint base = 0;
      if (tid == 0) {
        base = atomic_fetch_add_explicit(outCount, emitCount, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // 广播 base（简化：再次读取 outCount 减去 emitCount 获取 base）
      // 由于 Metal 无直接广播，这里通过共享内存再写一次
      threadgroup atomic_uint baseHolder;
      if (tid == 0) { atomic_store_explicit(&baseHolder, base, memory_order_relaxed); }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      base = atomic_load_explicit(&baseHolder, memory_order_relaxed);
      // 边界保护
      uint maxWrite = 0;
      if (base < capacity) {
        uint remain = capacity - base;
        maxWrite = emitCount < remain ? emitCount : remain;
      } else {
        maxWrite = 0;
      }
      // 协作写回
      for (uint k = tid; k < maxWrite; k += tgSize) {
        Pair2i p = localPairs[k];
        outPairs[base + k] = int2(p.x, p.y);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
)";
}

struct MetalRuntime::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    bool ok = false;
    // Global tuning/env
    NSUInteger tgOverride = 0;        // 0 = auto
    bool enableTiming = false;        // enable per-dispatch timing logs/CSV
    std::string timingCsvPath;        // if non-empty, append timing here
    // 缓存编译后的库与管线
    id<MTLLibrary> yzLib = nil;
    id<MTLComputePipelineState> yzCps = nil;
    id<MTLLibrary> sweepLib = nil;
    id<MTLComputePipelineState> sweepCps = nil;
    id<MTLLibrary> stqLib = nil;
    id<MTLComputePipelineState> stqCps = nil;
    id<MTLLibrary> npLib = nil;
    id<MTLComputePipelineState> npAddCps = nil;
    id<MTLComputePipelineState> npPackVfCps = nil;
    id<MTLComputePipelineState> npPackEeCps = nil;
    id<MTLComputePipelineState> ccdRunVfPlaceholderCps = nil;
    id<MTLComputePipelineState> vfRootSkeletonCps = nil;
    id<MTLComputePipelineState> eeRootSkeletonCps = nil;
    id<MTLComputePipelineState> reduceMinToiCps = nil;
    // Reusable buffers (staging/shared and device-private)
    id<MTLBuffer> sV0 = nil;   // shared staging V0
    id<MTLBuffer> sV1 = nil;   // shared staging V1
    id<MTLBuffer> sF  = nil;   // shared faces
    id<MTLBuffer> sE  = nil;   // shared edges
    id<MTLBuffer> sO  = nil;   // shared overlaps chunk
    id<MTLBuffer> sNO = nil;   // shared count
    id<MTLBuffer> sToi = nil;  // shared ToI chunk
    id<MTLBuffer> pV0 = nil;   // private V0
    id<MTLBuffer> pV1 = nil;   // private V1
    id<MTLBuffer> pF  = nil;   // private faces
    id<MTLBuffer> pE  = nil;   // private edges
    id<MTLBuffer> pCCD_VF = nil; // private CCD VF chunk
    id<MTLBuffer> pCCD_EE = nil; // private CCD EE chunk
    // Queue-based root finder data
    id<MTLComputePipelineState> vfQueueCps = nil;
    id<MTLComputePipelineState> eeQueueCps = nil;
    id<MTLComputePipelineState> ccdTolVfCps = nil;
    id<MTLComputePipelineState> ccdTolEeCps = nil;
    id<MTLComputePipelineState> ccdInitDomainCps = nil;
    id<MTLComputePipelineState> ccdResetCountersCps = nil;
    id<MTLComputePipelineState> ccdInitToiCps = nil;
    id<MTLBuffer> qDomainA = nil;
    id<MTLBuffer> qDomainB = nil;
    id<MTLBuffer> qToi = nil;
    id<MTLBuffer> qCounters = nil;   // popCounter, nextCounter, overflow flag
    id<MTLBuffer> qCurrentCount = nil;
    id<MTLBuffer> qMaxCount = nil;
    id<MTLBuffer> qAllowZero = nil;
    id<MTLBuffer> qCodTol = nil;
    id<MTLBuffer> qMaxIterations = nil;
    id<MTLBuffer> qEvalCounts = nil;
    id<MTLLibrary> queueLib = nil;
    id<MTLLibrary> tolLib = nil;
    size_t queueDomainCapacity = 0;
    size_t queueToiCapacity = 0;
    size_t queueEvalCapacity = 0;
    // Debug logging for queue path
    bool queueDebug = false;
    id<MTLBuffer> qDbgCount = nil;   // atomic counter (uint32_t)
    id<MTLBuffer> qDbgEvents = nil;  // float events [N x 8]
    id<MTLBuffer> qDbgMaxBuf = nil;  // uint32_t dbg max
    id<MTLBuffer> qDbgThrBuf = nil;  // float threshold for t

    bool ensureTolerancePipelines();
    bool ensureQueuePipelines();
    bool ensureQueueBuffers(size_t domainCapacity, size_t count);
    bool runQueueChunkVF(uint32_t baseIndex,
                         uint32_t count,
                         size_t domainCapacity,
                         float tolerance,
                         id<MTLBuffer> bTolScalar,
                         id<MTLBuffer> bAz,
                         id<MTLBuffer> bMin,
                         int maxIterations);
    bool runQueueChunkEE(uint32_t baseIndex,
                         uint32_t count,
                         size_t domainCapacity,
                         float tolerance,
                         id<MTLBuffer> bTolScalar,
                         id<MTLBuffer> bAz,
                         id<MTLBuffer> bMin,
                         int maxIterations);
};

// Append a line to CSV if path is provided. Columns: phase,count,tgWidth,ms
static inline void append_timing_csv_if_needed(const std::string& path,
                                               const char* phase,
                                               uint64_t count,
                                               uint32_t tgWidth,
                                               double ms)
{
    if (path.empty()) return;
    std::ofstream ofs(path, std::ios::out | std::ios::app);
    if (!ofs.is_open()) return;
    ofs << phase << "," << count << "," << tgWidth << "," << ms << "\n";
}

MetalRuntime& MetalRuntime::instance()
{
    static MetalRuntime inst;
    return inst;
}

static inline void ensureBuffer(id<MTLDevice> device, id<MTLBuffer>& buf, size_t length, MTLResourceOptions opts)
{
    if (buf == nil || [buf length] < length) {
        buf = [device newBufferWithLength:length options:opts];
    }
}

static inline bool env_flag_enabled(const char* name, bool defaultValue = false)
{
    const char* val = std::getenv(name);
    if (!val) return defaultValue;
    if (val[0] == '\0') return true;
    std::string str(val);
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (str == "0" || str == "false" || str == "off" || str == "no") {
        return false;
    }
    return true;
}

static inline size_t env_size_override(const char* name, size_t defaultValue, size_t minValue = 1)
{
    if (const char* val = std::getenv(name)) {
        try {
            size_t parsed = static_cast<size_t>(std::stoull(val));
            if (parsed < minValue) parsed = minValue;
            return parsed;
        } catch (...) {
            return defaultValue;
        }
    }
    return defaultValue;
}

static inline int env_int_override(const char* name, int defaultValue, int minValue = INT32_MIN, int maxValue = INT32_MAX)
{
    if (const char* val = std::getenv(name)) {
        try {
            long parsed = std::stol(val);
            if (parsed < minValue) parsed = minValue;
            if (parsed > maxValue) parsed = maxValue;
            return static_cast<int>(parsed);
        } catch (...) {
            return defaultValue;
        }
    }
    return defaultValue;
}

enum class RootMode {
    Legacy,
    Queue
};

static inline RootMode root_mode_from_env()
{
    const char* mode = std::getenv("SCALABLE_CCD_ROOT_MODE");
    if (mode && mode[0] != '\0') {
        std::string m(mode);
        std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (m == "queue") return RootMode::Queue;
        if (m == "legacy") return RootMode::Legacy;
    }
    // 缺省对齐 CUDA：默认使用队列版根查找（Queue）
    return env_flag_enabled("SCALABLE_CCD_ROOT_USE_QUEUE", true) ? RootMode::Queue : RootMode::Legacy;
}

static inline RootMode root_mode_vf_from_env()
{
    const char* mode = std::getenv("SCALABLE_CCD_VF_ROOT_MODE");
    if (mode && mode[0] != '\0') {
        std::string m(mode);
        std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (m == "queue") return RootMode::Queue;
        if (m == "legacy") return RootMode::Legacy;
    }
    // 默认 VF 用 Legacy 更稳健，EE 仍可用全局策略
    return RootMode::Legacy;
}

static inline RootMode root_mode_ee_from_env()
{
    const char* mode = std::getenv("SCALABLE_CCD_EE_ROOT_MODE");
    if (mode && mode[0] != '\0') {
        std::string m(mode);
        std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (m == "queue") return RootMode::Queue;
        if (m == "legacy") return RootMode::Legacy;
    }
    return root_mode_from_env();
}
struct QueueCountersHost {
    uint32_t popCounter = 0;
    uint32_t pushCounter = 0;
    uint32_t overflow = 0;
};

struct CCDDomainHost {
    float tuv[3][2];
    uint32_t query = 0;
    uint32_t padding = 0;
};

bool MetalRuntime::Impl::ensureTolerancePipelines()
{
    NSError* err = nil;
    if (!tolLib) {
        MTLCompileOptions* opts = [MTLCompileOptions new];
        tolLib = [device newLibraryWithSource:[NSString stringWithUTF8String:kCCDToleranceKernelSrc]
                                      options:opts
                                        error:&err];
        if (!tolLib) {
            scalable_ccd::logger().error("Metal queue: build tolerance library failed: {}",
                                         err ? err.localizedDescription.UTF8String : "unknown error");
            return false;
        }
    }
    if (!ccdTolVfCps) {
        id<MTLFunction> fn = [tolLib newFunctionWithName:@"ccdComputeTolVF"];
        if (!fn) {
            scalable_ccd::logger().error("Metal queue: create function ccdComputeTolVF failed");
            return false;
        }
        ccdTolVfCps = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!ccdTolVfCps) {
            scalable_ccd::logger().error("Metal queue: pipeline ccdComputeTolVF failed: {}",
                                         err ? err.localizedDescription.UTF8String : "unknown error");
            return false;
        }
    }
    if (!ccdTolEeCps) {
        id<MTLFunction> fn = [tolLib newFunctionWithName:@"ccdComputeTolEE"];
        if (!fn) {
            scalable_ccd::logger().error("Metal queue: create function ccdComputeTolEE failed");
            return false;
        }
        ccdTolEeCps = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!ccdTolEeCps) {
            scalable_ccd::logger().error("Metal queue: pipeline ccdComputeTolEE failed: {}",
                                         err ? err.localizedDescription.UTF8String : "unknown error");
            return false;
        }
    }
    return true;
}

bool MetalRuntime::Impl::ensureQueuePipelines()
{
    NSError* err = nil;
    if (!queueLib) {
        MTLCompileOptions* opts = [MTLCompileOptions new];
        queueLib = [device newLibraryWithSource:[NSString stringWithUTF8String:kCCDQueueKernelSrc]
                                        options:opts
                                          error:&err];
        if (!queueLib) {
            scalable_ccd::logger().error("Metal queue: build queue library failed: {}",
                                         err ? err.localizedDescription.UTF8String : "unknown error");
            return false;
        }
    }
    if (!ccdInitDomainCps) {
        id<MTLFunction> fn = [queueLib newFunctionWithName:@"ccdInitDomains"];
        if (!fn) {
            scalable_ccd::logger().error("Metal queue: create function ccdInitDomains failed");
            return false;
        }
        ccdInitDomainCps = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!ccdInitDomainCps) {
            scalable_ccd::logger().error("Metal queue: pipeline ccdInitDomains failed: {}",
                                         err ? err.localizedDescription.UTF8String : "unknown error");
            return false;
        }
    }
    if (!ccdResetCountersCps) {
        id<MTLFunction> fn = [queueLib newFunctionWithName:@"ccdResetCounters"];
        if (!fn) {
            scalable_ccd::logger().error("Metal queue: create function ccdResetCounters failed");
            return false;
        }
        ccdResetCountersCps = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!ccdResetCountersCps) {
            scalable_ccd::logger().error("Metal queue: pipeline ccdResetCounters failed: {}",
                                         err ? err.localizedDescription.UTF8String : "unknown error");
            return false;
        }
    }
    if (!ccdInitToiCps) {
        id<MTLFunction> fn = [queueLib newFunctionWithName:@"ccdInitToi"];
        if (!fn) {
            scalable_ccd::logger().error("Metal queue: create function ccdInitToi failed");
            return false;
        }
        ccdInitToiCps = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!ccdInitToiCps) {
            scalable_ccd::logger().error("Metal queue: pipeline ccdInitToi failed: {}",
                                         err ? err.localizedDescription.UTF8String : "unknown error");
            return false;
        }
    }
    if (!vfQueueCps) {
        id<MTLFunction> fn = [queueLib newFunctionWithName:@"vfQueueProcess"];
        if (!fn) {
            scalable_ccd::logger().error("Metal queue: create function vfQueueProcess failed");
            return false;
        }
        vfQueueCps = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!vfQueueCps) {
            scalable_ccd::logger().error("Metal queue: pipeline vfQueueProcess failed: {}",
                                         err ? err.localizedDescription.UTF8String : "unknown error");
            return false;
        }
    }
    if (!eeQueueCps) {
        id<MTLFunction> fn = [queueLib newFunctionWithName:@"eeQueueProcess"];
        if (!fn) {
            scalable_ccd::logger().error("Metal queue: create function eeQueueProcess failed");
            return false;
        }
        eeQueueCps = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!eeQueueCps) {
            scalable_ccd::logger().error("Metal queue: pipeline eeQueueProcess failed: {}",
                                         err ? err.localizedDescription.UTF8String : "unknown error");
            return false;
        }
    }
    return true;
}

bool MetalRuntime::Impl::ensureQueueBuffers(size_t domainCapacity, size_t count)
{
    const size_t domainBytes = domainCapacity * sizeof(CCDDomainHost);
    ensureBuffer(device, qDomainA, domainBytes, MTLResourceStorageModePrivate);
    ensureBuffer(device, qDomainB, domainBytes, MTLResourceStorageModePrivate);
    ensureBuffer(device, qCounters, sizeof(QueueCountersHost), MTLResourceStorageModeShared);
    ensureBuffer(device, qCurrentCount, sizeof(uint32_t), MTLResourceStorageModeShared);
    ensureBuffer(device, qMaxCount, sizeof(uint32_t), MTLResourceStorageModeShared);
    ensureBuffer(device, qCodTol, sizeof(float), MTLResourceStorageModeShared);
    ensureBuffer(device, qAllowZero, sizeof(uint32_t), MTLResourceStorageModeShared);
    ensureBuffer(device, qMaxIterations, sizeof(uint32_t), MTLResourceStorageModeShared);
    ensureBuffer(device, qToi, sizeof(uint32_t) * count, MTLResourceStorageModeShared);
    ensureBuffer(device, qEvalCounts, sizeof(uint32_t) * count, MTLResourceStorageModeShared);
    // Optional debug
    queueDebug = env_flag_enabled("SCALABLE_CCD_QUEUE_DEBUG", false);
    if (queueDebug) {
        uint32_t dbgMax = static_cast<uint32_t>(env_int_override("SCALABLE_CCD_QUEUE_DBG_MAX", 1024, 1, 1<<24));
        float dbgThr = 1e-8f;
        if (const char* e = std::getenv("SCALABLE_CCD_QUEUE_DBG_THR")) {
            try { dbgThr = std::stof(e); } catch (...) {}
        }
        ensureBuffer(device, qDbgCount, sizeof(uint32_t), MTLResourceStorageModeShared);
        ensureBuffer(device, qDbgEvents, static_cast<size_t>(dbgMax) * 12 * sizeof(float), MTLResourceStorageModeShared);
        ensureBuffer(device, qDbgMaxBuf, sizeof(uint32_t), MTLResourceStorageModeShared);
        ensureBuffer(device, qDbgThrBuf, sizeof(float), MTLResourceStorageModeShared);
        if (qDbgCount && qDbgEvents && qDbgMaxBuf && qDbgThrBuf) {
            uint32_t zero = 0u;
            memcpy([qDbgCount contents], &zero, sizeof(uint32_t));
            memcpy([qDbgMaxBuf contents], &dbgMax, sizeof(uint32_t));
            memcpy([qDbgThrBuf contents], &dbgThr, sizeof(float));
        }
    } else {
        qDbgCount = nil; qDbgEvents = nil; qDbgMaxBuf = nil; qDbgThrBuf = nil;
    }
    queueDomainCapacity = domainCapacity;
    queueToiCapacity = count;
    queueEvalCapacity = count;
    return qDomainA && qDomainB && qCounters && qToi && qEvalCounts;
}

bool MetalRuntime::Impl::runQueueChunkVF(
    uint32_t baseIndex,
    uint32_t count,
    size_t domainCapacity,
    float tolerance,
    id<MTLBuffer> bTolScalar,
    id<MTLBuffer> bAz,
    id<MTLBuffer> bMin,
    int maxIterations)
{
    if (count == 0) return true;
    if (!ensureTolerancePipelines()) {
        scalable_ccd::logger().error("Metal VF queue: tolerance pipeline init failed");
        return false;
    }
    if (!ensureQueuePipelines()) {
        scalable_ccd::logger().error("Metal VF queue: pipeline init failed");
        return false;
    }
    if (!ensureQueueBuffers(domainCapacity, count)) {
        scalable_ccd::logger().error("Metal VF queue: buffer allocation failed capacity={} count={}",
                                     static_cast<uint64_t>(domainCapacity),
                                     static_cast<uint64_t>(count));
        return false;
    }

    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ccdTolVfCps];
        [enc setBuffer:pCCD_VF offset:0 atIndex:0];
        [enc setBuffer:sNO offset:0 atIndex:1];
        [enc setBuffer:bTolScalar offset:0 atIndex:2];
        NSUInteger tgW = preferredThreadgroupWidth(ccdTolVfCps, 64, count);
        MTLSize tg = MTLSizeMake(tgW, 1, 1);
        MTLSize grid = MTLSizeMake(count, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ccdInitDomainCps];
            [enc setBuffer:qDomainA offset:0 atIndex:0];
            [enc setBuffer:sNO offset:0 atIndex:1];
            NSUInteger tgW = preferredThreadgroupWidth(ccdInitDomainCps, 64, count);
            MTLSize tg = MTLSizeMake(tgW,1,1);
            MTLSize grid = MTLSizeMake(count,1,1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ccdInitToiCps];
            [enc setBuffer:qToi offset:0 atIndex:0];
            [enc setBuffer:sNO offset:0 atIndex:1];
            NSUInteger tgW = preferredThreadgroupWidth(ccdInitToiCps, 64, count);
            MTLSize tg = MTLSizeMake(tgW,1,1);
            MTLSize grid = MTLSizeMake(count,1,1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];
    }

    uint32_t currentCount = count;
    id<MTLBuffer> currentDomains = qDomainA;
    id<MTLBuffer> nextDomains = qDomainB;
    const uint32_t queueCapacityU32 = static_cast<uint32_t>(domainCapacity);
    memcpy([qMaxCount contents], &queueCapacityU32, sizeof(uint32_t));
    memcpy([qCodTol contents], &tolerance, sizeof(float));
    if (qEvalCounts && [qEvalCounts length] >= sizeof(uint32_t) * count) {
        memset([qEvalCounts contents], 0, sizeof(uint32_t) * count);
    }
    {
        int qMax = env_int_override("SCALABLE_CCD_QUEUE_MAX_EVALS", maxIterations, 1, INT32_MAX);
        memcpy([qMaxIterations contents], &qMax, sizeof(int));
    }

    bool overflowed = false;
    while (currentCount > 0 && !overflowed) {
        memcpy([qCurrentCount contents], &currentCount, sizeof(uint32_t));
        {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ccdResetCountersCps];
            [enc setBuffer:qCounters offset:0 atIndex:0];
            MTLSize grid = MTLSizeMake(1,1,1);
            MTLSize tg = MTLSizeMake(1,1,1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:vfQueueCps];
            [enc setBuffer:pCCD_VF offset:0 atIndex:0];
            [enc setBuffer:currentDomains offset:0 atIndex:1];
            [enc setBuffer:qCurrentCount offset:0 atIndex:2];
            [enc setBuffer:nextDomains offset:0 atIndex:3];
            [enc setBuffer:qCounters offset:0 atIndex:4];
            [enc setBuffer:qCodTol offset:0 atIndex:5];
            [enc setBuffer:bAz offset:0 atIndex:6];
            [enc setBuffer:qToi offset:0 atIndex:7];
            [enc setBuffer:qMaxCount offset:0 atIndex:8];
            [enc setBuffer:qEvalCounts offset:0 atIndex:9];
            [enc setBuffer:qMaxIterations offset:0 atIndex:10];
            [enc setBuffer:bMin offset:0 atIndex:11];
            id<MTLBuffer> bBase = [device newBufferWithBytes:&baseIndex length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            [enc setBuffer:bBase offset:0 atIndex:16];
            float minAcceptT = 0.0f;
            if (const char* e = std::getenv("SCALABLE_CCD_QUEUE_ACCEPT_MIN_T")) {
                try { minAcceptT = std::stof(e); } catch (...) { minAcceptT = 0.0f; }
            }
            id<MTLBuffer> bMinAcc = [device newBufferWithBytes:&minAcceptT length:sizeof(float) options:MTLResourceStorageModeShared];
            [enc setBuffer:bMinAcc offset:0 atIndex:17];
            if (queueDebug) {
                [enc setBuffer:qDbgCount offset:0 atIndex:12];
                [enc setBuffer:qDbgEvents offset:0 atIndex:13];
                [enc setBuffer:qDbgMaxBuf offset:0 atIndex:14];
                [enc setBuffer:qDbgThrBuf offset:0 atIndex:15];
            } else {
                [enc setBuffer:nil offset:0 atIndex:12];
                [enc setBuffer:nil offset:0 atIndex:13];
                [enc setBuffer:nil offset:0 atIndex:14];
                [enc setBuffer:nil offset:0 atIndex:15];
            }
            NSUInteger workerCount = std::max<uint32_t>(1, std::min<uint32_t>(currentCount, 256u));
            NSUInteger tgW = preferredThreadgroupWidth(vfQueueCps, 64, workerCount);
            MTLSize tg = MTLSizeMake(tgW,1,1);
            MTLSize grid = MTLSizeMake(workerCount,1,1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        QueueCountersHost* ctr = static_cast<QueueCountersHost*>([qCounters contents]);
        if (ctr->overflow) {
            overflowed = true;
            break;
        }
        currentCount = ctr->pushCounter;
        std::swap(currentDomains, nextDomains);
    }
    if (overflowed) {
        scalable_ccd::logger().warn("Metal VF queue overflow at capacity={} count={}",
                                     static_cast<uint64_t>(domainCapacity),
                                     static_cast<uint64_t>(count));
        if (env_flag_enabled("SCALABLE_CCD_QUEUE_DEBUG", false) && qEvalCounts) {
            uint32_t* evals = static_cast<uint32_t*>([qEvalCounts contents]);
            uint32_t maxv = 0; uint32_t maxi = 0;
            for (uint32_t i = 0; i < count; ++i) { if (evals[i] > maxv) { maxv = evals[i]; maxi = i; } }
            scalable_ccd::logger().warn("Metal VF queue debug: max evals per query = {} (qid={})",
                                        static_cast<uint64_t>(maxv),
                                        static_cast<uint64_t>(maxi));
        }
        return false;
    }

    float* toiFloats = static_cast<float*>([sToi contents]);
    uint32_t* toiBits = static_cast<uint32_t*>([qToi contents]);
    for (uint32_t i = 0; i < count; ++i) {
        toiFloats[i] = float_from_bits(toiBits[i]);
    }
    // Debug dump (optional)
    if (queueDebug && qDbgCount && qDbgEvents) {
        uint32_t* pc = static_cast<uint32_t*>([qDbgCount contents]);
        uint32_t nlog = *pc;
        nlog = std::min(nlog, static_cast<uint32_t>(20));
        float* ev = static_cast<float*>([qDbgEvents contents]);
        for (uint32_t i = 0; i < nlog; ++i) {
            float* base = ev + i*12u;
            scalable_ccd::logger().warn("VF dbg accept: qid={} t=[{:.9f},{:.9f}] trueTol={:.3e} w=({:.3e},{:.3e},{:.3e}) tol=({:.3e},{:.3e},{:.3e}) mask={}",
                                        static_cast<int>(base[0]),
                                        static_cast<double>(base[1]),
                                        static_cast<double>(base[2]),
                                        static_cast<double>(base[3]),
                                        static_cast<double>(base[4]),
                                        static_cast<double>(base[5]),
                                        static_cast<double>(base[6]),
                                        static_cast<double>(base[8]),
                                        static_cast<double>(base[9]),
                                        static_cast<double>(base[10]),
                                        static_cast<int>(base[7]));
        }
    }

    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:reduceMinToiCps];
        [enc setBuffer:sToi offset:0 atIndex:0];
        [enc setBuffer:sNO offset:0 atIndex:1];
        [enc setBuffer:bMin offset:0 atIndex:2];
        NSUInteger tgW = preferredThreadgroupWidth(reduceMinToiCps, 64, count);
        MTLSize tg = MTLSizeMake(tgW,1,1);
        MTLSize grid = MTLSizeMake(count,1,1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    return true;
}

bool MetalRuntime::Impl::runQueueChunkEE(
    uint32_t baseIndex,
    uint32_t count,
    size_t domainCapacity,
    float tolerance,
    id<MTLBuffer> bTolScalar,
    id<MTLBuffer> bAz,
    id<MTLBuffer> bMin,
    int maxIterations)
{
    if (count == 0) return true;
    if (!ensureTolerancePipelines()) {
        scalable_ccd::logger().error("Metal EE queue: tolerance pipeline init failed");
        return false;
    }
    if (!ensureQueuePipelines()) {
        scalable_ccd::logger().error("Metal EE queue: pipeline init failed");
        return false;
    }
    if (!ensureQueueBuffers(domainCapacity, count)) {
        scalable_ccd::logger().error("Metal EE queue: buffer allocation failed capacity={} count={}",
                                     static_cast<uint64_t>(domainCapacity),
                                     static_cast<uint64_t>(count));
        return false;
    }

    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ccdTolEeCps];
        [enc setBuffer:pCCD_EE offset:0 atIndex:0];
        [enc setBuffer:sNO offset:0 atIndex:1];
        [enc setBuffer:bTolScalar offset:0 atIndex:2];
        NSUInteger tgW = preferredThreadgroupWidth(ccdTolEeCps, 64, count);
        MTLSize tg = MTLSizeMake(tgW,1,1);
        MTLSize grid = MTLSizeMake(count,1,1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ccdInitDomainCps];
            [enc setBuffer:qDomainA offset:0 atIndex:0];
            [enc setBuffer:sNO offset:0 atIndex:1];
            NSUInteger tgW = preferredThreadgroupWidth(ccdInitDomainCps, 64, count);
            MTLSize tg = MTLSizeMake(tgW,1,1);
            MTLSize grid = MTLSizeMake(count,1,1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ccdInitToiCps];
            [enc setBuffer:qToi offset:0 atIndex:0];
            [enc setBuffer:sNO offset:0 atIndex:1];
            NSUInteger tgW = preferredThreadgroupWidth(ccdInitToiCps, 64, count);
            MTLSize tg = MTLSizeMake(tgW,1,1);
            MTLSize grid = MTLSizeMake(count,1,1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];
    }

    uint32_t currentCount = count;
    id<MTLBuffer> currentDomains = qDomainA;
    id<MTLBuffer> nextDomains = qDomainB;
    const uint32_t queueCapacityU32 = static_cast<uint32_t>(domainCapacity);
    memcpy([qMaxCount contents], &queueCapacityU32, sizeof(uint32_t));
    memcpy([qCodTol contents], &tolerance, sizeof(float));
    if (qEvalCounts && [qEvalCounts length] >= sizeof(uint32_t) * count) {
        memset([qEvalCounts contents], 0, sizeof(uint32_t) * count);
    }
    {
        int qMax = env_int_override("SCALABLE_CCD_QUEUE_MAX_EVALS", maxIterations, 1, INT32_MAX);
        memcpy([qMaxIterations contents], &qMax, sizeof(int));
    }

    bool overflowed = false;
    while (currentCount > 0 && !overflowed) {
        memcpy([qCurrentCount contents], &currentCount, sizeof(uint32_t));
        {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ccdResetCountersCps];
            [enc setBuffer:qCounters offset:0 atIndex:0];
            MTLSize grid = MTLSizeMake(1,1,1);
            MTLSize tg = MTLSizeMake(1,1,1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:eeQueueCps];
            [enc setBuffer:pCCD_EE offset:0 atIndex:0];
            [enc setBuffer:currentDomains offset:0 atIndex:1];
            [enc setBuffer:qCurrentCount offset:0 atIndex:2];
            [enc setBuffer:nextDomains offset:0 atIndex:3];
            [enc setBuffer:qCounters offset:0 atIndex:4];
            [enc setBuffer:qCodTol offset:0 atIndex:5];
            [enc setBuffer:bAz offset:0 atIndex:6];
            [enc setBuffer:qToi offset:0 atIndex:7];
            [enc setBuffer:qMaxCount offset:0 atIndex:8];
            [enc setBuffer:qEvalCounts offset:0 atIndex:9];
            [enc setBuffer:qMaxIterations offset:0 atIndex:10];
            [enc setBuffer:bMin offset:0 atIndex:11];
            id<MTLBuffer> bBase = [device newBufferWithBytes:&baseIndex length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            [enc setBuffer:bBase offset:0 atIndex:16];
            float minAcceptT = 0.0f;
            if (const char* e = std::getenv("SCALABLE_CCD_QUEUE_ACCEPT_MIN_T")) {
                try { minAcceptT = std::stof(e); } catch (...) { minAcceptT = 0.0f; }
            }
            id<MTLBuffer> bMinAcc = [device newBufferWithBytes:&minAcceptT length:sizeof(float) options:MTLResourceStorageModeShared];
            [enc setBuffer:bMinAcc offset:0 atIndex:17];
            if (queueDebug) {
                [enc setBuffer:qDbgCount offset:0 atIndex:12];
                [enc setBuffer:qDbgEvents offset:0 atIndex:13];
                [enc setBuffer:qDbgMaxBuf offset:0 atIndex:14];
                [enc setBuffer:qDbgThrBuf offset:0 atIndex:15];
            } else {
                [enc setBuffer:nil offset:0 atIndex:12];
                [enc setBuffer:nil offset:0 atIndex:13];
                [enc setBuffer:nil offset:0 atIndex:14];
                [enc setBuffer:nil offset:0 atIndex:15];
            }
            NSUInteger workerCount = std::max<uint32_t>(1, std::min<uint32_t>(currentCount, 256u));
            NSUInteger tgW = preferredThreadgroupWidth(eeQueueCps, 64, workerCount);
            MTLSize tg = MTLSizeMake(tgW,1,1);
            MTLSize grid = MTLSizeMake(workerCount,1,1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        QueueCountersHost* ctr = static_cast<QueueCountersHost*>([qCounters contents]);
        if (ctr->overflow) {
            overflowed = true;
            break;
        }
        currentCount = ctr->pushCounter;
        std::swap(currentDomains, nextDomains);
    }
    if (overflowed) {
        scalable_ccd::logger().warn("Metal EE queue overflow at capacity={} count={}",
                                     static_cast<uint64_t>(domainCapacity),
                                     static_cast<uint64_t>(count));
        if (env_flag_enabled("SCALABLE_CCD_QUEUE_DEBUG", false) && qEvalCounts) {
            uint32_t* evals = static_cast<uint32_t*>([qEvalCounts contents]);
            uint32_t maxv = 0; uint32_t maxi = 0;
            for (uint32_t i = 0; i < count; ++i) { if (evals[i] > maxv) { maxv = evals[i]; maxi = i; } }
            scalable_ccd::logger().warn("Metal EE queue debug: max evals per query = {} (qid={})",
                                        static_cast<uint64_t>(maxv),
                                        static_cast<uint64_t>(maxi));
        }
        return false;
    }

    float* toiFloats = static_cast<float*>([sToi contents]);
    uint32_t* toiBits = static_cast<uint32_t*>([qToi contents]);
    for (uint32_t i = 0; i < count; ++i) {
        toiFloats[i] = float_from_bits(toiBits[i]);
    }
    if (queueDebug && qDbgCount && qDbgEvents) {
        uint32_t* pc = static_cast<uint32_t*>([qDbgCount contents]);
        uint32_t nlog = *pc;
        nlog = std::min(nlog, static_cast<uint32_t>(20));
        float* ev = static_cast<float*>([qDbgEvents contents]);
        for (uint32_t i = 0; i < nlog; ++i) {
            float* base = ev + i*12u;
            scalable_ccd::logger().warn("EE dbg accept: qid={} t=[{:.9f},{:.9f}] trueTol={:.3e} w=({:.3e},{:.3e},{:.3e}) tol=({:.3e},{:.3e},{:.3e}) mask={}",
                                        static_cast<int>(base[0]),
                                        static_cast<double>(base[1]),
                                        static_cast<double>(base[2]),
                                        static_cast<double>(base[3]),
                                        static_cast<double>(base[4]),
                                        static_cast<double>(base[5]),
                                        static_cast<double>(base[6]),
                                        static_cast<double>(base[8]),
                                        static_cast<double>(base[9]),
                                        static_cast<double>(base[10]),
                                        static_cast<int>(base[7]));
        }
    }

    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:reduceMinToiCps];
        [enc setBuffer:sToi offset:0 atIndex:0];
        [enc setBuffer:sNO offset:0 atIndex:1];
        [enc setBuffer:bMin offset:0 atIndex:2];
        NSUInteger tgW = preferredThreadgroupWidth(reduceMinToiCps, 64, count);
        MTLSize tg = MTLSizeMake(tgW,1,1);
        MTLSize grid = MTLSizeMake(count,1,1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    return true;
}


// Thin wrappers to align public API naming with implementation name.
bool MetalRuntime::runSweepAndTiniestQueue(
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
    bool twoLists)
{
    return sweepSTQSingleList(minX, maxX, minY, maxY, minZ, maxZ, v0, v1, v2,
                              capacity, outPairs, startJ, listTag, twoLists);
}

bool MetalRuntime::runSweepAndTiniestQueue(
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
    const STQConfig& cfg)
{
    return sweepSTQSingleList(minX, maxX, minY, maxY, minZ, maxZ, v0, v1, v2,
                              capacity, outPairs, startJ, listTag, twoLists, cfg);
}

bool MetalRuntime::sweepSTQSingleList(
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
    bool twoLists)
{
    if (!available()) return false;
    const size_t n = minX.size();
    scalable_ccd::logger().info("Metal STQ: enter n={} capacity={}", n, capacity);
    if (n == 0) {
        outPairs.clear();
        scalable_ccd::logger().info("Metal STQ: empty input, early-ok");
        return true;
    }
    if (maxX.size()!=n || minY.size()!=n || maxY.size()!=n || minZ.size()!=n || maxZ.size()!=n) return false;
    if (v0.size()!=n || v1.size()!=n || v2.size()!=n) return false;
    @autoreleasepool {
        NSError* err = nil;
        if (!impl->stqCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            impl->stqLib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kSTQKernelSrc]
                                                      options:opts
                                                        error:&err];
            if (!impl->stqLib) {
                scalable_ccd::logger().error("Metal STQ: newLibrary failed: {}", err ? err.localizedDescription.UTF8String : "unknown");
                return false;
            }
            id<MTLFunction> fn = [impl->stqLib newFunctionWithName:@"sweepSTQSingle"];
            if (!fn) {
                scalable_ccd::logger().error("Metal STQ: newFunction(sweepSTQSingle) failed");
                return false;
            }
            impl->stqCps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!impl->stqCps) {
                scalable_ccd::logger().error("Metal STQ: newComputePipelineState failed: {}", err ? err.localizedDescription.UTF8String : "unknown");
                return false;
            }
            scalable_ccd::logger().info("Metal STQ: pipeline ready");
        }

        id<MTLBuffer> bMinX = [impl->device newBufferWithBytes:minX.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxX = [impl->device newBufferWithBytes:maxX.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMinY = [impl->device newBufferWithBytes:minY.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxY = [impl->device newBufferWithBytes:maxY.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMinZ = [impl->device newBufferWithBytes:minZ.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxZ = [impl->device newBufferWithBytes:maxZ.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV0   = [impl->device newBufferWithBytes:v0.data()  length:sizeof(int32_t)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV1   = [impl->device newBufferWithBytes:v1.data()  length:sizeof(int32_t)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV2   = [impl->device newBufferWithBytes:v2.data()  length:sizeof(int32_t)*n options:MTLResourceStorageModeShared];
        uint32_t n32 = static_cast<uint32_t>(n);
        id<MTLBuffer> bN    = [impl->device newBufferWithBytes:&n32 length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bPairs= [impl->device newBufferWithLength:sizeof(int32_t)*2*capacity options:MTLResourceStorageModeShared];
        uint32_t zero = 0, cap=capacity;
        id<MTLBuffer> bCount= [impl->device newBufferWithBytes:&zero length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bCap  = [impl->device newBufferWithBytes:&cap length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        // epsScale（相对主轴容差）支持通过环境变量覆盖，默认 0（CUDA 同构）
        float epsScale = 0.0f;
        if (const char* env = std::getenv("SCALABLE_CCD_STQ_EPS")) {
            try {
                epsScale = std::stof(env);
            } catch (...) {
                epsScale = 0.0f;
            }
        }
        id<MTLBuffer> bEps  = [impl->device newBufferWithBytes:&epsScale length:sizeof(float) options:MTLResourceStorageModeShared];
        // yzEpsScale（YZ 相对容差），默认 0，与最终路径对齐；可用 SCALABLE_CCD_STQ_YZ_EPS 单独配置
        float yzEpsScale = 0.0f;
        if (const char* env2 = std::getenv("SCALABLE_CCD_STQ_YZ_EPS")) {
            try {
                yzEpsScale = std::stof(env2);
            } catch (...) {
                yzEpsScale = 0.0f;
            }
        }
        id<MTLBuffer> bYZEps  = [impl->device newBufferWithBytes:&yzEpsScale length:sizeof(float) options:MTLResourceStorageModeShared];

        // startJ buffer（可选）
        std::vector<uint32_t> tmpStartJ;
        id<MTLBuffer> bStartJ = nil;
        bool disableStartJ = false;
        if (const char* envSJ = std::getenv("SCALABLE_CCD_STQ_DISABLE_STARTJ")) {
            // any non-empty value means disable
            disableStartJ = (envSJ[0] != '\0');
        }
        if (disableStartJ || !(startJ && startJ->size() == n)) {
            tmpStartJ.resize(n);
            for (uint32_t i = 0; i < n32; ++i) tmpStartJ[i] = i + 1;
            bStartJ = [impl->device newBufferWithBytes:tmpStartJ.data() length:sizeof(uint32_t)*n options:MTLResourceStorageModeShared];
        } else {
            bStartJ = [impl->device newBufferWithBytes:startJ->data() length:sizeof(uint32_t)*n options:MTLResourceStorageModeShared];
        }

        // 列表标签（可选，双列表时用于内核快速过滤同源对）
        std::vector<uint8_t> tmpTags;
        id<MTLBuffer> bTags = nil;
        if (listTag && listTag->size() == n) {
            bTags = [impl->device newBufferWithBytes:listTag->data() length:sizeof(uint8_t)*n options:MTLResourceStorageModeShared];
        } else {
            tmpTags.assign(n, 0);
            bTags = [impl->device newBufferWithBytes:tmpTags.data() length:sizeof(uint8_t)*n options:MTLResourceStorageModeShared];
        }
        uint32_t twoListsFlag = twoLists ? 1u : 0u;
        id<MTLBuffer> bTwoLists = [impl->device newBufferWithBytes:&twoListsFlag length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl->stqCps];
        [enc setBuffer:bMinX offset:0 atIndex:0];
        [enc setBuffer:bMaxX offset:0 atIndex:1];
        [enc setBuffer:bMinY offset:0 atIndex:2];
        [enc setBuffer:bMaxY offset:0 atIndex:3];
        [enc setBuffer:bMinZ offset:0 atIndex:4];
        [enc setBuffer:bMaxZ offset:0 atIndex:5];
        [enc setBuffer:bV0   offset:0 atIndex:6];
        [enc setBuffer:bV1   offset:0 atIndex:7];
        [enc setBuffer:bV2   offset:0 atIndex:8];
        [enc setBuffer:bN    offset:0 atIndex:9];
        [enc setBuffer:bPairs offset:0 atIndex:10];
        [enc setBuffer:bCount offset:0 atIndex:11];
        [enc setBuffer:bCap   offset:0 atIndex:12];
        [enc setBuffer:bEps    offset:0 atIndex:13];
        [enc setBuffer:bStartJ offset:0 atIndex:14];
        [enc setBuffer:bTags   offset:0 atIndex:15];
        [enc setBuffer:bTwoLists offset:0 atIndex:16];
        [enc setBuffer:bYZEps  offset:0 atIndex:17];
        // maxSkipSteps（推进阶段最多尝试跳过多少次，默认 0）
        uint32_t maxSkip = 0;
        if (const char* env3 = std::getenv("SCALABLE_CCD_STQ_MAX_SKIP_STEPS")) {
            try {
                int v = std::stoi(env3);
                if (v > 0) maxSkip = static_cast<uint32_t>(v);
            } catch (...) { }
        }
        id<MTLBuffer> bMaxSkip = [impl->device newBufferWithBytes:&maxSkip length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        [enc setBuffer:bMaxSkip offset:0 atIndex:18];
        MTLSize grid = MTLSizeMake(n, 1, 1);
        // 线程组大小（默认 32，可通过 SCALABLE_CCD_STQ_TG 配置）
        NSUInteger simdPreferred = 32;
        if (const char* envTG = std::getenv("SCALABLE_CCD_STQ_TG")) {
            try {
                int v = std::stoi(envTG);
                if (v > 0) simdPreferred = static_cast<NSUInteger>(v);
            } catch (...) {}
        }
        NSUInteger maxTG = impl->stqCps.maxTotalThreadsPerThreadgroup;
        if (maxTG == 0) maxTG = simdPreferred;
        NSUInteger tgW = simdPreferred;
        if (tgW > maxTG) tgW = maxTG;
        if (tgW > n) tgW = static_cast<NSUInteger>(n);
        if (tgW == 0) tgW = 1;
        MTLSize tg = MTLSizeMake(tgW, 1, 1);
        scalable_ccd::logger().info("Metal STQ: dispatch grid={} tg={} epsScale={}", static_cast<uint64_t>(n), static_cast<uint64_t>(tg.width), static_cast<double>(epsScale));
        // 传递线程组大小给内核用于协作写回步长
        uint32_t tgWidthU32 = static_cast<uint32_t>(tg.width);
        id<MTLBuffer> bTGSize = [impl->device newBufferWithBytes:&tgWidthU32 length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        [enc setBuffer:bTGSize offset:0 atIndex:19];
        // 近边界严格化的 ulp 阈值（固定为 2 ULP，移除环境变量依赖）
        uint32_t yzTieUlps = 2;
        id<MTLBuffer> bTie = [impl->device newBufferWithBytes:&yzTieUlps length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        [enc setBuffer:bTie offset:0 atIndex:20];
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        uint32_t count = *static_cast<uint32_t*>([bCount contents]);
        if (count > capacity) count = capacity;
        scalable_ccd::logger().info("Metal STQ: gpu_count={}", count);
        outPairs.resize(count);
        struct Pair2i { int32_t x; int32_t y; };
        Pair2i* ptr = static_cast<Pair2i*>([bPairs contents]);
        for (uint32_t k = 0; k < count; ++k) {
            outPairs[k] = { ptr[k].x, ptr[k].y };
        }
        scalable_ccd::logger().info("Metal STQ: return pairs={}", outPairs.size());
        return true;
    }
}

bool MetalRuntime::getVFCCDData(
    const std::vector<float>& V0,
    const std::vector<float>& V1,
    uint32_t nV,
    const std::vector<int32_t>& facesFlat,
    const std::vector<std::pair<int,int>>& overlaps,
    float ms,
    std::vector<CCDDataMetalVF>& out)
{
    if (!available()) return false;
    @autoreleasepool {
        NSError* err = nil;
        if (!impl->npPackVfCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> lib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kNPPackVFKernelSrc]
                                                            options:opts
                                                              error:&err];
            if (!lib) return false;
            id<MTLFunction> fn = [lib newFunctionWithName:@"npPackVF"];
            if (!fn) return false;
            impl->npPackVfCps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!impl->npPackVfCps) return false;
        }
        const size_t nOver = overlaps.size();
        struct Pair2i { int32_t x; int32_t y; };
        std::vector<Pair2i> ovs(nOver);
        for (size_t i=0;i<nOver;++i){ ovs[i].x = overlaps[i].first; ovs[i].y = overlaps[i].second; }
        id<MTLBuffer> bV0 = [impl->device newBufferWithBytes:V0.data() length:sizeof(float)*V0.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV1 = [impl->device newBufferWithBytes:V1.data() length:sizeof(float)*V1.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bNV = [impl->device newBufferWithBytes:&nV length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bF  = [impl->device newBufferWithBytes:facesFlat.data() length:sizeof(int32_t)*facesFlat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bO  = [impl->device newBufferWithBytes:ovs.data() length:sizeof(Pair2i)*ovs.size() options:MTLResourceStorageModeShared];
        uint32_t nOverU32 = static_cast<uint32_t>(nOver);
        id<MTLBuffer> bNO = [impl->device newBufferWithBytes:&nOverU32 length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMs = [impl->device newBufferWithBytes:&ms length:sizeof(float) options:MTLResourceStorageModeShared];
        // out buffer matching CCDDataMetalVF
        id<MTLBuffer> bOut = [impl->device newBufferWithLength:sizeof(CCDDataMetalVF)*nOver options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl->npPackVfCps];
        [enc setBuffer:bV0 offset:0 atIndex:0];
        [enc setBuffer:bV1 offset:0 atIndex:1];
        [enc setBuffer:bNV offset:0 atIndex:2];
        [enc setBuffer:bF  offset:0 atIndex:3];
        [enc setBuffer:bO  offset:0 atIndex:4];
        [enc setBuffer:bNO offset:0 atIndex:5];
        [enc setBuffer:bMs offset:0 atIndex:6];
        [enc setBuffer:bOut offset:0 atIndex:7];
        MTLSize grid = MTLSizeMake(nOver,1,1);
        NSUInteger w = impl->npPackVfCps.maxTotalThreadsPerThreadgroup;
        if (w == 0) w = 64;
        MTLSize tg = MTLSizeMake(std::min<NSUInteger>(w, nOver), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        // read back
        out.resize(nOver);
        memcpy(out.data(), [bOut contents], sizeof(CCDDataMetalVF)*nOver);
        scalable_ccd::logger().info("Metal getVFCCDData: copied {} records", static_cast<uint64_t>(nOver));
        return true;
    }
}

bool MetalRuntime::getEECCDData(
    const std::vector<float>& V0,
    const std::vector<float>& V1,
    uint32_t nV,
    const std::vector<int32_t>& edgesFlat,
    const std::vector<std::pair<int,int>>& overlaps,
    float ms,
    std::vector<CCDDataMetalEE>& out)
{
    if (!available()) return false;
    @autoreleasepool {
        NSError* err = nil;
        if (!impl->npPackEeCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> lib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kNPPackEEKernelSrc]
                                                            options:opts
                                                              error:&err];
            if (!lib) return false;
            id<MTLFunction> fn = [lib newFunctionWithName:@"npPackEE"];
            if (!fn) return false;
            impl->npPackEeCps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!impl->npPackEeCps) return false;
        }
        const size_t nOver = overlaps.size();
        struct Pair2i { int32_t x; int32_t y; };
        std::vector<Pair2i> ovs(nOver);
        for (size_t i=0;i<nOver;++i){ ovs[i].x = overlaps[i].first; ovs[i].y = overlaps[i].second; }
        id<MTLBuffer> bV0 = [impl->device newBufferWithBytes:V0.data() length:sizeof(float)*V0.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV1 = [impl->device newBufferWithBytes:V1.data() length:sizeof(float)*V1.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bNV = [impl->device newBufferWithBytes:&nV length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bE  = [impl->device newBufferWithBytes:edgesFlat.data() length:sizeof(int32_t)*edgesFlat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bO  = [impl->device newBufferWithBytes:ovs.data() length:sizeof(Pair2i)*ovs.size() options:MTLResourceStorageModeShared];
        uint32_t nOverU32 = static_cast<uint32_t>(nOver);
        id<MTLBuffer> bNO = [impl->device newBufferWithBytes:&nOverU32 length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMs = [impl->device newBufferWithBytes:&ms length:sizeof(float) options:MTLResourceStorageModeShared];
        // out buffer matching CCDDataMetalEE
        id<MTLBuffer> bOut = [impl->device newBufferWithLength:sizeof(CCDDataMetalEE)*nOver options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl->npPackEeCps];
        [enc setBuffer:bV0 offset:0 atIndex:0];
        [enc setBuffer:bV1 offset:0 atIndex:1];
        [enc setBuffer:bNV offset:0 atIndex:2];
        [enc setBuffer:bE  offset:0 atIndex:3];
        [enc setBuffer:bO  offset:0 atIndex:4];
        [enc setBuffer:bNO offset:0 atIndex:5];
        [enc setBuffer:bMs offset:0 atIndex:6];
        [enc setBuffer:bOut offset:0 atIndex:7];
        MTLSize grid = MTLSizeMake(nOver,1,1);
        NSUInteger w = impl->npPackEeCps.maxTotalThreadsPerThreadgroup;
        if (w == 0) w = 64;
        MTLSize tg = MTLSizeMake(std::min<NSUInteger>(w, nOver), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        // read back
        out.resize(nOver);
        memcpy(out.data(), [bOut contents], sizeof(CCDDataMetalEE)*nOver);
        scalable_ccd::logger().info("Metal getEECCDData: copied {} records", static_cast<uint64_t>(nOver));
        return true;
    }
}
bool MetalRuntime::runVFPlaceholder(
    const std::vector<float>& vertices_t0_flat,
    const std::vector<float>& vertices_t1_flat,
    uint32_t num_vertices,
    const std::vector<int32_t>& faces_flat,
    const std::vector<std::pair<int,int>>& overlaps,
    float minimum_separation,
    float tolerance,
    int max_iterations,
    bool allow_zero_toi,
    uint32_t& out_processed)
{
    if (!available()) return false;
    const auto& V0 = vertices_t0_flat;
    const auto& V1 = vertices_t1_flat;
    const uint32_t nV = num_vertices;
    const auto& facesFlat = faces_flat;
    const float ms = minimum_separation;
    const float tol = tolerance;
    const int gpuMaxIter = env_int_override("SCALABLE_CCD_GPU_MAX_ITER", max_iterations, 1);
    const int maxIter = std::max(1, std::min(max_iterations, gpuMaxIter));
    const bool allowZeroToi = allow_zero_toi;
    @autoreleasepool {
        NSError* err = nil;
        if (!impl->ccdRunVfPlaceholderCps) {
            if (!impl->npLib) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl->npLib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kCCDRunVFPlaceholderKernelSrc]
                                                         options:opts
                                                           error:&err];
                if (!impl->npLib) return false;
            }
            id<MTLFunction> fn = [impl->npLib newFunctionWithName:@"ccdRunVFPlaceholder"];
            if (!fn) return false;
            impl->ccdRunVfPlaceholderCps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!impl->ccdRunVfPlaceholderCps) return false;
        }
        const size_t nOver = overlaps.size();
        struct Pair2i { int32_t x; int32_t y; };
        std::vector<Pair2i> ovs(nOver);
        for (size_t i=0;i<nOver;++i){ ovs[i].x = overlaps[i].first; ovs[i].y = overlaps[i].second; }
        id<MTLBuffer> bV0 = [impl->device newBufferWithBytes:V0.data() length:sizeof(float)*V0.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV1 = [impl->device newBufferWithBytes:V1.data() length:sizeof(float)*V1.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bNV = [impl->device newBufferWithBytes:&nV length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bF  = [impl->device newBufferWithBytes:facesFlat.data() length:sizeof(int32_t)*facesFlat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bO  = [impl->device newBufferWithBytes:ovs.data() length:sizeof(Pair2i)*ovs.size() options:MTLResourceStorageModeShared];
        uint32_t nOverU32 = static_cast<uint32_t>(nOver);
        id<MTLBuffer> bNO = [impl->device newBufferWithBytes:&nOverU32 length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMs = [impl->device newBufferWithBytes:&ms length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bTol= [impl->device newBufferWithBytes:&tol length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bIt = [impl->device newBufferWithBytes:&maxIter length:sizeof(int) options:MTLResourceStorageModeShared];
        uint32_t allow = allowZeroToi ? 1u : 0u;
        id<MTLBuffer> bAz = [impl->device newBufferWithBytes:&allow length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bToi= [impl->device newBufferWithLength:sizeof(float)*nOver options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl->ccdRunVfPlaceholderCps];
        [enc setBuffer:bV0 offset:0 atIndex:0];
        [enc setBuffer:bV1 offset:0 atIndex:1];
        [enc setBuffer:bNV offset:0 atIndex:2];
        [enc setBuffer:bF  offset:0 atIndex:3];
        [enc setBuffer:bO  offset:0 atIndex:4];
        [enc setBuffer:bNO offset:0 atIndex:5];
        [enc setBuffer:bMs offset:0 atIndex:6];
        [enc setBuffer:bTol offset:0 atIndex:7];
        [enc setBuffer:bIt  offset:0 atIndex:8];
        [enc setBuffer:bAz  offset:0 atIndex:9];
        [enc setBuffer:bToi offset:0 atIndex:10];
        MTLSize grid = MTLSizeMake(nOver,1,1);
        NSUInteger maxW = impl->ccdRunVfPlaceholderCps.maxTotalThreadsPerThreadgroup;
        if (maxW == 0) maxW = 64;
        NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, maxW) : maxW;
        if (tgW > nOver) tgW = static_cast<NSUInteger>(nOver);
        if (tgW == 0) tgW = 1;
        MTLSize tg = MTLSizeMake(tgW, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (impl->enableTiming) {
            CFTimeInterval t0 = cb.GPUStartTime;
            CFTimeInterval t1 = cb.GPUEndTime;
            if (t1 > t0 && t0 > 0) {
                double msElapsed = (t1 - t0) * 1000.0;
                scalable_ccd::logger().info("Metal CCD VF placeholder: gpu_time_ms={} count={} tg={}",
                                             msElapsed, static_cast<uint64_t>(nOver), static_cast<uint64_t>(tg.width));
                append_timing_csv_if_needed(impl->timingCsvPath, "ccd_vf_placeholder",
                                            static_cast<uint64_t>(nOver),
                                            static_cast<uint32_t>(tg.width),
                                            msElapsed);
            }
        }
        out_processed = static_cast<uint32_t>(nOver);
        scalable_ccd::logger().info("Metal CCD VF placeholder: processed {}", static_cast<uint64_t>(nOver));
        return true;
    }
}
bool MetalRuntime::sweepSTQSingleList(
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
    const STQConfig& cfg)
{
    if (!available()) return false;
    const size_t n = minX.size();
    scalable_ccd::logger().info("Metal STQ(cfg): enter n={} capacity={}", n, capacity);
    if (n == 0) { outPairs.clear(); return true; }
    if (maxX.size()!=n || minY.size()!=n || maxY.size()!=n || minZ.size()!=n || maxZ.size()!=n) return false;
    if (v0.size()!=n || v1.size()!=n || v2.size()!=n) return false;
    @autoreleasepool {
        NSError* err = nil;
        if (!impl->stqCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            impl->stqLib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kSTQKernelSrc]
                                                      options:opts
                                                        error:&err];
            if (!impl->stqLib) return false;
            id<MTLFunction> fn = [impl->stqLib newFunctionWithName:@"sweepSTQSingle"];
            if (!fn) return false;
            impl->stqCps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!impl->stqCps) return false;
        }
        id<MTLBuffer> bMinX = [impl->device newBufferWithBytes:minX.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxX = [impl->device newBufferWithBytes:maxX.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMinY = [impl->device newBufferWithBytes:minY.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxY = [impl->device newBufferWithBytes:maxY.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMinZ = [impl->device newBufferWithBytes:minZ.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxZ = [impl->device newBufferWithBytes:maxZ.data() length:sizeof(float)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV0   = [impl->device newBufferWithBytes:v0.data()  length:sizeof(int32_t)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV1   = [impl->device newBufferWithBytes:v1.data()  length:sizeof(int32_t)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV2   = [impl->device newBufferWithBytes:v2.data()  length:sizeof(int32_t)*n options:MTLResourceStorageModeShared];
        uint32_t n32 = static_cast<uint32_t>(n);
        id<MTLBuffer> bN    = [impl->device newBufferWithBytes:&n32 length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bPairs= [impl->device newBufferWithLength:sizeof(int32_t)*2*capacity options:MTLResourceStorageModeShared];
        uint32_t zero = 0, cap=capacity;
        id<MTLBuffer> bCount= [impl->device newBufferWithBytes:&zero length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bCap  = [impl->device newBufferWithBytes:&cap length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        // 默认值按环境变量读取，再被配置覆盖
        float epsScale = 0.0f;
        if (const char* env = std::getenv("SCALABLE_CCD_STQ_EPS")) {
            try { epsScale = std::stof(env); } catch (...) { epsScale = 0.0f; }
        }
        epsScale = cfg.epsScale;
        id<MTLBuffer> bEps  = [impl->device newBufferWithBytes:&epsScale length:sizeof(float) options:MTLResourceStorageModeShared];
        float yzEpsScale = 0.0f;
        if (const char* env2 = std::getenv("SCALABLE_CCD_STQ_YZ_EPS")) {
            try { yzEpsScale = std::stof(env2); } catch (...) { yzEpsScale = 0.0f; }
        }
        yzEpsScale = cfg.yzEpsScale;
        id<MTLBuffer> bYZEps  = [impl->device newBufferWithBytes:&yzEpsScale length:sizeof(float) options:MTLResourceStorageModeShared];
        // startJ
        std::vector<uint32_t> tmpStartJ;
        id<MTLBuffer> bStartJ = nil;
        bool disableStartJ = cfg.disableStartJ;
        if (disableStartJ || !(startJ && startJ->size() == n)) {
            tmpStartJ.resize(n);
            for (uint32_t i = 0; i < n32; ++i) tmpStartJ[i] = i + 1;
            bStartJ = [impl->device newBufferWithBytes:tmpStartJ.data() length:sizeof(uint32_t)*n options:MTLResourceStorageModeShared];
        } else {
            bStartJ = [impl->device newBufferWithBytes:startJ->data() length:sizeof(uint32_t)*n options:MTLResourceStorageModeShared];
        }
        // list tags
        std::vector<uint8_t> tmpTags;
        id<MTLBuffer> bTags = nil;
        if (listTag && listTag->size() == n) {
            bTags = [impl->device newBufferWithBytes:listTag->data() length:sizeof(uint8_t)*n options:MTLResourceStorageModeShared];
        } else {
            tmpTags.assign(n, 0);
            bTags = [impl->device newBufferWithBytes:tmpTags.data() length:sizeof(uint8_t)*n options:MTLResourceStorageModeShared];
        }
        uint32_t twoListsFlag = twoLists ? 1u : 0u;
        id<MTLBuffer> bTwoLists = [impl->device newBufferWithBytes:&twoListsFlag length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl->stqCps];
        [enc setBuffer:bMinX offset:0 atIndex:0];
        [enc setBuffer:bMaxX offset:0 atIndex:1];
        [enc setBuffer:bMinY offset:0 atIndex:2];
        [enc setBuffer:bMaxY offset:0 atIndex:3];
        [enc setBuffer:bMinZ offset:0 atIndex:4];
        [enc setBuffer:bMaxZ offset:0 atIndex:5];
        [enc setBuffer:bV0   offset:0 atIndex:6];
        [enc setBuffer:bV1   offset:0 atIndex:7];
        [enc setBuffer:bV2   offset:0 atIndex:8];
        [enc setBuffer:bN    offset:0 atIndex:9];
        [enc setBuffer:bPairs offset:0 atIndex:10];
        [enc setBuffer:bCount offset:0 atIndex:11];
        [enc setBuffer:bCap   offset:0 atIndex:12];
        [enc setBuffer:bEps    offset:0 atIndex:13];
        [enc setBuffer:bStartJ offset:0 atIndex:14];
        [enc setBuffer:bTags   offset:0 atIndex:15];
        [enc setBuffer:bTwoLists offset:0 atIndex:16];
        [enc setBuffer:bYZEps  offset:0 atIndex:17];
        // maxSkip
        uint32_t maxSkip = cfg.maxSkipSteps;
        id<MTLBuffer> bMaxSkip = [impl->device newBufferWithBytes:&maxSkip length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        [enc setBuffer:bMaxSkip offset:0 atIndex:18];
        MTLSize grid = MTLSizeMake(n, 1, 1);
        // tg size
        NSUInteger simdPreferred = static_cast<NSUInteger>(cfg.threadgroupWidth > 0 ? cfg.threadgroupWidth : 32u);
        NSUInteger maxTG = impl->stqCps.maxTotalThreadsPerThreadgroup;
        if (maxTG == 0) maxTG = simdPreferred;
        NSUInteger tgW = simdPreferred;
        if (tgW > maxTG) tgW = maxTG;
        if (tgW > n) tgW = static_cast<NSUInteger>(n);
        if (tgW == 0) tgW = 1;
        MTLSize tg = MTLSizeMake(tgW, 1, 1);
        uint32_t tgWidthU32 = static_cast<uint32_t>(tg.width);
        id<MTLBuffer> bTGSize = [impl->device newBufferWithBytes:&tgWidthU32 length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        [enc setBuffer:bTGSize offset:0 atIndex:19];
        // tie ULPs
        uint32_t yzTieUlps = cfg.yzTieUlps > 0 ? cfg.yzTieUlps : 2u;
        id<MTLBuffer> bTie = [impl->device newBufferWithBytes:&yzTieUlps length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        [enc setBuffer:bTie offset:0 atIndex:20];
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        uint32_t count = *static_cast<uint32_t*>([bCount contents]);
        if (count > capacity) count = capacity;
        outPairs.resize(count);
        struct Pair2i { int32_t x; int32_t y; };
        Pair2i* ptr = static_cast<Pair2i*>([bPairs contents]);
        for (uint32_t k = 0; k < count; ++k) {
            outPairs[k] = { ptr[k].x, ptr[k].y };
        }
        return true;
    }
}
MetalRuntime::MetalRuntime() : impl(new Impl())
{
    @autoreleasepool {
        impl->device = MTLCreateSystemDefaultDevice();
        if (!impl->device) {
            impl->ok = false;
            return;
        }
        impl->queue = [impl->device newCommandQueue];
        impl->ok = (impl->queue != nil);
        // Read env for tuning
        // Default TG to 128 unless overridden
        impl->tgOverride = 128;
        if (const char* tg = std::getenv("SCCD_METAL_TG")) {
            long v = 0;
            try { v = std::stol(std::string(tg)); } catch (...) { v = 0; }
            if (v > 0) impl->tgOverride = static_cast<NSUInteger>(v);
        }
        if (const char* tim = std::getenv("SCCD_METAL_TIME")) {
            // any non-empty enables timing
            impl->enableTiming = (tim[0] != '\0');
        }
        if (const char* csv = std::getenv("SCCD_METAL_TIMING_CSV")) {
            impl->timingCsvPath = std::string(csv);
        }
    }
}

MetalRuntime::~MetalRuntime() = default;

bool MetalRuntime::available() const
{
    return impl && impl->ok;
}

bool MetalRuntime::warmup()
{
    if (!available()) return false;
    @autoreleasepool {
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kNoopKernelSrc]
                                                        options:opts
                                                          error:&err];
        if (!lib) {
            return false;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"noop"];
        if (!fn) {
            return false;
        }
        id<MTLComputePipelineState> cps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
        if (!cps) {
            return false;
        }
        int value = 0;
        id<MTLBuffer> out = [impl->device newBufferWithBytes:&value length:sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:cps];
        [enc setBuffer:out offset:0 atIndex:0];
        MTLSize grid = MTLSizeMake(1, 1, 1);
        NSUInteger w = cps.maxTotalThreadsPerThreadgroup;
        MTLSize tg = MTLSizeMake(w > 1 ? 1 : 1, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        int* ptr = static_cast<int*>(out.contents);
        return ptr && (*ptr == 42);
    }
}

bool MetalRuntime::filterYZ(
    const std::vector<float>& minY,
    const std::vector<float>& maxY,
    const std::vector<float>& minZ,
    const std::vector<float>& maxZ,
    const std::vector<int32_t>& v0,
    const std::vector<int32_t>& v1,
    const std::vector<int32_t>& v2,
    const std::vector<std::pair<int, int>>& pairs,
    std::vector<uint8_t>& outMask)
{
    if (!available()) return false;
    if (minY.size() != maxY.size() || minZ.size() != maxZ.size()) return false;
    if (v0.size() != minY.size() || v1.size() != minY.size() || v2.size() != minY.size()) return false;
    @autoreleasepool {
        NSError* err = nil;
        if (!impl->yzCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            impl->yzLib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kYZFilterKernelSrc]
                                                     options:opts
                                                       error:&err];
            if (!impl->yzLib) {
                return false;
            }
            id<MTLFunction> fn = [impl->yzLib newFunctionWithName:@"yzFilter"];
            if (!fn) {
                return false;
            }
            impl->yzCps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!impl->yzCps) {
                return false;
            }
        }

        const size_t nBoxes = minY.size();
        const size_t nPairs = pairs.size();
        outMask.assign(nPairs, 0);
        if (nPairs == 0) return true;

        id<MTLBuffer> bMinY = [impl->device newBufferWithBytes:minY.data() length:sizeof(float)*nBoxes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxY = [impl->device newBufferWithBytes:maxY.data() length:sizeof(float)*nBoxes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMinZ = [impl->device newBufferWithBytes:minZ.data() length:sizeof(float)*nBoxes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxZ = [impl->device newBufferWithBytes:maxZ.data() length:sizeof(float)*nBoxes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV0   = [impl->device newBufferWithBytes:v0.data()  length:sizeof(int32_t)*nBoxes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV1   = [impl->device newBufferWithBytes:v1.data()  length:sizeof(int32_t)*nBoxes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV2   = [impl->device newBufferWithBytes:v2.data()  length:sizeof(int32_t)*nBoxes options:MTLResourceStorageModeShared];

        struct Pair2i { int32_t x; int32_t y; };
        std::vector<Pair2i> pairBuf(nPairs);
        for (size_t i = 0; i < nPairs; ++i) {
            pairBuf[i] = Pair2i{ static_cast<int32_t>(pairs[i].first), static_cast<int32_t>(pairs[i].second) };
        }
        id<MTLBuffer> bPairs = [impl->device newBufferWithBytes:pairBuf.data() length:sizeof(Pair2i)*nPairs options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMask  = [impl->device newBufferWithLength:sizeof(uint8_t)*nPairs options:MTLResourceStorageModeShared];
        // 固定容差：绝对 eps 与相对 eps 均为 0（后续由 narrow phase 收敛）
        float absEps = 0.0f;
        float relEps = 0.0f;
        id<MTLBuffer> bAbs  = [impl->device newBufferWithBytes:&absEps length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bRel  = [impl->device newBufferWithBytes:&relEps length:sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl->yzCps];
        [enc setBuffer:bMinY offset:0 atIndex:0];
        [enc setBuffer:bMaxY offset:0 atIndex:1];
        [enc setBuffer:bMinZ offset:0 atIndex:2];
        [enc setBuffer:bMaxZ offset:0 atIndex:3];
        [enc setBuffer:bV0   offset:0 atIndex:4];
        [enc setBuffer:bV1   offset:0 atIndex:5];
        [enc setBuffer:bV2   offset:0 atIndex:6];
        [enc setBuffer:bPairs offset:0 atIndex:7];
        [enc setBuffer:bMask  offset:0 atIndex:8];
        [enc setBuffer:bAbs   offset:0 atIndex:9];
        [enc setBuffer:bRel   offset:0 atIndex:10];
        MTLSize grid = MTLSizeMake(nPairs, 1, 1);
        NSUInteger w = impl->yzCps.maxTotalThreadsPerThreadgroup;
        if (w == 0) w = 64;
        MTLSize tg = MTLSizeMake(std::min<NSUInteger>(w, nPairs), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        memcpy(outMask.data(), [bMask contents], sizeof(uint8_t)*nPairs);
        return true;
    }
}

bool MetalRuntime::runVFRootSkeleton(
    const std::vector<float>& vertices_t0_flat,
    const std::vector<float>& vertices_t1_flat,
    uint32_t num_vertices,
    const std::vector<int32_t>& faces_flat,
    const std::vector<std::pair<int,int>>& overlaps,
    float minimum_separation,
    float tolerance,
    int max_iterations,
    bool allow_zero_toi,
    std::vector<float>& out_toi)
{
    if (!available()) return false;
    const float tol = tolerance;
    const int gpuMaxIter = env_int_override("SCALABLE_CCD_GPU_MAX_ITER", max_iterations, 1);
    const int maxIter = std::max(1, std::min(max_iterations, gpuMaxIter));
    const bool allowZeroToi = allow_zero_toi;
    auto& outToi = out_toi;
    @autoreleasepool {
        NSError* err = nil;
        // Force recompile to avoid stale pipeline after kernel signature changes
        impl->vfRootSkeletonCps = nil;
        // Ensure pack pipeline
        if (!impl->npPackVfCps) {
            if (!impl->npLib) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl->npLib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kNPPackVFKernelSrc]
                                                         options:opts
                                                           error:&err];
                if (!impl->npLib) {
                    scalable_ccd::logger().error("Metal runVFRootSkeleton: build npLib failed: {}",
                                                 err ? err.localizedDescription.UTF8String : "unknown error");
                    return false;
                }
            }
            id<MTLFunction> pfn = [impl->npLib newFunctionWithName:@"npPackVF"];
            if (!pfn) {
                scalable_ccd::logger().error("Metal runVFRootSkeleton: newFunction npPackVF failed");
                return false;
            }
            impl->npPackVfCps = [impl->device newComputePipelineStateWithFunction:pfn error:&err];
            if (!impl->npPackVfCps) {
                scalable_ccd::logger().error("Metal runVFRootSkeleton: newComputePipelineState(npPackVF) failed: {}",
                                             err ? err.localizedDescription.UTF8String : "unknown error");
                return false;
            }
        }
        // Ensure root skeleton pipeline
        if (!impl->vfRootSkeletonCps) {
            id<MTLLibrary> lib = impl->npLib;
            if (!lib) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                lib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kVFRootSkeletonKernelSrc]
                                                 options:opts
                                                   error:&err];
                if (!lib) {
                    scalable_ccd::logger().error("Metal runVFRootSkeleton: build vfRoot lib failed: {}",
                                                 err ? err.localizedDescription.UTF8String : "unknown error");
                    return false;
                }
            } else {
                // If npLib exists, extend with extra function by creating a new library
                MTLCompileOptions* opts = [MTLCompileOptions new];
                lib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kVFRootSkeletonKernelSrc]
                                                 options:opts
                                                   error:&err];
                if (!lib) {
                    scalable_ccd::logger().error("Metal runVFRootSkeleton: extend vfRoot lib failed: {}",
                                                 err ? err.localizedDescription.UTF8String : "unknown error");
                    return false;
                }
            }
            id<MTLFunction> rfn = [lib newFunctionWithName:@"vfRootSkeleton"];
            if (!rfn) {
                scalable_ccd::logger().error("Metal runVFRootSkeleton: newFunction vfRootSkeleton failed");
                return false;
            }
            impl->vfRootSkeletonCps = [impl->device newComputePipelineStateWithFunction:rfn error:&err];
            if (!impl->vfRootSkeletonCps) {
                scalable_ccd::logger().error("Metal runVFRootSkeleton: newComputePipelineState(vfRootSkeleton) failed: {}",
                                             err ? err.localizedDescription.UTF8String : "unknown error");
                return false;
            }
        }
        const size_t nOver = overlaps.size();
        struct Pair2i { int32_t x; int32_t y; };
        std::vector<Pair2i> ovs(nOver);
        for (size_t i=0;i<nOver;++i){ ovs[i].x = overlaps[i].first; ovs[i].y = overlaps[i].second; }
        id<MTLBuffer> bV0 = [impl->device newBufferWithBytes:vertices_t0_flat.data() length:sizeof(float)*vertices_t0_flat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV1 = [impl->device newBufferWithBytes:vertices_t1_flat.data() length:sizeof(float)*vertices_t1_flat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bNV = [impl->device newBufferWithBytes:&num_vertices length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bF  = [impl->device newBufferWithBytes:faces_flat.data() length:sizeof(int32_t)*faces_flat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMs = [impl->device newBufferWithBytes:&minimum_separation length:sizeof(float) options:MTLResourceStorageModeShared];
        // Private copies for heavy readers (once per call)
        id<MTLBuffer> pV0 = [impl->device newBufferWithLength:sizeof(float)*vertices_t0_flat.size() options:MTLResourceStorageModePrivate];
        id<MTLBuffer> pV1 = [impl->device newBufferWithLength:sizeof(float)*vertices_t1_flat.size() options:MTLResourceStorageModePrivate];
        id<MTLBuffer> pF  = [impl->device newBufferWithLength:sizeof(int32_t)*faces_flat.size() options:MTLResourceStorageModePrivate];
        {
            id<MTLCommandBuffer> cbu = [impl->queue commandBuffer];
            id<MTLBlitCommandEncoder> bl = [cbu blitCommandEncoder];
            [bl copyFromBuffer:bV0 sourceOffset:0 toBuffer:pV0 destinationOffset:0 size:sizeof(float)*vertices_t0_flat.size()];
            [bl copyFromBuffer:bV1 sourceOffset:0 toBuffer:pV1 destinationOffset:0 size:sizeof(float)*vertices_t1_flat.size()];
            [bl copyFromBuffer:bF  sourceOffset:0 toBuffer:pF  destinationOffset:0 size:sizeof(int32_t)*faces_flat.size()];
            [bl endEncoding];
            [cbu commit];
            [cbu waitUntilCompleted];
        }
        // Env overrides
        size_t chunkSize = 100000;
        if (const char* envChunk = std::getenv("SCALABLE_CCD_NP_CHUNK")) {
            try { chunkSize = std::max<size_t>(1, static_cast<size_t>(std::stoll(envChunk))); } catch (...) {}
        }
        int refineSteps = 16;
        if (const char* envRef = std::getenv("SCALABLE_CCD_REFINE_STEPS")) {
            try { refineSteps = std::max(1, std::stoi(envRef)); } catch (...) {}
        }
        // Outputs
        outToi.resize(nOver);
        uint64_t totalIter = 0, totalHit = 0;
        // Early-stop config
        double stopBelow = -1.0;
        if (const char* envStop = std::getenv("SCALABLE_CCD_STOP_BELOW")) {
            try { stopBelow = std::stod(envStop); } catch (...) { stopBelow = -1.0; }
        }
        bool stopFirstHit = false;
        if (const char* envFirst = std::getenv("SCALABLE_CCD_STOP_FIRSTHIT")) {
            stopFirstHit = (envFirst[0] != '\0');
        }
        bool stopped = false;
        // Reusable small constant buffers
        id<MTLBuffer> bTol = [impl->device newBufferWithBytes:&tol length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bIt  = [impl->device newBufferWithBytes:&maxIter length:sizeof(int) options:MTLResourceStorageModeShared];
        uint32_t allow = allowZeroToi ? 1u : 0u;
        id<MTLBuffer> bAz  = [impl->device newBufferWithBytes:&allow length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bRs   = [impl->device newBufferWithBytes:&refineSteps length:sizeof(int) options:MTLResourceStorageModeShared];
        uint32_t useTStack = 0u;
        if (const char* envT = std::getenv("SCALABLE_CCD_USE_TSTACK")) { if (envT[0] != '\0') useTStack = 1u; }
        id<MTLBuffer> bTst = [impl->device newBufferWithBytes:&useTStack length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        uint32_t useFastCull = 0u;
        if (const char* envFC = std::getenv("SCALABLE_CCD_NP_FASTCULL")) { if (envFC[0] != '\0') useFastCull = 1u; }
        id<MTLBuffer> bFC = [impl->device newBufferWithBytes:&useFastCull length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        uint32_t useQueueFlag = 0u;
        if (const char* envQ = std::getenv("SCALABLE_CCD_USE_QUEUE")) { if (envQ[0] != '\0') useQueueFlag = 1u; }
        id<MTLBuffer> bUQ = [impl->device newBufferWithBytes:&useQueueFlag length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        uint32_t zeroCtr = 0u;
        id<MTLBuffer> bCtr = [impl->device newBufferWithBytes:&zeroCtr length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        uint32_t useQueue = 0u;
        if (const char* envQ = std::getenv("SCALABLE_CCD_USE_QUEUE")) { if (envQ[0] != '\0') useQueue = 1u; }
        size_t queueChunk = 0;
        if (useQueue) {
            if (const char* envQC = std::getenv("SCALABLE_CCD_QUEUE_CHUNK")) {
                try { queueChunk = std::max<size_t>(1, static_cast<size_t>(std::stoll(envQC))); } catch (...) { queueChunk = 65536; }
            } else {
                queueChunk = 65536;
            }
        }
        scalable_ccd::logger().info(
            "Metal vfRootSkeleton cfg: chunkSize={} refineSteps={} tol={} allowZero={} stopBelow={} stopFirstHit={} useTStack={} fastCull={} useQueue={} queueChunk={}",
            static_cast<uint64_t>(chunkSize), refineSteps,
            static_cast<double>(tol), allowZeroToi ? 1 : 0,
            stopBelow, stopFirstHit ? 1 : 0,
            static_cast<uint64_t>(useTStack),
            static_cast<uint64_t>(useFastCull),
            static_cast<uint64_t>(useQueue),
            static_cast<uint64_t>(queueChunk));
        // Reusable per-chunk buffers (allocated at chunkSize capacity)
        id<MTLBuffer> bO   = [impl->device newBufferWithLength:sizeof(Pair2i)*chunkSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bNO  = [impl->device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bCCD = [impl->device newBufferWithLength:sizeof(CCDDataMetalVF)*chunkSize options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bToi = [impl->device newBufferWithLength:sizeof(float)*chunkSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bIter= [impl->device newBufferWithLength:sizeof(uint32_t)*chunkSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bHit = [impl->device newBufferWithLength:sizeof(uint32_t)*chunkSize options:MTLResourceStorageModeShared];
        const size_t effChunkVF = (useQueue && queueChunk > 0) ? queueChunk : chunkSize;
        for (size_t base = 0; base < nOver; base += effChunkVF) {
            const size_t cur = std::min(effChunkVF, nOver - base);
            uint32_t curU32 = static_cast<uint32_t>(cur);
            // Fill reusable buffers for current chunk
            memcpy([bO contents], (ovs.data()+base), sizeof(Pair2i)*cur);
            memcpy([bNO contents], &curU32, sizeof(uint32_t));
            // Pack + Root-finder in one command buffer for this chunk
            {
                id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
                // Pack VF -> CCDData
                {
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:impl->npPackVfCps];
                [enc setBuffer:pV0 offset:0 atIndex:0];
                [enc setBuffer:pV1 offset:0 atIndex:1];
                    [enc setBuffer:bNV offset:0 atIndex:2];
                [enc setBuffer:pF  offset:0 atIndex:3];
                    [enc setBuffer:bO  offset:0 atIndex:4];
                    [enc setBuffer:bNO offset:0 atIndex:5];
                    [enc setBuffer:bMs offset:0 atIndex:6];
                    [enc setBuffer:bCCD offset:0 atIndex:7];
                    MTLSize grid = MTLSizeMake(cur,1,1);
                    NSUInteger w = impl->npPackVfCps.maxTotalThreadsPerThreadgroup;
                    if (w == 0) w = 64;
                    NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, w) : w;
                    if (tgW > cur) tgW = static_cast<NSUInteger>(cur);
                    if (tgW == 0) tgW = 1;
                    MTLSize tg = MTLSizeMake(tgW, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    [enc endEncoding];
                }
                // Root-finder
                {
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:impl->vfRootSkeletonCps];
                    [enc setBuffer:bCCD offset:0 atIndex:0];
                    [enc setBuffer:bNO  offset:0 atIndex:1];
                    [enc setBuffer:bTol offset:0 atIndex:2];
                    [enc setBuffer:bIt  offset:0 atIndex:3];
                    [enc setBuffer:bAz  offset:0 atIndex:4];
                    [enc setBuffer:bToi offset:0 atIndex:5];
                    [enc setBuffer:bIter offset:0 atIndex:6];
                    [enc setBuffer:bHit  offset:0 atIndex:7];
                    [enc setBuffer:bRs   offset:0 atIndex:8];
                    [enc setBuffer:bTst  offset:0 atIndex:9];
                    [enc setBuffer:bFC   offset:0 atIndex:10];
                    // reset counter for this dispatch
                    auto* counter_ptr = static_cast<uint32_t*>([bCtr contents]);
                    *counter_ptr = 0u;
                    [enc setBuffer:bCtr  offset:0 atIndex:11];
                    [enc setBuffer:bUQ   offset:0 atIndex:12];
                    MTLSize grid = MTLSizeMake(cur,1,1);
                    NSUInteger w = impl->vfRootSkeletonCps.maxTotalThreadsPerThreadgroup;
                    if (w == 0) w = 64;
                    NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, w) : w;
                    if (tgW > cur) tgW = static_cast<NSUInteger>(cur);
                    if (tgW == 0) tgW = 1;
                    MTLSize tg = MTLSizeMake(tgW, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    [enc endEncoding];
                }
                [cb commit];
                [cb waitUntilCompleted];
                if (impl->enableTiming) {
                    CFTimeInterval t0 = cb.GPUStartTime;
                    CFTimeInterval t1 = cb.GPUEndTime;
                    if (t1 > t0 && t0 > 0) {
                        double msElapsed = (t1 - t0) * 1000.0;
                        scalable_ccd::logger().info("Metal VF chunk(pack+root): gpu_time_ms={} count={}",
                                                     msElapsed, static_cast<uint64_t>(cur));
                        append_timing_csv_if_needed(impl->timingCsvPath, "vf_chunk",
                                                    static_cast<uint64_t>(cur),
                                                    0u,
                                                    msElapsed);
                    }
                }
                // Aggregate diagnostics for chunk
                uint32_t* iterPtr = static_cast<uint32_t*>([bIter contents]);
                uint32_t* hitPtr  = static_cast<uint32_t*>([bHit contents]);
                for (size_t i=0;i<cur;++i){ totalIter += iterPtr[i]; totalHit += hitPtr[i]; }
            }
            // Copy out ToI chunk
            float* toiPtr = static_cast<float*>([bToi contents]);
            memcpy(outToi.data()+base, toiPtr, sizeof(float)*cur);
            // Early-stop check
            if (!stopped) {
                float minChunk = 1.0f;
                for (size_t i = 0; i < cur; ++i) {
                    if (toiPtr[i] < minChunk) minChunk = toiPtr[i];
                }
                if ((stopFirstHit && minChunk < 1.0f) ||
                    (stopBelow >= 0.0 && minChunk <= static_cast<float>(stopBelow))) {
                    scalable_ccd::logger().info("Metal vfRootSkeleton: early-stop at base={} cur={} minToi={}",
                                                static_cast<uint64_t>(base),
                                                static_cast<uint64_t>(cur),
                                                static_cast<double>(minChunk));
                    stopped = true;
                    break;
                }
            }
        }
        scalable_ccd::logger().info("Metal vfRootSkeleton: hits={} avgIters={}",
                                    static_cast<uint64_t>(totalHit),
                                    totalHit ? static_cast<double>(totalIter)/static_cast<double>(totalHit) : 0.0);
        scalable_ccd::logger().info("Metal vfRootSkeleton: wrote toi for {}", static_cast<uint64_t>(nOver));
        return true;
    }
}

bool MetalRuntime::runEERootSkeleton(
    const std::vector<float>& vertices_t0_flat,
    const std::vector<float>& vertices_t1_flat,
    uint32_t num_vertices,
    const std::vector<int32_t>& edges_flat,
    const std::vector<std::pair<int,int>>& overlaps,
    float minimum_separation,
    float tolerance,
    int max_iterations,
    bool allow_zero_toi,
    std::vector<float>& out_toi)
{
    if (!available()) return false;
    const float tol = tolerance;
    const int maxIter = max_iterations;
    const bool allowZeroToi = allow_zero_toi;
    auto& outToi = out_toi;
    @autoreleasepool {
        NSError* err = nil;
        // Force recompile to avoid stale pipeline after kernel signature changes
        impl->eeRootSkeletonCps = nil;
        // Ensure pack pipeline (build dedicated EE pack lib to avoid stale npLib contents)
        if (!impl->npPackEeCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> libPack = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kNPPackEEKernelSrc]
                                                                options:opts
                                                                  error:&err];
            if (!libPack) {
                scalable_ccd::logger().error("Metal runEERootSkeleton: build packEE lib failed: {}",
                                             err ? err.localizedDescription.UTF8String : "unknown error");
                return false;
            }
            id<MTLFunction> pfn = [libPack newFunctionWithName:@"npPackEE"];
            if (!pfn) {
                scalable_ccd::logger().error("Metal runEERootSkeleton: newFunction npPackEE failed");
                return false;
            }
            impl->npPackEeCps = [impl->device newComputePipelineStateWithFunction:pfn error:&err];
            if (!impl->npPackEeCps) {
                scalable_ccd::logger().error("Metal runEERootSkeleton: newComputePipelineState(npPackEE) failed: {}",
                                             err ? err.localizedDescription.UTF8String : "unknown error");
                return false;
            }
        }
        // Ensure root skeleton pipeline
        if (!impl->eeRootSkeletonCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> lib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kEERootSkeletonKernelSrc]
                                                            options:opts
                                                              error:&err];
            if (!lib) {
                scalable_ccd::logger().error("Metal runEERootSkeleton: build eeRoot lib failed: {}",
                                             err ? err.localizedDescription.UTF8String : "unknown error");
                return false;
            }
            id<MTLFunction> rfn = [lib newFunctionWithName:@"eeRootSkeleton"];
            if (!rfn) {
                scalable_ccd::logger().error("Metal runEERootSkeleton: newFunction eeRootSkeleton failed");
                return false;
            }
            impl->eeRootSkeletonCps = [impl->device newComputePipelineStateWithFunction:rfn error:&err];
            if (!impl->eeRootSkeletonCps) {
                scalable_ccd::logger().error("Metal runEERootSkeleton: newComputePipelineState(eeRootSkeleton) failed: {}",
                                             err ? err.localizedDescription.UTF8String : "unknown error");
                return false;
            }
        }
        const size_t nOver = overlaps.size();
        struct Pair2i { int32_t x; int32_t y; };
        std::vector<Pair2i> ovs(nOver);
        for (size_t i=0;i<nOver;++i){ ovs[i].x = overlaps[i].first; ovs[i].y = overlaps[i].second; }
        id<MTLBuffer> bV0 = [impl->device newBufferWithBytes:vertices_t0_flat.data() length:sizeof(float)*vertices_t0_flat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV1 = [impl->device newBufferWithBytes:vertices_t1_flat.data() length:sizeof(float)*vertices_t1_flat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bNV = [impl->device newBufferWithBytes:&num_vertices length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bE  = [impl->device newBufferWithBytes:edges_flat.data() length:sizeof(int32_t)*edges_flat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMs = [impl->device newBufferWithBytes:&minimum_separation length:sizeof(float) options:MTLResourceStorageModeShared];
        // Private copies for heavy readers (once per call)
        id<MTLBuffer> pV0 = [impl->device newBufferWithLength:sizeof(float)*vertices_t0_flat.size() options:MTLResourceStorageModePrivate];
        id<MTLBuffer> pV1 = [impl->device newBufferWithLength:sizeof(float)*vertices_t1_flat.size() options:MTLResourceStorageModePrivate];
        id<MTLBuffer> pE  = [impl->device newBufferWithLength:sizeof(int32_t)*edges_flat.size() options:MTLResourceStorageModePrivate];
        {
            id<MTLCommandBuffer> cbu = [impl->queue commandBuffer];
            id<MTLBlitCommandEncoder> bl = [cbu blitCommandEncoder];
            [bl copyFromBuffer:bV0 sourceOffset:0 toBuffer:pV0 destinationOffset:0 size:sizeof(float)*vertices_t0_flat.size()];
            [bl copyFromBuffer:bV1 sourceOffset:0 toBuffer:pV1 destinationOffset:0 size:sizeof(float)*vertices_t1_flat.size()];
            [bl copyFromBuffer:bE  sourceOffset:0 toBuffer:pE  destinationOffset:0 size:sizeof(int32_t)*edges_flat.size()];
            [bl endEncoding];
            [cbu commit];
            [cbu waitUntilCompleted];
        }
        size_t chunkSize = 100000;
        if (const char* envChunk = std::getenv("SCALABLE_CCD_NP_CHUNK")) {
            try { chunkSize = std::max<size_t>(1, static_cast<size_t>(std::stoll(envChunk))); } catch (...) {}
        }
        int refineSteps = 16;
        if (const char* envRef = std::getenv("SCALABLE_CCD_REFINE_STEPS")) {
            try { refineSteps = std::max(1, std::stoi(envRef)); } catch (...) {}
        }
        outToi.resize(nOver);
        uint64_t totalIter = 0, totalHit = 0;
        // Early-stop config
        double stopBelow = -1.0;
        if (const char* envStop = std::getenv("SCALABLE_CCD_STOP_BELOW")) {
            try { stopBelow = std::stod(envStop); } catch (...) { stopBelow = -1.0; }
        }
        bool stopFirstHit = false;
        if (const char* envFirst = std::getenv("SCALABLE_CCD_STOP_FIRSTHIT")) {
            stopFirstHit = (envFirst[0] != '\0');
        }
        bool stopped = false;
        // Reusable small constant buffers
        id<MTLBuffer> bTol = [impl->device newBufferWithBytes:&tol length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bIt  = [impl->device newBufferWithBytes:&maxIter length:sizeof(int) options:MTLResourceStorageModeShared];
        uint32_t allow = allowZeroToi ? 1u : 0u;
        id<MTLBuffer> bAz  = [impl->device newBufferWithBytes:&allow length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bRs   = [impl->device newBufferWithBytes:&refineSteps length:sizeof(int) options:MTLResourceStorageModeShared];
        uint32_t useTStack = 0u;
        if (const char* envT = std::getenv("SCALABLE_CCD_USE_TSTACK")) { if (envT[0] != '\0') useTStack = 1u; }
        id<MTLBuffer> bTst = [impl->device newBufferWithBytes:&useTStack length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        uint32_t useFastCull = 0u;
        if (const char* envFC = std::getenv("SCALABLE_CCD_NP_FASTCULL")) { if (envFC[0] != '\0') useFastCull = 1u; }
        id<MTLBuffer> bFC = [impl->device newBufferWithBytes:&useFastCull length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        uint32_t useQueue = 0u;
        if (const char* envQ = std::getenv("SCALABLE_CCD_USE_QUEUE")) { if (envQ[0] != '\0') useQueue = 1u; }
        size_t queueChunk = 0;
        if (useQueue) {
            if (const char* envQC = std::getenv("SCALABLE_CCD_QUEUE_CHUNK")) {
                try { queueChunk = std::max<size_t>(1, static_cast<size_t>(std::stoll(envQC))); } catch (...) { queueChunk = 65536; }
            } else {
                queueChunk = 65536;
            }
        }
        scalable_ccd::logger().info(
            "Metal eeRootSkeleton cfg: chunkSize={} refineSteps={} tol={} allowZero={} stopBelow={} stopFirstHit={} useTStack={} fastCull={} useQueue={} queueChunk={}",
            static_cast<uint64_t>(chunkSize), refineSteps,
            static_cast<double>(tol), allowZeroToi ? 1 : 0,
            stopBelow, stopFirstHit ? 1 : 0,
            static_cast<uint64_t>(useTStack),
            static_cast<uint64_t>(useFastCull),
            static_cast<uint64_t>(useQueue),
            static_cast<uint64_t>(queueChunk));
        // Reusable per-chunk buffers
        id<MTLBuffer> bO   = [impl->device newBufferWithLength:sizeof(Pair2i)*chunkSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bNO  = [impl->device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bCCD = [impl->device newBufferWithLength:sizeof(CCDDataMetalEE)*chunkSize options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bToi = [impl->device newBufferWithLength:sizeof(float)*chunkSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bIter= [impl->device newBufferWithLength:sizeof(uint32_t)*chunkSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bHit = [impl->device newBufferWithLength:sizeof(uint32_t)*chunkSize options:MTLResourceStorageModeShared];
        const size_t effChunkEE = (useQueue && queueChunk > 0) ? queueChunk : chunkSize;
        for (size_t base = 0; base < nOver; base += effChunkEE) {
            const size_t cur = std::min(effChunkEE, nOver - base);
            uint32_t curU32 = static_cast<uint32_t>(cur);
            memcpy([bO contents], (ovs.data()+base), sizeof(Pair2i)*cur);
            memcpy([bNO contents], &curU32, sizeof(uint32_t));
            // Pack + Root-finder for chunk in one command buffer
            {
                id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
                // Pack EE -> CCDData
                {
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:impl->npPackEeCps];
                [enc setBuffer:pV0 offset:0 atIndex:0];
                [enc setBuffer:pV1 offset:0 atIndex:1];
                    [enc setBuffer:bNV offset:0 atIndex:2];
                [enc setBuffer:pE  offset:0 atIndex:3];
                    [enc setBuffer:bO  offset:0 atIndex:4];
                    [enc setBuffer:bNO offset:0 atIndex:5];
                    [enc setBuffer:bMs offset:0 atIndex:6];
                    [enc setBuffer:bCCD offset:0 atIndex:7];
                    MTLSize grid = MTLSizeMake(cur,1,1);
                    NSUInteger w = impl->npPackEeCps.maxTotalThreadsPerThreadgroup;
                    if (w == 0) w = 64;
                    NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, w) : w;
                    if (tgW > cur) tgW = static_cast<NSUInteger>(cur);
                    if (tgW == 0) tgW = 1;
                    MTLSize tg = MTLSizeMake(tgW, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    [enc endEncoding];
                }
                // Root-finder
                {
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:impl->eeRootSkeletonCps];
                    [enc setBuffer:bCCD offset:0 atIndex:0];
                    [enc setBuffer:bNO  offset:0 atIndex:1];
                    [enc setBuffer:bTol offset:0 atIndex:2];
                    [enc setBuffer:bIt  offset:0 atIndex:3];
                    [enc setBuffer:bAz  offset:0 atIndex:4];
                    [enc setBuffer:bToi offset:0 atIndex:5];
                    [enc setBuffer:bIter offset:0 atIndex:6];
                    [enc setBuffer:bHit  offset:0 atIndex:7];
                    [enc setBuffer:bRs   offset:0 atIndex:8];
                    [enc setBuffer:bTst  offset:0 atIndex:9];
                    [enc setBuffer:bFC   offset:0 atIndex:10];
                    uint32_t useQueueFlag = 0u;
                    if (const char* envQ = std::getenv("SCALABLE_CCD_USE_QUEUE")) { if (envQ[0] != '\0') useQueueFlag = 1u; }
                    id<MTLBuffer> bUQ = [impl->device newBufferWithBytes:&useQueueFlag length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                    uint32_t zeroCtr = 0u;
                    id<MTLBuffer> bCtr = [impl->device newBufferWithBytes:&zeroCtr length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
                    [enc setBuffer:bCtr offset:0 atIndex:11];
                    [enc setBuffer:bUQ  offset:0 atIndex:12];
                    MTLSize grid = MTLSizeMake(cur,1,1);
                    NSUInteger w = impl->eeRootSkeletonCps.maxTotalThreadsPerThreadgroup;
                    if (w == 0) w = 64;
                    NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, w) : w;
                    if (tgW > cur) tgW = static_cast<NSUInteger>(cur);
                    if (tgW == 0) tgW = 1;
                    MTLSize tg = MTLSizeMake(tgW, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    [enc endEncoding];
                }
                [cb commit];
                [cb waitUntilCompleted];
                if (impl->enableTiming) {
                    CFTimeInterval t0 = cb.GPUStartTime;
                    CFTimeInterval t1 = cb.GPUEndTime;
                    if (t1 > t0 && t0 > 0) {
                        double msElapsed = (t1 - t0) * 1000.0;
                        scalable_ccd::logger().info("Metal EE chunk(pack+root): gpu_time_ms={} count={}",
                                                     msElapsed, static_cast<uint64_t>(cur));
                        append_timing_csv_if_needed(impl->timingCsvPath, "ee_chunk",
                                                    static_cast<uint64_t>(cur),
                                                    0u,
                                                    msElapsed);
                    }
                }
                uint32_t* iterPtr = static_cast<uint32_t*>([bIter contents]);
                uint32_t* hitPtr  = static_cast<uint32_t*>([bHit contents]);
                for (size_t i=0;i<cur;++i){ totalIter += iterPtr[i]; totalHit += hitPtr[i]; }
            }
            float* toiPtr = static_cast<float*>([bToi contents]);
            memcpy(outToi.data()+base, toiPtr, sizeof(float)*cur);
            // Early-stop check
            if (!stopped) {
                float minChunk = 1.0f;
                for (size_t i = 0; i < cur; ++i) {
                    if (toiPtr[i] < minChunk) minChunk = toiPtr[i];
                }
                if ((stopFirstHit && minChunk < 1.0f) ||
                    (stopBelow >= 0.0 && minChunk <= static_cast<float>(stopBelow))) {
                    scalable_ccd::logger().info("Metal eeRootSkeleton: early-stop at base={} cur={} minToi={}",
                                                static_cast<uint64_t>(base),
                                                static_cast<uint64_t>(cur),
                                                static_cast<double>(minChunk));
                    stopped = true;
                    break;
                }
            }
        }
        scalable_ccd::logger().info("Metal eeRootSkeleton: hits={} avgIters={}",
                                    static_cast<uint64_t>(totalHit),
                                    totalHit ? static_cast<double>(totalIter)/static_cast<double>(totalHit) : 0.0);
        scalable_ccd::logger().info("Metal eeRootSkeleton: wrote toi for {}", static_cast<uint64_t>(nOver));
        return true;
    }
}
bool MetalRuntime::runVFMin(
    const std::vector<float>& vertices_t0_flat,
    const std::vector<float>& vertices_t1_flat,
    uint32_t num_vertices,
    const std::vector<int32_t>& faces_flat,
    const std::vector<std::pair<int,int>>& overlaps,
    float minimum_separation,
    float tolerance,
    int max_iterations,
    bool allow_zero_toi,
    float& out_min_toi)
{
    auto& outMinToi = out_min_toi;
    outMinToi = 1.0f;
    if (!available()) return false;
    const auto& V0 = vertices_t0_flat;
    const auto& V1 = vertices_t1_flat;
    const uint32_t nV = num_vertices;
    const auto& facesFlat = faces_flat;
    const float ms = minimum_separation;
    const float tol = tolerance;
    const int maxIter = max_iterations;
    const bool allowZeroToi = allow_zero_toi;
    const size_t nOver = overlaps.size();
    if (nOver == 0) return true;
    @autoreleasepool {
        NSError* err = nil;
        // Ensure pack pipeline
        if (!impl->npPackVfCps) {
            if (!impl->npLib) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl->npLib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kNPPackVFKernelSrc]
                                                         options:opts
                                                           error:&err];
                if (!impl->npLib) return false;
            }
            id<MTLFunction> pfn = [impl->npLib newFunctionWithName:@"npPackVF"];
            if (!pfn) return false;
            impl->npPackVfCps = [impl->device newComputePipelineStateWithFunction:pfn error:&err];
            if (!impl->npPackVfCps) return false;
        }
        // Ensure root pipeline
        if (!impl->vfRootSkeletonCps) {
            id<MTLLibrary> lib = nil;
            MTLCompileOptions* opts = [MTLCompileOptions new];
            lib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kVFRootSkeletonKernelSrc]
                                             options:opts
                                               error:&err];
            if (!lib) return false;
            id<MTLFunction> rfn = [lib newFunctionWithName:@"vfRootSkeleton"];
            if (!rfn) return false;
            impl->vfRootSkeletonCps = [impl->device newComputePipelineStateWithFunction:rfn error:&err];
            if (!impl->vfRootSkeletonCps) return false;
        }
        // Ensure reduce pipeline
        if (!impl->reduceMinToiCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> lib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kReduceMinToiKernelSrc]
                                                            options:opts
                                                              error:&err];
            if (!lib) return false;
            id<MTLFunction> fn = [lib newFunctionWithName:@"reduceMinToi"];
            if (!fn) return false;
            impl->reduceMinToiCps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!impl->reduceMinToiCps) return false;
        }
        // Prepare overlaps and staging buffers (reusable)
        struct Pair2i { int32_t x; int32_t y; };
        std::vector<Pair2i> ovs(nOver);
        for (size_t i=0;i<nOver;++i){ ovs[i].x = overlaps[i].first; ovs[i].y = overlaps[i].second; }
        // Ensure reusable shared/private buffers for geometry
        ensureBuffer(impl->device, impl->sV0, sizeof(float)*V0.size(), MTLResourceStorageModeShared);
        ensureBuffer(impl->device, impl->sV1, sizeof(float)*V1.size(), MTLResourceStorageModeShared);
        ensureBuffer(impl->device, impl->sF,  sizeof(int32_t)*facesFlat.size(), MTLResourceStorageModeShared);
        memcpy([impl->sV0 contents], V0.data(), sizeof(float)*V0.size());
        memcpy([impl->sV1 contents], V1.data(), sizeof(float)*V1.size());
        memcpy([impl->sF contents],  facesFlat.data(), sizeof(int32_t)*facesFlat.size());
        ensureBuffer(impl->device, impl->pV0, sizeof(float)*V0.size(), MTLResourceStorageModePrivate);
        ensureBuffer(impl->device, impl->pV1, sizeof(float)*V1.size(), MTLResourceStorageModePrivate);
        ensureBuffer(impl->device, impl->pF,  sizeof(int32_t)*facesFlat.size(), MTLResourceStorageModePrivate);
        // Upload geometry via a single blit
        {
            id<MTLCommandBuffer> cbu = [impl->queue commandBuffer];
            id<MTLBlitCommandEncoder> bl = [cbu blitCommandEncoder];
            [bl copyFromBuffer:impl->sV0 sourceOffset:0 toBuffer:impl->pV0 destinationOffset:0 size:sizeof(float)*V0.size()];
            [bl copyFromBuffer:impl->sV1 sourceOffset:0 toBuffer:impl->pV1 destinationOffset:0 size:sizeof(float)*V1.size()];
            [bl copyFromBuffer:impl->sF  sourceOffset:0 toBuffer:impl->pF  destinationOffset:0 size:sizeof(int32_t)*facesFlat.size()];
            [bl endEncoding];
            [cbu commit];
            [cbu waitUntilCompleted];
        }
        // Small constants (per-call)
        id<MTLBuffer> bNV = [impl->device newBufferWithBytes:&nV length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMs = [impl->device newBufferWithBytes:&ms length:sizeof(float) options:MTLResourceStorageModeShared];
        // Min ToI buffer initialized to bit pattern of 1.0f
        // Init global min toi; allow env override (SCALABLE_CCD_INIT_MIN_TOI in [0,1])
        uint32_t initMin = 0x3F800000u;
        if (const char* envInit = std::getenv("SCALABLE_CCD_INIT_MIN_TOI")) {
            try {
                float v = std::stof(envInit);
                v = std::max(0.0f, std::min(1.0f, v));
                initMin = *reinterpret_cast<uint32_t*>(&v);
            } catch (...) {}
        }
        id<MTLBuffer> bMin = [impl->device newBufferWithBytes:&initMin length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        // Constants
        id<MTLBuffer> bTol = [impl->device newBufferWithBytes:&tol length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bIt  = [impl->device newBufferWithBytes:&maxIter length:sizeof(int) options:MTLResourceStorageModeShared];
        uint32_t allow = allowZeroToi ? 1u : 0u;
        id<MTLBuffer> bAz  = [impl->device newBufferWithBytes:&allow length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        int refineSteps = 16;
        if (const char* envRef = std::getenv("SCALABLE_CCD_REFINE_STEPS")) {
            try { refineSteps = std::max(1, std::stoi(envRef)); } catch (...) {}
        }
        id<MTLBuffer> bRs   = [impl->device newBufferWithBytes:&refineSteps length:sizeof(int) options:MTLResourceStorageModeShared];
        uint32_t useTStack = env_flag_enabled("SCALABLE_CCD_USE_TSTACK", false) ? 1u : 0u;
        id<MTLBuffer> bTst = [impl->device newBufferWithBytes:&useTStack length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        // Chunk buffers
        size_t chunkSize = env_size_override("SCALABLE_CCD_NP_CHUNK", static_cast<size_t>(100000), static_cast<size_t>(1));
        ensureBuffer(impl->device, impl->sO,   sizeof(Pair2i)*chunkSize, MTLResourceStorageModeShared);
        ensureBuffer(impl->device, impl->sNO,  sizeof(uint32_t),         MTLResourceStorageModeShared);
        ensureBuffer(impl->device, impl->pCCD_VF, sizeof(CCDDataMetalVF)*chunkSize, MTLResourceStorageModePrivate);
        ensureBuffer(impl->device, impl->sToi, sizeof(float)*chunkSize,  MTLResourceStorageModeShared);
        const bool legacyQueueEnabled = env_flag_enabled("SCALABLE_CCD_USE_QUEUE", true);
        uint32_t useQueueLegacy = legacyQueueEnabled ? 1u : 0u;
        size_t queueChunk = legacyQueueEnabled ? env_size_override("SCALABLE_CCD_QUEUE_CHUNK", static_cast<size_t>(65536), static_cast<size_t>(1)) : 0;
        if (queueChunk > chunkSize) queueChunk = chunkSize;
        const size_t effChunkVFM = (legacyQueueEnabled && queueChunk > 0) ? queueChunk : chunkSize;
        uint32_t useFastCullFlag = env_flag_enabled("SCALABLE_CCD_NP_FASTCULL", false) ? 1u : 0u;
        id<MTLBuffer> bFC = [impl->device newBufferWithBytes:&useFastCullFlag length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bCtr = nil;
        if (legacyQueueEnabled) {
            uint32_t zeroCtr = 0u;
            bCtr = [impl->device newBufferWithBytes:&zeroCtr length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> bUQ = [impl->device newBufferWithBytes:&useQueueLegacy length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        const RootMode rootMode = root_mode_vf_from_env();
        bool queueRootMode = (rootMode == RootMode::Queue);
        // Small-n heuristic: for very few overlaps, legacy root often faster
        int minQ = env_int_override("SCALABLE_CCD_QUEUE_MIN_OVERLAPS", 1000, 1, INT32_MAX);
        if (static_cast<int>(nOver) < minQ) queueRootMode = false;
        // 调大默认队列因子与重试次数，降低溢出概率（可用环境变量覆盖）
        const size_t queueFactor = env_size_override("SCALABLE_CCD_QUEUE_FACTOR", static_cast<size_t>(32), static_cast<size_t>(1));
        const int queueMaxRetries = std::max(1, env_int_override("SCALABLE_CCD_QUEUE_RETRIES", 5, 1, 16));
        scalable_ccd::logger().info("Metal VF Min: chunksize={} (eff={}) overlaps={} useQueue={} queueChunk={}",
                                    static_cast<uint64_t>(chunkSize), static_cast<uint64_t>(effChunkVFM),
                                    static_cast<uint64_t>(nOver), static_cast<uint64_t>(useQueueLegacy),
                                    static_cast<uint64_t>(queueChunk));
        for (size_t base = 0; base < nOver; base += effChunkVFM) {
            const size_t cur = std::min(effChunkVFM, nOver - base);
            uint32_t curU32 = static_cast<uint32_t>(cur);
            memcpy([impl->sO contents], (ovs.data()+base), sizeof(Pair2i)*cur);
            memcpy([impl->sNO contents], &curU32, sizeof(uint32_t));
            id<MTLCommandBuffer> cbPack = [impl->queue commandBuffer];
            {
                id<MTLComputeCommandEncoder> enc = [cbPack computeCommandEncoder];
                [enc setComputePipelineState:impl->npPackVfCps];
                [enc setBuffer:impl->pV0 offset:0 atIndex:0];
                [enc setBuffer:impl->pV1 offset:0 atIndex:1];
                [enc setBuffer:bNV offset:0 atIndex:2];
                [enc setBuffer:impl->pF  offset:0 atIndex:3];
                [enc setBuffer:impl->sO  offset:0 atIndex:4];
                [enc setBuffer:impl->sNO offset:0 atIndex:5];
                [enc setBuffer:bMs offset:0 atIndex:6];
                [enc setBuffer:impl->pCCD_VF offset:0 atIndex:7];
                MTLSize grid = MTLSizeMake(cur,1,1);
                NSUInteger w = impl->npPackVfCps.maxTotalThreadsPerThreadgroup;
                if (w == 0) w = 64;
                NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, w) : w;
                if (tgW > cur) tgW = static_cast<NSUInteger>(cur);
                if (tgW == 0) tgW = 1;
                MTLSize tg = MTLSizeMake(tgW, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
            }
            [cbPack commit];
            [cbPack waitUntilCompleted];

            bool processedViaQueue = false;
            if (queueRootMode) {
                auto safeMul = [](size_t a, size_t b) -> size_t {
                    if (a == 0 || b == 0) return 0;
                    if (a > SIZE_MAX / b) return SIZE_MAX;
                    return a * b;
                };
                size_t domainCapacity = std::max(cur, safeMul(cur, queueFactor));
                size_t chunkBasedCapacity = safeMul(chunkSize, queueFactor);
                if (chunkBasedCapacity > domainCapacity) domainCapacity = chunkBasedCapacity;
                if (domainCapacity == 0) domainCapacity = cur;
                for (int attempt = 0; attempt < queueMaxRetries; ++attempt) {
                    if (impl->runQueueChunkVF(static_cast<uint32_t>(base), static_cast<uint32_t>(cur), domainCapacity, tol, bTol, bAz, bMin, maxIter)) {
                        processedViaQueue = true;
                        break;
                    }
                    QueueCountersHost* ctrHost = impl->qCounters ? static_cast<QueueCountersHost*>([impl->qCounters contents]) : nullptr;
                    const bool overflowed = ctrHost && ctrHost->overflow;
                    if (!overflowed) {
                        scalable_ccd::logger().warn("Metal VF queue root failed without overflow (base={} size={} attempt={})",
                                                     static_cast<uint64_t>(base),
                                                     static_cast<uint64_t>(cur),
                                                     attempt);
                        break;
                    }
                    if (domainCapacity >= SIZE_MAX / 2) {
                        scalable_ccd::logger().warn("Metal VF queue domain capacity saturated (base={} size={} cap={})",
                                                     static_cast<uint64_t>(base),
                                                     static_cast<uint64_t>(cur),
                                                     static_cast<uint64_t>(domainCapacity));
                        break;
                    }
                    domainCapacity *= 2;
                    scalable_ccd::logger().warn("Metal VF queue overflow, retrying with domainCapacity={} (attempt {}/{})",
                                                 static_cast<uint64_t>(domainCapacity),
                                                 attempt + 1,
                                                 queueMaxRetries);
                }
                if (!processedViaQueue) {
                    scalable_ccd::logger().warn("Metal VF queue root fallback at chunk base={} size={}",
                                                 static_cast<uint64_t>(base), static_cast<uint64_t>(cur));
                    queueRootMode = false;
                }
            }
            if (processedViaQueue) {
                continue;
            }

            id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
            // Root
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:impl->vfRootSkeletonCps];
                [enc setBuffer:impl->pCCD_VF offset:0 atIndex:0];
                [enc setBuffer:impl->sNO  offset:0 atIndex:1];
                [enc setBuffer:bTol offset:0 atIndex:2];
                [enc setBuffer:bIt  offset:0 atIndex:3];
                [enc setBuffer:bAz  offset:0 atIndex:4];
                [enc setBuffer:impl->sToi offset:0 atIndex:5];
                [enc setBuffer:nil  offset:0 atIndex:6];
                [enc setBuffer:nil  offset:0 atIndex:7];
                [enc setBuffer:bRs  offset:0 atIndex:8];
                [enc setBuffer:bTst offset:0 atIndex:9];
                MTLSize grid = MTLSizeMake(cur,1,1);
                NSUInteger w = impl->vfRootSkeletonCps.maxTotalThreadsPerThreadgroup;
                if (w == 0) w = 64;
                NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, w) : w;
                if (tgW > cur) tgW = static_cast<NSUInteger>(cur);
                if (tgW == 0) tgW = 1;
                MTLSize tg = MTLSizeMake(tgW, 1, 1);
                if (bCtr) {
                    uint32_t zero = 0u;
                    memcpy([bCtr contents], &zero, sizeof(uint32_t));
                }
                [enc setBuffer:bFC  offset:0 atIndex:10];
                [enc setBuffer:bCtr offset:0 atIndex:11];
                [enc setBuffer:bUQ  offset:0 atIndex:12];
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
            }
            // Reduce
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:impl->reduceMinToiCps];
                [enc setBuffer:impl->sToi offset:0 atIndex:0];
                [enc setBuffer:impl->sNO  offset:0 atIndex:1];
                [enc setBuffer:bMin offset:0 atIndex:2];
                MTLSize grid = MTLSizeMake(cur,1,1);
                NSUInteger w = impl->reduceMinToiCps.maxTotalThreadsPerThreadgroup;
                if (w == 0) w = 64;
                NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, w) : w;
                if (tgW > cur) tgW = static_cast<NSUInteger>(cur);
                if (tgW == 0) tgW = 1;
                MTLSize tg = MTLSizeMake(tgW, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
            }
            [cb commit];
            [cb waitUntilCompleted];
            if (impl->enableTiming) {
                CFTimeInterval t0 = cb.GPUStartTime;
                CFTimeInterval t1 = cb.GPUEndTime;
                if (t0 > 0 && t1 > t0) {
                    double msElapsed = (t1 - t0) * 1000.0;
                    append_timing_csv_if_needed(impl->timingCsvPath, "vf_pack_root_reduce",
                                                static_cast<uint64_t>(cur), 0u, msElapsed);
                }
            }
        }
        // Read back minimal ToI
        uint32_t* umin = static_cast<uint32_t*>([bMin contents]);
        if (!umin) return false;
        union { uint32_t u; float f; } conv;
        conv.u = *umin;
        outMinToi = conv.f;
        outMinToi = std::max(0.0f, std::min(1.0f, outMinToi));
        return true;
    }
}
bool MetalRuntime::runEEMin(
    const std::vector<float>& vertices_t0_flat,
    const std::vector<float>& vertices_t1_flat,
    uint32_t num_vertices,
    const std::vector<int32_t>& edges_flat,
    const std::vector<std::pair<int,int>>& overlaps,
    float minimum_separation,
    float tolerance,
    int max_iterations,
    bool allow_zero_toi,
    float& out_min_toi)
{
    auto& outMinToi = out_min_toi;
    outMinToi = 1.0f;
    if (!available()) return false;
    const auto& V0 = vertices_t0_flat;
    const auto& V1 = vertices_t1_flat;
    const uint32_t nV = num_vertices;
    const auto& edgesFlat = edges_flat;
    const float ms = minimum_separation;
    const float tol = tolerance;
    const int maxIter = max_iterations;
    const bool allowZeroToi = allow_zero_toi;
    const size_t nOver = overlaps.size();
    if (nOver == 0) return true;
    @autoreleasepool {
        NSError* err = nil;
        // Ensure pack pipeline
        if (!impl->npPackEeCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> libPack = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kNPPackEEKernelSrc]
                                                                options:opts
                                                                  error:&err];
            if (!libPack) return false;
            id<MTLFunction> pfn = [libPack newFunctionWithName:@"npPackEE"];
            if (!pfn) return false;
            impl->npPackEeCps = [impl->device newComputePipelineStateWithFunction:pfn error:&err];
            if (!impl->npPackEeCps) return false;
        }
        // Ensure root pipeline
        if (!impl->eeRootSkeletonCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> lib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kEERootSkeletonKernelSrc]
                                                            options:opts
                                                              error:&err];
            if (!lib) return false;
            id<MTLFunction> rfn = [lib newFunctionWithName:@"eeRootSkeleton"];
            if (!rfn) return false;
            impl->eeRootSkeletonCps = [impl->device newComputePipelineStateWithFunction:rfn error:&err];
            if (!impl->eeRootSkeletonCps) return false;
        }
        // Ensure reduce pipeline
        if (!impl->reduceMinToiCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> lib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kReduceMinToiKernelSrc]
                                                            options:opts
                                                              error:&err];
            if (!lib) return false;
            id<MTLFunction> fn = [lib newFunctionWithName:@"reduceMinToi"];
            if (!fn) return false;
            impl->reduceMinToiCps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!impl->reduceMinToiCps) return false;
        }
        // Prepare overlaps and staging
        struct Pair2i { int32_t x; int32_t y; };
        std::vector<Pair2i> ovs(nOver);
        for (size_t i=0;i<nOver;++i){ ovs[i].x = overlaps[i].first; ovs[i].y = overlaps[i].second; }
        // Ensure reusable geometry buffers
        ensureBuffer(impl->device, impl->sV0, sizeof(float)*V0.size(), MTLResourceStorageModeShared);
        ensureBuffer(impl->device, impl->sV1, sizeof(float)*V1.size(), MTLResourceStorageModeShared);
        ensureBuffer(impl->device, impl->sE,  sizeof(int32_t)*edgesFlat.size(), MTLResourceStorageModeShared);
        memcpy([impl->sV0 contents], V0.data(), sizeof(float)*V0.size());
        memcpy([impl->sV1 contents], V1.data(), sizeof(float)*V1.size());
        memcpy([impl->sE contents],  edgesFlat.data(), sizeof(int32_t)*edgesFlat.size());
        ensureBuffer(impl->device, impl->pV0, sizeof(float)*V0.size(), MTLResourceStorageModePrivate);
        ensureBuffer(impl->device, impl->pV1, sizeof(float)*V1.size(), MTLResourceStorageModePrivate);
        ensureBuffer(impl->device, impl->pE,  sizeof(int32_t)*edgesFlat.size(), MTLResourceStorageModePrivate);
        {
            id<MTLCommandBuffer> cbu = [impl->queue commandBuffer];
            id<MTLBlitCommandEncoder> bl = [cbu blitCommandEncoder];
            [bl copyFromBuffer:impl->sV0 sourceOffset:0 toBuffer:impl->pV0 destinationOffset:0 size:sizeof(float)*V0.size()];
            [bl copyFromBuffer:impl->sV1 sourceOffset:0 toBuffer:impl->pV1 destinationOffset:0 size:sizeof(float)*V1.size()];
            [bl copyFromBuffer:impl->sE  sourceOffset:0 toBuffer:impl->pE  destinationOffset:0 size:sizeof(int32_t)*edgesFlat.size()];
            [bl endEncoding];
            [cbu commit];
            [cbu waitUntilCompleted];
        }
        id<MTLBuffer> bNV = [impl->device newBufferWithBytes:&nV length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMs = [impl->device newBufferWithBytes:&ms length:sizeof(float) options:MTLResourceStorageModeShared];
        // Init global min toi; allow env override from VF stage
        uint32_t initMin = 0x3F800000u;
        if (const char* envInit = std::getenv("SCALABLE_CCD_INIT_MIN_TOI")) {
            try {
                float v = std::stof(envInit);
                v = std::max(0.0f, std::min(1.0f, v));
                initMin = *reinterpret_cast<uint32_t*>(&v);
            } catch (...) {}
        }
        id<MTLBuffer> bMin = [impl->device newBufferWithBytes:&initMin length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bTol = [impl->device newBufferWithBytes:&tol length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bIt  = [impl->device newBufferWithBytes:&maxIter length:sizeof(int) options:MTLResourceStorageModeShared];
        uint32_t allow = allowZeroToi ? 1u : 0u;
        id<MTLBuffer> bAz  = [impl->device newBufferWithBytes:&allow length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        int refineSteps = 16;
        if (const char* envRef = std::getenv("SCALABLE_CCD_REFINE_STEPS")) {
            try { refineSteps = std::max(1, std::stoi(envRef)); } catch (...) {}
        }
        id<MTLBuffer> bRs   = [impl->device newBufferWithBytes:&refineSteps length:sizeof(int) options:MTLResourceStorageModeShared];
        uint32_t useTStack = env_flag_enabled("SCALABLE_CCD_USE_TSTACK", false) ? 1u : 0u;
        id<MTLBuffer> bTst = [impl->device newBufferWithBytes:&useTStack length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        size_t chunkSize = env_size_override("SCALABLE_CCD_NP_CHUNK", static_cast<size_t>(100000), static_cast<size_t>(1));
        ensureBuffer(impl->device, impl->sO,   sizeof(Pair2i)*chunkSize, MTLResourceStorageModeShared);
        ensureBuffer(impl->device, impl->sNO,  sizeof(uint32_t),         MTLResourceStorageModeShared);
        ensureBuffer(impl->device, impl->pCCD_EE, sizeof(CCDDataMetalEE)*chunkSize, MTLResourceStorageModePrivate);
        ensureBuffer(impl->device, impl->sToi, sizeof(float)*chunkSize,  MTLResourceStorageModeShared);
        const bool legacyQueueEnabled = env_flag_enabled("SCALABLE_CCD_USE_QUEUE", true);
        uint32_t useQueueLegacy = legacyQueueEnabled ? 1u : 0u;
        size_t queueChunk = legacyQueueEnabled ? env_size_override("SCALABLE_CCD_QUEUE_CHUNK", static_cast<size_t>(65536), static_cast<size_t>(1)) : 0;
        if (queueChunk > chunkSize) queueChunk = chunkSize;
        const size_t effChunkEEM = (legacyQueueEnabled && queueChunk > 0) ? queueChunk : chunkSize;
        uint32_t useFastCullFlag = env_flag_enabled("SCALABLE_CCD_NP_FASTCULL", false) ? 1u : 0u;
        id<MTLBuffer> bFC = [impl->device newBufferWithBytes:&useFastCullFlag length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bCtr = nil;
        if (legacyQueueEnabled) {
            uint32_t zeroCtr = 0u;
            bCtr = [impl->device newBufferWithBytes:&zeroCtr length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> bUQ = [impl->device newBufferWithBytes:&useQueueLegacy length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        const RootMode rootModeEE = root_mode_ee_from_env();
        bool queueRootModeEE = (rootModeEE == RootMode::Queue);
        int minQee = env_int_override("SCALABLE_CCD_QUEUE_MIN_OVERLAPS", 1000, 1, INT32_MAX);
        if (static_cast<int>(nOver) < minQee) queueRootModeEE = false;
        const size_t queueFactor = env_size_override("SCALABLE_CCD_QUEUE_FACTOR", static_cast<size_t>(32), static_cast<size_t>(1));
        const int queueMaxRetries = std::max(1, env_int_override("SCALABLE_CCD_QUEUE_RETRIES", 5, 1, 16));
        scalable_ccd::logger().info("Metal EE Min: chunksize={} (eff={}) overlaps={} useQueue={} queueChunk={}",
                                    static_cast<uint64_t>(chunkSize), static_cast<uint64_t>(effChunkEEM),
                                    static_cast<uint64_t>(nOver), static_cast<uint64_t>(useQueueLegacy),
                                    static_cast<uint64_t>(queueChunk));
        for (size_t base = 0; base < nOver; base += effChunkEEM) {
            const size_t cur = std::min(effChunkEEM, nOver - base);
            uint32_t curU32 = static_cast<uint32_t>(cur);
            memcpy([impl->sO contents], (ovs.data()+base), sizeof(Pair2i)*cur);
            memcpy([impl->sNO contents], &curU32, sizeof(uint32_t));
            id<MTLCommandBuffer> cbPack = [impl->queue commandBuffer];
            {
                id<MTLComputeCommandEncoder> enc = [cbPack computeCommandEncoder];
                [enc setComputePipelineState:impl->npPackEeCps];
                [enc setBuffer:impl->pV0 offset:0 atIndex:0];
                [enc setBuffer:impl->pV1 offset:0 atIndex:1];
                [enc setBuffer:bNV offset:0 atIndex:2];
                [enc setBuffer:impl->pE  offset:0 atIndex:3];
                [enc setBuffer:impl->sO  offset:0 atIndex:4];
                [enc setBuffer:impl->sNO offset:0 atIndex:5];
                [enc setBuffer:bMs offset:0 atIndex:6];
                [enc setBuffer:impl->pCCD_EE offset:0 atIndex:7];
                MTLSize grid = MTLSizeMake(cur,1,1);
                NSUInteger w = impl->npPackEeCps.maxTotalThreadsPerThreadgroup;
                if (w == 0) w = 64;
                NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, w) : w;
                if (tgW > cur) tgW = static_cast<NSUInteger>(cur);
                if (tgW == 0) tgW = 1;
                MTLSize tg = MTLSizeMake(tgW, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
            }
            [cbPack commit];
            [cbPack waitUntilCompleted];

            bool processedViaQueue = false;
            if (queueRootModeEE) {
                auto safeMul = [](size_t a, size_t b) -> size_t {
                    if (a == 0 || b == 0) return 0;
                    if (a > SIZE_MAX / b) return SIZE_MAX;
                    return a * b;
                };
                size_t domainCapacity = std::max(cur, safeMul(cur, queueFactor));
                size_t chunkBasedCapacity = safeMul(chunkSize, queueFactor);
                if (chunkBasedCapacity > domainCapacity) domainCapacity = chunkBasedCapacity;
                if (domainCapacity == 0) domainCapacity = cur;
                for (int attempt = 0; attempt < queueMaxRetries; ++attempt) {
                    if (impl->runQueueChunkEE(static_cast<uint32_t>(base), static_cast<uint32_t>(cur), domainCapacity, tol, bTol, bAz, bMin, maxIter)) {
                        processedViaQueue = true;
                        break;
                    }
                    QueueCountersHost* ctrHost = impl->qCounters ? static_cast<QueueCountersHost*>([impl->qCounters contents]) : nullptr;
                    const bool overflowed = ctrHost && ctrHost->overflow;
                    if (!overflowed) {
                        scalable_ccd::logger().warn("Metal EE queue root failed without overflow (base={} size={} attempt={})",
                                                     static_cast<uint64_t>(base),
                                                     static_cast<uint64_t>(cur),
                                                     attempt);
                        break;
                    }
                    if (domainCapacity >= SIZE_MAX / 2) {
                        scalable_ccd::logger().warn("Metal EE queue domain capacity saturated (base={} size={} cap={})",
                                                     static_cast<uint64_t>(base),
                                                     static_cast<uint64_t>(cur),
                                                     static_cast<uint64_t>(domainCapacity));
                        break;
                    }
                    domainCapacity *= 2;
                    scalable_ccd::logger().warn("Metal EE queue overflow, retrying with domainCapacity={} (attempt {}/{})",
                                                 static_cast<uint64_t>(domainCapacity),
                                                 attempt + 1,
                                                 queueMaxRetries);
                }
                if (!processedViaQueue) {
                    scalable_ccd::logger().warn("Metal EE queue root fallback at chunk base={} size={}",
                                                 static_cast<uint64_t>(base), static_cast<uint64_t>(cur));
                    queueRootModeEE = false;
                }
            }
            if (processedViaQueue) {
                continue;
            }

            id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
            // Root
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:impl->eeRootSkeletonCps];
                [enc setBuffer:impl->pCCD_EE offset:0 atIndex:0];
                [enc setBuffer:impl->sNO  offset:0 atIndex:1];
                [enc setBuffer:bTol offset:0 atIndex:2];
                [enc setBuffer:bIt  offset:0 atIndex:3];
                [enc setBuffer:bAz  offset:0 atIndex:4];
                [enc setBuffer:impl->sToi offset:0 atIndex:5];
                [enc setBuffer:nil  offset:0 atIndex:6];
                [enc setBuffer:nil  offset:0 atIndex:7];
                [enc setBuffer:bRs  offset:0 atIndex:8];
                [enc setBuffer:bTst offset:0 atIndex:9];
                if (bCtr) {
                    uint32_t zero = 0u;
                    memcpy([bCtr contents], &zero, sizeof(uint32_t));
                }
                [enc setBuffer:bFC  offset:0 atIndex:10];
                [enc setBuffer:bCtr offset:0 atIndex:11];
                [enc setBuffer:bUQ  offset:0 atIndex:12];
                MTLSize grid = MTLSizeMake(cur,1,1);
                NSUInteger w = impl->eeRootSkeletonCps.maxTotalThreadsPerThreadgroup;
                if (w == 0) w = 64;
                NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, w) : w;
                if (tgW > cur) tgW = static_cast<NSUInteger>(cur);
                if (tgW == 0) tgW = 1;
                MTLSize tg = MTLSizeMake(tgW, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
            }
            // Reduce
            {
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:impl->reduceMinToiCps];
                [enc setBuffer:impl->sToi offset:0 atIndex:0];
                [enc setBuffer:impl->sNO  offset:0 atIndex:1];
                [enc setBuffer:bMin offset:0 atIndex:2];
                MTLSize grid = MTLSizeMake(cur,1,1);
                NSUInteger w = impl->reduceMinToiCps.maxTotalThreadsPerThreadgroup;
                if (w == 0) w = 64;
                NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, w) : w;
                if (tgW > cur) tgW = static_cast<NSUInteger>(cur);
                if (tgW == 0) tgW = 1;
                MTLSize tg = MTLSizeMake(tgW, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
            }
            [cb commit];
            [cb waitUntilCompleted];
            if (impl->enableTiming) {
                CFTimeInterval t0 = cb.GPUStartTime;
                CFTimeInterval t1 = cb.GPUEndTime;
                if (t0 > 0 && t1 > t0) {
                    double msElapsed = (t1 - t0) * 1000.0;
                    append_timing_csv_if_needed(impl->timingCsvPath, "ee_pack_root_reduce",
                                                static_cast<uint64_t>(cur), 0u, msElapsed);
                }
            }
        }
        uint32_t* umin = static_cast<uint32_t*>([bMin contents]);
        if (!umin) return false;
        union { uint32_t u; float f; } conv2;
        conv2.u = *umin;
        outMinToi = conv2.f;
        outMinToi = std::max(0.0f, std::min(1.0f, outMinToi));
        return true;
    }
}
bool MetalRuntime::narrowAddData(
    const std::vector<std::pair<int,int>>& overlaps,
    float ms,
    std::vector<std::tuple<int,int,float>>& outRecords)
{
    if (!available()) return false;
    @autoreleasepool {
        NSError* err = nil;
        if (!impl->npAddCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            impl->npLib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kNPAddDataKernelSrc]
                                                     options:opts
                                                       error:&err];
            if (!impl->npLib) return false;
            id<MTLFunction> fn = [impl->npLib newFunctionWithName:@"npAddData"];
            if (!fn) return false;
            impl->npAddCps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!impl->npAddCps) return false;
        }
        const size_t n = overlaps.size();
        struct Pair2i { int32_t x; int32_t y; };
        std::vector<Pair2i> buf(n);
        for (size_t i=0;i<n;++i){ buf[i].x=overlaps[i].first; buf[i].y=overlaps[i].second; }
        id<MTLBuffer> bPairs = [impl->device newBufferWithBytes:buf.data() length:sizeof(Pair2i)*n options:MTLResourceStorageModeShared];
        uint32_t n32 = static_cast<uint32_t>(n);
        id<MTLBuffer> bN = [impl->device newBufferWithBytes:&n32 length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMs = [impl->device newBufferWithBytes:&ms length:sizeof(float) options:MTLResourceStorageModeShared];
        struct NPRecord { int32_t aid; int32_t bid; float ms; };
        id<MTLBuffer> bOut = [impl->device newBufferWithLength:sizeof(NPRecord)*n options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl->npAddCps];
        [enc setBuffer:bPairs offset:0 atIndex:0];
        [enc setBuffer:bN    offset:0 atIndex:1];
        [enc setBuffer:bMs   offset:0 atIndex:2];
        [enc setBuffer:bOut  offset:0 atIndex:3];
        MTLSize grid = MTLSizeMake(n,1,1);
        NSUInteger maxW = impl->npAddCps.maxTotalThreadsPerThreadgroup;
        if (maxW == 0) maxW = 64;
        NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, maxW) : maxW;
        if (tgW > n) tgW = static_cast<NSUInteger>(n);
        if (tgW == 0) tgW = 1;
        MTLSize tg = MTLSizeMake(tgW, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (impl->enableTiming) {
            CFTimeInterval t0 = cb.GPUStartTime;
            CFTimeInterval t1 = cb.GPUEndTime;
            if (t1 > t0 && t0 > 0) {
                double msElapsed = (t1 - t0) * 1000.0;
                scalable_ccd::logger().info("Metal NarrowAddData: gpu_time_ms={} count={} tg={}",
                                             msElapsed, static_cast<uint64_t>(n), static_cast<uint64_t>(tg.width));
                append_timing_csv_if_needed(impl->timingCsvPath, "narrow_add_data",
                                            static_cast<uint64_t>(n),
                                            static_cast<uint32_t>(tg.width),
                                            msElapsed);
            }
        }
        // Read back
        outRecords.clear();
        outRecords.reserve(n);
        NPRecord* rec = static_cast<NPRecord*>([bOut contents]);
        for (size_t i=0;i<n;++i){
            outRecords.emplace_back(static_cast<int>(rec[i].aid), static_cast<int>(rec[i].bid), rec[i].ms);
        }
        scalable_ccd::logger().info("Metal NarrowAddData: packed {}", n);
        return true;
    }
}

bool MetalRuntime::narrowPackVF(
    const std::vector<float>& vertices_t0_flat,
    const std::vector<float>& vertices_t1_flat,
    uint32_t num_vertices,
    const std::vector<int32_t>& faces_flat,
    const std::vector<std::pair<int,int>>& overlaps,
    float minimum_separation,
    uint32_t& out_packed_count)
{
    if (!available()) return false;
    @autoreleasepool {
        NSError* err = nil;
        scalable_ccd::logger().debug("Metal narrowPackVF: begin (nV={}, faces={}, overlaps={})",
                                     static_cast<uint64_t>(num_vertices),
                                     static_cast<uint64_t>(faces_flat.size()/3),
                                     static_cast<uint64_t>(overlaps.size()));
        if (!impl->npPackVfCps) {
            if (!impl->npLib) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl->npLib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kNPPackVFKernelSrc]
                                                         options:opts
                                                           error:&err];
                if (!impl->npLib) {
                    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown";
                    scalable_ccd::logger().error("Metal narrowPackVF: newLibraryWithSource failed: {}", msg);
                    return false;
                }
            }
            id<MTLFunction> fn = [impl->npLib newFunctionWithName:@"npPackVF"];
            if (!fn) {
                scalable_ccd::logger().error("Metal narrowPackVF: newFunctionWithName('npPackVF') failed");
                return false;
            }
            impl->npPackVfCps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!impl->npPackVfCps) {
                const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown";
                scalable_ccd::logger().error("Metal narrowPackVF: newComputePipelineState failed: {}", msg);
                return false;
            }
        }
        const size_t nOver = overlaps.size();
        struct Pair2i { int32_t x; int32_t y; };
        std::vector<Pair2i> ovs(nOver);
        for (size_t i=0;i<nOver;++i){ ovs[i].x = overlaps[i].first; ovs[i].y = overlaps[i].second; }
        // Staging (shared)
        id<MTLBuffer> bV0 = [impl->device newBufferWithBytes:vertices_t0_flat.data() length:sizeof(float)*vertices_t0_flat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV1 = [impl->device newBufferWithBytes:vertices_t1_flat.data() length:sizeof(float)*vertices_t1_flat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bNV = [impl->device newBufferWithBytes:&num_vertices length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bF  = [impl->device newBufferWithBytes:faces_flat.data() length:sizeof(int32_t)*faces_flat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bO  = [impl->device newBufferWithBytes:ovs.data() length:sizeof(Pair2i)*ovs.size() options:MTLResourceStorageModeShared];
        uint32_t nOverU32 = static_cast<uint32_t>(nOver);
        id<MTLBuffer> bNO = [impl->device newBufferWithBytes:&nOverU32 length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMs = [impl->device newBufferWithBytes:&minimum_separation length:sizeof(float) options:MTLResourceStorageModeShared];
        // Device-private for heavy readers
        id<MTLBuffer> pV0 = [impl->device newBufferWithLength:sizeof(float)*vertices_t0_flat.size() options:MTLResourceStorageModePrivate];
        id<MTLBuffer> pV1 = [impl->device newBufferWithLength:sizeof(float)*vertices_t1_flat.size() options:MTLResourceStorageModePrivate];
        id<MTLBuffer> pF  = [impl->device newBufferWithLength:sizeof(int32_t)*faces_flat.size() options:MTLResourceStorageModePrivate];
        id<MTLBuffer> pO  = [impl->device newBufferWithLength:sizeof(Pair2i)*ovs.size() options:MTLResourceStorageModePrivate];
        // out buffer
        id<MTLBuffer> bOut = [impl->device newBufferWithLength:sizeof(CCDDataMetalVF)*nOver options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
        // Blit uploads to private
        {
            id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
            [bl copyFromBuffer:bV0 sourceOffset:0 toBuffer:pV0 destinationOffset:0 size:sizeof(float)*vertices_t0_flat.size()];
            [bl copyFromBuffer:bV1 sourceOffset:0 toBuffer:pV1 destinationOffset:0 size:sizeof(float)*vertices_t1_flat.size()];
            [bl copyFromBuffer:bF  sourceOffset:0 toBuffer:pF  destinationOffset:0 size:sizeof(int32_t)*faces_flat.size()];
            [bl copyFromBuffer:bO  sourceOffset:0 toBuffer:pO  destinationOffset:0 size:sizeof(Pair2i)*ovs.size()];
            [bl endEncoding];
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl->npPackVfCps];
        [enc setBuffer:pV0 offset:0 atIndex:0];
        [enc setBuffer:pV1 offset:0 atIndex:1];
        [enc setBuffer:bNV offset:0 atIndex:2];
        [enc setBuffer:pF  offset:0 atIndex:3];
        [enc setBuffer:pO  offset:0 atIndex:4];
        [enc setBuffer:bNO offset:0 atIndex:5];
        [enc setBuffer:bMs offset:0 atIndex:6];
        [enc setBuffer:bOut offset:0 atIndex:7];
        MTLSize grid = MTLSizeMake(nOver,1,1);
        NSUInteger maxW = impl->npPackVfCps.maxTotalThreadsPerThreadgroup;
        if (maxW == 0) maxW = 64;
        NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, maxW) : maxW;
        if (tgW > nOver) tgW = static_cast<NSUInteger>(nOver);
        if (tgW == 0) tgW = 1;
        MTLSize tg = MTLSizeMake(tgW, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (impl->enableTiming) {
            CFTimeInterval t0 = cb.GPUStartTime;
            CFTimeInterval t1 = cb.GPUEndTime;
            if (t1 > t0 && t0 > 0) {
                double msElapsed = (t1 - t0) * 1000.0;
                scalable_ccd::logger().info("Metal NarrowPackVF: gpu_time_ms={} count={} tg={}",
                                             msElapsed, static_cast<uint64_t>(nOver), static_cast<uint64_t>(tg.width));
                append_timing_csv_if_needed(impl->timingCsvPath, "narrow_pack_vf",
                                            static_cast<uint64_t>(nOver),
                                            static_cast<uint32_t>(tg.width),
                                            msElapsed);
            }
        }
        out_packed_count = static_cast<uint32_t>(nOver);
        scalable_ccd::logger().info("Metal NarrowPackVF: packed {} records", static_cast<uint64_t>(nOver));
        return true;
    }
}

bool MetalRuntime::narrowPackEE(
    const std::vector<float>& vertices_t0_flat,
    const std::vector<float>& vertices_t1_flat,
    uint32_t num_vertices,
    const std::vector<int32_t>& edges_flat,
    const std::vector<std::pair<int,int>>& overlaps,
    float minimum_separation,
    uint32_t& out_packed_count)
{
    if (!available()) return false;
    @autoreleasepool {
        NSError* err = nil;
        scalable_ccd::logger().debug("Metal narrowPackEE: begin (nV={}, edges={}, overlaps={})",
                                     static_cast<uint64_t>(num_vertices),
                                     static_cast<uint64_t>(edges_flat.size()/2),
                                     static_cast<uint64_t>(overlaps.size()));
        if (!impl->npPackEeCps) {
            if (!impl->npLib) {
                MTLCompileOptions* opts = [MTLCompileOptions new];
                impl->npLib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kNPPackEEKernelSrc]
                                                         options:opts
                                                           error:&err];
                if (!impl->npLib) {
                    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown";
                    scalable_ccd::logger().error("Metal narrowPackEE: newLibraryWithSource failed: {}", msg);
                    return false;
                }
            }
            id<MTLFunction> fn = [impl->npLib newFunctionWithName:@"npPackEE"];
            if (!fn) {
                scalable_ccd::logger().error("Metal narrowPackEE: newFunctionWithName('npPackEE') failed");
                return false;
            }
            impl->npPackEeCps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!impl->npPackEeCps) {
                const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown";
                scalable_ccd::logger().error("Metal narrowPackEE: newComputePipelineState failed: {}", msg);
                return false;
            }
        }
        const size_t nOver = overlaps.size();
        struct Pair2i { int32_t x; int32_t y; };
        std::vector<Pair2i> ovs(nOver);
        for (size_t i=0;i<nOver;++i){ ovs[i].x = overlaps[i].first; ovs[i].y = overlaps[i].second; }
        // Staging (shared)
        id<MTLBuffer> bV0 = [impl->device newBufferWithBytes:vertices_t0_flat.data() length:sizeof(float)*vertices_t0_flat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV1 = [impl->device newBufferWithBytes:vertices_t1_flat.data() length:sizeof(float)*vertices_t1_flat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bNV = [impl->device newBufferWithBytes:&num_vertices length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bE  = [impl->device newBufferWithBytes:edges_flat.data() length:sizeof(int32_t)*edges_flat.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bO  = [impl->device newBufferWithBytes:ovs.data() length:sizeof(Pair2i)*ovs.size() options:MTLResourceStorageModeShared];
        uint32_t nOverU32 = static_cast<uint32_t>(nOver);
        id<MTLBuffer> bNO = [impl->device newBufferWithBytes:&nOverU32 length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMs = [impl->device newBufferWithBytes:&minimum_separation length:sizeof(float) options:MTLResourceStorageModeShared];
        // Device-private for heavy readers
        id<MTLBuffer> pV0 = [impl->device newBufferWithLength:sizeof(float)*vertices_t0_flat.size() options:MTLResourceStorageModePrivate];
        id<MTLBuffer> pV1 = [impl->device newBufferWithLength:sizeof(float)*vertices_t1_flat.size() options:MTLResourceStorageModePrivate];
        id<MTLBuffer> pE  = [impl->device newBufferWithLength:sizeof(int32_t)*edges_flat.size() options:MTLResourceStorageModePrivate];
        id<MTLBuffer> pO  = [impl->device newBufferWithLength:sizeof(Pair2i)*ovs.size() options:MTLResourceStorageModePrivate];
        // out buffer
        id<MTLBuffer> bOut = [impl->device newBufferWithLength:sizeof(CCDDataMetalEE)*nOver options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
        // Blit uploads to private
        {
            id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
            [bl copyFromBuffer:bV0 sourceOffset:0 toBuffer:pV0 destinationOffset:0 size:sizeof(float)*vertices_t0_flat.size()];
            [bl copyFromBuffer:bV1 sourceOffset:0 toBuffer:pV1 destinationOffset:0 size:sizeof(float)*vertices_t1_flat.size()];
            [bl copyFromBuffer:bE  sourceOffset:0 toBuffer:pE  destinationOffset:0 size:sizeof(int32_t)*edges_flat.size()];
            [bl copyFromBuffer:bO  sourceOffset:0 toBuffer:pO  destinationOffset:0 size:sizeof(Pair2i)*ovs.size()];
            [bl endEncoding];
        }
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl->npPackEeCps];
        [enc setBuffer:pV0 offset:0 atIndex:0];
        [enc setBuffer:pV1 offset:0 atIndex:1];
        [enc setBuffer:bNV offset:0 atIndex:2];
        [enc setBuffer:pE  offset:0 atIndex:3];
        [enc setBuffer:pO  offset:0 atIndex:4];
        [enc setBuffer:bNO offset:0 atIndex:5];
        [enc setBuffer:bMs offset:0 atIndex:6];
        [enc setBuffer:bOut offset:0 atIndex:7];
        MTLSize grid = MTLSizeMake(nOver,1,1);
        NSUInteger maxW = impl->npPackEeCps.maxTotalThreadsPerThreadgroup;
        if (maxW == 0) maxW = 64;
        NSUInteger tgW = impl->tgOverride > 0 ? std::min<NSUInteger>(impl->tgOverride, maxW) : maxW;
        if (tgW > nOver) tgW = static_cast<NSUInteger>(nOver);
        if (tgW == 0) tgW = 1;
        MTLSize tg = MTLSizeMake(tgW, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (impl->enableTiming) {
            CFTimeInterval t0 = cb.GPUStartTime;
            CFTimeInterval t1 = cb.GPUEndTime;
            if (t1 > t0 && t0 > 0) {
                double msElapsed = (t1 - t0) * 1000.0;
                scalable_ccd::logger().info("Metal NarrowPackEE: gpu_time_ms={} count={} tg={}",
                                             msElapsed, static_cast<uint64_t>(nOver), static_cast<uint64_t>(tg.width));
                append_timing_csv_if_needed(impl->timingCsvPath, "narrow_pack_ee",
                                            static_cast<uint64_t>(nOver),
                                            static_cast<uint32_t>(tg.width),
                                            msElapsed);
            }
        }
        out_packed_count = static_cast<uint32_t>(nOver);
        scalable_ccd::logger().info("Metal NarrowPackEE: packed {} EE records", static_cast<uint64_t>(nOver));
        return true;
    }
}
bool MetalRuntime::sweepSingleList(
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
    std::vector<std::pair<int, int>>& outPairs)
{
    if (!available()) return false;
    const size_t n = minX.size();
    if (n == 0) { outPairs.clear(); return true; }
    if (maxX.size()!=n || minY.size()!=n || maxY.size()!=n || minZ.size()!=n || maxZ.size()!=n) return false;
    if (v0.size()!=n || v1.size()!=n || v2.size()!=n) return false;
    @autoreleasepool {
        NSError* err = nil;
        if (!impl->sweepCps) {
            MTLCompileOptions* opts = [MTLCompileOptions new];
            impl->sweepLib = [impl->device newLibraryWithSource:[NSString stringWithUTF8String:kSweepSingleKernelSrc]
                                                       options:opts
                                                         error:&err];
            if (!impl->sweepLib) {
                return false;
            }
            id<MTLFunction> fn = [impl->sweepLib newFunctionWithName:@"sweepSingle"];
            if (!fn) {
                return false;
            }
            impl->sweepCps = [impl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!impl->sweepCps) {
                return false;
            }
        }

        id<MTLBuffer> bMinX = [impl->device newBufferWithBytes:minX.data() length:sizeof(double)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxX = [impl->device newBufferWithBytes:maxX.data() length:sizeof(double)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMinY = [impl->device newBufferWithBytes:minY.data() length:sizeof(double)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxY = [impl->device newBufferWithBytes:maxY.data() length:sizeof(double)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMinZ = [impl->device newBufferWithBytes:minZ.data() length:sizeof(double)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bMaxZ = [impl->device newBufferWithBytes:maxZ.data() length:sizeof(double)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV0   = [impl->device newBufferWithBytes:v0.data()  length:sizeof(int32_t)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV1   = [impl->device newBufferWithBytes:v1.data()  length:sizeof(int32_t)*n options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV2   = [impl->device newBufferWithBytes:v2.data()  length:sizeof(int32_t)*n options:MTLResourceStorageModeShared];
        uint32_t n32 = static_cast<uint32_t>(n);
        id<MTLBuffer> bN    = [impl->device newBufferWithBytes:&n32 length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bPairs= [impl->device newBufferWithLength:sizeof(int32_t)*2*capacity options:MTLResourceStorageModeShared];
        uint32_t zero = 0, cap=capacity;
        id<MTLBuffer> bCount= [impl->device newBufferWithBytes:&zero length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bCap  = [impl->device newBufferWithBytes:&cap length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:impl->sweepCps];
        [enc setBuffer:bMinX offset:0 atIndex:0];
        [enc setBuffer:bMaxX offset:0 atIndex:1];
        [enc setBuffer:bMinY offset:0 atIndex:2];
        [enc setBuffer:bMaxY offset:0 atIndex:3];
        [enc setBuffer:bMinZ offset:0 atIndex:4];
        [enc setBuffer:bMaxZ offset:0 atIndex:5];
        [enc setBuffer:bV0   offset:0 atIndex:6];
        [enc setBuffer:bV1   offset:0 atIndex:7];
        [enc setBuffer:bV2   offset:0 atIndex:8];
        [enc setBuffer:bN    offset:0 atIndex:9];
        [enc setBuffer:bPairs offset:0 atIndex:10];
        [enc setBuffer:bCount offset:0 atIndex:11];
        [enc setBuffer:bCap   offset:0 atIndex:12];
        MTLSize grid = MTLSizeMake(n, 1, 1);
        NSUInteger w = impl->sweepCps.maxTotalThreadsPerThreadgroup;
        if (w == 0) w = 64;
        MTLSize tg = MTLSizeMake(std::min<NSUInteger>(w, n), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        uint32_t count = *static_cast<uint32_t*>([bCount contents]);
        if (count > capacity) count = capacity;
        outPairs.resize(count);
        struct Pair2i { int32_t x; int32_t y; };
        Pair2i* ptr = static_cast<Pair2i*>([bPairs contents]);
        for (uint32_t k = 0; k < count; ++k) {
            outPairs[k] = { ptr[k].x, ptr[k].y };
        }
        return true;
    }
}

} // namespace scalable_ccd::metal

namespace scalable_ccd::metal {

std::string MetalRuntime::device_name() const
{
    if (!impl || !impl->device) return std::string();
    NSString* n = impl->device.name;
    if (!n) return std::string();
    return std::string([n UTF8String] ? [n UTF8String] : "");
}

} // namespace scalable_ccd::metal
