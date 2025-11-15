#pragma once

#include <scalable_ccd/broad_phase/aabb.hpp>
#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>
#include <scalable_ccd/utils/logger.hpp>

#include <scalable_ccd/metal2/broad_phase/aabb.hpp>
#include <scalable_ccd/metal2/runtime/runtime.hpp>

#include <memory>
#include <utility>
#include <vector>
#include <cstdlib>
#include <string>
#include <cctype>
#include <algorithm>

namespace scalable_ccd::metal2 {

// Metal v2 BroadPhase（全新实现骨架）
// 目标：默认“严格正确性模式”，最终结果直接回退 CPU sort_and_sweep；
// 后续逐步引入 GPU 路径（disable/enable 可控），并保持与 truth 对齐。
class BroadPhase {
public:
    BroadPhase() = default;

    void clear()
    {
        d_boxesA.reset();
        d_boxesB.reset();
        is_two_lists = false;
        m_overlaps.clear();
        built_ = false;
        ran_ = false;
    }

    void build(const std::shared_ptr<DeviceAABBs> boxes)
    {
        clear();
        d_boxesA = boxes;
        is_two_lists = false;
        built_ = true;
    }

    void build(const std::shared_ptr<DeviceAABBs> boxesA,
               const std::shared_ptr<DeviceAABBs> boxesB)
    {
        clear();
        d_boxesA = boxesA;
        d_boxesB = boxesB;
        is_two_lists = true;
        built_ = true;
    }

    struct AxisSorted {
        std::vector<AABB> boxes;
        int sort_axis = 0;
    };

    static void sort_boxes_along_axis(int& sort_axis, std::vector<AABB>& boxes)
    {
        sort_along_axis(sort_axis, boxes);
    }

    static void generate_axis_candidates(const std::vector<AABB>& boxes,
                                         int sort_axis,
                                         bool two_lists,
                                         std::vector<std::pair<int,int>>& out)
    {
        out.clear();
        out.reserve(boxes.size());
        auto is_cross = [&](long a, long b)->bool {
            if (!two_lists) return true;
            return (a >= 0 && b < 0) || (a < 0 && b >= 0);
        };
        for (int i=0;i<(int)boxes.size();++i){
            const auto& A = boxes[i];
            for (int j=i+1;j<(int)boxes.size();++j){
                const auto& B = boxes[j];
                if (A.max[sort_axis] < B.min[sort_axis]) break;
                if (is_cross(A.element_id, B.element_id)) {
                    out.emplace_back(i,j);
                }
            }
        }
    }

    static inline bool share_a_vertex(const std::array<long,3>& a, const std::array<long,3>& b)
    {
        return a[0]==b[0]||a[0]==b[1]||a[0]==b[2]
            || a[1]==b[0]||a[1]==b[1]||a[1]==b[2]
            || a[2]==b[0]||a[2]==b[1]||a[2]==b[2];
    }

    // CPU YZ 过滤（不收缩/不扩张），用于观测，不影响最终输出；返回掩码
    static void cpu_filter_yz(const std::vector<AABB>& boxes,
                              const std::vector<std::pair<int,int>>& pairs,
                              bool two_lists,
                              std::vector<uint8_t>& outMask)
    {
        outMask.assign(pairs.size(), 0);
        for (size_t k=0;k<pairs.size();++k){
            const auto& pr = pairs[k];
            const auto& A = boxes[pr.first];
            const auto& B = boxes[pr.second];
            bool overlapY = !(A.max[1] < B.min[1] || A.min[1] > B.max[1]);
            bool overlapZ = true;
            if (A.min.size() >= 3) {
                overlapZ = !(A.max[2] < B.min[2] || A.min[2] > B.max[2]);
            }
            if (!overlapY || !overlapZ) continue;
            // 与 CPU 路径一致：两列表/单列表均剔除共享顶点对
            if (share_a_vertex(A.vertex_ids, B.vertex_ids)) continue;
            outMask[k] = 1;
        }
    }

    // 一次性运行并返回 overlaps；v2 默认严格正确性（CPU）
    std::vector<std::pair<int,int>> detect_overlaps()
    {
        if (!built_) return {};
        if (ran_) return m_overlaps;

        // 环境开关：严格模式（默认开）
        const bool strict = env_flag_enabled("SCALABLE_CCD_METAL2_STRICT", true);
        const bool observe = env_flag_enabled("SCALABLE_CCD_METAL2_OBSERVE", true);
        // 预留：将来可逐步导入 GPU 路径；当前严格模式恒走 CPU。

        int sort_axis = 0;
        if (is_two_lists) {
            std::vector<AABB> a = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
            std::vector<AABB> b = d_boxesB ? d_boxesB->h_boxes : std::vector<AABB>{};
            if (observe && !a.empty() && !b.empty()) {
                // 观测：轴向候选 + CPU YZ filter 统计
                sort_boxes_along_axis(sort_axis, a);
                sort_boxes_along_axis(sort_axis, b);
                // Flip A ids and merge（与 CPU 路径一致）
                for (auto& ax : a) ax.element_id = -ax.element_id - 1;
                std::vector<AABB> merged(a.size()+b.size());
                auto less_min = [=](const AABB& x, const AABB& y){ return x.min[sort_axis] < y.min[sort_axis]; };
                std::merge(a.begin(), a.end(), b.begin(), b.end(), merged.begin(), less_min);
                std::vector<std::pair<int,int>> pairs;
                generate_axis_candidates(merged, sort_axis, /*two_lists*/true, pairs);
                // 准备 GPU/CPU YZ 过滤输入
                std::vector<float> minY(merged.size()), maxY(merged.size());
                std::vector<float> minZ(merged.size()), maxZ(merged.size());
                std::vector<int32_t> v0(merged.size()), v1(merged.size()), v2(merged.size());
                for (size_t i=0;i<merged.size();++i){
                    minY[i] = static_cast<float>(merged[i].min[1]);
                    maxY[i] = static_cast<float>(merged[i].max[1]);
                    // Z 维可能不存在（2D），做保护
                    double miZ = merged[i].min.size() >= 3 ? merged[i].min[2] : merged[i].min[1];
                    double maZ = merged[i].max.size() >= 3 ? merged[i].max[2] : merged[i].max[1];
                    minZ[i] = static_cast<float>(miZ);
                    maxZ[i] = static_cast<float>(maZ);
                    v0[i] = static_cast<int32_t>(merged[i].vertex_ids[0]);
                    v1[i] = static_cast<int32_t>(merged[i].vertex_ids[1]);
                    v2[i] = static_cast<int32_t>(merged[i].vertex_ids[2]);
                }
                // 选择过滤器：环境变量 SCALABLE_CCD_METAL2_FILTER={off|cpu|gpu}
                auto filterSel = [](const char* key)->int{
                    const char* v = std::getenv(key);
                    if (!v) return 1; // cpu by default
                    std::string s(v);
                    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    if (s=="off") return 0;
                    if (s=="gpu") return 2;
                    return 1; // cpu
                };
                const int sel = filterSel("SCALABLE_CCD_METAL2_FILTER");
                std::vector<uint8_t> mask;
                bool ok = false;
                if (sel == 2 && Metal2Runtime::instance().available() && Metal2Runtime::instance().warmup()) {
                    ok = Metal2Runtime::instance().filterYZ(minY,maxY,minZ,maxZ,v0,v1,v2,pairs,/*two_lists*/true, mask);
                }
                if (!ok) {
                    cpu_filter_yz(merged, pairs, /*two_lists*/true, mask);
                }
                size_t kept = 0; for (auto m:mask) if (m) ++kept;
                logger().info("Metal2 Broad-phase OBS(two): axis_pairs={} yz_kept={}", pairs.size(), kept);
                // 与最终集合做差集（最终集合稍后计算）
                }
            // 最终结果（严格模式下回退 CPU）
            sort_and_sweep(std::move(a), std::move(b), sort_axis, m_overlaps);
            if (observe) {
                // 生成与 CPU 最终结果的差集统计
                // 需要与上面的 merged/pairs/mask一致，故再次构造
                sort_boxes_along_axis(sort_axis, a = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{});
                sort_boxes_along_axis(sort_axis, b = d_boxesB ? d_boxesB->h_boxes : std::vector<AABB>{});
                for (auto& ax : a) ax.element_id = -ax.element_id - 1;
                std::vector<AABB> merged(a.size()+b.size());
                auto less_min = [=](const AABB& x, const AABB& y){ return x.min[sort_axis] < y.min[sort_axis]; };
                std::merge(a.begin(), a.end(), b.begin(), b.end(), merged.begin(), less_min);
                std::vector<std::pair<int,int>> pairs;
                generate_axis_candidates(merged, sort_axis, /*two_lists*/true, pairs);
                std::vector<uint8_t> mask;
                cpu_filter_yz(merged, pairs, /*two_lists*/true, mask);
                // 规范化为 element_id 对（小到大）
                std::vector<std::pair<int,int>> yzPairs;
                yzPairs.reserve(pairs.size());
                for (size_t i=0;i<pairs.size();++i){
                    if (!mask[i]) continue;
                    const auto& A = merged[pairs[i].first];
                    const auto& B = merged[pairs[i].second];
                    const bool a_from_A = (A.element_id < 0);
                    int ida = a_from_A ? -static_cast<int>(A.element_id) - 1 : static_cast<int>(A.element_id);
                    int idb = (B.element_id < 0) ? -static_cast<int>(B.element_id) - 1 : static_cast<int>(B.element_id);
                    int aN = std::min(ida, idb);
                    int bN = std::max(ida, idb);
                    yzPairs.emplace_back(aN, bN);
                }
                std::sort(yzPairs.begin(), yzPairs.end());
                yzPairs.erase(std::unique(yzPairs.begin(), yzPairs.end()), yzPairs.end());
                auto finalPairs = m_overlaps;
                std::sort(finalPairs.begin(), finalPairs.end());
                std::vector<std::pair<int,int>> inter, yz_only, final_only;
                std::set_intersection(yzPairs.begin(), yzPairs.end(), finalPairs.begin(), finalPairs.end(), std::back_inserter(inter));
                std::set_difference(yzPairs.begin(), yzPairs.end(), finalPairs.begin(), finalPairs.end(), std::back_inserter(yz_only));
                std::set_difference(finalPairs.begin(), finalPairs.end(), yzPairs.begin(), yzPairs.end(), std::back_inserter(final_only));
                logger().info("Metal2 Broad-phase OBS vs FINAL(two): inter={} yz_only={} final_only={}", inter.size(), yz_only.size(), final_only.size());
                // 打印样本，便于边界问题定位
                auto print_sample_two = [&](const std::pair<int,int>& pr, const char* tag){
                    const AABB* AA = nullptr; const AABB* BB = nullptr;
                    // 在原始 A/B 列表中查找匹配 element_id
                    if (d_boxesA) {
                        for (const auto& bx : d_boxesA->h_boxes) {
                            if (static_cast<int>(bx.element_id) == pr.first) { AA = &bx; break; }
                        }
                    }
                    if (d_boxesB) {
                        for (const auto& bx : d_boxesB->h_boxes) {
                            if (static_cast<int>(bx.element_id) == pr.second) { BB = &bx; break; }
                        }
                    }
                    if (!AA || !BB) {
                        logger().info("Metal2 OBS(two) sample[{}]: (a={},b={}) boxes not found", tag, pr.first, pr.second);
                        return;
                    }
                    auto rng = [](const AABB* P)->std::pair<double,double>{ return { P->max[1]-P->min[1], (P->min.size()>=3?P->max[2]-P->min[2]:P->max[1]-P->min[1]) }; };
                    auto [yA,zA] = rng(AA); auto [yB,zB] = rng(BB);
                    logger().info("Metal2 OBS(two) sample[{}]: (a={},b={}) Ay=[{:.6g},{:.6g}] By=[{:.6g},{:.6g}] Az=[{:.6g},{:.6g}] Bz=[{:.6g},{:.6g}]",
                        tag, pr.first, pr.second,
                        (double)AA->min[1], (double)AA->max[1], (double)BB->min[1], (double)BB->max[1],
                        (double)(AA->min.size()>=3?AA->min[2]:AA->min[1]), (double)(AA->max.size()>=3?AA->max[2]:AA->max[1]),
                        (double)(BB->min.size()>=3?BB->min[2]:BB->min[1]), (double)(BB->max.size()>=3?BB->max[2]:BB->max[1]));
                };
                auto env_sample = []()->int{
                    const char* v = std::getenv("SCALABLE_CCD_METAL2_SAMPLE");
                    if (!v) return 5;
                    try { int k = std::stoi(v); return k>0 ? k : 5; } catch(...) { return 5; }
                };
                int sample_n = env_sample();
                int c1 = 0; for (const auto& p : yz_only) { if (c1++ >= sample_n) break; print_sample_two(p, "yz_only"); }
                int c2 = 0; for (const auto& p : final_only) { if (c2++ >= sample_n) break; print_sample_two(p, "final_only"); }
            }
        } else {
            std::vector<AABB> a = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
            if (observe && !a.empty()) {
                sort_boxes_along_axis(sort_axis, a);
                std::vector<std::pair<int,int>> pairs;
                generate_axis_candidates(a, sort_axis, /*two_lists*/false, pairs);
                std::vector<float> minY(a.size()), maxY(a.size());
                std::vector<float> minZ(a.size()), maxZ(a.size());
                std::vector<int32_t> v0(a.size()), v1(a.size()), v2(a.size());
                for (size_t i=0;i<a.size();++i){
                    minY[i] = static_cast<float>(a[i].min[1]);
                    maxY[i] = static_cast<float>(a[i].max[1]);
                    double miZ = a[i].min.size() >= 3 ? a[i].min[2] : a[i].min[1];
                    double maZ = a[i].max.size() >= 3 ? a[i].max[2] : a[i].max[1];
                    minZ[i] = static_cast<float>(miZ);
                    maxZ[i] = static_cast<float>(maZ);
                    v0[i] = static_cast<int32_t>(a[i].vertex_ids[0]);
                    v1[i] = static_cast<int32_t>(a[i].vertex_ids[1]);
                    v2[i] = static_cast<int32_t>(a[i].vertex_ids[2]);
                }
                const int sel = [](){ const char* v=getenv("SCALABLE_CCD_METAL2_FILTER"); if(!v) return 1; std::string s(v); for(auto&c:s) c=(char)tolower((unsigned char)c); if(s=="gpu") return 2; if(s=="off") return 0; return 1;}();
                std::vector<uint8_t> mask;
                bool ok=false;
                if (sel==2 && Metal2Runtime::instance().available() && Metal2Runtime::instance().warmup()){
                    ok = Metal2Runtime::instance().filterYZ(minY,maxY,minZ,maxZ,v0,v1,v2,pairs,false,mask);
                }
                if (!ok){
                    cpu_filter_yz(a, pairs, /*two_lists*/false, mask);
                }
                size_t kept = 0; for (auto m:mask) if (m) ++kept;
                logger().info("Metal2 Broad-phase OBS(single): axis_pairs={} yz_kept={}", pairs.size(), kept);
            }
            sort_and_sweep(std::move(a), sort_axis, m_overlaps);
            if (observe) {
                // 规范化并做差集统计
                std::vector<AABB> a2 = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
                sort_boxes_along_axis(sort_axis, a2);
                std::vector<std::pair<int,int>> pairs;
                generate_axis_candidates(a2, sort_axis, /*two_lists*/false, pairs);
                std::vector<uint8_t> mask;
                cpu_filter_yz(a2, pairs, /*two_lists*/false, mask);
                std::vector<std::pair<int,int>> yzPairs;
                yzPairs.reserve(pairs.size());
                for (size_t i=0;i<pairs.size();++i){
                    if (!mask[i]) continue;
                    const auto& A = a2[pairs[i].first];
                    const auto& B = a2[pairs[i].second];
                    int ida = static_cast<int>(A.element_id);
                    int idb = static_cast<int>(B.element_id);
                    yzPairs.emplace_back(std::min(ida,idb), std::max(ida,idb));
                }
                std::sort(yzPairs.begin(), yzPairs.end());
                yzPairs.erase(std::unique(yzPairs.begin(), yzPairs.end()), yzPairs.end());
                auto finalPairs = m_overlaps;
                std::sort(finalPairs.begin(), finalPairs.end());
                std::vector<std::pair<int,int>> inter, yz_only, final_only;
                std::set_intersection(yzPairs.begin(), yzPairs.end(), finalPairs.begin(), finalPairs.end(), std::back_inserter(inter));
                std::set_difference(yzPairs.begin(), yzPairs.end(), finalPairs.begin(), finalPairs.end(), std::back_inserter(yz_only));
                std::set_difference(finalPairs.begin(), finalPairs.end(), yzPairs.begin(), yzPairs.end(), std::back_inserter(final_only));
                logger().info("Metal2 Broad-phase OBS vs FINAL(single): inter={} yz_only={} final_only={}", inter.size(), yz_only.size(), final_only.size());
                auto find_by_id = [&](int id)->const AABB*{
                    if (d_boxesA) {
                        for (const auto& bx : d_boxesA->h_boxes) if ((int)bx.element_id==id) return &bx;
                    }
                    return nullptr;
                };
                auto print_sample_single = [&](const std::pair<int,int>& pr, const char* tag){
                    const AABB* A = find_by_id(pr.first);
                    const AABB* B = find_by_id(pr.second);
                    if (!A || !B) {
                        logger().info("Metal2 OBS(single) sample[{}]: (a={},b={}) boxes not found", tag, pr.first, pr.second);
                        return;
                    }
                    logger().info("Metal2 OBS(single) sample[{}]: (a={},b={}) Ay=[{:.6g},{:.6g}] By=[{:.6g},{:.6g}] Az=[{:.6g},{:.6g}] Bz=[{:.6g},{:.6g}]",
                        tag, pr.first, pr.second,
                        (double)A->min[1], (double)A->max[1], (double)B->min[1], (double)B->max[1],
                        (double)(A->min.size()>=3?A->min[2]:A->min[1]), (double)(A->max.size()>=3?A->max[2]:A->max[1]),
                        (double)(B->min.size()>=3?B->min[2]:B->min[1]), (double)(B->max.size()>=3?B->max[2]:B->max[1]));
                };
                auto env_sample = []()->int{
                    const char* v = std::getenv("SCALABLE_CCD_METAL2_SAMPLE");
                    if (!v) return 5;
                    try { int k = std::stoi(v); return k>0 ? k : 5; } catch(...) { return 5; }
                };
                int sample_n = env_sample();
                int c1 = 0; for (const auto& p : yz_only) { if (c1++ >= sample_n) break; print_sample_single(p, "yz_only"); }
                int c2 = 0; for (const auto& p : final_only) { if (c2++ >= sample_n) break; print_sample_single(p, "final_only"); }
            }
        }
        ran_ = true;
        return m_overlaps;
    }

    const std::vector<std::pair<int,int>>& overlaps() const { return m_overlaps; }

private:
    static bool env_flag_enabled(const char* key, bool def)
    {
        const char* v = std::getenv(key);
        if (!v) return def;
        std::string s(v);
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (s.empty()) return true;
        if (s=="0" || s=="false" || s=="off" || s=="no") return false;
        return true;
    }

    std::shared_ptr<DeviceAABBs> d_boxesA;
    std::shared_ptr<DeviceAABBs> d_boxesB;
    bool is_two_lists = false;
    std::vector<std::pair<int,int>> m_overlaps;
    bool built_ = false;
    bool ran_ = false;
};

} // namespace scalable_ccd::metal2
