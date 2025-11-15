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
#include <chrono>

namespace scalable_ccd::metal2 {

// Metal v2 BroadPhase（全新实现骨架）
// 目标：默认“严格正确性模式”，最终结果直接回退 CPU sort_and_sweep；
// 后续逐步引入 GPU 路径（disable/enable 可控），并保持与 truth 对齐。
class BroadPhase {
public:
    BroadPhase() = default;

    struct Timing {
        // 细分计时（毫秒）；<0 表示未采集
        double total_ms = -1.0;
        double axis_merge_ms = -1.0; // 轴向排序/合并/准备数据
        double pairs_ms = -1.0;      // 候选生成（GPU STQ 或 CPU）
        double filter_ms = -1.0;     // YZ 过滤（GPU 或 CPU）
        double compose_ms = -1.0;    // 规范化/去重/排序
        int sort_axis = 0;
        std::string pairs_src = "none";   // "gpu" | "cpu" | "none"
        std::string filter_src = "none";  // "gpu" | "cpu" | "off"
    };

    void clear()
    {
        d_boxesA.reset();
        d_boxesB.reset();
        is_two_lists = false;
        m_overlaps.clear();
        built_ = false;
        ran_ = false;
        last_timing_ = Timing{};
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
            // 3D 轴向重叠：X/Y/Z 都用闭区间
            bool overlapX = !(A.max[0] < B.min[0] || A.min[0] > B.max[0]);
            bool overlapY = !(A.max[1] < B.min[1] || A.min[1] > B.max[1]);
            bool overlapZ = true;
            if (A.min.size() >= 3) {
                overlapZ = !(A.max[2] < B.min[2] || A.min[2] > B.max[2]);
            }
            if (!overlapX || !overlapY || !overlapZ) continue;
            // 与 CPU 路径一致：两列表/单列表均剔除共享顶点对
            if (share_a_vertex(A.vertex_ids, B.vertex_ids)) continue;
            outMask[k] = 1;
        }
    }

    // 一次性运行并返回 overlaps；v2 默认严格正确性（CPU）
    std::vector<std::pair<int,int>> detect_overlaps()
    {
        using Clock = std::chrono::high_resolution_clock;
        if (!built_) return {};
        if (ran_) return m_overlaps;

        // 环境开关：严格模式（默认开）
        const bool strict = env_flag_enabled("SCALABLE_CCD_METAL2_STRICT", true);
        const bool observe = env_flag_enabled("SCALABLE_CCD_METAL2_OBSERVE", true);
        const bool log_timing = env_flag_enabled("SCALABLE_CCD_METAL2_LOG_TIMING", false);
        // 预留：将来可逐步导入 GPU 路径；当前严格模式恒走 CPU。

        int sort_axis = 0;
        if (is_two_lists) {
            auto t_total_start = Clock::now();
            std::vector<AABB> a = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
            std::vector<AABB> b = d_boxesB ? d_boxesB->h_boxes : std::vector<AABB>{};
            if (observe && !a.empty() && !b.empty()) {
                // 观测：轴向候选 + CPU YZ filter 统计
                std::vector<AABB> a_obs = a;
                std::vector<AABB> b_obs = b;
                // 为了 GPU STQ 的正确性，这里统一按 X 轴(min[0])排序与合并
                auto less_min_x = [](const AABB& x, const AABB& y){ return x.min[0] < y.min[0]; };
                std::sort(a_obs.begin(), a_obs.end(), less_min_x);
                std::sort(b_obs.begin(), b_obs.end(), less_min_x);
                // Flip A ids and merge
                for (auto& ax : a_obs) ax.element_id = -ax.element_id - 1;
                std::vector<AABB> merged(a_obs.size()+b_obs.size());
                std::merge(a_obs.begin(), a_obs.end(), b_obs.begin(), b_obs.end(), merged.begin(), less_min_x);
                std::vector<std::pair<int,int>> pairs;
                bool stq_ok = false;
                if (Metal2Runtime::instance().available()) {
                    std::vector<double> minX(merged.size()), maxX(merged.size());
                    std::vector<double> minYd(merged.size()), maxYd(merged.size());
                    std::vector<double> minZd(merged.size()), maxZd(merged.size());
                    std::vector<int32_t> v0(merged.size()), v1(merged.size()), v2(merged.size());
                    std::vector<uint8_t> listTag(merged.size());
                    for (size_t i=0;i<merged.size();++i){
                        minX[i] = merged[i].min[0]; maxX[i] = merged[i].max[0];
                        minYd[i] = merged[i].min[1]; maxYd[i] = merged[i].max[1];
                        double miZ = merged[i].min.size() >= 3 ? merged[i].min[2] : merged[i].min[1];
                        double maZ = merged[i].max.size() >= 3 ? merged[i].max[2] : merged[i].max[1];
                        minZd[i] = miZ; maxZd[i] = maZ;
                        v0[i] = static_cast<int32_t>(merged[i].vertex_ids[0]);
                        v1[i] = static_cast<int32_t>(merged[i].vertex_ids[1]);
                        v2[i] = static_cast<int32_t>(merged[i].vertex_ids[2]);
                        listTag[i] = merged[i].element_id < 0 ? 1 : 0;
                    }
                    stq_ok = Metal2Runtime::instance().stqTwoLists(minX,maxX,minYd,maxYd,minZd,maxZd,v0,v1,v2,listTag,pairs);
                }
                if (!stq_ok) {
                    // 回退 CPU 候选（同样沿 X 轴）
                    generate_axis_candidates(merged, /*sort_axis*/0, /*two_lists*/true, pairs);
                }
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
            // 最终结果：严格或未启用 STQ -> 回退 CPU；否则使用 轴候选+过滤 产出
            if (!strict && env_flag_enabled("SCALABLE_CCD_METAL2_USE_STQ", false)) {
                auto t_axis_start = Clock::now();
                // 使用与观测一致的管线生成最终输出（沿 X 轴排序与合并）
                std::vector<AABB> a_final = a;
                std::vector<AABB> b_final = b;
                auto less_min_x = [](const AABB& x, const AABB& y){ return x.min[0] < y.min[0]; };
                std::sort(a_final.begin(), a_final.end(), less_min_x);
                std::sort(b_final.begin(), b_final.end(), less_min_x);
                for (auto& ax : a_final) ax.element_id = -ax.element_id - 1;
                std::vector<AABB> merged(a_final.size()+b_final.size());
                std::merge(a_final.begin(), a_final.end(), b_final.begin(), b_final.end(), merged.begin(), less_min_x);
                auto t_axis_end = Clock::now();
                std::vector<std::pair<int,int>> pairs;
                // 优先 GPU STQ 候选生成
                bool stq_ok = false;
                last_timing_.pairs_src = "cpu";
                if (Metal2Runtime::instance().available()) {
                    std::vector<double> minX(merged.size()), maxX(merged.size());
                    std::vector<double> minYd(merged.size()), maxYd(merged.size());
                    std::vector<double> minZd(merged.size()), maxZd(merged.size());
                    std::vector<int32_t> v0(merged.size()), v1(merged.size()), v2(merged.size());
                    std::vector<uint8_t> listTag(merged.size());
                    for (size_t i=0;i<merged.size();++i){
                        minX[i] = merged[i].min[0]; maxX[i] = merged[i].max[0];
                        minYd[i] = merged[i].min[1]; maxYd[i] = merged[i].max[1];
                        double miZ = merged[i].min.size() >= 3 ? merged[i].min[2] : merged[i].min[1];
                        double maZ = merged[i].max.size() >= 3 ? merged[i].max[2] : merged[i].max[1];
                        minZd[i] = miZ; maxZd[i] = maZ;
                        v0[i] = static_cast<int32_t>(merged[i].vertex_ids[0]);
                        v1[i] = static_cast<int32_t>(merged[i].vertex_ids[1]);
                        v2[i] = static_cast<int32_t>(merged[i].vertex_ids[2]);
                        listTag[i] = merged[i].element_id < 0 ? 1 : 0;
                    }
                    stq_ok = Metal2Runtime::instance().stqTwoLists(minX,maxX,minYd,maxYd,minZd,maxZd,v0,v1,v2,listTag,pairs);
                    if (stq_ok) last_timing_.pairs_src = "gpu";
                }
                auto t_pairs_cpu_start = Clock::now();
                if (!stq_ok){
                    generate_axis_candidates(merged, /*sort_axis*/0, /*two_lists*/true, pairs);
                }
                auto t_pairs_cpu_end = Clock::now();
                std::vector<uint8_t> mask;
                bool ok=false;
                // 过滤使用 CPU 或 GPU（默认 CPU）
                std::vector<float> minY(merged.size()), maxY(merged.size());
                std::vector<float> minZ(merged.size()), maxZ(merged.size());
                std::vector<int32_t> v0(merged.size()), v1(merged.size()), v2(merged.size());
                for (size_t i=0;i<merged.size();++i){
                    minY[i] = static_cast<float>(merged[i].min[1]);
                    maxY[i] = static_cast<float>(merged[i].max[1]);
                    double miZ = merged[i].min.size() >= 3 ? merged[i].min[2] : merged[i].min[1];
                    double maZ = merged[i].max.size() >= 3 ? merged[i].max[2] : merged[i].max[1];
                    minZ[i] = static_cast<float>(miZ);
                    maxZ[i] = static_cast<float>(maZ);
                    v0[i] = static_cast<int32_t>(merged[i].vertex_ids[0]);
                    v1[i] = static_cast<int32_t>(merged[i].vertex_ids[1]);
                    v2[i] = static_cast<int32_t>(merged[i].vertex_ids[2]);
                }
                const int sel = [](){ const char* v=getenv("SCALABLE_CCD_METAL2_FILTER"); if(!v) return 1; std::string s(v); for(auto&c:s) c=(char)tolower((unsigned char)c); if(s=="gpu") return 2; if(s=="off") return 0; return 1;}();
                auto t_filter_cpu_start = Clock::now();
                if (sel==2 && Metal2Runtime::instance().available() && Metal2Runtime::instance().warmup()){
                    ok = Metal2Runtime::instance().filterYZ(minY,maxY,minZ,maxZ,v0,v1,v2,pairs,true,mask);
                    if (ok) last_timing_.filter_src = "gpu"; else last_timing_.filter_src = "cpu";
                }
                if (!ok){
                    cpu_filter_yz(merged, pairs, /*two_lists*/true, mask);
                    last_timing_.filter_src = "cpu";
                }
                auto t_filter_cpu_end = Clock::now();
                auto t_compose_start = Clock::now();
                m_overlaps.clear();
                m_overlaps.reserve(pairs.size());
                for (size_t i=0;i<pairs.size();++i){
                    if (!mask[i]) continue;
                    const AABB& A0 = merged[pairs[i].first];
                    const AABB& B0 = merged[pairs[i].second];
                    const bool a_from_A = (A0.element_id < 0);
                    const int aid = a_from_A ? (-static_cast<int>(A0.element_id) - 1)
                                             : (-static_cast<int>(B0.element_id) - 1);
                    const int bid = a_from_A ? static_cast<int>(B0.element_id)
                                             : static_cast<int>(A0.element_id);
                    m_overlaps.emplace_back(aid, bid);
                }
                std::sort(m_overlaps.begin(), m_overlaps.end());
                m_overlaps.erase(std::unique(m_overlaps.begin(), m_overlaps.end()), m_overlaps.end());
                auto t_compose_end = Clock::now();
                // 记录计时
                last_timing_.sort_axis = 0;
                last_timing_.axis_merge_ms = std::chrono::duration<double, std::milli>(t_axis_end - t_axis_start).count();
                if (last_timing_.pairs_src == "gpu") {
                    last_timing_.pairs_ms = Metal2Runtime::instance().lastSTQPairsMs();
                } else {
                    last_timing_.pairs_ms = std::chrono::duration<double, std::milli>(t_pairs_cpu_end - t_pairs_cpu_start).count();
                }
                if (last_timing_.filter_src == "gpu") {
                    last_timing_.filter_ms = Metal2Runtime::instance().lastYZFilterMs();
                } else if (sel != 0) {
                    last_timing_.filter_ms = std::chrono::duration<double, std::milli>(t_filter_cpu_end - t_filter_cpu_start).count();
                } else {
                    last_timing_.filter_src = "off";
                    last_timing_.filter_ms = 0.0;
                }
                last_timing_.compose_ms = std::chrono::duration<double, std::milli>(t_compose_end - t_compose_start).count();
                last_timing_.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_total_start).count();
                if (log_timing) {
                    logger().info("Metal2 Timing(two): axis_merge_ms={:.3f} pairs_ms({})={:.3f} filter_ms({})={:.3f} compose_ms={:.3f} total_ms={:.3f}",
                                  last_timing_.axis_merge_ms,
                                  last_timing_.pairs_src.c_str(), last_timing_.pairs_ms,
                                  last_timing_.filter_src.c_str(), last_timing_.filter_ms,
                                  last_timing_.compose_ms, last_timing_.total_ms);
                }
            } else {
                sort_and_sweep(std::move(a), std::move(b), sort_axis, m_overlaps);
            }
            if (observe) {
                // 生成与 CPU 最终结果的差集统计
                // 与最终路径保持一致：统一沿 X 轴排序与合并
                a = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
                b = d_boxesB ? d_boxesB->h_boxes : std::vector<AABB>{};
                auto less_min_x = [](const AABB& x, const AABB& y){ return x.min[0] < y.min[0]; };
                std::sort(a.begin(), a.end(), less_min_x);
                std::sort(b.begin(), b.end(), less_min_x);
                for (auto& ax : a) ax.element_id = -ax.element_id - 1;
                std::vector<AABB> merged(a.size()+b.size());
                std::merge(a.begin(), a.end(), b.begin(), b.end(), merged.begin(), less_min_x);
                std::vector<std::pair<int,int>> pairs;
                generate_axis_candidates(merged, /*sort_axis*/0, /*two_lists*/true, pairs);
                std::vector<uint8_t> mask;
                cpu_filter_yz(merged, pairs, /*two_lists*/true, mask);
                // 规范化为 element_id 对（双列表：有向对 <aid from A, bid from B>）
                std::vector<std::pair<int,int>> yzPairs;
                yzPairs.reserve(pairs.size());
                for (size_t i=0;i<pairs.size();++i){
                    if (!mask[i]) continue;
                    const auto& A = merged[pairs[i].first];
                    const auto& B = merged[pairs[i].second];
                    // exactly one is negative (from list A)
                    const bool a_from_A = (A.element_id < 0);
                    const int aid = a_from_A ? (-static_cast<int>(A.element_id) - 1)
                                             : (-static_cast<int>(B.element_id) - 1);
                    const int bid = a_from_A ? static_cast<int>(B.element_id)
                                             : static_cast<int>(A.element_id);
                    yzPairs.emplace_back(aid, bid);
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
            auto t_total_start = Clock::now();
            std::vector<AABB> a = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
            const bool use_stq = (!strict && env_flag_enabled("SCALABLE_CCD_METAL2_USE_STQ", false));
            if (observe && !a.empty()) {
                // OBS 统计：若使用 STQ，与最终路径保持一致（沿 X 轴排序）
                if (use_stq) {
                    auto less_min_x = [](const AABB& x, const AABB& y){ return x.min[0] < y.min[0]; };
                    std::sort(a.begin(), a.end(), less_min_x);
                    sort_axis = 0;
                } else {
                    sort_boxes_along_axis(sort_axis, a);
                }
                std::vector<std::pair<int,int>> pairs;
                generate_axis_candidates(a, /*sort_axis*/use_stq?0:sort_axis, /*two_lists*/false, pairs);
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
            if (!use_stq) {
                sort_and_sweep(std::move(a), sort_axis, m_overlaps);
            } else {
                auto t_axis_start = Clock::now();
                // 轴向排序（沿 X）
                auto less_min_x = [](const AABB& x, const AABB& y){ return x.min[0] < y.min[0]; };
                std::sort(a.begin(), a.end(), less_min_x);
                sort_axis = 0;
                auto t_axis_end = Clock::now();
                // 候选生成：GPU 优先
                std::vector<std::pair<int,int>> pairs;
                bool stq_ok = false;
                last_timing_.pairs_src = "cpu";
                if (Metal2Runtime::instance().available()) {
                    std::vector<double> minX(a.size()), maxX(a.size());
                    std::vector<double> minYd(a.size()), maxYd(a.size());
                    std::vector<double> minZd(a.size()), maxZd(a.size());
                    std::vector<int32_t> v0(a.size()), v1(a.size()), v2(a.size());
                    for (size_t i=0;i<a.size();++i){
                        minX[i] = a[i].min[0]; maxX[i] = a[i].max[0];
                        minYd[i] = a[i].min[1]; maxYd[i] = a[i].max[1];
                        double miZ = a[i].min.size() >= 3 ? a[i].min[2] : a[i].min[1];
                        double maZ = a[i].max.size() >= 3 ? a[i].max[2] : a[i].max[1];
                        minZd[i] = miZ; maxZd[i] = maZ;
                        v0[i] = static_cast<int32_t>(a[i].vertex_ids[0]);
                        v1[i] = static_cast<int32_t>(a[i].vertex_ids[1]);
                        v2[i] = static_cast<int32_t>(a[i].vertex_ids[2]);
                    }
                    stq_ok = Metal2Runtime::instance().stqSingleList(minX,maxX,minYd,maxYd,minZd,maxZd,v0,v1,v2,pairs);
                    if (stq_ok) last_timing_.pairs_src = "gpu";
                }
                auto t_pairs_cpu_start = Clock::now();
                if (!stq_ok) {
                    generate_axis_candidates(a, /*sort_axis*/0, /*two_lists*/false, pairs);
                }
                auto t_pairs_cpu_end = Clock::now();
                // 过滤（默认 CPU）
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
                auto t_filter_cpu_start = Clock::now();
                if (sel==2 && Metal2Runtime::instance().available() && Metal2Runtime::instance().warmup()){
                    ok = Metal2Runtime::instance().filterYZ(minY,maxY,minZ,maxZ,v0,v1,v2,pairs,false,mask);
                    if (ok) last_timing_.filter_src = "gpu"; else last_timing_.filter_src = "cpu";
                }
                if (!ok) {
                    cpu_filter_yz(a, pairs, /*two_lists*/false, mask);
                    last_timing_.filter_src = "cpu";
                }
                auto t_filter_cpu_end = Clock::now();
                // compose
                auto t_compose_start = Clock::now();
                m_overlaps.clear();
                m_overlaps.reserve(pairs.size());
                for (size_t i=0;i<pairs.size();++i){
                    if (!mask[i]) continue;
                    const auto& A = a[pairs[i].first];
                    const auto& B = a[pairs[i].second];
                    int ida = static_cast<int>(A.element_id);
                    int idb = static_cast<int>(B.element_id);
                    m_overlaps.emplace_back(std::min(ida,idb), std::max(ida,idb));
                }
                std::sort(m_overlaps.begin(), m_overlaps.end());
                m_overlaps.erase(std::unique(m_overlaps.begin(), m_overlaps.end()), m_overlaps.end());
                auto t_compose_end = Clock::now();

                last_timing_.sort_axis = sort_axis;
                last_timing_.axis_merge_ms = std::chrono::duration<double, std::milli>(t_axis_end - t_axis_start).count();
                if (last_timing_.pairs_src == "gpu") {
                    last_timing_.pairs_ms = Metal2Runtime::instance().lastSTQPairsMs();
                } else {
                    last_timing_.pairs_ms = std::chrono::duration<double, std::milli>(t_pairs_cpu_end - t_pairs_cpu_start).count();
                }
                if (last_timing_.filter_src == "gpu") {
                    last_timing_.filter_ms = Metal2Runtime::instance().lastYZFilterMs();
                } else if (sel != 0) {
                    last_timing_.filter_ms = std::chrono::duration<double, std::milli>(t_filter_cpu_end - t_filter_cpu_start).count();
                } else {
                    last_timing_.filter_src = "off";
                    last_timing_.filter_ms = 0.0;
                }
                last_timing_.compose_ms = std::chrono::duration<double, std::milli>(t_compose_end - t_compose_start).count();
                last_timing_.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_total_start).count();
                if (log_timing) {
                    logger().info("Metal2 Timing(single): axis_merge_ms={:.3f} pairs_ms({})={:.3f} filter_ms({})={:.3f} compose_ms={:.3f} total_ms={:.3f}",
                                  last_timing_.axis_merge_ms,
                                  last_timing_.pairs_src.c_str(), last_timing_.pairs_ms,
                                  last_timing_.filter_src.c_str(), last_timing_.filter_ms,
                                  last_timing_.compose_ms, last_timing_.total_ms);
                }
            }
            if (observe) {
                // 规范化并做差集统计
                std::vector<AABB> a2 = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
                if (use_stq) {
                    auto less_min_x = [](const AABB& x, const AABB& y){ return x.min[0] < y.min[0]; };
                    std::sort(a2.begin(), a2.end(), less_min_x);
                } else {
                    sort_boxes_along_axis(sort_axis, a2);
                }
                std::vector<std::pair<int,int>> pairs;
                generate_axis_candidates(a2, /*sort_axis*/use_stq?0:sort_axis, /*two_lists*/false, pairs);
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

    const Timing& last_timing() const { return last_timing_; }

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
    Timing last_timing_;
};

} // namespace scalable_ccd::metal2
