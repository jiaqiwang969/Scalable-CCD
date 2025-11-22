#pragma once

#include <scalable_ccd/broad_phase/aabb.hpp>
#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>
#include <scalable_ccd/utils/logger.hpp>

#include <scalable_ccd/metalcpp/broad_phase/aabb.hpp>
#include <scalable_ccd/metalcpp/runtime/runtime.hpp>

#include <memory>
#include <utility>
#include <vector>
#include <cstdlib>
#include <string>
#include <cctype>
#include <algorithm>

namespace scalable_ccd::metalcpp {

// Metal-cpp BroadPhase（全新起步：默认严格正确性，最终回退 CPU；OBS 可走 GPU 过滤）
class BroadPhase {
public:
    BroadPhase() = default;

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
    static int filterSel(const char* key)
    {
        const char* v = std::getenv(key);
        if (!v) return 0; // default off for safety
        std::string s(v);
        for (auto& c : s) c = (char)std::tolower((unsigned char)c);
        if (s=="gpu") return 2;
        if (s=="cpu") return 1;
        return 0; // off
    }

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

    static inline bool share_a_vertex(const std::array<long,3>& a, const std::array<long,3>& b)
    {
        return a[0]==b[0]||a[0]==b[1]||a[0]==b[2]
            || a[1]==b[0]||a[1]==b[1]||a[1]==b[2]
            || a[2]==b[0]||a[2]==b[1]||a[2]==b[2];
    }

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
            bool overlapX = !(A.max[0] < B.min[0] || A.min[0] > B.max[0]);
            bool overlapY = !(A.max[1] < B.min[1] || A.min[1] > B.max[1]);
            bool overlapZ = true;
            if (A.min.size() >= 3) {
                overlapZ = !(A.max[2] < B.min[2] || A.min[2] > B.max[2]);
            }
            if (!overlapX || !overlapY || !overlapZ) continue;
            if (share_a_vertex(A.vertex_ids, B.vertex_ids)) continue;
            outMask[k] = 1;
        }
    }

    static void sort_along_x(std::vector<AABB>& boxes)
    {
        std::sort(boxes.begin(), boxes.end(), [](const AABB& a, const AABB& b){ return a.min[0] < b.min[0]; });
    }

    static void axis_candidates(const std::vector<AABB>& boxes,
                                bool two_lists,
                                std::vector<std::pair<int,int>>& out)
    {
        out.clear();
        for (int i=0;i<(int)boxes.size();++i){
            const auto& A = boxes[i];
            for (int j=i+1;j<(int)boxes.size();++j){
                const auto& B = boxes[j];
                if (A.max[0] < B.min[0]) break;
                if (!two_lists) {
                    out.emplace_back(i,j);
                } else {
                    bool cross = (A.element_id < 0) ^ (B.element_id < 0);
                    if (cross) out.emplace_back(i,j);
                }
            }
        }
    }

    std::vector<std::pair<int,int>> detect_overlaps()
    {
        if (!built_) return {};
        if (ran_) return m_overlaps;
        const bool use_stq = env_flag_enabled("SCALABLE_CCD_METALCPP_USE_STQ", false);
        const bool strict = !use_stq;
        if (is_two_lists) {
            std::vector<AABB> a = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
            std::vector<AABB> b = d_boxesB ? d_boxesB->h_boxes : std::vector<AABB>{};
            // OBS：沿 X 轴排序 + 候选 + 过滤（仅观测；默认关闭，需 SCALABLE_CCD_METALCPP_OBSERVE=1 手动开启）
            const bool observe = env_flag_enabled("SCALABLE_CCD_METALCPP_OBSERVE", false);
            if (observe && !a.empty() && !b.empty()) {
                std::vector<AABB> a_obs = a, b_obs = b;
                sort_along_x(a_obs); sort_along_x(b_obs);
                for (auto& ax : a_obs) ax.element_id = -ax.element_id - 1;
                std::vector<AABB> merged(a_obs.size()+b_obs.size());
                std::merge(a_obs.begin(), a_obs.end(), b_obs.begin(), b_obs.end(), merged.begin(),
                           [](const AABB& x, const AABB& y){ return x.min[0] < y.min[0]; });
                std::vector<std::pair<int,int>> pairs;
                axis_candidates(merged, /*two*/true, pairs);
                std::vector<uint8_t> mask;
                bool ok = false;
                const int sel = filterSel("SCALABLE_CCD_METALCPP_FILTER"); // off|cpu|gpu; default off
                if (sel == 2 && MetalCppRuntime::instance().available() && MetalCppRuntime::instance().warmup()) {
                    std::vector<float> minY(merged.size()), maxY(merged.size());
                    std::vector<float> minZ(merged.size()), maxZ(merged.size());
                    std::vector<int32_t> v0(merged.size()), v1(merged.size()), v2(merged.size());
                    for (size_t i=0;i<merged.size();++i){
                        minY[i] = static_cast<float>(merged[i].min[1]);
                        maxY[i] = static_cast<float>(merged[i].max[1]);
                        double miZ = merged[i].min.size()>=3 ? merged[i].min[2] : merged[i].min[1];
                        double maZ = merged[i].max.size()>=3 ? merged[i].max[2] : merged[i].max[1];
                        minZ[i] = static_cast<float>(miZ);
                        maxZ[i] = static_cast<float>(maZ);
                        v0[i] = static_cast<int32_t>(merged[i].vertex_ids[0]);
                        v1[i] = static_cast<int32_t>(merged[i].vertex_ids[1]);
                        v2[i] = static_cast<int32_t>(merged[i].vertex_ids[2]);
                    }
                    ok = MetalCppRuntime::instance().filterYZ(minY,maxY,minZ,maxZ,v0,v1,v2,pairs,true,mask);
                }
                if (!ok && sel != 0) cpu_filter_yz(merged, pairs, true, mask);
                if (sel == 0) {
                    // 只计数 axis_pairs，避免大规模分配
                    logger().info("Metal-cpp Broad-phase OBS(two): axis_pairs={} (filter=off)", pairs.size());
                } else {
                    size_t kept = 0; for (auto m:mask) if (m) ++kept;
                    logger().info("Metal-cpp Broad-phase OBS(two): axis_pairs={} yz_kept={} (filter={})", pairs.size(), kept, (sel==2?"gpu":"cpu"));
                }
            }
            if (use_stq && !a.empty() && !b.empty()) {
                std::vector<AABB> a_final = a;
                std::vector<AABB> b_final = b;
                sort_along_x(a_final);
                sort_along_x(b_final);
                for (auto& ax : a_final) ax.element_id = -ax.element_id - 1;
                std::vector<AABB> merged(a_final.size()+b_final.size());
                std::merge(a_final.begin(), a_final.end(), b_final.begin(), b_final.end(), merged.begin(),
                           [](const AABB& x, const AABB& y){ return x.min[0] < y.min[0]; });
                std::vector<std::pair<int,int>> pairs;
                bool stq_ok = false;
                if (MetalCppRuntime::instance().available() && MetalCppRuntime::instance().warmup()) {
                    std::vector<double> minX(merged.size()), maxX(merged.size());
                    std::vector<uint8_t> listTag(merged.size());
                    for (size_t i=0;i<merged.size();++i){
                        minX[i] = merged[i].min[0]; maxX[i] = merged[i].max[0];
                        listTag[i] = merged[i].element_id < 0 ? 1 : 0;
                    }
                    stq_ok = MetalCppRuntime::instance().stqTwoLists(minX,maxX,listTag,pairs);
                }
                if (!stq_ok) axis_candidates(merged, /*two*/true, pairs);
                // filter
                std::vector<uint8_t> mask;
                bool ok = false;
                const int sel = filterSel("SCALABLE_CCD_METALCPP_FILTER");
                if (sel==2 && MetalCppRuntime::instance().available() && MetalCppRuntime::instance().warmup()) {
                    std::vector<float> minY(merged.size()), maxY(merged.size());
                    std::vector<float> minZ(merged.size()), maxZ(merged.size());
                    std::vector<int32_t> v0(merged.size()), v1(merged.size()), v2(merged.size());
                    for (size_t i=0;i<merged.size();++i){
                        minY[i] = static_cast<float>(merged[i].min[1]);
                        maxY[i] = static_cast<float>(merged[i].max[1]);
                        double miZ = merged[i].min.size()>=3 ? merged[i].min[2] : merged[i].min[1];
                        double maZ = merged[i].max.size()>=3 ? merged[i].max[2] : merged[i].max[1];
                        minZ[i] = static_cast<float>(miZ);
                        maxZ[i] = static_cast<float>(maZ);
                        v0[i] = static_cast<int32_t>(merged[i].vertex_ids[0]);
                        v1[i] = static_cast<int32_t>(merged[i].vertex_ids[1]);
                        v2[i] = static_cast<int32_t>(merged[i].vertex_ids[2]);
                    }
                    ok = MetalCppRuntime::instance().filterYZ(minY,maxY,minZ,maxZ,v0,v1,v2,pairs,true,mask);
                }
                if (!ok) cpu_filter_yz(merged, pairs, true, mask);
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
            } else {
                int sort_axis = 0;
                sort_and_sweep(std::move(a), std::move(b), sort_axis, m_overlaps);
            }
        } else {
            std::vector<AABB> a = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
            const bool observe = env_flag_enabled("SCALABLE_CCD_METALCPP_OBSERVE", false);
            if (observe && !a.empty()) {
                std::vector<AABB> a_obs = a;
                sort_along_x(a_obs);
                std::vector<std::pair<int,int>> pairs;
                axis_candidates(a_obs, /*two*/false, pairs);
                std::vector<float> minY(a_obs.size()), maxY(a_obs.size());
                std::vector<float> minZ(a_obs.size()), maxZ(a_obs.size());
                std::vector<int32_t> v0(a_obs.size()), v1(a_obs.size()), v2(a_obs.size());
                for (size_t i=0;i<a_obs.size();++i){
                    minY[i] = static_cast<float>(a_obs[i].min[1]);
                    maxY[i] = static_cast<float>(a_obs[i].max[1]);
                    double miZ = a_obs[i].min.size()>=3 ? a_obs[i].min[2] : a_obs[i].min[1];
                    double maZ = a_obs[i].max.size()>=3 ? a_obs[i].max[2] : a_obs[i].max[1];
                    minZ[i] = static_cast<float>(miZ);
                    maxZ[i] = static_cast<float>(maZ);
                    v0[i] = static_cast<int32_t>(a_obs[i].vertex_ids[0]);
                    v1[i] = static_cast<int32_t>(a_obs[i].vertex_ids[1]);
                    v2[i] = static_cast<int32_t>(a_obs[i].vertex_ids[2]);
                }
                std::vector<uint8_t> mask;
                bool ok = false;
                const int sel = filterSel("SCALABLE_CCD_METALCPP_FILTER"); // default off
                if (sel == 2 && MetalCppRuntime::instance().available() && MetalCppRuntime::instance().warmup()) {
                    ok = MetalCppRuntime::instance().filterYZ(minY,maxY,minZ,maxZ,v0,v1,v2,pairs,false,mask);
                }
                if (!ok && sel!=0) cpu_filter_yz(a_obs, pairs, false, mask);
                if (sel==0) {
                    logger().info("Metal-cpp Broad-phase OBS(single): axis_pairs={} (filter=off)", pairs.size());
                } else {
                    size_t kept = 0; for (auto m:mask) if (m) ++kept;
                    logger().info("Metal-cpp Broad-phase OBS(single): axis_pairs={} yz_kept={} (filter={})", pairs.size(), kept, (sel==2?"gpu":"cpu"));
                }
            }
            int sort_axis = 0;
            sort_and_sweep(std::move(a), sort_axis, m_overlaps);
        }
        ran_ = true;
        return m_overlaps;
    }

    const std::vector<std::pair<int,int>>& overlaps() const { return m_overlaps; }

private:
    std::shared_ptr<DeviceAABBs> d_boxesA;
    std::shared_ptr<DeviceAABBs> d_boxesB;
    bool is_two_lists = false;
    std::vector<std::pair<int,int>> m_overlaps;
    bool built_ = false;
    bool ran_ = false;
};

} // namespace scalable_ccd::metalcpp
