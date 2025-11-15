#pragma once

#include <scalable_ccd/broad_phase/aabb.hpp>
#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>
#include <scalable_ccd/utils/logger.hpp>

#include <scalable_ccd/metal/runtime/runtime.hpp>
#include <scalable_ccd/metal/broad_phase/aabb.hpp>
#include <scalable_ccd/metal/broad_phase/utils.hpp>
#include <scalable_ccd/metal/broad_phase/sweep.hpp>

#include <memory>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <cctype>

namespace scalable_ccd::metal {

class BroadPhase {
public:
    BroadPhase()
    {
#ifdef SCALABLE_CCD_USE_CUDA_SAP
        use_sweep_and_prune = true;
#endif
    }

    /// 设置 Metal STQ 运行配置（便于与 CUDA 外围一致化）。不设置则使用默认/环境值。
    void set_stq_config(const STQConfig& cfg)
    {
        stq_cfg_ = cfg;
        has_stq_cfg_ = true;
    }

    void clear()
    {
        d_boxesA.reset();
        d_boxesB.reset();
        is_two_lists = false;
        m_overlaps.clear();
        built_ = false;
        ran_ = false;
        thread_start_box_id = 0;
    }

    /// @brief Build the broad phase data structure for a single list of boxes.
    void build(const std::shared_ptr<DeviceAABBs> boxes)
    {
        logger().debug("Metal Broad-phase (stub): building (# boxes: {:d})", static_cast<int>(boxes ? boxes->size() : 0));
        clear();
        d_boxesA = boxes;
        is_two_lists = false;
        built_ = true;
    }

    /// @brief Build the broad phase data structure for two lists of boxes.
    void build(
        const std::shared_ptr<DeviceAABBs> boxesA,
        const std::shared_ptr<DeviceAABBs> boxesB)
    {
        logger().debug("Metal Broad-phase (stub): building two lists (#A: {:d}, #B: {:d})",
            static_cast<int>(boxesA ? boxesA->size() : 0),
            static_cast<int>(boxesB ? boxesB->size() : 0));
        clear();
        d_boxesA = boxesA;
        d_boxesB = boxesB;
        is_two_lists = true;
        built_ = true;
    }

    /// @brief CUDA 对齐：分步运行（Metal 实现为一次运行后缓存结果）
    /// @return 返回本次（或最近一次）检测得到的重叠对引用
    const std::vector<std::pair<int, int>>& detect_overlaps_partial()
    {
        if (!ran_) {
            auto v = detect_overlaps(); // 计算一次并缓存到 m_overlaps
            (void)v;
        }
        return m_overlaps;
    }

    /// @brief CUDA 对齐：一次性运行并返回完整 overlaps（Metal 走混合/CPU 路径）
    std::vector<std::pair<int, int>> detect_overlaps()
    {
        if (!built_) {
            logger().error("Metal Broad-phase (stub): detect_overlaps called before build()");
            return {};
        }
        // 环境变量开关：SCALABLE_CCD_DISABLE_METAL=1/true/on/yes 时，完全禁用 Metal 路径
        auto env_metal_disabled = []() -> bool {
            const char* v = std::getenv("SCALABLE_CCD_DISABLE_METAL");
            if (!v) return false;
            std::string s(v);
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return (s == "1" || s == "true" || s == "on" || s == "yes");
        };
        const bool metal_disabled = env_metal_disabled();
        if (metal_disabled) {
            logger().info("Metal Broad-phase: disabled by env SCALABLE_CCD_DISABLE_METAL");
        }
        // 预热一次 Metal 管道，验证 GPU 可用
        static bool warmed = false;
        static bool metal_ok = false;
        if (!warmed) {
            if (!metal_disabled) {
                metal_ok = MetalRuntime::instance().warmup();
                logger().info("Metal warmup: {}", metal_ok ? "ok" : "failed");
            } else {
                metal_ok = false;
            }
            warmed = true;
        }
        // 一次性预热（仅用于编译/验证内核，不影响结果）
        static bool warmed_once = false;
        if (metal_ok && !warmed_once) {
            std::vector<float> dx, ex;
            std::vector<float> fy, gy, fz, gz;
            std::vector<int32_t> v0e, v1e, v2e;
            std::vector<std::pair<int,int>> tmp;
            bool ok = false;
            if (use_sweep_and_prune) {
                ok = bp::sweep_and_prune(
                    dx, ex, fy, gy, fz, gz, v0e, v1e, v2e, 0u, tmp);
                logger().info("Metal Broad-phase: SAP warmup {}", ok ? "ok" : "failed");
            } else {
                ok = MetalRuntime::instance().runSweepAndTiniestQueue(
                    dx, ex, fy, gy, fz, gz, v0e, v1e, v2e, 0u, tmp);
                logger().info("Metal Broad-phase: STQ warmup {}", ok ? "ok" : "failed");
            }
            warmed_once = true;
        }
        if (ran_) {
            return m_overlaps;
        }

        int sort_axis = 0;

        std::vector<AABB> boxes;
        bp::build_sorted_boxes(sort_axis, is_two_lists, d_boxesA, d_boxesB, boxes);
        if (boxes.empty()) {
            logger().info("Metal Broad-phase: no boxes to process (skip)");
            ran_ = true;
            thread_start_box_id = num_boxes();
            return m_overlaps;
        }

        // 为保证与 ground truth 一致性，当前禁用 GPU sweep 结果参与最终输出（仅使用 GPU 做 yz+共享顶点过滤）
        if (metal_ok && !is_two_lists) {
            logger().info("Metal Broad-phase: skip GPU sweep for correctness, using hybrid path");
        }

        // 生成主轴候选（CPU），后续用 GPU 做 yz+共享顶点过滤（或回退 CPU）
        std::vector<std::pair<int, int>> candidates;
        bp::generate_axis_candidates(boxes, sort_axis, is_two_lists, candidates);
        logger().info(
            "Metal Broad-phase: axis={} candidates={}",
            sort_axis, candidates.size());

        // 观测：尝试使用 Metal STQ 与混合路径结果对比（不作为最终输出）
        std::vector<std::pair<int, int>> stqPairsElemIds; // 以 element_id（已做 min-max）存储，便于与最终输出对比
        // 路径探测日志（无论是否可用都打印一次）
        {
            const uint64_t maxPairs_log = static_cast<uint64_t>(boxes.size()) * (boxes.size() - 1) / 2;
            const uint32_t capacity_log = static_cast<uint32_t>(std::min<uint64_t>(maxPairs_log, candidates.size()));
            logger().info("Metal Broad-phase: STQ path check metal_ok={} n={} capacity={} two_lists={}",
                metal_ok, boxes.size(), capacity_log, is_two_lists);
        }
        if (metal_ok) {
            const uint64_t maxPairs = static_cast<uint64_t>(boxes.size()) * (boxes.size() - 1) / 2;
            const uint32_t capacity = static_cast<uint32_t>(std::min<uint64_t>(maxPairs, candidates.size()));
            std::vector<std::pair<int,int>> stqPairs; // 索引空间(i,j)
            logger().info(
                "Metal Broad-phase: invoking {} observation, n={}, capacity={}",
                use_sweep_and_prune ? "SAP" : "STQ", boxes.size(), capacity);
            // 行扫描起点预处理：仅在双列表时，为每个 i 提供起始 j0（跨列表且满足主轴重叠）
            const char* epsEnv = std::getenv("SCALABLE_CCD_STQ_EPS");
            const double epsScale = epsEnv ? std::strtod(epsEnv, nullptr) : 1e-8;
            std::vector<uint32_t> startJ;
            std::vector<uint8_t> listTag;
            if (is_two_lists) {
                const size_t nsz = boxes.size();
                startJ.assign(nsz, static_cast<uint32_t>(nsz)); // 默认无起点
                listTag.assign(nsz, 0);
                for (size_t i = 0; i < nsz; ++i) {
                    // A 列表标 1，B 列表标 0（或反之均可）
                    listTag[i] = boxes[i].element_id < 0 ? 1 : 0;
                }
                for (size_t i = 0; i + 1 < nsz; ++i) {
                    const double amax = static_cast<double>(boxes[i].max[sort_axis]);
                    for (size_t j = i + 1; j < nsz; ++j) {
                        const double bmin = static_cast<double>(boxes[j].min[sort_axis]);
                        const double eps = epsScale * std::max(1.0, std::max(std::abs(amax), std::abs(bmin)));
                        if (amax + eps < bmin) break; // 主轴分离，提前结束
                        const bool a_from_A = boxes[i].element_id < 0;
                        const bool b_from_A = boxes[j].element_id < 0;
                        if (a_from_A != b_from_A) {
                            startJ[i] = static_cast<uint32_t>(j);
                            break;
                        }
                    }
                }
            }
            // 组装 STQ 所需的数组
            std::vector<float> minX(boxes.size()), maxX(boxes.size());
            std::vector<float> minY(boxes.size()), maxY(boxes.size());
            std::vector<float> minZ(boxes.size()), maxZ(boxes.size());
            std::vector<int32_t> v0(boxes.size()), v1(boxes.size()), v2(boxes.size());
            for (size_t i = 0; i < boxes.size(); ++i) {
                minX[i] = static_cast<float>(boxes[i].min[sort_axis]);
                maxX[i] = static_cast<float>(boxes[i].max[sort_axis]);
                minY[i] = static_cast<float>(boxes[i].min[1]);
                maxY[i] = static_cast<float>(boxes[i].max[1]);
                minZ[i] = static_cast<float>(boxes[i].min[2 < boxes[i].min.size() ? 2 : 1]);
                maxZ[i] = static_cast<float>(boxes[i].max[2 < boxes[i].max.size() ? 2 : 1]);
                v0[i] = static_cast<int32_t>(boxes[i].vertex_ids[0]);
                v1[i] = static_cast<int32_t>(boxes[i].vertex_ids[1]);
                v2[i] = static_cast<int32_t>(boxes[i].vertex_ids[2]);
            }
            // 将 threads_per_block 透传到 Metal STQ 的 threadgroupWidth，保持与 CUDA 参数名一致
            STQConfig cfg_local = has_stq_cfg_ ? stq_cfg_ : STQConfig{};
            cfg_local.threadgroupWidth = static_cast<uint32_t>(std::max(1, threads_per_block));
            bool stq_ok = false;
            if (use_sweep_and_prune) {
                stq_ok = bp::sweep_and_prune(
                    minX, maxX, minY, maxY, minZ, maxZ, v0, v1, v2, capacity, stqPairs,
                    is_two_lists ? &startJ : nullptr,
                    is_two_lists ? &listTag : nullptr,
                    is_two_lists);
            } else {
                stq_ok = bp::sweep_and_tiniest_queue(
                    minX, maxX, minY, maxY, minZ, maxZ, v0, v1, v2, capacity, stqPairs,
                    is_two_lists ? &startJ : nullptr,
                    is_two_lists ? &listTag : nullptr,
                    is_two_lists, cfg_local);
            }
            if (stq_ok) {
                logger().info("Metal Broad-phase: {} observed overlaps={}", use_sweep_and_prune ? "SAP" : "STQ", stqPairs.size());
                // 映射到 element_id，并规范化为 (min,max)
                stqPairsElemIds.clear();
                stqPairsElemIds.reserve(stqPairs.size());
                for (const auto& p : stqPairs) {
                    const int i = p.first;
                    const int j = p.second;
                    if (i < 0 || j < 0 || i >= static_cast<int>(boxes.size()) || j >= static_cast<int>(boxes.size())) continue;
                    const int ida = boxes[static_cast<size_t>(i)].element_id;
                    const int idb = boxes[static_cast<size_t>(j)].element_id;
                    if (is_two_lists) {
                        const bool a_from_A = (ida < 0);
                        const bool b_from_A = (idb < 0);
                        // 两列表观测：只保留跨列表的对
                        if (a_from_A == b_from_A) continue;
                        const int ida_pos = a_from_A ? -ida - 1 : ida;
                        const int idb_pos = b_from_A ? -idb - 1 : idb;
                        const int a = std::min(ida_pos, idb_pos);
                        const int b = std::max(ida_pos, idb_pos);
                        stqPairsElemIds.emplace_back(a, b);
                    } else {
                        // 单列表：按元素 id 升序
                        const int a = std::min(ida, idb);
                        const int b = std::max(ida, idb);
                        stqPairsElemIds.emplace_back(a, b);
                    }
                }
                // 排序去重，便于集合对比
                std::sort(stqPairsElemIds.begin(), stqPairsElemIds.end());
                stqPairsElemIds.erase(std::unique(stqPairsElemIds.begin(), stqPairsElemIds.end()), stqPairsElemIds.end());
            } else {
                logger().warn("Metal Broad-phase: STQ observation failed");
            }
        }

        // 若 Metal 可用，使用 GPU 做 yz 过滤；否则回退 CPU 完整路径
        if (metal_ok && !candidates.empty()) {
            // 组装 YZ 与顶点 ID
            std::vector<float> minY(boxes.size()), maxY(boxes.size());
            std::vector<float> minZ(boxes.size()), maxZ(boxes.size());
            for (size_t i = 0; i < boxes.size(); ++i) {
                minY[i] = static_cast<float>(boxes[i].min[1]);
                maxY[i] = static_cast<float>(boxes[i].max[1]);
                minZ[i] = static_cast<float>(boxes[i].min[2 < boxes[i].min.size() ? 2 : 1]);
                maxZ[i] = static_cast<float>(boxes[i].max[2 < boxes[i].max.size() ? 2 : 1]);
            }
            std::vector<int32_t> v0(boxes.size()), v1(boxes.size()), v2(boxes.size());
            for (size_t i = 0; i < boxes.size(); ++i) {
                v0[i] = static_cast<int32_t>(boxes[i].vertex_ids[0]);
                v1[i] = static_cast<int32_t>(boxes[i].vertex_ids[1]);
                v2[i] = static_cast<int32_t>(boxes[i].vertex_ids[2]);
            }
            std::vector<uint8_t> mask;
            const bool ok = bp::filter_yz(minY, maxY, minZ, maxZ, v0, v1, v2, candidates, mask);
            if (ok) {
                m_overlaps.clear();
                m_overlaps.reserve(candidates.size());
                size_t accepted = 0;
                for (size_t k = 0; k < candidates.size(); ++k) {
                    if (!mask[k]) continue;
                    ++accepted;
                    const int i = candidates[k].first;
                    const int j = candidates[k].second;
                    const AABB& a = boxes[i];
                    const AABB& b = boxes[j];
                    if (is_two_lists) {
                        const int idA = a.element_id < 0 ? -a.element_id - 1 : a.element_id;
                        const int idB = b.element_id < 0 ? -b.element_id - 1 : b.element_id;
                        const bool a_from_A = (a.element_id < 0);
                        m_overlaps.emplace_back(
                            a_from_A ? idA : idB,
                            a_from_A ? idB : idA);
                    } else {
                        m_overlaps.emplace_back(
                            std::min<int>(a.element_id, b.element_id),
                            std::max<int>(a.element_id, b.element_id));
                    }
                }
                logger().info(
                    "Metal Broad-phase: yz GPU filter used, accepted={} overlaps={}",
                    accepted, m_overlaps.size());
                // 若有 STQ 观测结果，则与最终输出集合对比并打日志（单列表与双列表均对比）
                if (!stqPairsElemIds.empty()) {
                    // 规范化最终输出对为(小,大)，便于与 STQ 对比（双列表也统一用 min/max 形式）
                    std::vector<std::pair<int,int>> finalPairsNorm;
                    finalPairsNorm.reserve(m_overlaps.size());
                    for (const auto& p : m_overlaps) {
                        const int a = std::min(p.first, p.second);
                        const int b = std::max(p.first, p.second);
                        finalPairsNorm.emplace_back(a, b);
                    }
                    std::sort(finalPairsNorm.begin(), finalPairsNorm.end());
                    finalPairsNorm.erase(std::unique(finalPairsNorm.begin(), finalPairsNorm.end()), finalPairsNorm.end());
                    // 交集与差集规模
                    std::vector<std::pair<int,int>> inters, stq_minus_final, final_minus_stq;
                    inters.reserve(std::min(finalPairsNorm.size(), stqPairsElemIds.size()));
                    std::set_intersection(
                        finalPairsNorm.begin(), finalPairsNorm.end(),
                        stqPairsElemIds.begin(), stqPairsElemIds.end(),
                        std::back_inserter(inters));
                    std::set_difference(
                        stqPairsElemIds.begin(), stqPairsElemIds.end(),
                        finalPairsNorm.begin(), finalPairsNorm.end(),
                        std::back_inserter(stq_minus_final));
                    std::set_difference(
                        finalPairsNorm.begin(), finalPairsNorm.end(),
                        stqPairsElemIds.begin(), stqPairsElemIds.end(),
                        std::back_inserter(final_minus_stq));
                    logger().info(
                        "Metal Broad-phase: STQ vs final — final={}, stq={}, inter={}, stq-only={}, final-only={}",
                        finalPairsNorm.size(), stqPairsElemIds.size(), inters.size(),
                        stq_minus_final.size(), final_minus_stq.size());
                    // 若存在 stq-only 差异，按环境变量采样输出若干示例，便于诊断
                    auto env_sample_n = []() -> int {
                        const char* v = std::getenv("SCALABLE_CCD_STQ_LOG_SAMPLES");
                        if (!v) return 0;
                        try {
                            return std::max(0, std::stoi(std::string(v)));
                        } catch (...) { return 0; }
                    };
                    const int sample_n = env_sample_n();
                    if (sample_n > 0 && (!stq_minus_final.empty() || !final_minus_stq.empty())) {
                        // 复核：使用 CPU/double 精度在 yz 与共享顶点上再次检查，并打印主轴 X 区间
                        auto print_recheck = [&](const scalable_ccd::AABB* A, const scalable_ccd::AABB* B, const char* tag) {
                            if (!A || !B) {
                                logger().info("Metal Broad-phase: recheck[{}]: AABB not found for mapping", tag ? tag : "n/a");
                                return;
                            }
                            const double a_min_y = A->min[1], a_max_y = A->max[1];
                            const double b_min_y = B->min[1], b_max_y = B->max[1];
                            const double a_min_z = A->min[2 < A->min.size() ? 2 : 1], a_max_z = A->max[2 < A->max.size() ? 2 : 1];
                            const double b_min_z = B->min[2 < B->min.size() ? 2 : 1], b_max_z = B->max[2 < B->max.size() ? 2 : 1];
                            const bool overlapY = !(a_max_y < b_min_y || a_min_y > b_max_y);
                            const bool overlapZ = !(a_max_z < b_min_z || a_min_z > b_max_z);
                            const auto& av = A->vertex_ids;
                            const auto& bv = B->vertex_ids;
                            const bool share =
                                (av[0] == bv[0]) || (av[0] == bv[1]) || (av[0] == bv[2]) ||
                                (av[1] == bv[0]) || (av[1] == bv[1]) || (av[1] == bv[2]) ||
                                (av[2] == bv[0]) || (av[2] == bv[1]) || (av[2] == bv[2]);
                            logger().info(
                                "Metal Broad-phase: recheck[{}]: a_id={} b_id={} "
                                "Y:[{:.6f},{:.6f}] vs [{:.6f},{:.6f}] Z:[{:.6f},{:.6f}] vs [{:.6f},{:.6f}] "
                                "overlapY={} overlapZ={} share={}",
                                tag ? tag : "n/a",
                                A->element_id, B->element_id,
                                a_min_y, a_max_y, b_min_y, b_max_y,
                                a_min_z, a_max_z, b_min_z, b_max_z,
                                overlapY, overlapZ, share);
                            const double a_min_x = A->min[0], a_max_x = A->max[0];
                            const double b_min_x = B->min[0], b_max_x = B->max[0];
                            logger().info(
                                "Metal Broad-phase: recheck[{}]: X:[{:.6f},{:.6f}] vs [{:.6f},{:.6f}]",
                                tag ? tag : "n/a", a_min_x, a_max_x, b_min_x, b_max_x);
                        };
                        // STQ-only 样本
                        int printed = 0;
                        for (const auto& pr : stq_minus_final) {
                            if (printed >= sample_n) break;
                            logger().info("Metal Broad-phase: STQ-only sample pair=({}, {})", pr.first, pr.second);
                            if (!is_two_lists) {
                                const scalable_ccd::AABB* Aptr = nullptr;
                                const scalable_ccd::AABB* Bptr = nullptr;
                                if (d_boxesA) {
                                    for (const auto& bx : d_boxesA->h_boxes) {
                                        if (bx.element_id == pr.first) Aptr = &bx;
                                        if (bx.element_id == pr.second) Bptr = &bx;
                                    }
                                }
                                print_recheck(Aptr, Bptr, "single");
                            } else {
                                const scalable_ccd::AABB* AA = nullptr;
                                const scalable_ccd::AABB* AB = nullptr;
                                const scalable_ccd::AABB* BA = nullptr;
                                const scalable_ccd::AABB* BB = nullptr;
                                if (d_boxesA) {
                                    for (const auto& bx : d_boxesA->h_boxes) {
                                        if (bx.element_id == pr.first) AA = &bx;
                                        if (bx.element_id == pr.second) BA = &bx;
                                    }
                                }
                                if (d_boxesB) {
                                    for (const auto& bx : d_boxesB->h_boxes) {
                                        if (bx.element_id == pr.second) BB = &bx;
                                        if (bx.element_id == pr.first) AB = &bx;
                                    }
                                }
                                if (AA && BB) {
                                    print_recheck(AA, BB, "two(A,B)");
                                } else if (AB && BA) {
                                    print_recheck(BA, AB, "two(B,A)");
                                } else {
                                    logger().info("Metal Broad-phase: STQ-only recheck[two]: mapping not found (a={}, b={})", pr.first, pr.second);
                                }
                            }
                            ++printed;
                        }
                        // FINAL-only 样本
                        int printed2 = 0;
                        for (const auto& pr : final_minus_stq) {
                            if (printed2 >= sample_n) break;
                            logger().info("Metal Broad-phase: FINAL-only sample pair=({}, {})", pr.first, pr.second);
                            if (!is_two_lists) {
                                const scalable_ccd::AABB* Aptr = nullptr;
                                const scalable_ccd::AABB* Bptr = nullptr;
                                if (d_boxesA) {
                                    for (const auto& bx : d_boxesA->h_boxes) {
                                        if (bx.element_id == pr.first) Aptr = &bx;
                                        if (bx.element_id == pr.second) Bptr = &bx;
                                    }
                                }
                                print_recheck(Aptr, Bptr, "single");
                            } else {
                                const scalable_ccd::AABB* AA = nullptr;
                                const scalable_ccd::AABB* AB = nullptr;
                                const scalable_ccd::AABB* BA = nullptr;
                                const scalable_ccd::AABB* BB = nullptr;
                                if (d_boxesA) {
                                    for (const auto& bx : d_boxesA->h_boxes) {
                                        if (bx.element_id == pr.first) AA = &bx;
                                        if (bx.element_id == pr.second) BA = &bx;
                                    }
                                }
                                if (d_boxesB) {
                                    for (const auto& bx : d_boxesB->h_boxes) {
                                        if (bx.element_id == pr.second) BB = &bx;
                                        if (bx.element_id == pr.first) AB = &bx;
                                    }
                                }
                                if (AA && BB) {
                                    print_recheck(AA, BB, "two(A,B)");
                                } else if (AB && BA) {
                                    print_recheck(BA, AB, "two(B,A)");
                                } else {
                                    logger().info("Metal Broad-phase: FINAL-only recheck[two]: mapping not found (a={}, b={})", pr.first, pr.second);
                                }
                            }
                            ++printed2;
                        }
                    }
                }
            } else {
                // Metal 失败，回退 CPU
                logger().warn("Metal Broad-phase: yz GPU filter failed, falling back to CPU path");
                if (is_two_lists) {
                    std::vector<AABB> a = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
                    std::vector<AABB> b = d_boxesB ? d_boxesB->h_boxes : std::vector<AABB>{};
                    sort_and_sweep(std::move(a), std::move(b), sort_axis, m_overlaps);
                } else {
                    std::vector<AABB> a = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
                    sort_and_sweep(std::move(a), sort_axis, m_overlaps);
                }
                logger().info("Metal Broad-phase: CPU sort_and_sweep overlaps={}", m_overlaps.size());
            }
        } else {
            // 没有 Metal 或没有候选：走 CPU 路径
            if (!metal_ok) {
                logger().info("Metal Broad-phase: Metal unavailable, falling back to CPU path");
            } else if (candidates.empty()) {
                logger().info("Metal Broad-phase: no candidates, skip GPU filter");
            }
            if (is_two_lists) {
                std::vector<AABB> a = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
                std::vector<AABB> b = d_boxesB ? d_boxesB->h_boxes : std::vector<AABB>{};
                sort_and_sweep(std::move(a), std::move(b), sort_axis, m_overlaps);
            } else {
                std::vector<AABB> a = d_boxesA ? d_boxesA->h_boxes : std::vector<AABB>{};
                sort_and_sweep(std::move(a), sort_axis, m_overlaps);
            }
            logger().info("Metal Broad-phase: CPU sort_and_sweep overlaps={}", m_overlaps.size());
        }

        // 不再提供 CPU 覆盖式对齐方案；YZ 容差固定为 0，后续由 narrow phase 收敛

        // 标记完成
        ran_ = true;
        thread_start_box_id = num_boxes();
        return m_overlaps;
    }

    /// @brief CUDA 对齐：是否完成（Metal 为一次性，计算后即完成）
    bool is_complete() const { return thread_start_box_id >= num_boxes(); }

    /// @brief CUDA 对齐：返回当前 boxes（双列表时返回 A；如需合并请参考 utils::build_sorted_boxes）
    std::shared_ptr<DeviceAABBs> boxes() { return d_boxesA; }

    /// @brief CUDA 对齐：返回上次计算得到的 overlaps 引用
    const std::vector<std::pair<int, int>>& overlaps() const { return m_overlaps; }

    /// @brief CUDA 对齐：box 数量（双列表返回 A+B）
    size_t num_boxes() const
    {
        if (!d_boxesA) return 0;
        if (!is_two_lists) return d_boxesA->size();
        return (d_boxesA ? d_boxesA->size() : 0) + (d_boxesB ? d_boxesB->size() : 0);
    }

private:
    std::shared_ptr<DeviceAABBs> d_boxesA;
    std::shared_ptr<DeviceAABBs> d_boxesB;
    bool is_two_lists = false;

    std::vector<std::pair<int, int>> m_overlaps;
    bool built_ = false;
    bool ran_ = false;
    size_t thread_start_box_id = 0;
    bool has_stq_cfg_ = false;
    STQConfig stq_cfg_{};

public:
    // CUDA 对齐：公开线程组大小参数名保持一致（Metal 实际为 threadgroup 宽度）
    int threads_per_block = 32;
    // CUDA 对齐：选择 SAP 路径的开关（默认与宏/环境一致）
    bool use_sweep_and_prune = false;

    /// @brief CUDA 对齐：合并后的 boxes 访问器；返回经必要翻转与排序合并后的 host 侧 boxes。
    /// @param sort_axis 主轴，默认 0
    std::vector<AABB> merged_boxes(int sort_axis = 0) const
    {
        std::vector<AABB> bx;
        bp::build_sorted_boxes(sort_axis, is_two_lists, d_boxesA, d_boxesB, bx);
        return bx;
    }
};

} // namespace scalable_ccd::metal
