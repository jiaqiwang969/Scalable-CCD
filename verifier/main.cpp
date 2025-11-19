// Scalable-CCD Verifier
// - 枚举一组代表性场景
// - CPU/CUDA 分别运行 Broad Phase
// - 打印环境、阶段用时、对比真值
// - 生成 JSON + HTML 报告

#include "env_info.hpp"
#include "io.hpp"
#include "compare.hpp"
#include "report.hpp"
#include "queries.hpp"

#include <scalable_ccd/config.hpp>
#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>
#include <scalable_ccd/utils/timer.hpp>
#include <scalable_ccd/utils/logger.hpp>
#include <scalable_ccd/utils/profiler.hpp>

#ifdef SCALABLE_CCD_WITH_CUDA
#include <scalable_ccd/cuda/broad_phase/broad_phase.cuh>
#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/cuda/memory_handler.hpp>
#endif
#ifdef SCALABLE_CCD_WITH_METAL
#include <scalable_ccd/metal/broad_phase_metal.hpp>
#endif

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <tbb/global_control.h>
#include <tbb/info.h>

namespace fs = std::filesystem;
using nlohmann::json;
using Pair = std::pair<int, int>;

struct SceneCase {
    std::string scene;
    std::string t0;
    std::string t1;
    // expected files for truth
    std::string vf_json;
    std::string ee_json;
};

static std::vector<SceneCase> default_scenes(const fs::path& data)
{
    // 选择与现有单测一致的若干场景与帧
    return {
        { "armadillo-rollers", "326.ply", "327.ply", "326vf.json", "326ee.json" },
        { "cloth-ball", "frames/cloth_ball92.ply", "frames/cloth_ball93.ply", "boxes/92vf.json", "boxes/92ee.json" },
        { "cloth-funnel", "227.ply", "228.ply", "227vf.json", "227ee.json" },
        { "n-body-simulation", "frames/balls16_18.ply", "frames/balls16_19.ply", "boxes/18vf.json", "boxes/18ee.json" },
        { "rod-twist", "3036.ply", "3037.ply", "3036vf.json", "3036ee.json" },
    };
}

static bool load_scenarios_json(const fs::path& path, std::vector<SceneCase>& out)
{
    std::ifstream in(path);
    if (!in.good()) return false;
    json j = json::parse(in, nullptr, true, true);
    if (!j.contains("scenes") || !j["scenes"].is_array()) return false;
    out.clear();
    for (const auto& s : j["scenes"]) {
        SceneCase sc;
        sc.scene = s.value("scene", "");
        sc.t0 = s.value("t0", "");
        sc.t1 = s.value("t1", "");
        sc.vf_json = s.value("vf_json", "");
        sc.ee_json = s.value("ee_json", "");
        if (!sc.scene.empty() && !sc.t0.empty() && !sc.t1.empty() && !sc.vf_json.empty() && !sc.ee_json.empty()) {
            out.push_back(std::move(sc));
        }
    }
    return !out.empty();
}

static void usage(const char* argv0)
{
    std::cout << "用法: " << argv0 << " [--backend cpu|cuda|both|metal] [--out DIR] [--log N] [--threads N]\n";
    std::cout << "                 [--repeat N] [--warmup N] [--scenarios FILE] [--scan]\n";
    std::cout << "                 [--max-per-scene N] [--tag NAME] [--data DIR] [--metal-stq]\n";
    std::cout << "说明: 读取 SCALABLE_CCD_DATA_DIR 下的样例场景，生成验证报告。\n";
}

static bool is_numeric(char c) { return c >= '0' && c <= '9'; }

// 提取帧文件名中的“末尾数字”作为时间步（如 balls16_18 -> 18, 3036 -> 3036, cloth_ball92 -> 92）
static bool extract_step_from_filename(const std::string& stem, int& step_out)
{
    int i = static_cast<int>(stem.size()) - 1;
    while (i >= 0 && !is_numeric(stem[i])) --i;
    if (i < 0) return false;
    int end = i;
    while (i >= 0 && is_numeric(stem[i])) --i;
    int start = i + 1;
    if (start > end) return false;
    try {
        step_out = std::stoi(stem.substr(start, end - start + 1));
        return true;
    } catch (...) {
        return false;
    }
}

static std::vector<SceneCase> scan_scenarios_from_dataset(const fs::path& data, int max_per_scene, int min_step, const std::vector<std::string>& scene_filters)
{
    std::vector<SceneCase> out;
    if (!fs::exists(data) || !fs::is_directory(data)) return out;
    for (auto& scene_dir_entry : fs::directory_iterator(data)) {
        if (!scene_dir_entry.is_directory()) continue;
        std::string scene = scene_dir_entry.path().filename().string();
        if (scene == "_src" || scene == "_loose" || scene == ".git") continue;
        if (!scene_filters.empty()) {
            bool match = false;
            for (const auto& sf : scene_filters) {
                if (scene == sf) { match = true; break; }
            }
            if (!match) continue;
        }
        fs::path scene_dir = scene_dir_entry.path();
        fs::path frames_dir = scene_dir / "frames";
        fs::path boxes_dir = scene_dir / "boxes";
        if (!fs::exists(frames_dir) || !fs::is_directory(frames_dir)) continue;
        if (!fs::exists(boxes_dir) || !fs::is_directory(boxes_dir)) continue;
        // 收集所有帧文件，按步号排序
        struct FrameItem { int step; fs::path path; };
        std::vector<FrameItem> frames;
        for (auto& f : fs::directory_iterator(frames_dir)) {
            if (!f.is_regular_file()) continue;
            auto ext = f.path().extension().string();
            if (ext != ".ply" && ext != ".obj") continue;
            int step;
            if (extract_step_from_filename(f.path().stem().string(), step)) {
                frames.push_back({ step, f.path() });
            }
        }
        std::sort(frames.begin(), frames.end(), [](const FrameItem& a, const FrameItem& b) {
            if (a.step != b.step) return a.step < b.step;
            return a.path.string() < b.path.string();
        });
        if (frames.size() < 2) continue;
        int added = 0;
        for (size_t i = 0; i + 1 < frames.size(); ++i) {
            int s = frames[i].step;
            int nexts = frames[i + 1].step;
            if (nexts != s + 1) continue; // 只匹配相邻的时间步
            if (min_step >= 0 && s < min_step) continue;
            // 推断 boxes json 名称
            fs::path vf = boxes_dir / (std::to_string(s) + "vf.json");
            fs::path ee = boxes_dir / (std::to_string(s) + "ee.json");
            // 仅选择存在真值的时间步，避免早期帧缺少真值导致不一致
            if (!fs::exists(vf) || !fs::exists(ee)) {
                continue;
            }
            SceneCase sc;
            sc.scene = scene;
            sc.t0 = fs::relative(frames[i].path, scene_dir).string();
            sc.t1 = fs::relative(frames[i + 1].path, scene_dir).string();
            sc.vf_json = fs::relative(vf, scene_dir).string();
            sc.ee_json = fs::relative(ee, scene_dir).string();
            out.push_back(std::move(sc));
            if (max_per_scene > 0 && ++added >= max_per_scene) break;
        }
    }
    return out;
}

static int auto_select_axis(
    const std::vector<AABB>& boxesA,
    const std::vector<AABB>* boxesB = nullptr)
{
    double sum[3] = { 0.0, 0.0, 0.0 };
    double sumsq[3] = { 0.0, 0.0, 0.0 };
    size_t total = 0;
    auto accumulate = [&](const std::vector<AABB>& boxes) {
        for (const auto& box : boxes) {
            const auto center = (box.min + box.max) / 2.0;
            for (int i = 0; i < 3; ++i) {
                const double c = center[i];
                sum[i] += c;
                sumsq[i] += c * c;
            }
            ++total;
        }
    };
    accumulate(boxesA);
    if (boxesB) accumulate(*boxesB);
    if (total == 0) {
        return 0;
    }
    int axis = 0;
    double best_var = -1.0;
    for (int i = 0; i < 3; ++i) {
        const double mean = sum[i] / total;
        const double var = sumsq[i] / total - mean * mean;
        if (var > best_var) {
            best_var = var;
            axis = i;
        }
    }
    return axis;
}

static size_t load_truth_size(const fs::path& path)
{
    std::ifstream in(path);
    if (!in.good()) return 0;
    const json j = json::parse(in, nullptr, true, true);
    return j.size();
}

static uint32_t estimate_capacity(size_t truth_size, size_t soa_size)
{
    size_t base = truth_size > 0 ? truth_size + truth_size / 10 + 4096 : soa_size * 4 + 4096;
    if (base > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        base = static_cast<size_t>(std::numeric_limits<uint32_t>::max());
    }
    return static_cast<uint32_t>(base);
}

static double average_all(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

static double average_nonnegative(const std::vector<double>& v)
{
    double s = 0.0;
    int c = 0;
    for (double x : v) {
        if (x >= 0.0) {
            s += x;
            ++c;
        }
    }
    return c > 0 ? s / static_cast<double>(c) : -1.0;
}

struct MetalStageResult {
    std::vector<Pair> overlaps;
    double avg_host_ms = 0.0;
    double avg_gpu_ms = -1.0;
    double e2e_ms = 0.0;
    int sort_axis = 0;
    uint32_t cutoff = 0;
    uint32_t capacity = 0;
};

#ifdef SCALABLE_CCD_WITH_METAL
static MetalStageResult run_metal_stage(
    const std::vector<AABB>& boxesA,
    const std::vector<AABB>* boxesB,
    bool two_lists,
    bool use_stq,
    int warmup,
    int repeat,
    size_t truth_size)
{
    MetalStageResult res{};
    res.sort_axis = auto_select_axis(boxesA, boxesB);
    scalable_ccd::metal::MetalAABBsSoA soa;
    if (two_lists) {
        if (!boxesB) throw std::runtime_error("Two-list stage requires secondary boxes");
        scalable_ccd::metal::make_soa_two_lists(boxesA, *boxesB, res.sort_axis, soa);
    } else {
        scalable_ccd::metal::make_soa_one_list(boxesA, res.sort_axis, soa);
    }

    scalable_ccd::metal::BroadPhase bp;
    bp.set_use_stq(use_stq);
    bp.upload(soa);
    res.cutoff = static_cast<uint32_t>(soa.size());
    res.capacity = estimate_capacity(truth_size, soa.size());

    for (int w = 0; w < warmup; ++w) {
        (void)bp.detect_overlaps(two_lists, res.cutoff, res.capacity);
    }

    std::vector<double> host_times;
    std::vector<double> gpu_times;
    host_times.reserve(std::max(1, repeat));
    gpu_times.reserve(std::max(1, repeat));
    scalable_ccd::Timer total;
    total.start();
    for (int r = 0; r < repeat; ++r) {
        scalable_ccd::Timer ht;
        ht.start();
        res.overlaps = bp.detect_overlaps(two_lists, res.cutoff, res.capacity);
        ht.stop();
        host_times.push_back(ht.getElapsedTimeInMilliSec());
        gpu_times.push_back(bp.last_gpu_ms());
    }
    total.stop();
    res.avg_host_ms = average_all(host_times);
    res.avg_gpu_ms = average_nonnegative(gpu_times);
    res.e2e_ms = total.getElapsedTimeInMilliSec();
    return res;
}
#endif
int main(int argc, char** argv)
{
    std::string backend = "both";
    fs::path out_dir = fs::path("build") / "verifier_report";
    int log_level = spdlog::level::info;
    int num_threads = tbb::info::default_concurrency();
    int warmup = 0;
    int repeat = 1;
    std::string scenarios_path;
    std::string env_tag;
    std::string data_dir_cli;
    bool metal_use_stq = false;
    std::string metal_mode = "sap";
    bool do_scan = false;
    int max_per_scene = -1;
    int min_step = -1;
    std::vector<std::string> scene_filters;
    bool run_queries = false;
    int max_queries = -1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else if (arg == "--backend" && i + 1 < argc) {
            backend = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_dir = fs::path(argv[++i]);
        } else if (arg == "--log" && i + 1 < argc) {
            log_level = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
            if (num_threads <= 0) num_threads = tbb::info::default_concurrency();
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
            if (warmup < 0) warmup = 0;
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
            if (repeat < 1) repeat = 1;
        } else if (arg == "--scenarios" && i + 1 < argc) {
            scenarios_path = argv[++i];
        } else if (arg == "--scan") {
            do_scan = true;
        } else if (arg == "--max-per-scene" && i + 1 < argc) {
            max_per_scene = std::atoi(argv[++i]);
        } else if (arg == "--min-step" && i + 1 < argc) {
            min_step = std::atoi(argv[++i]);
        } else if (arg == "--scenes" && i + 1 < argc) {
            // 逗号分隔的场景过滤
            std::string list = argv[++i];
            size_t start = 0;
            while (true) {
                size_t pos = list.find(',', start);
                std::string token = list.substr(start, pos == std::string::npos ? std::string::npos : pos - start);
                if (!token.empty()) scene_filters.push_back(token);
                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        } else if (arg == "--tag" && i + 1 < argc) {
            env_tag = argv[++i];
        } else if (arg == "--data" && i + 1 < argc) {
            data_dir_cli = argv[++i];
        } else if (arg == "--metal-stq") {
            metal_use_stq = true;
            metal_mode = "stq";
        } else if (arg == "--queries") {
            run_queries = true;
        } else if (arg == "--max-queries" && i + 1 < argc) {
            max_queries = std::atoi(argv[++i]);
        }
    }
    spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
    scalable_ccd::logger().set_level(static_cast<spdlog::level::level_enum>(log_level));

    tbb::global_control thread_limiter(
        tbb::global_control::max_allowed_parallelism, num_threads);

    // 数据目录：优先命令行 --data ；否则尝试环境变量；否则使用编译期默认 SCALABLE_CCD_DATA_DIR
    fs::path data = fs::path(SCALABLE_CCD_DATA_DIR);
    if (!data_dir_cli.empty()) {
        data = fs::path(data_dir_cli);
    } else {
        const char* env_p = std::getenv("SCALABLE_CCD_DATA_DIR");
        if (env_p && *env_p) data = fs::path(env_p);
    }
    // 兼容不同子路径命名：有些场景将帧/真值放在 scene/frames 与 scene/boxes 下
    std::vector<SceneCase> scenes;
    if (do_scan) {
        scenes = scan_scenarios_from_dataset(data, max_per_scene, min_step, scene_filters);
        if (scenes.empty()) {
            spdlog::warn("扫描数据集未发现场景：{}", data.string());
        } else {
            spdlog::info("扫描到场景条目数量：{}", scenes.size());
        }
    } else if (!scenarios_path.empty()) {
        if (!load_scenarios_json(scenarios_path, scenes)) {
            spdlog::warn("无法加载scenarios文件: {}", scenarios_path);
        }
    }
    if (scenes.empty()) {
        scenes = default_scenes(data);
    }

    json aggregate;
    aggregate["env"] = verifier::collect_env_info();
    if (!env_tag.empty()) {
        aggregate["env"]["tag"] = env_tag;
    }
    aggregate["env"]["threads"] = num_threads;
    aggregate["env"]["repeat"] = repeat;
    aggregate["env"]["warmup"] = warmup;
    aggregate["env"]["data_dir"] = data.string();
    aggregate["runs"] = json::array();
    aggregate["query_runs"] = json::array();

    for (const auto& s : scenes) {
        fs::path base = data / s.scene;
        // 路径兼容：有的条目给出完整子路径，有的仅文件名
        auto make_path = [&](const std::string& p, const std::string& subdir) -> fs::path {
            fs::path path = base / p;
            if (!fs::exists(path)) {
                path = base / subdir / p;
            }
            return path;
        };

        fs::path file_t0 = make_path(s.t0, "frames");
        fs::path file_t1 = make_path(s.t1, "frames");
        fs::path vf_gt = make_path(s.vf_json, "boxes");
        fs::path ee_gt = make_path(s.ee_json, "boxes");

        verifier::MeshPair mp;
        if (!verifier::read_mesh_pair(file_t0, file_t1, mp)) {
            spdlog::warn("读取场景失败: {} t0={} t1={}", s.scene, file_t0.string(), file_t1.string());
            continue;
        }

        // ---------------- CPU Broad Phase ----------------
        if (backend == "cpu" || backend == "both") {
            std::vector<scalable_ccd::AABB> vboxes, eboxes, fboxes;
            double build_boxes_ms = 0.0;
            {
                scalable_ccd::Timer t;
                t.start();
                verifier::build_cpu_boxes(mp, vboxes, eboxes, fboxes);
                t.stop();
                build_boxes_ms = t.getElapsedTimeInMilliSec();
                spdlog::info("[CPU] {}: 构建AABB {:.3f} ms (V={},E={},F={})",
                             s.scene, t.getElapsedTimeInMilliSec(), vboxes.size(), eboxes.size(), fboxes.size());
            }

            int sort_axis = 0;
            std::vector<std::pair<int, int>> vf_overlaps, ee_overlaps;
            {
                // warmup
                for (int w = 0; w < warmup; ++w) {
                    std::vector<std::pair<int, int>> tmp;
                    int ax = 0;
                    scalable_ccd::sort_and_sweep(vboxes, fboxes, ax, tmp);
                }
                // repeats
                std::vector<double> times;
                times.reserve(repeat);
                for (int r = 0; r < repeat; ++r) {
                    spdlog::debug("[CPU][{}] broad_vf run {}/{}: V={},F={}", s.scene, r+1, repeat, vboxes.size(), fboxes.size());
                    scalable_ccd::Timer t; t.start();
                    scalable_ccd::sort_and_sweep(vboxes, fboxes, sort_axis, vf_overlaps);
                    t.stop();
                    spdlog::debug("[CPU][{}] broad_vf done: overlaps={}", s.scene, vf_overlaps.size());
                    times.push_back(t.getElapsedTimeInMilliSec());
                }
                double avg_ms = 0.0;
                for (double x : times) avg_ms += x;
                avg_ms /= std::max(1, (int)times.size());
                // Offset to match ground-truth encoding (see tests)
                {
                    int offset = static_cast<int>(vboxes.size() + eboxes.size());
                    for (auto& p : vf_overlaps) {
                        p.second += offset;
                    }
                }
                json run;
                run["scene"] = s.scene;
                run["t0"] = file_t0.filename().string();
                run["t1"] = file_t1.filename().string();
                run["backend"] = "cpu";
                run["steps"] = {
                    {"stage","broad_vf"},
                    {"avg_ms", avg_ms},
                    {"repeats", repeat},
                    {"warmup", warmup},
                    {"build_boxes_ms", build_boxes_ms},
                    {"sort_axis_end", sort_axis}
                };
                run["threads"] = num_threads;
                auto cmp = verifier::compare_overlaps_with_truth(vf_overlaps, {}, vf_gt);
                run["compare"] = { {"true_positives", cmp.true_positives}, {"truth_total", cmp.truth_total}, {"algo_total", cmp.algo_total}, {"covers_truth", cmp.covers_truth} };
                aggregate["runs"].push_back(run);
            }
            sort_axis = 0;
            {
                for (int w = 0; w < warmup; ++w) {
                    std::vector<std::pair<int, int>> tmp;
                    int ax = 0;
                    scalable_ccd::sort_and_sweep(eboxes, ax, tmp);
                }
                std::vector<double> times;
                times.reserve(repeat);
                for (int r = 0; r < repeat; ++r) {
                    spdlog::debug("[CPU][{}] broad_ee run {}/{}: E={}", s.scene, r+1, repeat, eboxes.size());
                    scalable_ccd::Timer t; t.start();
                    scalable_ccd::sort_and_sweep(eboxes, sort_axis, ee_overlaps);
                    t.stop();
                    spdlog::debug("[CPU][{}] broad_ee done: overlaps={}", s.scene, ee_overlaps.size());
                    times.push_back(t.getElapsedTimeInMilliSec());
                }
                double avg_ms = 0.0;
                for (double x : times) avg_ms += x;
                avg_ms /= std::max(1, (int)times.size());
                // Offset to match ground-truth encoding (see tests)
                {
                    int offset = static_cast<int>(vboxes.size());
                    for (auto& p : ee_overlaps) {
                        p.first += offset;
                        p.second += offset;
                    }
                }
                json run;
                run["scene"] = s.scene;
                run["t0"] = file_t0.filename().string();
                run["t1"] = file_t1.filename().string();
                run["backend"] = "cpu";
                run["steps"] = {
                    {"stage","broad_ee"},
                    {"avg_ms", avg_ms},
                    {"repeats", repeat},
                    {"warmup", warmup},
                    {"build_boxes_ms", build_boxes_ms},
                    {"sort_axis_end", sort_axis}
                };
                run["threads"] = num_threads;
                auto cmp = verifier::compare_overlaps_with_truth(ee_overlaps, {}, ee_gt);
                run["compare"] = { {"true_positives", cmp.true_positives}, {"truth_total", cmp.truth_total}, {"algo_total", cmp.algo_total}, {"covers_truth", cmp.covers_truth} };
                aggregate["runs"].push_back(run);
            }
        }

#ifdef SCALABLE_CCD_WITH_CUDA
        // ---------------- CUDA Broad Phase ----------------
        if (backend == "cuda" || backend == "both") {
            using namespace scalable_ccd::cuda;

            std::vector<scalable_ccd::cuda::AABB> vertex_boxes, edge_boxes, face_boxes;
            build_vertex_boxes(mp.V0, mp.V1, vertex_boxes);
            build_edge_boxes(vertex_boxes, mp.E, edge_boxes);
            build_face_boxes(vertex_boxes, mp.F, face_boxes);

            BroadPhase broad;

            // VF
            scalable_ccd::profiler().clear();
            {
                scalable_ccd::Timer tb; tb.start();
                broad.build(std::make_shared<DeviceAABBs>(vertex_boxes),
                            std::make_shared<DeviceAABBs>(face_boxes));
                tb.stop();
                // warmup
                for (int w = 0; w < warmup; ++w) {
                    (void)broad.detect_overlaps();
#ifdef SCALABLE_CCD_WITH_CUDA
                    cudaDeviceSynchronize();
#endif
                }
                std::vector<double> times;
                times.reserve(repeat);
                std::vector<std::pair<int,int>> vf_overlaps;
                for (int r = 0; r < repeat; ++r) {
                    scalable_ccd::Timer t; t.start();
                    vf_overlaps = broad.detect_overlaps();
#ifdef SCALABLE_CCD_WITH_CUDA
                    cudaDeviceSynchronize();
#endif
                    t.stop();
                    times.push_back(t.getElapsedTimeInMilliSec());
                }
                double avg_ms = 0.0;
                for (double x : times) avg_ms += x;
                avg_ms /= std::max(1, (int)times.size());

                // Offset for comparing with Mathematica ground-truth (same as tests)
                int offset = static_cast<int>(vertex_boxes.size()) + static_cast<int>(edge_boxes.size());
                for (auto& p : vf_overlaps) {
                    p.second += offset;
                }

                json run;
                run["scene"] = s.scene;
                run["t0"] = file_t0.filename().string();
                run["t1"] = file_t1.filename().string();
                run["backend"] = "cuda";
                run["steps"] = {
                    {"stage","broad_vf"},
                    {"avg_ms", avg_ms},
                    {"repeats", repeat},
                    {"warmup", warmup},
                    {"gpu_build_ms", tb.getElapsedTimeInMilliSec()}
                };
                auto cmp = verifier::compare_overlaps_with_truth(vf_overlaps, {}, vf_gt);
                run["compare"] = { {"true_positives", cmp.true_positives}, {"truth_total", cmp.truth_total}, {"algo_total", cmp.algo_total}, {"covers_truth", cmp.covers_truth} };
                run["profiler"] = scalable_ccd::profiler().data();
                aggregate["runs"].push_back(run);
            }

            // EE
            scalable_ccd::profiler().clear();
            {
                scalable_ccd::Timer tb; tb.start();
                broad.build(std::make_shared<DeviceAABBs>(edge_boxes));
                tb.stop();
                for (int w = 0; w < warmup; ++w) {
                    (void)broad.detect_overlaps();
#ifdef SCALABLE_CCD_WITH_CUDA
                    cudaDeviceSynchronize();
#endif
                }
                std::vector<double> times;
                times.reserve(repeat);
                std::vector<std::pair<int,int>> ee_overlaps;
                for (int r = 0; r < repeat; ++r) {
                    scalable_ccd::Timer t; t.start();
                    ee_overlaps = broad.detect_overlaps();
#ifdef SCALABLE_CCD_WITH_CUDA
                    cudaDeviceSynchronize();
#endif
                    t.stop();
                    times.push_back(t.getElapsedTimeInMilliSec());
                }
                double avg_ms = 0.0;
                for (double x : times) avg_ms += x;
                avg_ms /= std::max(1, (int)times.size());

                // Offset (same as tests)
                int offset = static_cast<int>(vertex_boxes.size());
                for (auto& p : ee_overlaps) {
                    p.first += offset;
                    p.second += offset;
                }

                json run;
                run["scene"] = s.scene;
                run["t0"] = file_t0.filename().string();
                run["t1"] = file_t1.filename().string();
                run["backend"] = "cuda";
                run["steps"] = {
                    {"stage","broad_ee"},
                    {"avg_ms", avg_ms},
                    {"repeats", repeat},
                    {"warmup", warmup},
                    {"gpu_build_ms", tb.getElapsedTimeInMilliSec()}
                };
                auto cmp = verifier::compare_overlaps_with_truth(ee_overlaps, {}, ee_gt);
                run["compare"] = { {"true_positives", cmp.true_positives}, {"truth_total", cmp.truth_total}, {"algo_total", cmp.algo_total}, {"covers_truth", cmp.covers_truth} };
                run["profiler"] = scalable_ccd::profiler().data();
                aggregate["runs"].push_back(run);
            }
        }
#else
        (void)backend;
#endif

                // ---------------- Metal Broad Phase ----------------
        if (backend == "metal") {
#ifdef SCALABLE_CCD_WITH_METAL
            std::vector<AABB> vboxes, eboxes, fboxes;
            double build_boxes_ms = 0.0;
            {
                scalable_ccd::Timer t;
                t.start();
                verifier::build_cpu_boxes(mp, vboxes, eboxes, fboxes);
                t.stop();
                build_boxes_ms = t.getElapsedTimeInMilliSec();
                spdlog::info("[METAL] {}: 构建AABB {:.3f} ms (V={},E={},F={})",
                    s.scene, build_boxes_ms, vboxes.size(), eboxes.size(), fboxes.size());
            }

            const size_t vf_truth = load_truth_size(vf_gt);
            const size_t ee_truth = load_truth_size(ee_gt);

            {
                auto vf_res = run_metal_stage(vboxes, &fboxes, true, metal_use_stq, warmup, repeat, vf_truth);
                int offset = static_cast<int>(vboxes.size() + eboxes.size());
                for (auto& p : vf_res.overlaps) {
                    p.second += offset;
                }
                json run;
                run["scene"] = s.scene;
                run["t0"] = file_t0.filename().string();
                run["t1"] = file_t1.filename().string();
                run["backend"] = metal_use_stq ? "metal_stq" : "metal";
                run["steps"] = {
                    {"stage","broad_vf"},
                    {"avg_ms", vf_res.avg_host_ms},
                    {"gpu_ms", vf_res.avg_gpu_ms},
                    {"e2e_ms", vf_res.e2e_ms},
                    {"repeats", repeat},
                    {"warmup", warmup},
                    {"build_boxes_ms", build_boxes_ms},
                    {"sort_axis_end", vf_res.sort_axis},
                    {"mode", metal_mode}
                };
                run["threads"] = num_threads;
                auto cmp = verifier::compare_overlaps_with_truth(vf_res.overlaps, {}, vf_gt);
                run["compare"] = { {"true_positives", cmp.true_positives}, {"truth_total", cmp.truth_total}, {"algo_total", cmp.algo_total}, {"covers_truth", cmp.covers_truth} };
                aggregate["runs"].push_back(run);
            }

            {
                auto ee_res = run_metal_stage(eboxes, nullptr, false, metal_use_stq, warmup, repeat, ee_truth);
                int offset = static_cast<int>(vboxes.size());
                for (auto& p : ee_res.overlaps) {
                    p.first += offset;
                    p.second += offset;
                }
                json run;
                run["scene"] = s.scene;
                run["t0"] = file_t0.filename().string();
                run["t1"] = file_t1.filename().string();
                run["backend"] = metal_use_stq ? "metal_stq" : "metal";
                run["steps"] = {
                    {"stage","broad_ee"},
                    {"avg_ms", ee_res.avg_host_ms},
                    {"gpu_ms", ee_res.avg_gpu_ms},
                    {"e2e_ms", ee_res.e2e_ms},
                    {"repeats", repeat},
                    {"warmup", warmup},
                    {"build_boxes_ms", build_boxes_ms},
                    {"sort_axis_end", ee_res.sort_axis},
                    {"mode", metal_mode}
                };
                run["threads"] = num_threads;
                auto cmp = verifier::compare_overlaps_with_truth(ee_res.overlaps, {}, ee_gt);
                run["compare"] = { {"true_positives", cmp.true_positives}, {"truth_total", cmp.truth_total}, {"algo_total", cmp.algo_total}, {"covers_truth", cmp.covers_truth} };
                aggregate["runs"].push_back(run);
            }
#else
            spdlog::warn("Metal 后端未编译（缺少 SCALABLE_CCD_WITH_METAL 或非 Apple 平台），跳过 Metal 验证。");
#endif
        }

// ---------------- Per-Query Verification (optional) ----------------
        if (run_queries) {
#ifdef SCALABLE_CCD_WITH_CUDA
            auto step_from_path = [&](const fs::path& p)->int{
                int st=-1; extract_step_from_filename(p.stem().string(), st); return st;
            };
            int st0 = step_from_path(file_t0);
            if (st0 >= 0) {
                auto qrs = verifier::verify_queries_for_step(base, s.scene, st0, max_queries, /*on_cuda_only*/true);
                for (const auto& qr : qrs) {
                    json jq;
                    jq["scene"] = qr.scene;
                    jq["step"] = qr.step;
                    jq["type"] = qr.type;
                    jq["csv_path"] = fs::relative(qr.csv_path, base).string();
                    jq["truth_positives"] = qr.truth_positives;
                    jq["algo_positives"] = qr.algo_positives;
                    jq["mismatches"] = qr.mismatches;
                    jq["total"] = qr.total;
                    jq["avg_ms"] = qr.avg_ms;
                    aggregate["query_runs"].push_back(jq);
                }
            }
#else
            (void)max_queries;
#endif
        }
    }

    // 写出报告
    fs::create_directories(out_dir);
    auto summary_path = out_dir / "summary.json";
    verifier::write_json(summary_path, aggregate);
    auto html = verifier::make_html_report_with_queries(aggregate);
    {
        std::ofstream out(out_dir / "report.html");
        out << html;
    }

    spdlog::info("报告已生成: {}", (out_dir / "report.html").string());
    return 0;
}
