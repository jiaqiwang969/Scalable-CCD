// Implementation of per-query verification by calling GPU narrow-phase on tiny meshes.

#include "queries.hpp"

#include <scalable_ccd/config.hpp>

#ifdef SCALABLE_CCD_WITH_CUDA
#include <scalable_ccd/utils/timer.hpp>
#include <scalable_ccd/cuda/ccd.cuh>
#endif

#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <fstream>
#include <sstream>

namespace verifier {

namespace {

struct Rat3 {
    double x, y, z;
};

// Parse one CSV row with 6 integers: x_num, x_den, y_num, y_den, z_num, z_den
static bool parse_rat3(const std::string& line, Rat3& out)
{
    std::istringstream iss(line);
    std::string tok;
    long long vals[6];
    int idx = 0;
    while (std::getline(iss, tok, ',')) {
        // trim
        size_t b = tok.find_first_not_of(" \t\r\n");
        size_t e = tok.find_last_not_of(" \t\r\n");
        if (b == std::string::npos) tok.clear(); else tok = tok.substr(b, e - b + 1);
        if (tok.empty()) return false;
        try {
            vals[idx++] = std::stoll(tok);
        } catch (...) {
            return false;
        }
        if (idx == 6) break;
    }
    if (idx != 6) return false;
    auto safe_div = [](long long num, long long den) -> double {
        if (den == 0) return 0.0;
        return static_cast<double>(num) / static_cast<double>(den);
    };
    out.x = safe_div(vals[0], vals[1]);
    out.y = safe_div(vals[2], vals[3]);
    out.z = safe_div(vals[4], vals[5]);
    return true;
}

static std::vector<int> load_truth_bools(const std::filesystem::path& truth_json)
{
    std::vector<int> out;
    std::ifstream in(truth_json);
    if (!in.good()) return out;
    nlohmann::json j = nlohmann::json::parse(in, nullptr, true, true);
    if (!j.is_array()) return out;
    out.reserve(j.size());
    for (auto& v : j) {
        out.push_back(v.get<bool>() ? 1 : 0);
    }
    return out;
}

#ifdef SCALABLE_CCD_WITH_CUDA
static double eval_query_ee(const Rat3 ea0_t0, const Rat3 ea1_t0, const Rat3 eb0_t0, const Rat3 eb1_t0,
                            const Rat3 ea0_t1, const Rat3 ea1_t1, const Rat3 eb0_t1, const Rat3 eb1_t1,
                            int max_iter, double tol, double min_sep)
{
    using namespace scalable_ccd;
    Eigen::MatrixXd V0(4,3), V1(4,3);
    V0 << ea0_t0.x, ea0_t0.y, ea0_t0.z,
          ea1_t0.x, ea1_t0.y, ea1_t0.z,
          eb0_t0.x, eb0_t0.y, eb0_t0.z,
          eb1_t0.x, eb1_t0.y, eb1_t0.z;
    V1 << ea0_t1.x, ea0_t1.y, ea0_t1.z,
          ea1_t1.x, ea1_t1.y, ea1_t1.z,
          eb0_t1.x, eb0_t1.y, eb0_t1.z,
          eb1_t1.x, eb1_t1.y, eb1_t1.z;
    Eigen::MatrixXi E(2,2), F(0,3);
    E << 0,1, 2,3;
    double toi = scalable_ccd::cuda::ccd(V0, V1, E, F, /*min_distance*/min_sep,
                                         /*max_iterations*/max_iter, /*tolerance*/tol,
                                         /*allow_zero_toi*/true,
    #ifdef SCALABLE_CCD_TOI_PER_QUERY
                                         // collisions
    #endif
                                         /*memory_limit_GB*/0);
    return toi;
}

static double eval_query_vf(const Rat3 v_t0, const Rat3 f0_t0, const Rat3 f1_t0, const Rat3 f2_t0,
                            const Rat3 v_t1, const Rat3 f0_t1, const Rat3 f1_t1, const Rat3 f2_t1,
                            int max_iter, double tol, double min_sep)
{
    using namespace scalable_ccd;
    Eigen::MatrixXd V0(4,3), V1(4,3);
    V0 << v_t0.x,  v_t0.y,  v_t0.z,
          f0_t0.x, f0_t0.y, f0_t0.z,
          f1_t0.x, f1_t0.y, f1_t0.z,
          f2_t0.x, f2_t0.y, f2_t0.z;
    V1 << v_t1.x,  v_t1.y,  v_t1.z,
          f0_t1.x, f0_t1.y, f0_t1.z,
          f1_t1.x, f1_t1.y, f1_t1.z,
          f2_t1.x, f2_t1.y, f2_t1.z;
    Eigen::MatrixXi E(0,2), F(1,3);
    F << 1,2,3;
    double toi = scalable_ccd::cuda::ccd(V0, V1, E, F, /*min_distance*/min_sep,
                                         /*max_iterations*/max_iter, /*tolerance*/tol,
                                         /*allow_zero_toi*/true,
    #ifdef SCALABLE_CCD_TOI_PER_QUERY
                                         // collisions
    #endif
                                         /*memory_limit_GB*/0);
    return toi;
}
#endif

static bool ends_with(const std::string& s, const std::string& suf)
{
    return s.size() >= suf.size() && s.compare(s.size()-suf.size(), suf.size(), suf) == 0;
}

static std::vector<QueryFileResult> run_one(const std::filesystem::path& scene_dir,
                                            const std::string& scene_name,
                                            int step,
                                            const std::string& type,
                                            int max_queries,
                                            bool on_cuda_only)
{
    std::vector<QueryFileResult> out;
    std::filesystem::path csv = scene_dir / "queries" / (std::to_string(step) + type + ".csv");
    std::filesystem::path truth = scene_dir / "mma_bool" / (std::to_string(step) + type + "_mma_bool.json");
    if (!std::filesystem::exists(csv) || !std::filesystem::exists(truth)) {
        return out;
    }
    QueryFileResult r;
    r.scene = scene_name;
    r.step = step;
    r.type = type;
    r.csv_path = csv;
    r.truth_path = truth;
    std::ifstream in(csv);
    if (!in.good()) {
        return out;
    }
    std::vector<int> truth_bools = load_truth_bools(truth);
    std::string line;
    std::vector<std::string> rows;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        rows.push_back(line);
    }
    if (rows.size() % 8 != 0) {
        // malformed; ignore
        return out;
    }
    int N = static_cast<int>(rows.size() / 8);
    if (max_queries > 0) N = std::min(N, max_queries);
    r.total = N;
    int tp_truth = 0, tp_algo = 0, mism = 0;
    double total_ms = 0.0;
#ifdef SCALABLE_CCD_WITH_CUDA
    for (int i = 0; i < N; ++i) {
        bool truth_collide = (i < static_cast<int>(truth_bools.size()) ? (truth_bools[i] != 0) : false);
        tp_truth += (truth_collide ? 1 : 0);
        auto R = [&](int idx)->Rat3 { Rat3 r{}; parse_rat3(rows[idx], r); return r; };
        double toi = 1.0;
        scalable_ccd::Timer t; t.start();
        if (type == "ee") {
            // ea0_t0, ea1_t0, eb0_t0, eb1_t0, ea0_t1, ea1_t1, eb0_t1, eb1_t1
            toi = eval_query_ee(R(i*8+0), R(i*8+1), R(i*8+2), R(i*8+3),
                                R(i*8+4), R(i*8+5), R(i*8+6), R(i*8+7),
                                /*max_iter*/-1, /*tol*/1e-6, /*min_sep*/0.0);
        } else { // vf
            // v_t0, f0_t0, f1_t0, f2_t0, v_t1, f0_t1, f1_t1, f2_t1
            toi = eval_query_vf(R(i*8+0), R(i*8+1), R(i*8+2), R(i*8+3),
                                R(i*8+4), R(i*8+5), R(i*8+6), R(i*8+7),
                                /*max_iter*/-1, /*tol*/1e-6, /*min_sep*/0.0);
        }
        t.stop();
        total_ms += t.getElapsedTimeInMilliSec();
        bool algo_collide = (toi < 1.0);
        tp_algo += (algo_collide ? 1 : 0);
        if (algo_collide != truth_collide) ++mism;
    }
#else
    (void)on_cuda_only;
    // Without CUDA build, skip evaluation
    return out;
#endif
    r.truth_positives = tp_truth;
    r.algo_positives = tp_algo;
    r.mismatches = mism;
    if (N > 0) r.avg_ms = total_ms / static_cast<double>(N);
    out.push_back(r);
    return out;
}

} // namespace

std::vector<QueryFileResult> verify_queries_for_step(
    const std::filesystem::path& scene_dir,
    const std::string& scene_name,
    int step,
    int max_queries,
    bool on_cuda_only)
{
    (void)on_cuda_only;
    std::vector<QueryFileResult> agg;
    // try VF and EE
    auto r1 = run_one(scene_dir, scene_name, step, "vf", max_queries, on_cuda_only);
    agg.insert(agg.end(), r1.begin(), r1.end());
    auto r2 = run_one(scene_dir, scene_name, step, "ee", max_queries, on_cuda_only);
    agg.insert(agg.end(), r2.begin(), r2.end());
    return agg;
}

} // namespace verifier
