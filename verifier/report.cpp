// Minimal standalone HTML generator (no external assets) and JSON writer.

#include "report.hpp"

#include <fstream>
#include <sstream>

namespace verifier {

void write_json(const std::filesystem::path& path, const nlohmann::json& j)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    out << j.dump(2) << std::endl;
}

static std::string esc(const std::string& s)
{
    std::string o;
    o.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '&': o += "&amp;"; break;
        case '<': o += "&lt;"; break;
        case '>': o += "&gt;"; break;
        case '"': o += "&quot;"; break;
        case '\'': o += "&#39;"; break;
        default: o += c; break;
        }
    }
    return o;
}

std::string make_html_report(const nlohmann::json& a)
{
    std::ostringstream os;
    os << "<!doctype html><html><head><meta charset=\"utf-8\"/>";
    os << "<title>Scalable-CCD Verifier Report</title>";
    os << "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:6px 10px}th{background:#f4f4f4}code{background:#f6f8fa;padding:2px 4px;border-radius:4px} .ok{color:#117733} .fail{color:#aa2222} .muted{color:#777}</style>";
    os << "</head><body>";
    // 顶部摘要
    size_t total_runs = a.contains("runs") && a["runs"].is_array() ? a["runs"].size() : 0;
    size_t cpu_runs = 0, cuda_runs = 0;
    if (a.contains("runs") && a["runs"].is_array()) {
        for (const auto& r : a["runs"]) {
            std::string be = r.value("backend", "");
            if (be == "cpu") ++cpu_runs; else if (be == "cuda") ++cuda_runs;
        }
    }
    os << "<h1>Scalable-CCD 验证报告</h1>";
    os << "<p class=\"muted\">总运行条目: " << total_runs << "，CPU: " << cpu_runs << "，CUDA: " << cuda_runs << "</p>";
    if (a.contains("envs")) {
        os << "<h2>环境信息（矩阵）</h2><pre><code>" << esc(a["envs"].dump(2)) << "</code></pre>";
    } else if (a.contains("env")) {
        os << "<h2>环境信息</h2><pre><code>" << esc(a["env"].dump(2)) << "</code></pre>";
    }
    os << "<h2>场景结果</h2>";
    os << "<table><tr><th>场景</th><th>帧t0</th><th>帧t1</th><th>标签</th><th>后端</th><th>阶段</th><th>均值(ms)</th><th>重复</th><th>预热</th><th>线程</th><th>真值</th><th>算法</th><th>命中</th><th>覆盖</th></tr>";
    if (a.contains("runs")) {
        for (const auto& r : a["runs"]) {
            std::string scene = r.value("scene", "");
            std::string t0 = r.value("t0", "");
            std::string t1 = r.value("t1", "");
            std::string be = r.value("backend", "");
            const auto& steps = r["steps"];
            const auto& cmp = r["compare"];
            os << "<tr>";
            os << "<td>" << esc(scene) << "</td>";
            os << "<td>" << esc(t0) << "</td>";
            os << "<td>" << esc(t1) << "</td>";
            std::string tag = r.value("env_tag", "");
            os << "<td>" << esc(tag) << "</td>";
            os << "<td>" << esc(be) << "</td>";
            os << "<td>" << esc(steps.value("stage", "")) << "</td>";
            double tm = steps.contains("avg_ms") ? steps.value("avg_ms", 0.0) : steps.value("time_ms", 0.0);
            os << "<td>" << tm << "</td>";
            os << "<td>" << steps.value("repeats", 1) << "</td>";
            os << "<td>" << steps.value("warmup", 0) << "</td>";
            int threads = r.value("threads", -1);
            os << "<td>" << (threads >= 0 ? std::to_string(threads) : "") << "</td>";
            os << "<td>" << cmp.value("truth_total", 0) << "</td>";
            os << "<td>" << cmp.value("algo_total", 0) << "</td>";
            os << "<td>" << cmp.value("true_positives", 0) << "</td>";
            bool ok = cmp.value("covers_truth", false);
            os << "<td class=\"" << (ok ? "ok" : "fail") << "\">" << (ok ? "是" : "否") << "</td>";
            os << "</tr>";
            // 可选：CUDA profiler 展开
            if (r.value("backend", "") == "cuda" && r.contains("profiler")) {
                os << "<tr><td colspan=\"13\"><details><summary>GPU Profiler</summary><pre><code>" << esc(r["profiler"].dump(2)) << "</code></pre></details></td></tr>";
            }
        }
    }
    os << "</table>";
    os << "</body></html>";
    return os.str();
}

std::string make_html_report_with_queries(const nlohmann::json& a)
{
    // Reuse base report generation then append query results if present
    std::string base = make_html_report(a);
    // cheap append by replacing closing tags
    std::string marker = "</body></html>";
    auto pos = base.rfind(marker);
    if (pos == std::string::npos) return base;
    std::ostringstream os;
    os << base.substr(0, pos);
    if (a.contains("query_runs") && a["query_runs"].is_array() && !a["query_runs"].empty()) {
        os << "<h2>逐 Query 验证</h2>";
        os << "<table><tr><th>场景</th><th>步</th><th>类型</th><th>文件</th><th>真值阳性</th><th>算法阳性</th><th>不一致</th><th>总数</th><th>平均耗时(ms/Query)</th></tr>";
        for (const auto& r : a["query_runs"]) {
            os << "<tr>";
            os << "<td>" << esc(r.value("scene","")) << "</td>";
            os << "<td>" << r.value("step",-1) << "</td>";
            os << "<td>" << esc(r.value("type","")) << "</td>";
            os << "<td>" << esc(r.value("csv_path","")) << "</td>";
            os << "<td>" << r.value("truth_positives",0) << "</td>";
            os << "<td>" << r.value("algo_positives",0) << "</td>";
            os << "<td>" << r.value("mismatches",0) << "</td>";
            os << "<td>" << r.value("total",0) << "</td>";
            os << "<td>" << r.value("avg_ms",0.0) << "</td>";
            os << "</tr>";
        }
        os << "</table>";
    }
    os << marker;
    return os.str();
}

} // namespace verifier
