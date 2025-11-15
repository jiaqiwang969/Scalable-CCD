// Aggregate multiple summary.json files into one JSON and HTML using the same report generator.
#include "report.hpp"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "用法: " << argv[0] << " OUT_DIR summary1.json [summary2.json ...]\n";
        return 1;
    }
    fs::path out_dir = argv[1];
    std::vector<json> inputs;
    for (int i = 2; i < argc; ++i) {
        std::ifstream in(argv[i]);
        if (!in.good()) {
            std::cerr << "跳过不存在: " << argv[i] << "\n";
            continue;
        }
        json j = json::parse(in, nullptr, true, true);
        inputs.push_back(std::move(j));
    }
    json agg;
    agg["envs"] = json::array();
    agg["runs"] = json::array();
    for (const auto& s : inputs) {
        if (s.contains("env")) {
            agg["envs"].push_back(s["env"]);
        }
        if (s.contains("runs") && s["runs"].is_array()) {
            for (const auto& r : s["runs"]) {
                json rr = r;
                if (s.contains("env") && s["env"].contains("tag")) {
                    rr["env_tag"] = s["env"]["tag"];
                }
                agg["runs"].push_back(std::move(rr));
            }
        }
    }
    fs::create_directories(out_dir);
    verifier::write_json(out_dir / "aggregate.json", agg);
    auto html = verifier::make_html_report(agg);
    std::ofstream(out_dir / "aggregate.html") << html;
    std::cout << "聚合报告生成于: " << (out_dir / "aggregate.html").string() << "\n";
    return 0;
}

