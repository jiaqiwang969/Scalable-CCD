// test_performance_metal2.cpp
// Standalone performance test for Metal2 broad phase with JSON output

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <ctime>
#include <sstream>

#include <scalable_ccd/config.hpp>
#ifdef SCALABLE_CCD_WITH_METAL2
#include <scalable_ccd/metal2/broad_phase/broad_phase.hpp>
#include <scalable_ccd/metal2/broad_phase/aabb.hpp>
#include <scalable_ccd/metal2/runtime/runtime.hpp>
#endif

// Local mesh parsing without Catch2 dependency
#include <scalable_ccd/broad_phase/aabb.hpp>
#include <igl/edges.h>
#include <igl/read_triangle_mesh.h>
#include <Eigen/Core>

namespace fs = std::filesystem;

namespace {
void parse_mesh_local(
    const fs::path& file_t0,
    const fs::path& file_t1,
    Eigen::MatrixXd& V0,
    Eigen::MatrixXd& V1,
    Eigen::MatrixXi& F,
    Eigen::MatrixXi& E)
{
    if (!igl::read_triangle_mesh(file_t0.string(), V0, F)) {
        throw std::runtime_error("Failed to read mesh: " + file_t0.string());
    }
    if (!igl::read_triangle_mesh(file_t1.string(), V1, F)) {
        throw std::runtime_error("Failed to read mesh: " + file_t1.string());
    }
    igl::edges(F, E);
}
} // anonymous namespace

using Clock = std::chrono::high_resolution_clock;

struct TestResult {
    std::string section;
    std::string slug;
    std::string case_name;
    int vf_pairs;
    int ee_pairs;
    double vf_total_ms;
    double ee_total_ms;
    double host_total_ms;
    double vf_stq_ms;
    double ee_stq_ms;
    std::string notes;
};

std::string get_device_name() {
#ifdef __APPLE__
    // Try to get GPU name from system_profiler
    FILE* pipe = popen("system_profiler SPDisplaysDataType 2>/dev/null | grep 'Chipset Model' | head -1 | cut -d: -f2", "r");
    if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            pclose(pipe);
            std::string result(buffer);
            // Trim whitespace
            result.erase(0, result.find_first_not_of(" \t\n\r"));
            result.erase(result.find_last_not_of(" \t\n\r") + 1);
            if (!result.empty()) return result;
        }
        pclose(pipe);
    }
#endif
    return "Apple GPU";
}

std::string get_device_alias(const std::string& name) {
    std::string alias = name;
    for (char& c : alias) {
        c = std::tolower(c);
        if (c == ' ' || c == '-') c = '_';
    }
    return alias;
}

void write_json(const TestResult& result, const std::string& device, const std::string& alias, const fs::path& output_dir) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"backend\": \"metal\",\n";
    json << "  \"device\": \"" << device << "\",\n";
    json << "  \"device_alias\": \"" << alias << "\",\n";
    json << "  \"category\": \"broad_phase_sap\",\n";
    json << "  \"case_name\": \"" << result.case_name << "\",\n";
    json << "  \"section\": \"" << result.section << "\",\n";
    json << "  \"slug\": \"" << result.slug << "_" << alias << "\",\n";
    json << "  \"vf_pairs\": " << result.vf_pairs << ",\n";
    json << "  \"ee_pairs\": " << result.ee_pairs << ",\n";
    json << "  \"host_total_ms\": " << std::fixed << std::setprecision(3) << result.host_total_ms << ",\n";
    json << "  \"gpu_ms\": " << std::fixed << std::setprecision(3) << result.host_total_ms << ",\n";
    json << "  \"vf_total_ms\": " << std::fixed << std::setprecision(3) << result.vf_total_ms << ",\n";
    json << "  \"ee_total_ms\": " << std::fixed << std::setprecision(3) << result.ee_total_ms << ",\n";
    json << "  \"vf_stq_ms\": " << std::fixed << std::setprecision(3) << result.vf_stq_ms << ",\n";
    json << "  \"ee_stq_ms\": " << std::fixed << std::setprecision(3) << result.ee_stq_ms << ",\n";
    json << "  \"notes\": \"" << result.notes << "\",\n";
    json << "  \"timestamp\": " << std::time(nullptr) << "\n";
    json << "}\n";

    fs::path out_path = output_dir / ("metal_sap_" + result.slug + "_" + alias + ".json");
    std::ofstream ofs(out_path);
    ofs << json.str();
    ofs.close();
    std::cout << "  -> Saved to " << out_path << "\n";
}

TestResult run_test(const std::string& section, const std::string& slug, const std::string& case_name,
                    const fs::path& t0_path, const fs::path& t1_path, const std::string& notes) {
    TestResult result;
    result.section = section;
    result.slug = slug;
    result.case_name = case_name;
    result.notes = notes;
    result.vf_pairs = 0;
    result.ee_pairs = 0;
    result.vf_total_ms = 0;
    result.ee_total_ms = 0;
    result.host_total_ms = 0;
    result.vf_stq_ms = 0;
    result.ee_stq_ms = 0;

#if defined(SCALABLE_CCD_WITH_METAL2) && defined(__APPLE__)
    using namespace scalable_ccd;

    std::cout << "\n========================================\n";
    std::cout << "Test: " << section << "\n";
    std::cout << "========================================\n";

    auto total_start = Clock::now();

    // Read mesh
    Eigen::MatrixXd V0, V1;
    Eigen::MatrixXi F, E;
    parse_mesh_local(t0_path, t1_path, V0, V1, F, E);

    std::cout << "Vertices: " << V0.rows() << "\n";
    std::cout << "Faces: " << F.rows() << "\n";
    std::cout << "Edges: " << E.rows() << "\n";

    // Build boxes
    std::vector<AABB> vboxes, eboxes, fboxes;
    build_vertex_boxes(V0, V1, vboxes);
    build_edge_boxes(vboxes, E, eboxes);
    build_face_boxes(vboxes, F, fboxes);

    std::cout << "Vertex boxes: " << vboxes.size() << "\n";
    std::cout << "Edge boxes: " << eboxes.size() << "\n";
    std::cout << "Face boxes: " << fboxes.size() << "\n";

    // VF test
    {
        std::cout << "\n--- VF (Vertex-Face) ---\n";
        auto start = Clock::now();

        scalable_ccd::metal2::BroadPhase bp;
        auto dV = std::make_shared<scalable_ccd::metal2::DeviceAABBs>(vboxes);
        auto dF = std::make_shared<scalable_ccd::metal2::DeviceAABBs>(fboxes);
        bp.build(dV, dF);

        auto build_time = Clock::now();
        auto vf_overlaps = bp.detect_overlaps();
        auto end = Clock::now();

        double build_ms = std::chrono::duration<double, std::milli>(build_time - start).count();
        double detect_ms = std::chrono::duration<double, std::milli>(end - build_time).count();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        result.vf_pairs = static_cast<int>(vf_overlaps.size());
        result.vf_total_ms = total_ms;

        std::cout << "  Overlaps found: " << vf_overlaps.size() << "\n";
        std::cout << "  Build time: " << std::fixed << std::setprecision(2) << build_ms << " ms\n";
        std::cout << "  Detect time: " << std::fixed << std::setprecision(2) << detect_ms << " ms\n";
        std::cout << "  Total time: " << std::fixed << std::setprecision(2) << total_ms << " ms\n";

        // Get GPU timing from runtime
        double stq_ms = scalable_ccd::metal2::Metal2Runtime::instance().lastSTQPairsMs();
        double yz_ms = scalable_ccd::metal2::Metal2Runtime::instance().lastYZFilterMs();
        if (stq_ms >= 0) {
            std::cout << "  GPU STQ time: " << stq_ms << " ms\n";
            result.vf_stq_ms = stq_ms;
        }
        if (yz_ms >= 0) std::cout << "  GPU YZ filter time: " << yz_ms << " ms\n";
    }

    // EE test
    {
        std::cout << "\n--- EE (Edge-Edge) ---\n";
        auto start = Clock::now();

        scalable_ccd::metal2::BroadPhase bp;
        auto dE = std::make_shared<scalable_ccd::metal2::DeviceAABBs>(eboxes);
        bp.build(dE);

        auto build_time = Clock::now();
        auto ee_overlaps = bp.detect_overlaps();
        auto end = Clock::now();

        double build_ms = std::chrono::duration<double, std::milli>(build_time - start).count();
        double detect_ms = std::chrono::duration<double, std::milli>(end - build_time).count();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        result.ee_pairs = static_cast<int>(ee_overlaps.size());
        result.ee_total_ms = total_ms;

        std::cout << "  Overlaps found: " << ee_overlaps.size() << "\n";
        std::cout << "  Build time: " << std::fixed << std::setprecision(2) << build_ms << " ms\n";
        std::cout << "  Detect time: " << std::fixed << std::setprecision(2) << detect_ms << " ms\n";
        std::cout << "  Total time: " << std::fixed << std::setprecision(2) << total_ms << " ms\n";

        double stq_ms = scalable_ccd::metal2::Metal2Runtime::instance().lastSTQPairsMs();
        double yz_ms = scalable_ccd::metal2::Metal2Runtime::instance().lastYZFilterMs();
        if (stq_ms >= 0) {
            std::cout << "  GPU STQ time: " << stq_ms << " ms\n";
            result.ee_stq_ms = stq_ms;
        }
        if (yz_ms >= 0) std::cout << "  GPU YZ filter time: " << yz_ms << " ms\n";
    }

    auto total_end = Clock::now();
    result.host_total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << "\nTotal test time: " << std::fixed << std::setprecision(2) << result.host_total_ms << " ms\n";
#else
    std::cout << "Metal2 not enabled\n";
#endif
    return result;
}

int main(int argc, char* argv[]) {
    // Use strict mode (CPU path) which is currently faster
    // GPU fused path has performance issues to investigate
    // setenv("SCALABLE_CCD_METAL2_USE_STQ", "1", 1);
    // setenv("SCALABLE_CCD_METAL2_FILTER", "gpu", 1);
    // setenv("SCALABLE_CCD_METAL2_STQ_MAX_NEIGHBORS", "512", 1);

    fs::path data("/Users/jqwang/128-ccd-cuda2metal/Scalable-CCD/tests/data-full");
    fs::path output_dir("/Users/jqwang/128-ccd-cuda2metal/Scalable-CCD/tests/results");

    // Allow override via command line
    if (argc > 1) {
        data = argv[1];
    }
    if (argc > 2) {
        output_dir = argv[2];
    }

    // Ensure output directory exists
    fs::create_directories(output_dir);

    std::string device = get_device_name();
    std::string alias = get_device_alias(device);

    std::cout << "Metal2 Broad Phase Performance Test\n";
    std::cout << "====================================\n";
    std::cout << "Device: " << device << " (" << alias << ")\n";
    std::cout << "Data path: " << data << "\n";
    std::cout << "Output path: " << output_dir << "\n";

    std::vector<TestResult> results;

    // Armadillo Rollers
    auto r1 = run_test("Armadillo-Rollers", "armadillo_rollers", "Armadillo-Rollers：宽阶段",
        data / "armadillo-rollers/frames/326.ply",
        data / "armadillo-rollers/frames/327.ply",
        "犰狳滚轮模拟；Metal2 SAP 模式；包含 mesh 读取 / AABB 构建 / 两次检测");
    write_json(r1, device, alias, output_dir);
    results.push_back(r1);

    // Cloth Funnel
    auto r2 = run_test("Cloth-Funnel", "cloth_funnel", "Cloth-Funnel：宽阶段",
        data / "cloth-funnel/frames/227.ply",
        data / "cloth-funnel/frames/228.ply",
        "布料漏斗；Metal2 SAP 模式；包含 mesh 读取 / AABB 构建 / 两次检测");
    write_json(r2, device, alias, output_dir);
    results.push_back(r2);

    // N-Body Simulation
    auto r3 = run_test("N-Body", "n_body", "N-Body：宽阶段",
        data / "n-body-simulation/frames/balls16_18.ply",
        data / "n-body-simulation/frames/balls16_19.ply",
        "N体模拟；Metal2 SAP 模式；包含 mesh 读取 / AABB 构建 / 两次检测");
    write_json(r3, device, alias, output_dir);
    results.push_back(r3);

    std::cout << "\n========================================\n";
    std::cout << "Summary\n";
    std::cout << "========================================\n";
    std::cout << std::left << std::setw(20) << "Scenario"
              << std::right << std::setw(10) << "VF pairs"
              << std::setw(10) << "EE pairs"
              << std::setw(12) << "Total (ms)" << "\n";
    std::cout << std::string(52, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(20) << r.section
                  << std::right << std::setw(10) << r.vf_pairs
                  << std::setw(10) << r.ee_pairs
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.host_total_ms << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "All tests complete! JSON files saved to " << output_dir << "\n";
    std::cout << "========================================\n";

    return 0;
}
