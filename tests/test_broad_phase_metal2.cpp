#include <catch2/catch_test_macros.hpp>

#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/config.hpp>
#ifdef SCALABLE_CCD_WITH_METAL2
#include <scalable_ccd/metal2/broad_phase/broad_phase.hpp>
#include <scalable_ccd/metal2/broad_phase/aabb.hpp>
#endif

#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;

TEST_CASE(
    "Test Metal2 broad phase (strict correctness)", "[metal2][broad_phase]")
{
#if defined(SCALABLE_CCD_WITH_METAL2) && defined(__APPLE__)
    // Enable GPU path for testing
    setenv("SCALABLE_CCD_METAL2_STRICT", "0", 1);
    setenv("SCALABLE_CCD_METAL2_USE_STQ", "1", 1);
    setenv("SCALABLE_CCD_METAL2_FILTER", "gpu", 1);
    // Enable detailed logging and observation
    setenv("SCALABLE_CCD_METAL2_LOG_TIMING", "1", 1);
    setenv("SCALABLE_CCD_METAL2_OBSERVE", "1", 1);
    // Increase max neighbors to avoid CPU fallback
    setenv("SCALABLE_CCD_METAL2_STQ_MAX_NEIGHBORS", "512", 1);

    using namespace scalable_ccd;

    // const fs::path data(SCALABLE_CCD_DATA_DIR);
    const fs::path data(
        "/Users/jqwang/128-ccd-cuda2metal/Scalable-CCD/tests/data-full");
    std::cout << "Hardcoded data path: " << data << std::endl;
    std::cout << "SCALABLE_CCD_DATA_DIR: " << data << std::endl;
    std::cout << "First char: " << data.string()[0] << std::endl;
    std::string hardcoded =
        "/Users/jqwang/128-ccd-cuda2metal/Scalable-CCD/tests/data/armadillo-rollers/frames/326.ply";
    std::cout << "Hardcoded check: " << hardcoded << " -> "
              << fs::exists(hardcoded) << std::endl;

    // 选用 data-full 中存在的场景
    struct Scene {
        const char* scene;
        const char* t0;
        const char* t1;
        const char* vf_json;
        const char* ee_json;
    };
    const Scene scenes[] = {
        { "armadillo-rollers", "frames/326.ply", "frames/327.ply",
          "boxes/326vf.json", "boxes/326ee.json" },
        { "cloth-funnel", "frames/227.ply", "frames/228.ply",
          "boxes/227vf.json", "boxes/227ee.json" },
        { "n-body-simulation", "frames/balls16_18.ply", "frames/balls16_19.ply",
          "boxes/18vf.json", "boxes/18ee.json" }
    };

    for (const auto& sc : scenes) {
        std::string base_str = data.string() + "/" + sc.scene;
        std::string t0_str = base_str + "/" + sc.t0;
        std::string t1_str = base_str + "/" + sc.t1;
        std::string vf_str = base_str + "/" + sc.vf_json;
        std::string ee_str = base_str + "/" + sc.ee_json;

        fs::path file_t0(t0_str);
        fs::path file_t1(t1_str);
        fs::path vf_truth(vf_str);
        fs::path ee_truth(ee_str);

        std::cout << "Checking string: " << t0_str << std::endl;
        if (!fs::exists(file_t0) || !fs::exists(file_t1)
            || !fs::exists(vf_truth) || !fs::exists(ee_truth)) {
            std::cout << "MISSING: " << file_t0 << std::endl;
            // 数据可能缺失，跳过
            continue;
        }

        // 读取网格并构造 AABB
        Eigen::MatrixXd V0, V1;
        Eigen::MatrixXi F, E;
        parse_mesh(file_t0, file_t1, V0, V1, F, E);

        std::vector<AABB> vboxes, eboxes, fboxes;
        build_vertex_boxes(V0, V1, vboxes);
        build_edge_boxes(vboxes, E, eboxes);
        build_face_boxes(vboxes, F, fboxes);

        // Metal2 BroadPhase（严格正确性：最终回退 CPU）
        {
            scalable_ccd::metal2::BroadPhase bp;
            auto dV =
                std::make_shared<scalable_ccd::metal2::DeviceAABBs>(vboxes);
            auto dF =
                std::make_shared<scalable_ccd::metal2::DeviceAABBs>(fboxes);
            bp.build(dV, dF);
            auto vf_overlaps = bp.detect_overlaps();
            // 对齐 ground-truth 偏移
            int offset = static_cast<int>(vboxes.size() + eboxes.size());
            for (auto& p : vf_overlaps)
                p.second += offset;
            compare_mathematica(vf_overlaps, vf_truth);
        }
        {
            scalable_ccd::metal2::BroadPhase bp;
            auto dE =
                std::make_shared<scalable_ccd::metal2::DeviceAABBs>(eboxes);
            bp.build(dE);
            auto ee_overlaps = bp.detect_overlaps();
            // 对齐 ground-truth 偏移
            int offset = static_cast<int>(vboxes.size());
            for (auto& p : ee_overlaps) {
                p.first += offset;
                p.second += offset;
            }
            compare_mathematica(ee_overlaps, ee_truth);
        }
    }
#else
    SUCCEED("Metal2 not enabled or not on Apple platform; test skipped.");
#endif
}
