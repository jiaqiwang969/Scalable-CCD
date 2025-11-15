#include <catch2/catch_test_macros.hpp>

#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/config.hpp>
#ifdef SCALABLE_CCD_WITH_METAL2
#include <scalable_ccd/metal2/broad_phase/broad_phase.hpp>
#include <scalable_ccd/metal2/broad_phase/aabb.hpp>
#endif

#include <filesystem>
namespace fs = std::filesystem;

TEST_CASE("Test Metal2 broad phase (strict correctness)", "[metal2][broad_phase]")
{
#if defined(SCALABLE_CCD_WITH_METAL2) && defined(__APPLE__)
    using namespace scalable_ccd;

    const fs::path data(SCALABLE_CCD_DATA_DIR);

    // 选用 data-full 中存在的场景
    struct Scene {
        const char* scene;
        const char* t0;
        const char* t1;
        const char* vf_json;
        const char* ee_json;
    };
    const Scene scenes[] = {
        { "armadillo-rollers", "326.ply", "327.ply", "326vf.json", "326ee.json" },
        { "cloth-funnel", "227.ply", "228.ply", "227vf.json", "227ee.json" },
        { "n-body-simulation", "frames/balls16_18.ply", "frames/balls16_19.ply", "boxes/18vf.json", "boxes/18ee.json" },
    };

    for (const auto& sc : scenes) {
        fs::path base = data / sc.scene;
        fs::path file_t0 = base / sc.t0;
        fs::path file_t1 = base / sc.t1;
        fs::path vf_truth = base / "boxes" / sc.vf_json;
        fs::path ee_truth = base / "boxes" / sc.ee_json;
        if (!fs::exists(file_t0) || !fs::exists(file_t1) || !fs::exists(vf_truth) || !fs::exists(ee_truth)) {
            // 数据可能缺失，跳过
            continue;
        }

        // 读取网格并构造 AABB
        Eigen::MatrixXd V0, V1;
        Eigen::MatrixXi F, E;
        verifier::parse_mesh(file_t0, file_t1, V0, V1, F, E);

        std::vector<AABB> vboxes, eboxes, fboxes;
        build_vertex_boxes(V0, V1, vboxes);
        build_edge_boxes(vboxes, E, eboxes);
        build_face_boxes(vboxes, F, fboxes);

        // Metal2 BroadPhase（严格正确性：最终回退 CPU）
        {
            scalable_ccd::metal2::BroadPhase bp;
            auto dV = std::make_shared<scalable_ccd::metal2::DeviceAABBs>(vboxes);
            auto dF = std::make_shared<scalable_ccd::metal2::DeviceAABBs>(fboxes);
            bp.build(dV, dF);
            auto vf_overlaps = bp.detect_overlaps();
            // 对齐 ground-truth 偏移
            int offset = static_cast<int>(vboxes.size() + eboxes.size());
            for (auto& p : vf_overlaps) p.second += offset;
            compare_mathematica(vf_overlaps, vf_truth);
        }
        {
            scalable_ccd::metal2::BroadPhase bp;
            auto dE = std::make_shared<scalable_ccd::metal2::DeviceAABBs>(eboxes);
            bp.build(dE);
            auto ee_overlaps = bp.detect_overlaps();
            // 对齐 ground-truth 偏移
            int offset = static_cast<int>(vboxes.size());
            for (auto& p : ee_overlaps) { p.first += offset; p.second += offset; }
            compare_mathematica(ee_overlaps, ee_truth);
        }
    }
#else
    SUCCEED("Metal2 not enabled or not on Apple platform; test skipped.");
#endif
}

