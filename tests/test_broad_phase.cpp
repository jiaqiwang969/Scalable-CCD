#include <catch2/catch_test_macros.hpp>

#include "io.hpp"
#include "ground_truth.hpp"

#include <scalable_ccd/broad_phase/sort_and_sweep.hpp>
#include <scalable_ccd/utils/timer.hpp>
#include <scalable_ccd/utils/logger.hpp>

#include <vector>
#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;

TEST_CASE("Test CPU broad phase", "[cpu][broad_phase]")
{
    using namespace scalable_ccd;

    const fs::path data(SCALABLE_CCD_DATA_DIR);

    const fs::path file_t0 =
        data / "cloth-ball" / "frames" / "cloth_ball92.ply";
    const fs::path file_t1 =
        data / "cloth-ball" / "frames" / "cloth_ball93.ply";

    const fs::path vf_ground_truth =
        data / "cloth-ball" / "boxes" / "92vf.json";
    const fs::path ee_ground_truth =
        data / "cloth-ball" / "boxes" / "92ee.json";

    // ------------------------------------------------------------------------
    // Load meshes

    std::vector<AABB> vertex_boxes, edge_boxes, face_boxes;
    parse_mesh(file_t0, file_t1, vertex_boxes, edge_boxes, face_boxes);

    CHECK(vertex_boxes.size() == 46'598);
    CHECK(edge_boxes.size() == 138'825);
    CHECK(face_boxes.size() == 92'230);

    // ------------------------------------------------------------------------
    // Run

    double vf_ms = 0.0, ee_ms = 0.0;

    int sort_axis = 0;
    std::vector<std::pair<int, int>> vf_overlaps;
    {
        Timer t;
        t.start();
        sort_and_sweep(vertex_boxes, face_boxes, sort_axis, vf_overlaps);
        t.stop();
        vf_ms = t.getElapsedTimeInMilliSec();
    }
    sort_and_sweep(vertex_boxes, face_boxes, sort_axis, vf_overlaps);
    CHECK(sort_axis == 0); // check output sort axis

    sort_axis = 0; // Reset sort axis
    std::vector<std::pair<int, int>> ee_overlaps;
    {
        Timer t;
        t.start();
        sort_and_sweep(edge_boxes, sort_axis, ee_overlaps);
        t.stop();
        ee_ms = t.getElapsedTimeInMilliSec();
    }
    CHECK(sort_axis == 0); // check output sort axis

    // 输出便于对比 CUDA/Metal
    std::cout << "[CPU-SAP] VF_MS=" << vf_ms << " EE_MS=" << ee_ms << std::endl;

    // ------------------------------------------------------------------------
    // Compare

    CHECK(vf_overlaps.size() == 1'655'541);
    CHECK(ee_overlaps.size() == 5'197'332);

    // Offset the boxes to match the way ground truth was originally generated.
    int offset = vertex_boxes.size();
    for (auto& [a, b] : ee_overlaps) {
        a += offset;
        b += offset;
    }
    offset += edge_boxes.size();
    for (auto& [v, f] : vf_overlaps) {
        f += offset;
    }

    compare_mathematica(vf_overlaps, vf_ground_truth);
    compare_mathematica(ee_overlaps, ee_ground_truth);
}
