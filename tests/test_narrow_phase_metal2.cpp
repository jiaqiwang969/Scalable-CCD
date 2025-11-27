#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>

#include "../src/scalable_ccd/metal2/runtime/runtime.hpp"

// Simple Vector3 for test
struct Vector3 {
    double x, y, z;
};

void run_test_vf()
{
    std::cout << "Running VF Test..." << std::endl;

    // Vertex moving from (0, 1, 0) to (0, -1, 0)
    // Triangle at y=0 plane: (-1, 0, -1), (1, 0, -1), (0, 0, 1)

    std::vector<double> v_t0 = {
        0.0,  1.0, 0.0,  // v0 (moving vertex)
        -1.0, 0.0, -1.0, // f0
        1.0,  0.0, -1.0, // f1
        0.0,  0.0, 1.0   // f2
    };

    std::vector<double> v_t1 = {
        0.0,  -1.0, 0.0,  // v0 (moved)
        -1.0, 0.0,  -1.0, // f0 (static)
        1.0,  0.0,  -1.0, // f1 (static)
        0.0,  0.0,  1.0   // f2 (static)
    };

    std::vector<int32_t> indices = {
        1, 2, 3 // Triangle face
    };

    std::vector<std::pair<int, int>> overlaps = {
        { 0, 0 } // Vertex 0 vs Face 0
    };

    double toi = 1.0;
    bool result = scalable_ccd::metal2::Metal2Runtime::instance().narrowPhase(
        v_t0, v_t1, indices, overlaps, true, 1e-6f, 1e-6f, 1000000, false, toi);

    if (result) {
        std::cout << "VF TOI: " << toi << std::endl;
        if (std::abs(toi - 0.5) < 1e-5) {
            std::cout << "VF Test PASSED" << std::endl;
        } else {
            std::cout << "VF Test FAILED (Expected 0.5)" << std::endl;
        }
    } else {
        std::cout << "VF Test FAILED (Runtime Error)" << std::endl;
    }
}

void run_test_ee()
{
    std::cout << "Running EE Test..." << std::endl;

    // Edge 1 moving from (0, 1, -1)-(0, 1, 1) to (0, -1, -1)-(0, -1, 1)
    // Edge 2 static at (-1, 0, 0)-(1, 0, 0)

    std::vector<double> v_t0 = {
        0.0,  1.0, -1.0, // e1_v0
        0.0,  1.0, 1.0,  // e1_v1
        -1.0, 0.0, 0.0,  // e2_v0
        1.0,  0.0, 0.0   // e2_v1
    };

    std::vector<double> v_t1 = {
        0.0,  -1.0, -1.0, // e1_v0
        0.0,  -1.0, 1.0,  // e1_v1
        -1.0, 0.0,  0.0,  // e2_v0
        1.0,  0.0,  0.0   // e2_v1
    };

    std::vector<int32_t> indices = {
        0, 1, // Edge 1
        2, 3  // Edge 2
    };

    std::vector<std::pair<int, int>> overlaps = {
        { 0, 1 } // Edge 0 vs Edge 1
    };

    double toi = 1.0;
    bool result = scalable_ccd::metal2::Metal2Runtime::instance().narrowPhase(
        v_t0, v_t1, indices, overlaps, false, 1e-6f, 1e-6f, 1000000, false,
        toi);

    if (result) {
        std::cout << "EE TOI: " << toi << std::endl;
        if (std::abs(toi - 0.5) < 1e-5) {
            std::cout << "EE Test PASSED" << std::endl;
        } else {
            std::cout << "EE Test FAILED (Expected 0.5)" << std::endl;
        }
    } else {
        std::cout << "EE Test FAILED (Runtime Error)" << std::endl;
    }
}

void run_test_miss()
{
    std::cout << "Running Miss Test..." << std::endl;

    // Vertex far away from triangle
    std::vector<double> v_t0 = {
        0.0,  10.0, 0.0,  // v0
        -1.0, 0.0,  -1.0, // f0
        1.0,  0.0,  -1.0, // f1
        0.0,  0.0,  1.0   // f2
    };

    std::vector<double> v_t1 = {
        0.0,  9.0, 0.0, // v0 (moving down but still far)
        -1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0
    };

    std::vector<int32_t> indices = { 1, 2, 3 };
    std::vector<std::pair<int, int>> overlaps = { { 0, 0 } };

    double toi = 1.0;
    bool result = scalable_ccd::metal2::Metal2Runtime::instance().narrowPhase(
        v_t0, v_t1, indices, overlaps, true, 1e-6f, 1e-6f, 1000000, false, toi);

    if (result) {
        if (toi == 1.0) { // Or whatever default/no-collision value is.
            // Actually, if no collision, toi should remain initial value (1.0)
            // or be untouched? The narrowPhase implementation initializes
            // atomic toi to 1.0 (implied? No, we pass it in). Wait, in
            // runtime.mm: float toi_f = (float)toi; ... bToI initialized with
            // it. In shader: atomic_min_float(toi, ...). So if no collision, it
            // stays 1.0.
            std::cout << "Miss Test PASSED (TOI: " << toi << ")" << std::endl;
        } else {
            std::cout << "Miss Test FAILED (TOI: " << toi << ", Expected: 1.0)"
                      << std::endl;
        }
    } else {
        std::cout << "Miss Test FAILED (Runtime Error)" << std::endl;
    }
}

void run_batch_test()
{
    std::cout << "Running Batch Test (1,000 queries)..." << std::endl;

    int n_queries = 1000;
    std::vector<double> v_t0;
    std::vector<double> v_t1;
    std::vector<int32_t> indices;
    std::vector<std::pair<int, int>> overlaps;

    v_t0.reserve(n_queries * 4 * 3);
    v_t1.reserve(n_queries * 4 * 3);
    indices.reserve(n_queries * 3);
    overlaps.reserve(n_queries);

    for (int i = 0; i < n_queries; ++i) {
        // Offset each query slightly to avoid identical data (though it doesn't
        // strictly matter for perf)
        double offset = i * 0.001;

        // VF Collision case
        // v0: (offset, 1, 0) -> (offset, -1, 0)
        // tri: (offset-1, 0, -1), (offset+1, 0, -1), (offset, 0, 1)

        // Vertices
        v_t0.push_back(offset);
        v_t0.push_back(1.0);
        v_t0.push_back(0.0);
        v_t0.push_back(offset - 1.0);
        v_t0.push_back(0.0);
        v_t0.push_back(-1.0);
        v_t0.push_back(offset + 1.0);
        v_t0.push_back(0.0);
        v_t0.push_back(-1.0);
        v_t0.push_back(offset);
        v_t0.push_back(0.0);
        v_t0.push_back(1.0);

        v_t1.push_back(offset);
        v_t1.push_back(-1.0);
        v_t1.push_back(0.0);
        v_t1.push_back(offset - 1.0);
        v_t1.push_back(0.0);
        v_t1.push_back(-1.0);
        v_t1.push_back(offset + 1.0);
        v_t1.push_back(0.0);
        v_t1.push_back(-1.0);
        v_t1.push_back(offset);
        v_t1.push_back(0.0);
        v_t1.push_back(1.0);

        // Indices (local to this group of 4 vertices)
        int base = i * 4;
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 3);

        // Overlap (Vertex 0 vs Face 0)
        // Note: In our flattened data logic in runtime.mm,
        // vi is the index in vertices array.
        // fi is the index in indices array (which is triplet index).
        // Here we have 1 face per query.
        overlaps.push_back({ base, i });
    }

    double toi = 1.0;
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    bool result = scalable_ccd::metal2::Metal2Runtime::instance().narrowPhase(
        v_t0, v_t1, indices, overlaps, true, 1e-6f, 1e-6f, 1000000, false, toi);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;

    if (result) {
        std::cout << "Batch Test Finished in " << ms.count() << " ms"
                  << std::endl;
        std::cout << "Batch TOI: " << toi << std::endl;
        // Since all collide at 0.5, min TOI should be ~0.5
        if (std::abs(toi - 0.5) < 1e-5) {
            std::cout << "Batch Test PASSED" << std::endl;
        } else {
            std::cout << "Batch Test FAILED (Expected ~0.5)" << std::endl;
        }
    } else {
        std::cout << "Batch Test FAILED (Runtime Error)" << std::endl;
    }
}

void run_batch_test_opt()
{
    std::cout << "\n=== Running Optimized Batch Test (1,000 queries) ===" << std::endl;

    int n_queries = 1000;
    std::vector<double> v_t0;
    std::vector<double> v_t1;
    std::vector<int32_t> indices;
    std::vector<std::pair<int, int>> overlaps;

    v_t0.reserve(n_queries * 4 * 3);
    v_t1.reserve(n_queries * 4 * 3);
    indices.reserve(n_queries * 3);
    overlaps.reserve(n_queries);

    for (int i = 0; i < n_queries; ++i) {
        double offset = i * 0.001;
        v_t0.push_back(offset);    v_t0.push_back(1.0);  v_t0.push_back(0.0);
        v_t0.push_back(offset - 1.0); v_t0.push_back(0.0); v_t0.push_back(-1.0);
        v_t0.push_back(offset + 1.0); v_t0.push_back(0.0); v_t0.push_back(-1.0);
        v_t0.push_back(offset);    v_t0.push_back(0.0);  v_t0.push_back(1.0);

        v_t1.push_back(offset);    v_t1.push_back(-1.0); v_t1.push_back(0.0);
        v_t1.push_back(offset - 1.0); v_t1.push_back(0.0); v_t1.push_back(-1.0);
        v_t1.push_back(offset + 1.0); v_t1.push_back(0.0); v_t1.push_back(-1.0);
        v_t1.push_back(offset);    v_t1.push_back(0.0);  v_t1.push_back(1.0);

        int base = i * 4;
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
        overlaps.push_back({ base, i });
    }

    // Run original version first for comparison
    double toi_orig = 1.0;
    auto start_orig = std::chrono::high_resolution_clock::now();
    bool result_orig = scalable_ccd::metal2::Metal2Runtime::instance().narrowPhase(
        v_t0, v_t1, indices, overlaps, true, 1e-6f, 1e-6f, 1000000, false, toi_orig);
    auto end_orig = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_orig = end_orig - start_orig;

    // Run optimized version
    double toi_opt = 1.0;
    auto start_opt = std::chrono::high_resolution_clock::now();
    bool result_opt = scalable_ccd::metal2::Metal2Runtime::instance().narrowPhaseOpt(
        v_t0, v_t1, indices, overlaps, true, 1e-6f, 1e-6f, 1000000, false, toi_opt);
    auto end_opt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_opt = end_opt - start_opt;

    std::cout << "Original:  " << ms_orig.count() << " ms, TOI=" << toi_orig << std::endl;
    std::cout << "Optimized: " << ms_opt.count() << " ms, TOI=" << toi_opt << std::endl;

    if (result_opt && result_orig) {
        double speedup = ms_orig.count() / ms_opt.count();
        std::cout << "Speedup: " << speedup << "x" << std::endl;

        if (std::abs(toi_opt - toi_orig) < 1e-5) {
            std::cout << "Result Consistency: PASSED" << std::endl;
        } else {
            std::cout << "Result Consistency: FAILED (toi_opt=" << toi_opt << " vs toi_orig=" << toi_orig << ")" << std::endl;
        }
    } else {
        std::cout << "Test FAILED (Runtime Error)" << std::endl;
    }
}

// 单独测试优化版
void run_single_test_opt()
{
    std::cout << "\n=== Running Single Query Optimized Test ===" << std::endl;

    // 与 run_test_vf 相同的数据
    std::vector<double> v_t0 = {
        0.0,  1.0, 0.0,  // v0 (moving vertex)
        -1.0, 0.0, -1.0, // f0
        1.0,  0.0, -1.0, // f1
        0.0,  0.0, 1.0   // f2
    };

    std::vector<double> v_t1 = {
        0.0,  -1.0, 0.0,  // v0 (moved)
        -1.0, 0.0,  -1.0, // f0 (static)
        1.0,  0.0,  -1.0, // f1 (static)
        0.0,  0.0,  1.0   // f2 (static)
    };

    std::vector<int32_t> indices = { 1, 2, 3 };
    std::vector<std::pair<int, int>> overlaps = { { 0, 0 } };

    double toi = 1.0;
    bool result = scalable_ccd::metal2::Metal2Runtime::instance().narrowPhaseOpt(
        v_t0, v_t1, indices, overlaps, true, 1e-6f, 1e-6f, 1000000, false, toi);

    std::cout << "Single Query Opt TOI: " << toi << std::endl;
    if (result && std::abs(toi - 0.5) < 1e-5) {
        std::cout << "Single Query Opt Test PASSED" << std::endl;
    } else {
        std::cout << "Single Query Opt Test FAILED (Expected 0.5, got " << toi << ")" << std::endl;
    }
}

void run_batch_test_large()
{
    std::cout << "\n=== Running Large Batch Test (10,000 queries) ===" << std::endl;

    int n_queries = 10000;
    std::vector<double> v_t0;
    std::vector<double> v_t1;
    std::vector<int32_t> indices;
    std::vector<std::pair<int, int>> overlaps;

    v_t0.reserve(n_queries * 4 * 3);
    v_t1.reserve(n_queries * 4 * 3);
    indices.reserve(n_queries * 3);
    overlaps.reserve(n_queries);

    for (int i = 0; i < n_queries; ++i) {
        double offset = i * 0.001;
        v_t0.push_back(offset);    v_t0.push_back(1.0);  v_t0.push_back(0.0);
        v_t0.push_back(offset - 1.0); v_t0.push_back(0.0); v_t0.push_back(-1.0);
        v_t0.push_back(offset + 1.0); v_t0.push_back(0.0); v_t0.push_back(-1.0);
        v_t0.push_back(offset);    v_t0.push_back(0.0);  v_t0.push_back(1.0);

        v_t1.push_back(offset);    v_t1.push_back(-1.0); v_t1.push_back(0.0);
        v_t1.push_back(offset - 1.0); v_t1.push_back(0.0); v_t1.push_back(-1.0);
        v_t1.push_back(offset + 1.0); v_t1.push_back(0.0); v_t1.push_back(-1.0);
        v_t1.push_back(offset);    v_t1.push_back(0.0);  v_t1.push_back(1.0);

        int base = i * 4;
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
        overlaps.push_back({ base, i });
    }

    // Run original version
    double toi_orig = 1.0;
    auto start_orig = std::chrono::high_resolution_clock::now();
    bool result_orig = scalable_ccd::metal2::Metal2Runtime::instance().narrowPhase(
        v_t0, v_t1, indices, overlaps, true, 1e-6f, 1e-6f, 1000000, false, toi_orig);
    auto end_orig = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_orig = end_orig - start_orig;

    // Run optimized version
    double toi_opt = 1.0;
    auto start_opt = std::chrono::high_resolution_clock::now();
    bool result_opt = scalable_ccd::metal2::Metal2Runtime::instance().narrowPhaseOpt(
        v_t0, v_t1, indices, overlaps, true, 1e-6f, 1e-6f, 1000000, false, toi_opt);
    auto end_opt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_opt = end_opt - start_opt;

    std::cout << "Original:  " << ms_orig.count() << " ms, TOI=" << toi_orig << std::endl;
    std::cout << "Optimized: " << ms_opt.count() << " ms, TOI=" << toi_opt << std::endl;

    if (result_opt && result_orig) {
        double speedup = ms_orig.count() / ms_opt.count();
        std::cout << "Speedup: " << speedup << "x" << std::endl;

        if (std::abs(toi_opt - 0.5) < 1e-4) {
            std::cout << "Large Batch Test PASSED" << std::endl;
        } else {
            std::cout << "Large Batch Test FAILED (Expected ~0.5)" << std::endl;
        }
    } else {
        std::cout << "Large Batch Test FAILED (Runtime Error)" << std::endl;
    }
}

void run_batch_test_v2()
{
    std::cout << "\n=== Running V2 Optimized Batch Test (1,000 queries) ===" << std::endl;

    int n_queries = 1000;
    std::vector<double> v_t0;
    std::vector<double> v_t1;
    std::vector<int32_t> indices;
    std::vector<std::pair<int, int>> overlaps;

    v_t0.reserve(n_queries * 4 * 3);
    v_t1.reserve(n_queries * 4 * 3);
    indices.reserve(n_queries * 3);
    overlaps.reserve(n_queries);

    for (int i = 0; i < n_queries; ++i) {
        double offset = i * 0.001;
        v_t0.push_back(offset);    v_t0.push_back(1.0);  v_t0.push_back(0.0);
        v_t0.push_back(offset - 1.0); v_t0.push_back(0.0); v_t0.push_back(-1.0);
        v_t0.push_back(offset + 1.0); v_t0.push_back(0.0); v_t0.push_back(-1.0);
        v_t0.push_back(offset);    v_t0.push_back(0.0);  v_t0.push_back(1.0);

        v_t1.push_back(offset);    v_t1.push_back(-1.0); v_t1.push_back(0.0);
        v_t1.push_back(offset - 1.0); v_t1.push_back(0.0); v_t1.push_back(-1.0);
        v_t1.push_back(offset + 1.0); v_t1.push_back(0.0); v_t1.push_back(-1.0);
        v_t1.push_back(offset);    v_t1.push_back(0.0);  v_t1.push_back(1.0);

        int base = i * 4;
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
        overlaps.push_back({ base, i });
    }

    // Run V1 optimized version
    double toi_v1 = 1.0;
    auto start_v1 = std::chrono::high_resolution_clock::now();
    bool result_v1 = scalable_ccd::metal2::Metal2Runtime::instance().narrowPhaseOpt(
        v_t0, v_t1, indices, overlaps, true, 1e-6f, 1e-6f, 1000000, false, toi_v1);
    auto end_v1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_v1 = end_v1 - start_v1;

    // Run V2 optimized version
    double toi_v2 = 1.0;
    auto start_v2 = std::chrono::high_resolution_clock::now();
    bool result_v2 = scalable_ccd::metal2::Metal2Runtime::instance().narrowPhaseOptV2(
        v_t0, v_t1, indices, overlaps, true, 1e-6f, 1e-6f, 1000000, false, toi_v2);
    auto end_v2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_v2 = end_v2 - start_v2;

    std::cout << "V1 Optimized: " << ms_v1.count() << " ms, TOI=" << toi_v1 << std::endl;
    std::cout << "V2 Optimized: " << ms_v2.count() << " ms, TOI=" << toi_v2 << std::endl;

    if (result_v2 && result_v1) {
        double speedup = ms_v1.count() / ms_v2.count();
        std::cout << "V2 vs V1 Speedup: " << speedup << "x" << std::endl;

        if (std::abs(toi_v2 - toi_v1) < 1e-4) {
            std::cout << "V2 Result Consistency: PASSED" << std::endl;
        } else {
            std::cout << "V2 Result Consistency: FAILED (toi_v2=" << toi_v2 << " vs toi_v1=" << toi_v1 << ")" << std::endl;
        }
    } else {
        std::cout << "V2 Test FAILED (Runtime Error)" << std::endl;
    }
}

int main()
{
    if (!scalable_ccd::metal2::Metal2Runtime::instance().available()) {
        std::cout << "Metal runtime not available. Skipping tests."
                  << std::endl;
        return 0;
    }

    run_test_vf();
    run_test_ee();
    run_test_miss();
    run_single_test_opt();
    run_batch_test();
    run_batch_test_opt();
    run_batch_test_large();

    return 0;
}
