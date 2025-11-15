// Mesh and boxes IO helpers for the verifier (no Catch2 dependency).
#pragma once

#include <scalable_ccd/broad_phase/aabb.hpp>

#include <Eigen/Core>

#include <filesystem>
#include <vector>

namespace verifier {

struct MeshPair {
    Eigen::MatrixXd V0;
    Eigen::MatrixXd V1;
    Eigen::MatrixXi F;
    Eigen::MatrixXi E;
};

// Read two frames and build E from F. Return true on success.
bool read_mesh_pair(
    const std::filesystem::path& file_t0,
    const std::filesystem::path& file_t1,
    MeshPair& out);

// Build CPU AABBs for broad-phase from a mesh pair.
void build_cpu_boxes(
    const MeshPair& mp,
    std::vector<scalable_ccd::AABB>& vertex_boxes,
    std::vector<scalable_ccd::AABB>& edge_boxes,
    std::vector<scalable_ccd::AABB>& face_boxes);

} // namespace verifier

