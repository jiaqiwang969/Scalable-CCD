// Minimal IO using libigl to read meshes; helpers to build CPU AABBs.

#include "io.hpp"

#include <igl/edges.h>
#include <igl/read_triangle_mesh.h>

namespace verifier {

bool read_mesh_pair(
    const std::filesystem::path& file_t0,
    const std::filesystem::path& file_t1,
    MeshPair& out)
{
    using Eigen::MatrixXd;
    using Eigen::MatrixXi;
    MatrixXd V0, V1;
    MatrixXi F, E;
    if (!igl::read_triangle_mesh(file_t0.string(), V0, F)) {
        return false;
    }
    if (!igl::read_triangle_mesh(file_t1.string(), V1, F)) {
        return false;
    }
    igl::edges(F, E);
    out.V0 = std::move(V0);
    out.V1 = std::move(V1);
    out.F = std::move(F);
    out.E = std::move(E);
    return true;
}

void build_cpu_boxes(
    const MeshPair& mp,
    std::vector<scalable_ccd::AABB>& vertex_boxes,
    std::vector<scalable_ccd::AABB>& edge_boxes,
    std::vector<scalable_ccd::AABB>& face_boxes)
{
    using namespace scalable_ccd;
    build_vertex_boxes(mp.V0, mp.V1, vertex_boxes);
    build_edge_boxes(vertex_boxes, mp.E, edge_boxes);
    build_face_boxes(vertex_boxes, mp.F, face_boxes);
}

} // namespace verifier

