#include "gpu_query.cuh"

#include <scalable_ccd/config.hpp>
#ifdef SCALABLE_CCD_WITH_CUDA
#include <scalable_ccd/cuda/narrow_phase/narrow_phase.cuh>
#include <scalable_ccd/cuda/utils/device_matrix.cuh>
#include <scalable_ccd/cuda/broad_phase/aabb.cuh>
#include <scalable_ccd/cuda/memory_handler.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace verifier::gpu {

static double eval_impl(bool is_vf,
                        const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V1,
                        int max_iter, double tol, double min_sep)
{
    using namespace scalable_ccd::cuda;
    const DeviceMatrix<scalable_ccd::Scalar> dV0(V0), dV1(V1);
    scalable_ccd::Scalar toi = 1.0;
    std::shared_ptr<MemoryHandler> mem = std::make_shared<MemoryHandler>();
    if (!is_vf) {
        // EE: E has two edges, F empty
        Eigen::MatrixXi E(2,2), F(0,3);
        E << 0,1, 2,3;
        const DeviceMatrix<int> dE(E), dF(F);
        thrust::host_vector<int2> h(1); h[0] = make_int2(0,1);
        thrust::device_vector<int2> d = h;
        narrow_phase</*is_vf=*/false>(dV0, dV1, dE, dF, d, 1024, max_iter, tol, min_sep, true, mem,
    #ifdef SCALABLE_CCD_TOI_PER_QUERY
                                      // collisions
    #endif
                                      toi);
    } else {
        // VF: F has one face, E empty
        Eigen::MatrixXi E(0,2), F(1,3);
        F << 1,2,3;
        const DeviceMatrix<int> dE(E), dF(F);
        thrust::host_vector<int2> h(1); h[0] = make_int2(0,0);
        thrust::device_vector<int2> d = h;
        narrow_phase</*is_vf=*/true>(dV0, dV1, dE, dF, d, 1024, max_iter, tol, min_sep, true, mem,
    #ifdef SCALABLE_CCD_TOI_PER_QUERY
                                     // collisions
    #endif
                                     toi);
    }
    return static_cast<double>(toi);
}

double eval_query_ee(const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V1,
                     int max_iter, double tol, double min_sep)
{
    return eval_impl(false, V0, V1, max_iter, tol, min_sep);
}

double eval_query_vf(const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V1,
                     int max_iter, double tol, double min_sep)
{
    return eval_impl(true, V0, V1, max_iter, tol, min_sep);
}

} // namespace verifier::gpu

#else
namespace verifier::gpu {
double eval_query_ee(const Eigen::MatrixXd&, const Eigen::MatrixXd&, int, double, double){return 1.0;}
double eval_query_vf(const Eigen::MatrixXd&, const Eigen::MatrixXd&, int, double, double){return 1.0;}
} // namespace verifier::gpu
#endif
