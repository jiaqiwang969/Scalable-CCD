#pragma once

#include <scalable_ccd/utils/logger.hpp>

#define gpuErrchk(ans)                                                         \
    {                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                  \
    }

namespace scalable_ccd::cuda {

inline void gpuAssert(
    cudaError_t code,
    const std::string& file,
    int line,
    bool throw_on_error = true)
{
    if (code != cudaSuccess) {
        logger().error(
            "{}: {} ({}:{:d})", cudaGetErrorName(code),
            cudaGetErrorString(code), file, line);
        if (throw_on_error) {
            assert(code == cudaSuccess);
            throw std::runtime_error(
                fmt::format("CUDA error {}", static_cast<int>(code)));
        }
    }
}
} // namespace scalable_ccd::cuda