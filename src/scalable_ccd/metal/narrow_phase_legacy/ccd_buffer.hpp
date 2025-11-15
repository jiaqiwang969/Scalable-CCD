#pragma once

// 与 CUDA 端 ccd_buffer.cuh 对齐的占位类型。
// Metal 端实际缓冲由 MTLBuffer 管理；此处仅对齐命名以便调用侧统一。

namespace scalable_ccd::metal {

struct CCDBuffer {
    // 仅作为命名占位，不持有实际资源
    int reserved = 0;
};

} // namespace scalable_ccd::metal
 #pragma once
