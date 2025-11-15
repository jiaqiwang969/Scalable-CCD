#pragma once

// 与 CUDA 端 interval.cuh 对齐的占位定义/工具。

namespace scalable_ccd::metal {

struct Interval {
    float lower;
    float upper;
};

inline Interval make_interval(float lower, float upper)
{
    return Interval{lower, upper};
}

} // namespace scalable_ccd::metal
 #pragma once
