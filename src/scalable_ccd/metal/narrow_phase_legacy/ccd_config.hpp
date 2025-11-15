#pragma once

// 与 CUDA 端 ccd_config.cuh 对齐的占位配置，集中放置窄相位可调参数。

namespace scalable_ccd::metal {

struct CCDConfig {
    // 与 CUDA 保持同名/同义：最大迭代次数，<0 表示使用内部默认
    int max_iterations = -1;
    // 根查找容差（域/值的等效使用由实现决定）
    double tolerance = 1e-6;
    // 是否允许 toi=0
    bool allow_zero_toi = true;
    // 每次命中后的二分细化步数（与 CUDA 的 refine 有关，Metal 端用于近似）
    int refine_steps = 5;
};

} // namespace scalable_ccd::metal
 #pragma once
