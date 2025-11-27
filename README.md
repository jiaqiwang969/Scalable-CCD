# Scalable CCD

[![Build](https://github.com/continuous-collision-detection/scalable-ccd/actions/workflows/continuous.yml/badge.svg)](https://github.com/continuous-collision-detection/scalable-ccd/actions/workflows/continuous.yml)
[![License](https://img.shields.io/github/license/continuous-collision-detection/scalable-ccd.svg?color=blue)](https://github.com/continuous-collision-detection/scalable-ccd/blob/main/LICENSE)

GPU-accelerated Continuous Collision Detection using Sweep and Tiniest Queue (STQ) algorithm with Tight-Inclusion root finding.

## Supported Backends

| Backend | Platform | Status |
|:--------|:---------|:-------|
| **CUDA** | NVIDIA GPU | ✅ Original Implementation |
| **Metal** | Apple Silicon (M1/M2/M3) | ✅ Ported Implementation |
| **CPU** | All Platforms | ✅ Fallback |

## Quick Start

### Prerequisites

- C++17 compatible compiler
- CMake 3.18+
- **For CUDA**: NVIDIA GPU + CUDA Toolkit
- **For Metal**: macOS with Apple Silicon or AMD GPU

### Build

```bash
mkdir build && cd build

# CUDA build (Linux/Windows with NVIDIA GPU)
cmake .. -DSCALABLE_CCD_WITH_CUDA=ON
make -j

# Metal build (macOS)
cmake .. -DSCALABLE_CCD_WITH_METAL=ON
make -j
```

### Run Benchmarks

```bash
# Run all broad phase tests
./build/tests/scalable_ccd_tests "[broad_phase]"

# Run Metal-specific tests
./build/tests/scalable_ccd_tests "[metal2]"

# Run CUDA-specific tests
./build/tests/scalable_ccd_tests "[cuda]"

# Generate JSON benchmark results
./scripts/run_benchmarks.sh
```

## Benchmark Results

See [BENCHMARK.md](BENCHMARK.md) for detailed performance comparison between CUDA and Metal implementations.

### Summary

| Phase | Test Case | CUDA | Metal | Match |
|:------|:----------|:-----|:------|:-----:|
| Broad Phase SAP | Cloth-Ball VF | ✅ | ✅ | ✅ |
| Broad Phase SAP | Cloth-Ball EE | ✅ | ✅ | ✅ |
| Broad Phase STQ | All tests | ✅ | ✅ | ✅ |
| Narrow Phase | VF/EE queries | ✅ | ✅ | ✅ |

## Project Structure

```
Scalable-CCD/
├── src/scalable_ccd/
│   ├── cuda/           # CUDA implementation
│   └── metal2/         # Metal implementation
│       ├── broad_phase/
│       ├── narrow_phase/
│       └── runtime/
├── tests/
│   ├── results/        # Benchmark JSON outputs
│   │   ├── cuda_*.json
│   │   ├── metal_*.json
│   │   └── comparison_report.json
│   └── data/           # Test datasets
├── scripts/
│   └── run_benchmarks.sh
├── BENCHMARK.md        # Performance comparison
└── README.md
```

## JSON Output Format

All benchmark results are saved in JSON format for easy comparison:

```json
{
  "backend": "cuda|metal_sap|metal_stq",
  "category": "broad_phase_sap|broad_phase_stq|narrow_phase",
  "case_name": "Test case description",
  "slug": "test_case_id",
  "overlaps_count": 12345,
  "gpu_ms": 10.5,
  "cpu_ms": 100.0,
  "passed": true,
  "timestamp": 1234567890
}
```

## Adding New Backend Results

1. Run benchmarks on target platform:
   ```bash
   ./scripts/run_benchmarks.sh
   ```

2. Copy JSON results to `tests/results/`:
   ```bash
   cp *.json tests/results/
   ```

3. Run comparison script:
   ```bash
   python scripts/compare_results.py
   ```

## Dependencies

| Library | Purpose | Required |
|:--------|:--------|:---------|
| [Eigen](https://eigen.tuxfamily.org/) | Linear algebra | Yes |
| [oneTBB](https://github.com/oneapi-src/oneTBB) | CPU multi-threading | Yes |
| [spdlog](https://github.com/gabime/spdlog) | Logging | Yes |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON output | Optional |
| CUDA Toolkit | NVIDIA GPU support | Optional |
| Metal Framework | Apple GPU support | Optional |

## CUDA / Metal 验证

### 测试场景与规模

| 场景 | VF 碰撞对 | EE 碰撞对 | 备注 |
| --- | --- | --- | --- |
| Armadillo-Rollers | 4,652 | 19,313 | 犰狳滚轮模拟 |
| Cloth-Funnel | 92 | 263 | 布料漏斗模拟 |
| N-Body | 9,460 | 41,036 | N 体模拟（大算力） |

> Cloth-Ball（VF=1,655,541、EE=5,197,332）与 Rod-Twist 也包含在宽阶段测试，与上表同样依赖 `tests/data` 内的帧与碰撞对。

### CUDA 结果导出

使用 `tests/export_cuda_results.py` 可以批量运行 Catch2 的 CUDA 宽阶段测试，并将结果写入 `tests/results/cuda_sap_<case>_<alias>.json`：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSCALABLE_CCD_WITH_CUDA=ON
cmake --build build -j
python3 tests/export_cuda_results.py \
    --device 0:rtx2000ada:"NVIDIA RTX 2000 Ada Generation Laptop GPU" \
    --device 1:rtx3090:"NVIDIA GeForce RTX 3090"
```

脚本会针对五个 Section（Armadillo-Rollers / Cloth-Ball / Cloth-Funnel / N-Body / Rod-Twist）逐一执行 `./tests/scalable_ccd_tests "Test CUDA broad phase" -c <Section> --durations yes`，并把 Catch2 输出的壁钟时间填入 JSON：`host_total_ms` 与 `gpu_ms` 字段均为整段运行所耗时间（含 mesh 读取、AABB 构建与两次 SAP 检测）。如需限制 GPU，可仅传入一个 `--device` 或自行设置 `CUDA_VISIBLE_DEVICES`。

### N-Body 高负载细节

- VF 阶段：需要处理约 9.4M 候选对，在默认缓冲区溢出后扩容至 16.7M。
- EE 阶段：需要处理约 22.4M 候选对，缓冲区扩容至 22.4M。
- 单次完整测试（含文件读取、AABB 构建、GPU 计算）约 66 s，可作为重负载性能基准。
- 结果与 `tests/data/n-body-simulation/boxes/18*.json` 的 ground truth 完全一致。

### Metal 数据

macOS + Metal 的宽阶段对拍结果参考 `tests/results/metal_sap_*.json`（例如 cloth-ball vf/ee），结构与 CUDA JSON 对齐，可直接进行后端比较。若要重新生成 Metal 数据，可在 macOS 上开启 `SCALABLE_CCD_WITH_METAL` 并运行对应的 Catch2 Section。

## Citation

```bibtex
@misc{belgrod2023time,
    title        = {Time of Impact Dataset for Continuous Collision Detection and a Scalable Conservative Algorithm},
    author       = {David Belgrod and Bolun Wang and Zachary Ferguson and Xin Zhao and Marco Attene and Daniele Panozzo and Teseo Schneider},
    year         = 2023,
    eprint       = {2112.06300},
    archiveprefix = {arXiv},
    primaryclass = {cs.GR}
}
```

## License

<<<<<<< HEAD
Apache-2.0 - see [LICENSE](LICENSE) for details.
=======
This project is licensed under the Apache-2.0 license - see the [LICENSE](https://github.com/continuous-collision-detection/scalable-ccd/blob/main/LICENSE) file for details.
>>>>>>> 272cdf0 (Add CUDA benchmarking exports and docs)
