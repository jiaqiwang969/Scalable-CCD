# Scalable CCD

[![Build](https://github.com/continuous-collision-detection/scalable-ccd/actions/workflows/continuous.yml/badge.svg)](https://github.com/continuous-collision-detection/scalable-ccd/actions/workflows/continuous.yml)
[![License](https://img.shields.io/github/license/continuous-collision-detection/scalable-ccd.svg?color=blue)](https://github.com/continuous-collision-detection/scalable-ccd/blob/main/LICENSE)

Sweep and Tiniest Queue & Tight-Inclusion GPU CCD

## Getting Started

### Prerequisites

* A C/C++ compiler (at least support for C++17)
* CMake (version 3.18 or newer)
* Optionally: A CUDA-compatible GPU and the CUDA toolkit installed

### Building

The easiest way to add this project to an existing CMake project is to download it through CMake. Here is an example of how to add this project to your CMake project using [CPM](https://github.com/cpm-cmake/CPM.cmake):

```cmake
# Scalable CCD (https://github.com/continuous-collision-detection/scalable-ccd)
# License: Apache 2.0
if(TARGET scalable_ccd::scalable_ccd)
    return()
endif()

message(STATUS "Third-party: creating target 'scalable_ccd::scalable_ccd'")

set(SCALABLE_CCD_WITH_CUDA ${MY_PROJECT_WITH_CUDA} CACHE BOOL "Enable CUDA CCD" FORCE)

include(CPM)
CPMAddPackage("gh:continuous-collision-detection/scalable-ccd#${SCALABLE_CCD_GIT_TAG}")
```

where `MY_PROJECT_WITH_CUDA` is an example variable set in your project and  `SCALABLE_CCD_GIT_TAG` is set to the version of this project you want to use. This will download and add this project to CMake. You can then be linked against it using

```cmake
# Link against the Scalable CCD
target_link_libraries(my_target PRIVATE scalable_ccd::scalable_ccd)
```

where `my_target` is the name of your library/binary.

#### Dependencies

**All required dependencies are downloaded through CMake** depending on the build options.

The following libraries are used in this project:

* [Eigen](https://eigen.tuxfamily.org/): linear algebra
* [oneTBB](https://github.com/oneapi-src/oneTBB): CPU multi-threading
* [spdlog](https://github.com/gabime/spdlog): logging

##### Optional

* [CUDA](https://developer.nvidia.com/cuda-toolkit): GPU acceleration
	* Required when using the CMake option `SCALABLE_CCD_WITH_CUDA`
* [nlohmann/json](https://github.com/nlohmann/json): saving profiler data
    * Required when using the CMake option `SCALABLE_CCD_WITH_PROFILER`

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

If you use this code in your project, please consider citing our paper:

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

This project is licensed under the Apache-2.0 license - see the [LICENSE](https://github.com/continuous-collision-detection/scalable-ccd/blob/main/LICENSE) file for details.
