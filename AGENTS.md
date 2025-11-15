# Scalable-CCD 验证与迁移准备（AGENTS 报告）

本报告总结了为 Metal 移植而搭建的“验证优先”流程、当前在两张 GPU（RTX 3090 与 RTX 2000 Ada）上的完整验证结果，以及后续迁移建议与使用方法。

## 一、验证工具与输出
- 可执行：
  - `verifier/scalable_ccd_verifier`：运行场景验证，输出 JSON + HTML（含环境、阶段计时、真值对比、CUDA Profiler）
  - `verifier/scalable_ccd_aggregate`：聚合多次 `summary.json`，生成总表 HTML
- 重要参数：
  - `--data DIR` 选择数据目录（本机完整数据集位于 `tests/data-full`）
  - `--backend cpu|cuda|both`；`--threads N`；`--repeat N`；`--warmup N`
  - `--scan / --scenes / --min-step / --max-per-scene` 用于自动扫描大数据集
  - `--tag NAME` 为当前环境打标签（聚合对比用）
- 产物：
  - `summary.json`：结构化数据（环境、阶段计时、真值对齐、CUDA 细分阶段）
  - `report.html`：用户可读页面（表格 + CUDA Profiler 折叠详情）

## 二、完整数据集 + 双 GPU 最终结果
本次使用完整数据集 `tests/data-full`，在两张 GPU 上分别运行，并聚合对比：
- GPU0（标签 `gpu0-rtx2000-r3w1`）：NVIDIA GeForce RTX 3090
- GPU1（标签 `gpu1-rtx3090-r3w1`）：NVIDIA RTX 2000 Ada Generation Laptop GPU
- 参数：`threads=8, repeat=3, warmup=1, Release, double`
- 输出位置：
  - 单 GPU 报告：
    - `build/gpu-benchmark/gpu0-r3w1/report.html`
    - `build/gpu-benchmark/gpu1-r3w1/report.html`
  - 聚合报告（同页对比，两次运行均带标签列）：
    - `build/gpu-benchmark/aggregate-r3w1/aggregate.html`
    - `build/gpu-benchmark/aggregate-r3w1/aggregate.json`

### 2.1 Broad-phase 阶段（CPU）
（单位：ms，repeat=3 的均值）
```
场景                | 阶段      | GPU0-3090   | GPU1-2000Ada
--------------------|-----------|-------------|--------------
armadillo-rollers   | broad_vf  | 61.31       | 61.99
armadillo-rollers   | broad_ee  | 65.62       | 66.57
cloth-ball          | broad_vf  | 282.11      | 284.19
cloth-ball          | broad_ee  | 306.66      | 310.73
cloth-funnel        | broad_vf  | 9.03        | 8.42
cloth-funnel        | broad_ee  | 10.93       | 11.02
n-body-simulation   | broad_vf  | 437.46      | 442.62
n-body-simulation   | broad_ee  | 544.20      | 561.39
rod-twist           | broad_vf  | 38.60       | 40.16
rod-twist           | broad_ee  | 55.00       | 55.39
```
结论：CPU 结果跨两次运行非常接近，适合作为跨后端的参考基线。

### 2.2 Broad-phase 阶段（CUDA，细分阶段）
说明：由于主流程计时会受调度影响，我们采信 CUDA Profiler 的细分阶段“sweep_and_tiniest_queue”的 `time_ms` 作为 GPU 核心计算耗时对比（已显示在 HTML 可展开详情中）。
（单位：ms，总计时，repeat=3, warmup=1 条件下采集）
```
场景                | 阶段      | GPU0-3090 | GPU1-2000Ada
--------------------|-----------|-----------|-------------
armadillo-rollers   | broad_vf  |  9.11     | 21.33
armadillo-rollers   | broad_ee  |  9.39     | 21.59
cloth-ball          | broad_vf  | 29.93     | 90.07
cloth-ball          | broad_ee  | 28.57     | 78.85
cloth-funnel        | broad_vf  |  7.39     |  8.54
cloth-funnel        | broad_ee  |  6.71     |  8.01
n-body-simulation   | broad_vf  | 58.34     |205.87
n-body-simulation   | broad_ee  | 53.42     |172.23
rod-twist           | broad_vf  |  5.65     | 12.83
rod-twist           | broad_ee  |  6.05     | 14.08
```
结论：3090 在大多数场景/阶段显著快于 2000 Ada，速度优势在大型场景（如 n-body）上尤为明显。两次运行均覆盖真值（`covers_truth=true`），验证可靠。

## 三、为 Metal 移植做的准备
- 验证框架：
  - 统一的验证器（`scalable_ccd_verifier`）可按“后端/精度/构建/线程/数据集/重复/预热”切分环境，生成可比对的报告（HTML+JSON）。
  - 强化的环境打印：OS/CPU/内存/编译器/构建参数/CUDA 驱动与设备细节，便于 Metal 端建立对齐环境。
  - 数据集扫描与场景配置：支持 `--scan` 自动生成场景矩阵（相邻帧 + 真值存在检查），也支持 `--scenarios` 明确列表；便于 Metal 侧逐步放量。
- 阶段化与计时：
  - CPU 路线：统计 `avg_ms`（repeat/warmup）；作为跨后端的参考基线。
  - CUDA 路线：保留 CUDA Profiler 的细分阶段（如 `sweep_and_tiniest_queue`），作为 Metal 对齐优化的关键指标。
- 逐 Query 验证（已打通）：
  - 基于 `queries/*.csv` 与 `mma_bool/*.json`，可逐 query 执行 GPU 窄阶段并对比真值（开启 `--queries` 选项），输出“真值阳性/算法阳性/不一致/平均ms/Query”。
  - 用于 Metal 侧聚焦窄阶段内核的正确性与性能对齐（默认未并入本次大矩阵，需按需开启）。
- 报告与聚合：
  - 聚合器将多环境 `summary.json` 合并为 `aggregate.json/html`，并携带 `env_tag` 标签，便于“3090 vs 2000Ada vs Metal”多维对比。

> Metal 迁移建议：在 `verifier/` 工具之上新增统一后端接口（如 ICCDBackend），首期提供 `CPU`、`CUDA`、`MetalStub`（打印环境/参数，跑通路径），再迭代 Metal 实现；验证器按接口调度，保持同一报告模板（含分步计时与细分阶段），形成跨后端同页对比。

## 四、如何本地复现
1) 构建（Release + CUDA + 测试）
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSCALABLE_CCD_WITH_CUDA=ON -DSCALABLE_CCD_BUILD_TESTS=ON -DSCALABLE_CCD_USE_EXISTING_DATA_DIR=ON
cmake --build build --parallel
```
2) 单环境运行（完整案例）
```
./build/verifier/scalable_ccd_verifier --backend both --threads 8 --repeat 1 --warmup 0 --data tests/data-full --tag local --out build/report_full
```
3) 双 GPU 对比（3090 与 2000 Ada），重复=3、预热=1
```
CUDA_VISIBLE_DEVICES=0 ./build/verifier/scalable_ccd_verifier --backend both --threads 8 --repeat 3 --warmup 1 --data tests/data-full --tag gpu0-rtx2000-r3w1 --out build/gpu-benchmark/gpu0-r3w1
CUDA_VISIBLE_DEVICES=1 ./build/verifier/scalable_ccd_verifier --backend both --threads 8 --repeat 3 --warmup 1 --data tests/data-full --tag gpu1-rtx3090-r3w1 --out build/gpu-benchmark/gpu1-r3w1
./build/verifier/scalable_ccd_aggregate build/gpu-benchmark/aggregate-r3w1 build/gpu-benchmark/gpu0-r3w1/summary.json build/gpu-benchmark/gpu1-r3w1/summary.json
```
查看：
- 单 GPU 页面：
  - `build/gpu-benchmark/gpu0-r3w1/report.html`
  - `build/gpu-benchmark/gpu1-r3w1/report.html`
- 聚合对比页面：`build/gpu-benchmark/aggregate-r3w1/aggregate.html`

## 五、附：本次关键路径与代码
- 验证器：`verifier/main.cpp`
- 环境采集：`verifier/env_info.{hpp,cpp}`
- Mesh/Boxes IO：`verifier/io.{hpp,cpp}`
- 真值对比：`verifier/compare.{hpp,cpp}`
- 报告生成：`verifier/report.{hpp,cpp}`
- 场景样例：`verifier/scenarios/default.json`
- 脚本：`scripts/run_matrix.sh`、`scripts/aggregate_reports.sh`、`scripts/fetch_full_dataset.sh`

如需我将 MetalStub 接入并把逐 Query 验证加入默认大矩阵，请告知优先级；我会保持同一报告模板，产出“Metal vs CUDA vs CPU”的同页对比，便于移植阶段快速回归与优化定位。 

