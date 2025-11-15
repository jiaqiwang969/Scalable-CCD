# Scalable-CCD Verifier

验证型可执行，面向“分环境、分阶段、可读报告”的移植/性能/正确性验证。

可执行
- `scalable_ccd_verifier`：单环境、单构建的验证，输出 JSON 与 HTML 报告
- `scalable_ccd_aggregate`：聚合多个 summary.json，生成总表 HTML

用法
```
scalable_ccd_verifier \
  [--backend cpu|cuda|both] \
  [--out DIR] \
  [--log N] \
  [--threads N] \
  [--repeat N] \
  [--warmup N] \
  [--scenarios FILE] \
  [--tag NAME] \
  [--data DIR]
```
- 报告输出：`DIR/summary.json` 与 `DIR/report.html`
- `--scenarios`：读取 JSON 场景文件（示例：`verifier/scenarios/default.json`）
- `--tag`：为当前环境打标签，方便聚合报告展示来源
- `--data`：运行时覆盖数据目录（默认使用编译期的 `SCALABLE_CCD_DATA_DIR`，也可用环境变量同名覆盖）

场景文件结构（示例）
```json
{
  "scenes": [
    { "scene": "cloth-ball",
      "t0": "frames/cloth_ball92.ply",
      "t1": "frames/cloth_ball93.ply",
      "vf_json": "boxes/92vf.json",
      "ee_json": "boxes/92ee.json" }
  ]
}
```

矩阵运行（多构建）
```
scripts/run_matrix.sh [OUT_DIR] [THREADS] [REPEAT] [WARMUP]
```
- 预置 CMakePresets：`cpu/cuda × float/double × Release（含 Debug 示例）`
- 每个预置会生成一个子目录报告；可用 `scalable_ccd_aggregate` 聚合：
  ```
  build/<some-preset>/verifier/scalable_ccd_aggregate \
    OUT_DIR_AGG build/reports-matrix/*/summary.json
  ```
  输出：`OUT_DIR_AGG/aggregate.json` 和 `aggregate.html`

报告内容
- 环境信息：OS/CPU/内存/编译器/构建类型/精度/CUDA驱动与设备/线程/重复与预热次数
- 场景结果（每步）：
  - `stage`：broad_vf / broad_ee
  - `avg_ms`：重复后的平均时间；同时记录 `repeats`、`warmup`
  - `compare`：`truth_total`、`algo_total`、`true_positives`、`covers_truth`
  - CUDA 时附带库内 `profiler` 的 JSON（细分阶段）

注意
- 本工具优先对齐已有 `tests/data` 的 `boxes` 真值，先保障 Broad-Phase 一致性；
  若需要 Narrow-Phase 的逐 query 验证，可在后续版本基于 `queries` 与 `mma_bool` 衔接。

使用完整数据集
- 推荐两种方式之一：
  1) 配置期指定（推荐稳定）：
     cmake -S . -B build -DSCALABLE_CCD_USE_EXISTING_DATA_DIR=ON -DSCALABLE_CCD_DATA_DIR=/path/to/full-dataset
     然后正常编译运行（无需 --data）。
  2) 运行期覆盖：
     ./build/verifier/scalable_ccd_verifier --data /path/to/full-dataset ...
     或设置环境变量 SCALABLE_CCD_DATA_DIR=/path/to/full-dataset
