# SAP 对比报告 - Cloth-Ball：顶点-面 (STQ)

- 用例标识: `cloth_ball_vf`
- 类别: `broad_phase_sap`
- 结果生成时间: 2025-11-19 09:53:08

## 计时对比
- CUDA Host(ms): 46.942
- Metal Host(ms): 87.927  (相对CUDA: 87.31%)
- CUDA E2E Host(ms): 258.146
- Metal E2E Host(ms): 182.174  (相对CUDA: -29.43%)
- CUDA GPU(ms): 46.940
- Metal GPU(ms): 86.646  (相对CUDA: 84.59%)

## 重叠数量对比
- CUDA overlaps: 1655541
- Metal overlaps: 1655541
- 数量一致: 是

## 测试通过状态
- CUDA passed: 是
- Metal passed: 是

## 结论
- 结果一致，功能对齐。

## 元信息
- CUDA 时间戳: 2025-11-18 19:04:47
- Metal 时间戳: 2025-11-19 01:12:02
