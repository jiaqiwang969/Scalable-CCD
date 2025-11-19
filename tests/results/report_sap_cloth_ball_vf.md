# SAP 对比报告 - Cloth-Ball：顶点-面 (SAP)

- 用例标识: `cloth_ball_vf`
- 类别: `broad_phase_sap`
- 结果生成时间: 2025-11-19 10:48:06

## 计时对比
- CUDA Host(ms): 46.942
- Metal Host(ms): 11.024  (相对CUDA: -76.52%)
- CUDA E2E Host(ms): 258.146
- Metal E2E Host(ms): 67.640  (相对CUDA: -73.80%)
- CUDA GPU(ms): 46.940
- Metal GPU(ms): 9.786  (相对CUDA: -79.15%)

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
- Metal 时间戳: 2025-11-19 10:47:38
