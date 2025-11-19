# SAP 对比报告 - Cloth-Ball：边-边 (STQ)

- 用例标识: `cloth_ball_ee`
- 类别: `broad_phase_sap`
- 结果生成时间: 2025-11-19 10:48:06

## 计时对比
- CUDA Host(ms): 68.003
- Metal Host(ms): 65.458  (相对CUDA: -3.74%)
- CUDA E2E Host(ms): 83.292
- Metal E2E Host(ms): 135.627  (相对CUDA: 62.83%)
- CUDA GPU(ms): 68.000
- Metal GPU(ms): 61.881  (相对CUDA: -9.00%)

## 重叠数量对比
- CUDA overlaps: 5197332
- Metal overlaps: 5197332
- 数量一致: 是

## 测试通过状态
- CUDA passed: 是
- Metal passed: 是

## 结论
- 结果一致，功能对齐。

## 元信息
- CUDA 时间戳: 2025-11-18 19:04:47
- Metal 时间戳: 2025-11-19 10:47:41
