# SAP 对比报告 - 双列表：仅跨列表配对

- 用例标识: `two_lists_cross_only`
- 类别: `broad_phase_sap`
- 结果生成时间: 2025-11-19 00:36:03

## 计时对比
- CUDA Host(ms): 11.530
- Metal Host(ms): 3.038  (相对CUDA: -73.65%)
- CUDA E2E Host(ms): 222.333
- Metal E2E Host(ms): 26.790  (相对CUDA: -87.95%)
- CUDA GPU(ms): 11.526
- Metal GPU(ms): 0.016  (相对CUDA: -99.86%)

## 重叠数量对比
- CUDA overlaps: 2
- Metal overlaps: 2
- 数量一致: 是

## 测试通过状态
- CUDA passed: 是
- Metal passed: 是

## 结论
- 结果一致，功能对齐。

## 元信息
- CUDA 时间戳: 2025-11-18 19:04:47
- Metal 时间戳: 2025-11-19 00:35:28
