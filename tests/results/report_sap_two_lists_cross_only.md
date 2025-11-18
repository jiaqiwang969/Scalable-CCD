# SAP 对比报告 - 双列表：仅跨列表配对

- 用例标识: `two_lists_cross_only`
- 类别: `broad_phase_sap`
- 结果生成时间: 2025-11-18 18:20:37

## 计时对比
- CUDA Host(ms): 11.283
- Metal Host(ms): 2.721  (相对CUDA: -75.89%)
- CUDA E2E Host(ms): 211.017
- Metal E2E Host(ms): 30.682  (相对CUDA: -85.46%)
- CUDA GPU(ms): 11.279
- Metal GPU(ms): 0.016  (相对CUDA: -99.85%)

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
- CUDA 时间戳: 2025-11-18 18:19:16
- Metal 时间戳: 2025-11-18 17:52:37
