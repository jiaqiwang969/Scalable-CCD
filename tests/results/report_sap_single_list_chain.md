# SAP 对比报告 - 单列表：链式重叠

- 用例标识: `single_list_chain`
- 类别: `broad_phase_sap`
- 结果生成时间: 2025-11-19 00:16:49

## 计时对比
- CUDA Host(ms): 11.170
- Metal Host(ms): 2.761  (相对CUDA: -75.28%)
- CUDA E2E Host(ms): 2310.286
- Metal E2E Host(ms): 30.346  (相对CUDA: -98.69%)
- CUDA GPU(ms): 11.167
- Metal GPU(ms): 0.016  (相对CUDA: -99.86%)

## 重叠数量对比
- CUDA overlaps: 3
- Metal overlaps: 3
- 数量一致: 是

## 测试通过状态
- CUDA passed: 是
- Metal passed: 是

## 结论
- 结果一致，功能对齐。

## 元信息
- CUDA 时间戳: 2025-11-18 19:04:46
- Metal 时间戳: 2025-11-19 00:07:46
