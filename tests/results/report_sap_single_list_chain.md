# SAP 对比报告 - 单列表：链式重叠

- 用例标识: `single_list_chain`
- 类别: `broad_phase_sap`
- 结果生成时间: 2025-11-18 18:20:37

## 计时对比
- CUDA Host(ms): 11.819
- Metal Host(ms): 2.949  (相对CUDA: -75.05%)
- CUDA E2E Host(ms): 327.593
- Metal E2E Host(ms): 35.171  (相对CUDA: -89.26%)
- CUDA GPU(ms): 11.814
- Metal GPU(ms): 0.014  (相对CUDA: -99.88%)

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
- CUDA 时间戳: 2025-11-18 18:19:15
- Metal 时间戳: 2025-11-18 17:52:37
