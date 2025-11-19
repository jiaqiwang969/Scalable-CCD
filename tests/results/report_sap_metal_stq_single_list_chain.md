# SAP 对比报告 - 单列表：链式重叠 (STQ)

- 用例标识: `single_list_chain`
- 类别: `broad_phase_sap`
- 结果生成时间: 2025-11-19 10:48:06

## 计时对比
- CUDA Host(ms): 11.170
- Metal Host(ms): 3.475  (相对CUDA: -68.89%)
- CUDA E2E Host(ms): 2310.286
- Metal E2E Host(ms): 3.901  (相对CUDA: -99.83%)
- CUDA GPU(ms): 11.167
- Metal GPU(ms): 0.019  (相对CUDA: -99.83%)

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
- Metal 时间戳: 2025-11-19 10:47:38
