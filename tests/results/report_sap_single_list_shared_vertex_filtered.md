# SAP 对比报告 - 单列表：共享顶点过滤 (SAP)

- 用例标识: `single_list_shared_vertex_filtered`
- 类别: `broad_phase_sap`
- 结果生成时间: 2025-11-19 11:29:26

## 计时对比
- CUDA Host(ms): 11.053
- Metal Host(ms): 3.080  (相对CUDA: -72.13%)
- CUDA E2E Host(ms): 225.155
- Metal E2E Host(ms): 26.418  (相对CUDA: -88.27%)
- CUDA GPU(ms): 11.050
- Metal GPU(ms): 0.013  (相对CUDA: -99.88%)

## 重叠数量对比
- CUDA overlaps: 0
- Metal overlaps: 0
- 数量一致: 是

## 测试通过状态
- CUDA passed: 是
- Metal passed: 是

## 结论
- 结果一致，功能对齐。

## 元信息
- CUDA 时间戳: 2025-11-18 19:04:46
- Metal 时间戳: 2025-11-19 11:29:08
