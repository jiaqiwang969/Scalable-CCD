# SAP 对比报告索引

| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA E2E(ms) | Metal E2E(ms) | ΔE2E(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 单列表：链式重叠 | 11.170 | 2.949 | -73.60% | 2310.286 | 35.171 | -98.48% | 11.167 | 0.014 | -99.87% | ✅ | [详情](report_sap_single_list_chain.md) |
| 单列表：共享顶点过滤 | 11.053 | 2.899 | -73.77% | 225.155 | 30.576 | -86.42% | 11.050 | 0.015 | -99.87% | ✅ | [详情](report_sap_single_list_shared_vertex_filtered.md) |
| 双列表：仅跨列表配对 | 11.530 | 2.721 | -76.40% | 222.333 | 30.682 | -86.20% | 11.526 | 0.016 | -99.86% | ✅ | [详情](report_sap_two_lists_cross_only.md) |