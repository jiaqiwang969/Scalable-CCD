# SAP 对比报告索引

| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA E2E(ms) | Metal E2E(ms) | ΔE2E(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 单列表：链式重叠 | 11.819 | 2.949 | -75.05% | 327.593 | 35.171 | -89.26% | 11.814 | 0.014 | -99.88% | ✅ | [详情](report_sap_single_list_chain.md) |
| 单列表：共享顶点过滤 | 11.304 | 2.899 | -74.36% | 212.550 | 30.576 | -85.61% | 11.301 | 0.015 | -99.87% | ✅ | [详情](report_sap_single_list_shared_vertex_filtered.md) |
| 双列表：仅跨列表配对 | 11.283 | 2.721 | -75.89% | 211.017 | 30.682 | -85.46% | 11.279 | 0.016 | -99.85% | ✅ | [详情](report_sap_two_lists_cross_only.md) |