# SAP 对比报告索引

| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA E2E(ms) | Metal E2E(ms) | ΔE2E(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 单列表：链式重叠 | 11.320 | 2.949 | -73.95% | 11.316 | 0.014 | -99.88% | ✅ | [详情](report_sap_single_list_chain.md) |
| 单列表：共享顶点过滤 | 11.439 | 2.899 | -74.66% | 11.436 | 0.015 | -99.87% | ✅ | [详情](report_sap_single_list_shared_vertex_filtered.md) |
| 双列表：仅跨列表配对 | 11.266 | 2.721 | -75.85% | 11.262 | 0.016 | -99.85% | ✅ | [详情](report_sap_two_lists_cross_only.md) |