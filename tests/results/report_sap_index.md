# SAP 对比报告索引

| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 单列表：链式重叠 | 11.320 | 2.962 | -73.83% | 11.316 | 0.015 | -99.87% | ✅ | [详情](report_sap_single_list_chain.md) |
| 单列表：共享顶点过滤 | 11.439 | 3.203 | -72.00% | 11.436 | 0.015 | -99.87% | ✅ | [详情](report_sap_single_list_shared_vertex_filtered.md) |
| 双列表：仅跨列表配对 | 11.266 | 2.931 | -73.98% | 11.262 | 0.017 | -99.85% | ✅ | [详情](report_sap_two_lists_cross_only.md) |