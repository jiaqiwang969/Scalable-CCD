# SAP/STQ 对比报告索引

## Metal 变体：SAP (metal_sap)

| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA E2E(ms) | Metal E2E(ms) | ΔE2E(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| Cloth-Ball：边-边 | 68.003 | 13.081 | -80.76% | 83.292 | 30.375 | -63.53% | 68.000 | 8.052 | -88.16% | ✅ | [详情](report_sap_cloth_ball_ee.md) |
| Cloth-Ball：顶点-面 | 46.942 | 10.937 | -76.70% | 258.146 | 67.326 | -73.92% | 46.940 | 9.726 | -79.28% | ✅ | [详情](report_sap_cloth_ball_vf.md) |
| 单列表：链式重叠 | 11.170 | 2.925 | -73.81% | 2310.286 | 27.111 | -98.83% | 11.167 | 0.015 | -99.87% | ✅ | [详情](report_sap_single_list_chain.md) |
| 单列表：共享顶点过滤 | 11.053 | 3.047 | -72.44% | 225.155 | 25.863 | -88.51% | 11.050 | 0.015 | -99.87% | ✅ | [详情](report_sap_single_list_shared_vertex_filtered.md) |
| 双列表：仅跨列表配对 | 11.530 | 3.019 | -73.82% | 222.333 | 25.335 | -88.60% | 11.526 | 0.017 | -99.85% | ✅ | [详情](report_sap_two_lists_cross_only.md) |

## Metal 变体：STQ (metal_stq)

| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA E2E(ms) | Metal E2E(ms) | ΔE2E(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| Cloth-Ball：边-边 | 68.003 | 71.617 | 5.31% | 83.292 | 141.893 | 70.36% | 68.000 | 67.819 | -0.27% | ✅ | [详情](report_sap_metal_stq_cloth_ball_ee.md) |
| Cloth-Ball：顶点-面 | 46.942 | 76.464 | 62.89% | 258.146 | 170.736 | -33.86% | 46.940 | 75.200 | 60.21% | ✅ | [详情](report_sap_metal_stq_cloth_ball_vf.md) |
| 单列表：链式重叠 | 11.170 | 3.398 | -69.58% | 2310.286 | 3.834 | -99.83% | 11.167 | 0.019 | -99.83% | ✅ | [详情](report_sap_metal_stq_single_list_chain.md) |
| 单列表：共享顶点过滤 | 11.053 | 3.371 | -69.50% | 225.155 | 3.786 | -98.32% | 11.050 | 0.019 | -99.83% | ✅ | [详情](report_sap_metal_stq_single_list_shared_vertex_filtered.md) |
| 双列表：仅跨列表配对 | 11.530 | 3.348 | -70.96% | 222.333 | 3.812 | -98.29% | 11.526 | 0.022 | -99.81% | ✅ | [详情](report_sap_metal_stq_two_lists_cross_only.md) |
