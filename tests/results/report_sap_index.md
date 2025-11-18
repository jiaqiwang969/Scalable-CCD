# SAP/STQ 对比报告索引

## Metal 变体：SAP (metal_sap)

| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA E2E(ms) | Metal E2E(ms) | ΔE2E(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| Cloth-Ball：边-边 | 68.003 | 12.984 | -80.91% | 83.292 | 30.025 | -63.95% | 68.000 | 7.915 | -88.36% | ✅ | [详情](report_sap_cloth_ball_ee.md) |
| Cloth-Ball：顶点-面 | 46.942 | 10.986 | -76.60% | 258.146 | 70.651 | -72.63% | 46.940 | 9.793 | -79.14% | ✅ | [详情](report_sap_cloth_ball_vf.md) |
| 单列表：链式重叠 | 11.170 | 3.904 | -65.05% | 2310.286 | 29.468 | -98.72% | 11.167 | 0.015 | -99.87% | ✅ | [详情](report_sap_single_list_chain.md) |
| 单列表：共享顶点过滤 | 11.053 | 2.971 | -73.12% | 225.155 | 27.620 | -87.73% | 11.050 | 0.019 | -99.83% | ✅ | [详情](report_sap_single_list_shared_vertex_filtered.md) |
| 双列表：仅跨列表配对 | 11.530 | 2.543 | -77.95% | 222.333 | 26.543 | -88.06% | 11.526 | 0.015 | -99.87% | ✅ | [详情](report_sap_two_lists_cross_only.md) |

## Metal 变体：STQ (metal_stq)

| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA E2E(ms) | Metal E2E(ms) | ΔE2E(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| Cloth-Ball：边-边 | 68.003 | 72.918 | 7.23% | 83.292 | 143.995 | 72.88% | 68.000 | 69.029 | 1.51% | ✅ | [详情](report_sap_metal_stq_cloth_ball_ee.md) |
| Cloth-Ball：顶点-面 | 46.942 | 87.927 | 87.31% | 258.146 | 182.174 | -29.43% | 46.940 | 86.646 | 84.59% | ✅ | [详情](report_sap_metal_stq_cloth_ball_vf.md) |
| 单列表：链式重叠 | 11.170 | 3.524 | -68.45% | 2310.286 | 3.965 | -99.83% | 11.167 | 0.020 | -99.82% | ✅ | [详情](report_sap_metal_stq_single_list_chain.md) |
| 单列表：共享顶点过滤 | 11.053 | 3.500 | -68.33% | 225.155 | 4.072 | -98.19% | 11.050 | 0.034 | -99.69% | ✅ | [详情](report_sap_metal_stq_single_list_shared_vertex_filtered.md) |
| 双列表：仅跨列表配对 | 11.530 | 3.013 | -73.87% | 222.333 | 4.535 | -97.96% | 11.526 | 0.023 | -99.80% | ✅ | [详情](report_sap_metal_stq_two_lists_cross_only.md) |
