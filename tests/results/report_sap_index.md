# SAP/STQ 对比报告索引

## Metal 变体：SAP (metal_sap)

| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA E2E(ms) | Metal E2E(ms) | ΔE2E(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| Cloth-Ball：边-边 | 68.003 | 13.350 | -80.37% | 83.292 | 30.413 | -63.49% | 68.000 | 8.165 | -87.99% | ✅ | [详情](report_sap_cloth_ball_ee.md) |
| Cloth-Ball：顶点-面 | 46.942 | 11.024 | -76.52% | 258.146 | 67.640 | -73.80% | 46.940 | 9.786 | -79.15% | ✅ | [详情](report_sap_cloth_ball_vf.md) |
| 单列表：链式重叠 | 11.170 | 3.033 | -72.85% | 2310.286 | 27.070 | -98.83% | 11.167 | 0.015 | -99.87% | ✅ | [详情](report_sap_single_list_chain.md) |
| 单列表：共享顶点过滤 | 11.053 | 2.944 | -73.37% | 225.155 | 25.808 | -88.54% | 11.050 | 0.013 | -99.89% | ✅ | [详情](report_sap_single_list_shared_vertex_filtered.md) |
| 双列表：仅跨列表配对 | 11.530 | 2.908 | -74.78% | 222.333 | 25.483 | -88.54% | 11.526 | 0.015 | -99.87% | ✅ | [详情](report_sap_two_lists_cross_only.md) |

## Metal 变体：STQ (metal_stq)

| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA E2E(ms) | Metal E2E(ms) | ΔE2E(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| Cloth-Ball：边-边 | 68.003 | 65.458 | -3.74% | 83.292 | 135.627 | 62.83% | 68.000 | 61.881 | -9.00% | ✅ | [详情](report_sap_metal_stq_cloth_ball_ee.md) |
| Cloth-Ball：顶点-面 | 46.942 | 74.936 | 59.64% | 258.146 | 168.674 | -34.66% | 46.940 | 73.652 | 56.91% | ✅ | [详情](report_sap_metal_stq_cloth_ball_vf.md) |
| 单列表：链式重叠 | 11.170 | 3.475 | -68.89% | 2310.286 | 3.901 | -99.83% | 11.167 | 0.019 | -99.83% | ✅ | [详情](report_sap_metal_stq_single_list_chain.md) |
| 单列表：共享顶点过滤 | 11.053 | 3.418 | -69.08% | 225.155 | 4.398 | -98.05% | 11.050 | 0.017 | -99.84% | ✅ | [详情](report_sap_metal_stq_single_list_shared_vertex_filtered.md) |
| 双列表：仅跨列表配对 | 11.530 | 3.544 | -69.26% | 222.333 | 4.074 | -98.17% | 11.526 | 0.019 | -99.84% | ✅ | [详情](report_sap_metal_stq_two_lists_cross_only.md) |
