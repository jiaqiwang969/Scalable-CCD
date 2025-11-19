# SAP/STQ 对比报告索引

## Metal 变体：SAP (metal_sap)

| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA E2E(ms) | Metal E2E(ms) | ΔE2E(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| Cloth-Ball：边-边 | 68.003 | 13.142 | -80.68% | 83.292 | 30.228 | -63.71% | 68.000 | 8.018 | -88.21% | ✅ | [详情](report_sap_cloth_ball_ee.md) |
| Cloth-Ball：顶点-面 | 46.942 | 10.927 | -76.72% | 258.146 | 66.929 | -74.07% | 46.940 | 9.600 | -79.55% | ✅ | [详情](report_sap_cloth_ball_vf.md) |
| 单列表：链式重叠 | 11.170 | 3.182 | -71.51% | 2310.286 | 27.128 | -98.83% | 11.167 | 0.017 | -99.85% | ✅ | [详情](report_sap_single_list_chain.md) |
| 单列表：共享顶点过滤 | 11.053 | 3.080 | -72.13% | 225.155 | 26.418 | -88.27% | 11.050 | 0.013 | -99.88% | ✅ | [详情](report_sap_single_list_shared_vertex_filtered.md) |
| 双列表：仅跨列表配对 | 11.530 | 2.828 | -75.47% | 222.333 | 29.115 | -86.90% | 11.526 | 0.015 | -99.87% | ✅ | [详情](report_sap_two_lists_cross_only.md) |

## Metal 变体：STQ (metal_stq)

| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA E2E(ms) | Metal E2E(ms) | ΔE2E(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| Cloth-Ball：边-边 | 68.003 | 65.087 | -4.29% | 83.292 | 135.769 | 63.00% | 68.000 | 61.559 | -9.47% | ✅ | [详情](report_sap_metal_stq_cloth_ball_ee.md) |
| Cloth-Ball：顶点-面 | 46.942 | 69.905 | 48.92% | 258.146 | 164.630 | -36.23% | 46.940 | 68.648 | 46.25% | ✅ | [详情](report_sap_metal_stq_cloth_ball_vf.md) |
| 单列表：链式重叠 | 11.170 | 3.492 | -68.74% | 2310.286 | 3.985 | -99.83% | 11.167 | 0.019 | -99.83% | ✅ | [详情](report_sap_metal_stq_single_list_chain.md) |
| 单列表：共享顶点过滤 | 11.053 | 2.925 | -73.53% | 225.155 | 3.427 | -98.48% | 11.050 | 0.053 | -99.52% | ✅ | [详情](report_sap_metal_stq_single_list_shared_vertex_filtered.md) |
| 双列表：仅跨列表配对 | 11.530 | 2.952 | -74.40% | 222.333 | 3.456 | -98.45% | 11.526 | 0.019 | -99.84% | ✅ | [详情](report_sap_metal_stq_two_lists_cross_only.md) |
