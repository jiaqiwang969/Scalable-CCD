# Benchmark Results: CUDA vs Metal

This document contains performance comparison between CUDA and Metal implementations of Scalable CCD.

## Test Environment

| | CUDA | Metal |
|:---|:---|:---|
| **Platform** | Linux/Windows | macOS |
| **GPU** | NVIDIA (TBD) | Apple Silicon |
| **API Version** | CUDA 11+ | Metal 2 |

---

## 1. Broad Phase - SAP (Sweep and Prune)

### Correctness

| Test Case | CUDA Count | Metal Count | Match |
|:----------|:----------:|:-----------:|:-----:|
| cloth_ball_vf | 1,655,541 | 1,655,541 | ✅ |
| cloth_ball_ee | 5,197,332 | 5,197,332 | ✅ |
| single_list_chain | 3 | 3 | ✅ |
| single_list_shared_vertex_filtered | 0 | 0 | ✅ |
| two_lists_cross_only | 2 | 2 | ✅ |

### Performance

| Test Case | CUDA GPU (ms) | Metal GPU (ms) | Speedup |
|:----------|:-------------:|:--------------:|:-------:|
| cloth_ball_vf | 46.94 | 11.08 | Metal 4.2x |
| cloth_ball_ee | 68.00 | 8.16 | Metal 8.3x |
| single_list_chain | 11.17 | 0.014 | Metal 798x |
| single_list_shared_vertex_filtered | 11.05 | 0.012 | Metal 921x |
| two_lists_cross_only | 11.53 | 0.014 | Metal 824x |

> Note: Small dataset speedups may be affected by startup overhead differences.

---

## 2. Broad Phase - STQ (Sweep and Tiniest Queue)

### Correctness

| Test Case | Metal STQ Count | Metal SAP Count | Match |
|:----------|:---------------:|:---------------:|:-----:|
| cloth_ball_vf | 1,655,541 | 1,655,541 | ✅ |
| cloth_ball_ee | 5,197,332 | 5,197,332 | ✅ |
| single_list_chain | 3 | 3 | ✅ |
| single_list_shared_vertex_filtered | 0 | 0 | ✅ |
| two_lists_cross_only | 2 | 2 | ✅ |

### Performance (Metal SAP vs STQ)

| Test Case | SAP (ms) | STQ (ms) | Notes |
|:----------|:--------:|:--------:|:------|
| cloth_ball_vf | 11.08 | 73.78 | SAP faster for normal density |
| cloth_ball_ee | 8.16 | 60.30 | SAP faster for normal density |

> STQ is designed for extreme density scenarios where memory is limited.

---

## 3. Narrow Phase (Root Finding)

### Correctness

| Test Type | Expected TOI | CUDA TOI | Metal TOI | Match |
|:----------|:------------:|:--------:|:---------:|:-----:|
| VF single query | ~0.5 | TBD | 0.499997 | TBD |
| EE single query | ~0.5 | TBD | 0.499997 | TBD |
| Batch 1000 queries | ~0.5 | TBD | 0.499997 | TBD |

### Performance

| Test Type | CUDA (ms) | Metal (ms) | Speedup |
|:----------|:---------:|:----------:|:-------:|
| Batch 1000 queries | TBD | 16.8 | TBD |

---

## 4. Architecture Comparison

### Overflow Handling

| Feature | CUDA | Metal |
|:--------|:-----|:------|
| Strategy | Dynamic expand + re-run | Dynamic expand + re-run ✅ |
| Memory query | cudaMemGetInfo | recommendedMaxWorkingSetSize ✅ |
| Timeout | None (infinite) | None (infinite) ✅ |
| Per-i truncation | No | No ✅ |

### Queue Implementation

| Feature | CUDA | Metal |
|:--------|:-----|:------|
| Queue model | Circular + Persistent Threads | Circular + Persistent Threads ✅ |
| Batch subdivision | Auto on overflow | Auto on overflow ✅ |
| Atomic float min | atomicMin | CAS-based atomicMin ✅ |

---

## 5. How to Update Results

### On CUDA Machine

```bash
# Build with CUDA
cmake .. -DSCALABLE_CCD_WITH_CUDA=ON
make -j

# Run benchmarks
./build/tests/scalable_ccd_tests "[cuda]"

# Copy results
cp tests/results/cuda_*.json /path/to/share/
```

### On Metal Machine

```bash
# Build with Metal
cmake .. -DSCALABLE_CCD_WITH_METAL=ON
make -j

# Run benchmarks
./build/tests/scalable_ccd_tests "[metal2]"

# Copy results
cp tests/results/metal_*.json /path/to/share/
```

### Generate Comparison

```bash
python scripts/compare_results.py tests/results/
```

---

## 6. JSON Files

### Available Results

| File | Status |
|:-----|:-------|
| `cuda_sap_cloth_ball_vf.json` | ✅ |
| `cuda_sap_cloth_ball_ee.json` | ✅ |
| `cuda_sap_single_list_chain.json` | ✅ |
| `cuda_sap_single_list_shared_vertex_filtered.json` | ✅ |
| `cuda_sap_two_lists_cross_only.json` | ✅ |
| `cuda_stq_*.json` | ⏳ Pending |
| `cuda_narrow_*.json` | ⏳ Pending |
| `metal_sap_*.json` | ✅ |
| `metal_stq_*.json` | ✅ |
| `metal_narrow_*.json` | ⏳ Pending |

---

## 7. Summary

| Component | CUDA | Metal | Status |
|:----------|:----:|:-----:|:------:|
| Broad Phase SAP | ✅ | ✅ | **Complete** |
| Broad Phase STQ | ⏳ | ✅ | Metal ready |
| Narrow Phase | ⏳ | ✅ | Metal ready |
| JSON Export | ✅ | ✅ | **Complete** |

**Legend**: ✅ Tested | ⏳ Pending | ❌ Not implemented
