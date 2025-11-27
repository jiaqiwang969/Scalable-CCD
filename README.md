# Scalable CCD

[![Build](https://github.com/continuous-collision-detection/scalable-ccd/actions/workflows/continuous.yml/badge.svg)](https://github.com/continuous-collision-detection/scalable-ccd/actions/workflows/continuous.yml)
[![License](https://img.shields.io/github/license/continuous-collision-detection/scalable-ccd.svg?color=blue)](https://github.com/continuous-collision-detection/scalable-ccd/blob/main/LICENSE)

GPU-accelerated Continuous Collision Detection using Sweep and Tiniest Queue (STQ) algorithm with Tight-Inclusion root finding.

## Supported Backends

| Backend | Platform | Status |
|:--------|:---------|:-------|
| **CUDA** | NVIDIA GPU | ✅ Original Implementation |
| **Metal** | Apple Silicon (M1/M2/M3) | ✅ Ported Implementation |
| **CPU** | All Platforms | ✅ Fallback |

## Quick Start

### Prerequisites

- C++17 compatible compiler
- CMake 3.18+
- **For CUDA**: NVIDIA GPU + CUDA Toolkit
- **For Metal**: macOS with Apple Silicon or AMD GPU

### Build

```bash
mkdir build && cd build

# CUDA build (Linux/Windows with NVIDIA GPU)
cmake .. -DSCALABLE_CCD_WITH_CUDA=ON
make -j

# Metal build (macOS)
cmake .. -DSCALABLE_CCD_WITH_METAL=ON
make -j
```

### Run Benchmarks

```bash
# Run all broad phase tests
./build/tests/scalable_ccd_tests "[broad_phase]"

# Run Metal-specific tests
./build/tests/scalable_ccd_tests "[metal2]"

# Run CUDA-specific tests
./build/tests/scalable_ccd_tests "[cuda]"

# Generate JSON benchmark results
./scripts/run_benchmarks.sh
```

## Benchmark Results

See [BENCHMARK.md](BENCHMARK.md) for detailed performance comparison between CUDA and Metal implementations.

### Summary

| Phase | Test Case | CUDA | Metal | Match |
|:------|:----------|:-----|:------|:-----:|
| Broad Phase SAP | Cloth-Ball VF | ✅ | ✅ | ✅ |
| Broad Phase SAP | Cloth-Ball EE | ✅ | ✅ | ✅ |
| Broad Phase STQ | All tests | ✅ | ✅ | ✅ |
| Narrow Phase | VF/EE queries | ✅ | ✅ | ✅ |

## Project Structure

```
Scalable-CCD/
├── src/scalable_ccd/
│   ├── cuda/           # CUDA implementation
│   └── metal2/         # Metal implementation
│       ├── broad_phase/
│       ├── narrow_phase/
│       └── runtime/
├── tests/
│   ├── results/        # Benchmark JSON outputs
│   │   ├── cuda_*.json
│   │   ├── metal_*.json
│   │   └── comparison_report.json
│   └── data/           # Test datasets
├── scripts/
│   └── run_benchmarks.sh
├── BENCHMARK.md        # Performance comparison
└── README.md
```

## JSON Output Format

All benchmark results are saved in JSON format for easy comparison:

```json
{
  "backend": "cuda|metal_sap|metal_stq",
  "category": "broad_phase_sap|broad_phase_stq|narrow_phase",
  "case_name": "Test case description",
  "slug": "test_case_id",
  "overlaps_count": 12345,
  "gpu_ms": 10.5,
  "cpu_ms": 100.0,
  "passed": true,
  "timestamp": 1234567890
}
```

## Adding New Backend Results

1. Run benchmarks on target platform:
   ```bash
   ./scripts/run_benchmarks.sh
   ```

2. Copy JSON results to `tests/results/`:
   ```bash
   cp *.json tests/results/
   ```

3. Run comparison script:
   ```bash
   python scripts/compare_results.py
   ```

## Dependencies

| Library | Purpose | Required |
|:--------|:--------|:---------|
| [Eigen](https://eigen.tuxfamily.org/) | Linear algebra | Yes |
| [oneTBB](https://github.com/oneapi-src/oneTBB) | CPU multi-threading | Yes |
| [spdlog](https://github.com/gabime/spdlog) | Logging | Yes |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON output | Optional |
| CUDA Toolkit | NVIDIA GPU support | Optional |
| Metal Framework | Apple GPU support | Optional |

## Citation

```bibtex
@misc{belgrod2023time,
    title        = {Time of Impact Dataset for Continuous Collision Detection and a Scalable Conservative Algorithm},
    author       = {David Belgrod and Bolun Wang and Zachary Ferguson and Xin Zhao and Marco Attene and Daniele Panozzo and Teseo Schneider},
    year         = 2023,
    eprint       = {2112.06300},
    archiveprefix = {arXiv},
    primaryclass = {cs.GR}
}
```

## License

Apache-2.0 - see [LICENSE](LICENSE) for details.
