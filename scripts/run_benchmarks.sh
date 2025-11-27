#!/bin/bash
# run_benchmarks.sh - Run CCD benchmarks and export JSON results
#
# Usage:
#   ./scripts/run_benchmarks.sh [cuda|metal|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
RESULTS_DIR="${PROJECT_DIR}/tests/results"

# Detect platform
detect_backend() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "metal"
    else
        echo "cuda"
    fi
}

BACKEND="${1:-$(detect_backend)}"

echo "=========================================="
echo "Scalable CCD Benchmark Runner"
echo "=========================================="
echo "Backend: $BACKEND"
echo "Build dir: $BUILD_DIR"
echo "Results dir: $RESULTS_DIR"
echo ""

# Check if test binary exists
TEST_BIN="${BUILD_DIR}/tests/scalable_ccd_tests"
if [[ ! -x "$TEST_BIN" ]]; then
    echo "Error: Test binary not found at $TEST_BIN"
    echo "Please build the project first:"
    echo "  mkdir build && cd build"
    echo "  cmake .. -DSCALABLE_CCD_WITH_${BACKEND^^}=ON"
    echo "  make -j"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "Running benchmarks..."
echo ""

case "$BACKEND" in
    metal)
        echo "[Metal] Running broad phase tests..."
        "$TEST_BIN" "[broad_phase][metal2]" 2>&1 | tee "${RESULTS_DIR}/metal_benchmark.log"

        echo ""
        echo "[Metal] Running narrow phase tests..."
        if [[ -x "${BUILD_DIR}/tests/test_narrow_phase_metal2" ]]; then
            "${BUILD_DIR}/tests/test_narrow_phase_metal2" 2>&1 | tee -a "${RESULTS_DIR}/metal_benchmark.log"
        fi
        ;;

    cuda)
        echo "[CUDA] Running broad phase tests..."
        "$TEST_BIN" "[broad_phase][cuda]" 2>&1 | tee "${RESULTS_DIR}/cuda_benchmark.log"

        echo ""
        echo "[CUDA] Running narrow phase tests..."
        "$TEST_BIN" "[narrow_phase][cuda]" 2>&1 | tee -a "${RESULTS_DIR}/cuda_benchmark.log"
        ;;

    all)
        echo "[All] Running all available tests..."
        "$TEST_BIN" "[broad_phase]" 2>&1 | tee "${RESULTS_DIR}/all_benchmark.log"
        ;;

    *)
        echo "Unknown backend: $BACKEND"
        echo "Usage: $0 [cuda|metal|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "JSON files:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "  (no JSON files found)"
echo ""
echo "Next steps:"
echo "  1. Copy JSON files to shared location"
echo "  2. Run: python scripts/compare_results.py"
