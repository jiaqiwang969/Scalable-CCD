#!/usr/bin/env bash
set -euo pipefail

# Run build matrix via CMakePresets and execute verifier, generating per-preset reports.
# Usage: scripts/run_matrix.sh [OUTPUT_DIR] [THREADS] [REPEAT] [WARMUP]

OUT_ROOT="${1:-build/reports-matrix}"
THREADS="${2:-$(nproc)}"
REPEAT="${3:-3}"
WARMUP="${4:-1}"

PRESETS=(
  cpu-double-release
  cpu-float-release
  cuda-double-release
  cuda-float-release
)

mkdir -p "${OUT_ROOT}"

for P in "${PRESETS[@]}"; do
  echo "[run_matrix] Configure & Build ${P}"
  cmake --preset "${P}"
  cmake --build --preset "${P}" --parallel
  # Determine backend by preset name
  BACKEND="cpu"
  if [[ "${P}" == cuda-* ]]; then BACKEND="cuda"; fi
  BIN="build/${P}/verifier/scalable_ccd_verifier"
  OUT_DIR="${OUT_ROOT}/${P}"
  echo "[run_matrix] Execute verifier for ${P} (backend=${BACKEND})"
  "${BIN}" --backend "${BACKEND}" --threads "${THREADS}" --repeat "${REPEAT}" --warmup "${WARMUP}" --tag "${P}" --out "${OUT_DIR}"
done

echo "[run_matrix] Done. Reports under ${OUT_ROOT}"

