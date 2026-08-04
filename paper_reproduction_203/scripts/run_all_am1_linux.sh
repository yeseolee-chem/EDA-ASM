#!/bin/bash
# =====================================================================
# Linux bash script to run Gaussian AM1 SPE on 1015 .gjf files.
#
# Prereq:
#   - Gaussian 16 or 09 installed and `g16` (or `g09`) on PATH
#   - CD to the paper_reproduction_203/ root before running
#
# Usage:
#   bash scripts/run_all_am1_linux.sh              # serial
#   bash scripts/run_all_am1_linux.sh --parallel N # N parallel workers via GNU parallel
#
# Output:
#   gaussian_inputs/<type>/<name>.log  (Gaussian output)
# =====================================================================
set -uo pipefail

# Pick Gaussian binary
if command -v g16 >/dev/null 2>&1; then
    GAUSS_CMD=g16
elif command -v g09 >/dev/null 2>&1; then
    GAUSS_CMD=g09
else
    echo "ERROR: neither g16 nor g09 found on PATH. Install Gaussian." >&2
    exit 1
fi
echo "Using Gaussian: $GAUSS_CMD"

# Parse args
PARALLEL=1
if [ "${1:-}" = "--parallel" ]; then
    PARALLEL="${2:-4}"
    if ! command -v parallel >/dev/null 2>&1; then
        echo "ERROR: GNU parallel not found (needed for --parallel)" >&2
        exit 1
    fi
fi

run_one() {
    local gjf="$1"
    local stem="${gjf%.gjf}"
    if [ -f "${stem}.log" ] && grep -q "Normal termination" "${stem}.log" 2>/dev/null; then
        echo "  [SKIP] ${stem}"
        return 0
    fi
    echo "  [RUN]  ${stem}"
    "$GAUSS_CMD" "$gjf" || echo "  [FAIL] ${stem}"
}
export -f run_one
export GAUSS_CMD

TOTAL=0
FAILED=0
for TYPE in ts gs dist_gs; do
    echo "=== Running $TYPE jobs ==="
    cd "gaussian_inputs/$TYPE"
    files=(*.gjf)
    TOTAL=$((TOTAL + ${#files[@]}))
    if [ "$PARALLEL" -gt 1 ]; then
        parallel -j "$PARALLEL" run_one ::: "${files[@]}"
    else
        for f in "${files[@]}"; do
            run_one "$f" || FAILED=$((FAILED + 1))
        done
    fi
    cd ../..
done

echo ""
echo "=== DONE: $TOTAL jobs attempted ==="
echo "Verify with: python scripts/verify_am1_results.py"
