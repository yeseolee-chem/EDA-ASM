#!/bin/bash
# Run OptTS + EDA-NOCV SPE for a single reaction.
# Idempotent: skips if channels_dft.json already exists.
#
# Usage: run_one.sh <rxn_id>   (rxn_id is a plain int, e.g. 3, 56, 1217)

set -uo pipefail

RXN_ID=$1
PILOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/dft_pilot
SCRIPTS=$PILOT/scripts
WORK_ROOT=$PILOT/work
RXN_TAG=$(printf "rxn_%04d" "$RXN_ID")
WORK=$WORK_ROOT/$RXN_TAG
ORCA=$HOME/orca_6_1_1_avx2/orca

mkdir -p "$WORK"
cd "$WORK"

# Idempotency gate
if [ -f channels_dft.json ]; then
    echo "SKIP $RXN_TAG: channels_dft.json exists"
    exit 0
fi

STAMP=$(date -Is)
echo "=== $RXN_TAG  start=$STAMP  host=$(hostname) ==="

# --- Stage 1: TS reoptimization at DFT ---
if [ ! -f dft_optts.done ]; then
    if [ ! -f dft_optts.inp ]; then
        echo "FATAL: dft_optts.inp missing (prepare_inputs.py not run for this rxn)"
        exit 2
    fi
    echo "--- stage 1: OptTS  time=$(date -Is) ---"
    $ORCA dft_optts.inp > dft_optts.out 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "OptTS failed rc=$RC"
        exit $RC
    fi
    if ! grep -q "OPTIMIZATION HAS CONVERGED" dft_optts.out; then
        echo "OptTS did not converge (no 'OPTIMIZATION HAS CONVERGED' in output)"
        exit 10
    fi
    # Verify exactly 1 imaginary frequency (TS check)
    NIMAG=$(grep -c "imaginary" dft_optts.out || echo 0)
    echo "NIMAG hint (grep count) = $NIMAG"
    touch dft_optts.done
    echo "--- stage 1 done  time=$(date -Is) ---"
else
    echo "--- stage 1: skip (dft_optts.done exists) ---"
fi

# --- Stage 2: build EDA input from optimized geometry ---
if [ ! -f eda_tz.inp ]; then
    echo "--- stage 2: build EDA input ---"
    python3 $SCRIPTS/build_eda_input.py "$WORK" || exit 11
fi

# --- Stage 3: EDA-NOCV SPE at def2-TZVP ---
if [ ! -f eda_tz.done ]; then
    echo "--- stage 3: EDA SPE  time=$(date -Is) ---"
    $ORCA eda_tz.inp > eda_tz.out 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "EDA SPE failed rc=$RC"
        exit $RC
    fi
    if ! grep -q "ORCA TERMINATED NORMALLY" eda_tz.out; then
        echo "EDA did not terminate normally"
        exit 12
    fi
    touch eda_tz.done
    echo "--- stage 3 done  time=$(date -Is) ---"
fi

# --- Stage 4: parse channels (uses existing kisti parser) ---
if [ ! -f channels_dft.json ]; then
    echo "--- stage 4: parse EDA ---"
    PARSER_DIR=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/kisti_results/scripts
    python3 -c "
import sys, json
sys.path.insert(0, '$PARSER_DIR')
from orca_eda_parser import parse_eda_out
r = parse_eda_out('eda_tz.out')
r['rxn_id'] = $RXN_ID
r['protocol'] = 'dft_optts_svp_then_eda_tzvp'
with open('channels_dft.json','w') as f: json.dump(r, f, indent=2, default=float)
print(f\"parsed: ok={r['ok']}  eint={r.get('eint_dft','?')}\")
" || exit 13
fi

# --- Cleanup scratch (keep only .out + .xyz + .json) ---
find . -maxdepth 1 -type f \( \
    -name "*.tmp" -o -name "*.hess" -o -name "*.densities" \
    -o -name "*.gbw" -o -name "*_frag*.gbw" -o -name "*_frag*.densities" \
    -o -name "*.frag*.tmp" -o -name "*.cpcm*" -o -name "*.bibtex" \
    -o -name "*.densitiesinfo" -o -name "*.property.txt" \
    \) -delete 2>/dev/null

END=$(date -Is)
echo "=== $RXN_TAG  end=$END  rc=0 ==="
