#!/bin/bash
# Compute distortion (strain) energies d1, d2 for one reaction on AM1 TS geometry.
#
# Per fragment i (i=1,2):
#   Stage A: opt at B3LYP-D3BJ/def2-SVP CPCM(SMD water)  → fragi_opt.xyz + E_opt_svp
#   Stage B: SPE at def2-TZVP on relaxed geom              → E_rel_tzvp_i
#   Stage C: SPE at def2-TZVP on original AM1 coords       → E_dist_tzvp_i
#
# d_i (kcal/mol) = (E_dist_tzvp_i − E_rel_tzvp_i) × 627.5094740631
#
# Writes strain.json. Idempotent: skip if strain.json exists.
#
# Usage: strain_one_rxn.sh <rxn_id>

set -uo pipefail

RXN_ID=$1
STRAIN=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1
RXN_TAG=$(printf "rxn_%04d" "$RXN_ID")
DATA=$STRAIN/data/$RXN_TAG
WORK=$STRAIN/work/$RXN_TAG
ORCA=$HOME/orca_6_1_1_avx2/orca
K=627.5094740631

mkdir -p "$WORK"
cd "$WORK"

if [ -f strain.json ]; then
    echo "SKIP $RXN_TAG: strain.json exists"
    exit 0
fi

if [ ! -f "$DATA/frag1_am1.xyz" ] || [ ! -f "$DATA/frag2_am1.xyz" ]; then
    echo "FAIL $RXN_TAG: missing frag xyz in $DATA"
    exit 2
fi

echo "=== $RXN_TAG  start=$(date -Is)  host=$(hostname) ==="

# ORCA header helpers -------------------------------------------------
opt_header() { cat <<EOF
! Opt B3LYP D3BJ def2-SVP CPCM(water) NoSym TightSCF
%maxcore 3500
%pal nprocs 4 end
%cpcm
  smd true
  smdsolvent "water"
end
EOF
}
sp_tzvp_header() { cat <<EOF
! SP B3LYP D3BJ def2-TZVP CPCM(water) NoSym TightSCF
%maxcore 3500
%pal nprocs 4 end
%cpcm
  smd true
  smdsolvent "water"
end
EOF
}

xyz_body() {
    # append xyz coords (without header) as * xyz 0 1 ... * block
    local xyz="$1"
    echo "* xyz 0 1"
    tail -n +3 "$xyz"    # skip the 2-line xyz header
    echo "*"
}

# Extract "FINAL SINGLE POINT ENERGY" (Hartree) from ORCA out
get_fspe() {
    grep -a "FINAL SINGLE POINT ENERGY" "$1" | tail -1 | awk '{print $NF}'
}

# Per-fragment loop --------------------------------------------------
declare -A E_DIST_EH E_REL_EH
for I in 1 2; do
    SRC_XYZ=$DATA/frag${I}_am1.xyz

    # --- Stage A: opt at SVP ---
    OPT_INP=frag${I}_opt.inp
    OPT_OUT=frag${I}_opt.out
    if [ ! -f frag${I}_opt.done ]; then
        {
            opt_header
            xyz_body "$SRC_XYZ"
        } > "$OPT_INP"
        echo "--- frag$I stage A (opt SVP) $(date -Is) ---"
        $ORCA "$OPT_INP" > "$OPT_OUT" 2>&1
        if ! grep -q "ORCA TERMINATED NORMALLY" "$OPT_OUT"; then
            echo "FAIL frag$I opt"; tail -20 "$OPT_OUT"; exit 10
        fi
        if ! grep -q "OPTIMIZATION HAS CONVERGED" "$OPT_OUT"; then
            echo "FAIL frag$I opt did not converge"; exit 11
        fi
        # ORCA writes optimized geometry to <basename>.xyz
        [ -f frag${I}_opt.xyz ] || { echo "FAIL: frag${I}_opt.xyz not produced"; exit 12; }
        touch frag${I}_opt.done
    fi

    # --- Stage B: SPE at TZVP on relaxed geom ---
    REL_INP=frag${I}_rel.inp
    REL_OUT=frag${I}_rel.out
    if [ ! -f frag${I}_rel.done ]; then
        {
            sp_tzvp_header
            xyz_body frag${I}_opt.xyz
        } > "$REL_INP"
        echo "--- frag$I stage B (SPE TZVP relaxed) $(date -Is) ---"
        $ORCA "$REL_INP" > "$REL_OUT" 2>&1
        if ! grep -q "ORCA TERMINATED NORMALLY" "$REL_OUT"; then
            echo "FAIL frag$I rel"; tail -20 "$REL_OUT"; exit 20
        fi
        touch frag${I}_rel.done
    fi

    # --- Stage C: SPE at TZVP on AM1-frozen geom ---
    DIST_INP=frag${I}_dist.inp
    DIST_OUT=frag${I}_dist.out
    if [ ! -f frag${I}_dist.done ]; then
        {
            sp_tzvp_header
            xyz_body "$SRC_XYZ"
        } > "$DIST_INP"
        echo "--- frag$I stage C (SPE TZVP AM1-frozen) $(date -Is) ---"
        $ORCA "$DIST_INP" > "$DIST_OUT" 2>&1
        if ! grep -q "ORCA TERMINATED NORMALLY" "$DIST_OUT"; then
            echo "FAIL frag$I dist"; tail -20 "$DIST_OUT"; exit 30
        fi
        touch frag${I}_dist.done
    fi

    # capture energies
    E_REL_EH[$I]=$(get_fspe "$REL_OUT")
    E_DIST_EH[$I]=$(get_fspe "$DIST_OUT")
done

# --- Compute distortion energies + emit JSON ---
python3 - <<PY
import json
K = $K
e_rel = {1: float("${E_REL_EH[1]}"), 2: float("${E_REL_EH[2]}")}
e_dist = {1: float("${E_DIST_EH[1]}"), 2: float("${E_DIST_EH[2]}")}
d1 = (e_dist[1] - e_rel[1]) * K
d2 = (e_dist[2] - e_rel[2]) * K
out = {
    "rxn_id": $RXN_ID,
    "protocol": "am1_ts_distortion_frag_opt_svp_spe_tzvp",
    "e_frag1_rel_tzvp_eh": e_rel[1],
    "e_frag2_rel_tzvp_eh": e_rel[2],
    "e_frag1_dist_tzvp_eh": e_dist[1],
    "e_frag2_dist_tzvp_eh": e_dist[2],
    "distortion_energy_1_dft": d1,
    "distortion_energy_2_dft": d2,
    "sum_distortion_energies_dft": d1 + d2,
    "ok": True,
}
with open("strain.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"OK $RXN_TAG  d1={d1:.4f}  d2={d2:.4f}  sum={d1+d2:.4f}")
PY

# --- Cleanup scratch (keep .out, .inp, .xyz [not _trj.xyz], .done, strain.json/labels.json) ---
find . -maxdepth 1 -type f \( \
    -name "*.tmp" -o -name "*.tmpg" -o -name "*.tmp[0-9]" -o -name "*.hess" \
    -o -name "*.densities" -o -name "*.densitiesinfo" \
    -o -name "*.gbw" -o -name "*.cpcm*" -o -name "*.bibtex" \
    -o -name "*.property.txt" -o -name "*.bas[0-9]" \
    -o -name "*.SHARKINP.tmp" -o -name "*.opt" -o -name "*.citations.tmp" \
    -o -name "*.ginp.tmp" -o -name "*.int.tmp" \
    -o -name "*.smd" -o -name "*.smd.grd" \
    -o -name "*.engrad" -o -name "*.carthess" \
    -o -name "*.hostnames" -o -name "*.0" \
    -o -name "*_trj.xyz" \
    \) -delete 2>/dev/null

echo "=== $RXN_TAG  end=$(date -Is)  rc=0 ==="
