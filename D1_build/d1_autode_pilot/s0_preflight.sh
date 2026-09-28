#!/bin/bash
# s0_preflight.sh — gate before any D1 job (submit_pilot.sh chains the array with afterok on this job).
# Exit 0 only if every check passes; the report goes to $SCRATCH/s0/S0_REPORT.md.
#SBATCH --job-name=d1p_s0
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
set -uo pipefail
source "$PILOT_DIR/pilot_env.sh"
S0=$SCRATCH/s0; mkdir -p "$S0"; REP=$S0/S0_REPORT.md
cd "$PILOT_DIR"
echo "# S0 preflight $(date -Is) on $(hostname)" > "$REP"
FAIL=0
step() {   # step NAME cmd...
    local name=$1; shift
    echo "## $name" >> "$REP"; echo '```' >> "$REP"
    "$@" >> "$REP" 2>&1; local rc=$?
    echo '```' >> "$REP"
    if [ $rc -eq 0 ]; then echo "- **PASS** $name" >> "$REP"; else echo "- **FAIL** $name (rc=$rc)" >> "$REP"; FAIL=1; fi
}
versions() {
    python - <<'PY'
import sys, importlib
print("python", sys.version.split()[0])
for m in ("autode", "rdkit", "networkx", "numpy", "pandas", "yaml"):
    mod = importlib.import_module(m)
    print(m, getattr(mod, "__version__", "?"))
import autode, rdkit
from rdkit.Chem import rdDetermineBonds  # noqa: F401  (needs RDKit >= 2022.09)
assert autode.__version__ == "1.4.5", "autodE must be 1.4.5 (the version the pilot code was checked against)"
PY
    "$XTB_BIN" --version 2>&1 | grep -m1 -i "xtb version"
    [ -x "$ORCA_BIN" ] && echo "orca binary: $ORCA_BIN" || { echo "orca binary missing"; return 1; }
    command -v g16 >/dev/null 2>&1 && echo "g16 found: $(command -v g16)  (engine stays as in config.yaml)" || echo "g16: not on PATH"
}
step "versions"                     versions
step "config tags on all D0 TS geometries"  python tests/validate_config_on_coley.py --profiles "$(cfgget d0_profiles)" \
        --csv "$(cfgget d0_csv)" --labels "$(cfgget repo_root)/$(cfgget labels_all)" --repo "$(cfgget repo_root)" \
        --out "$S0/cfg_validation.csv"
CTRL=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="d0_rxn_id") c=i} NR>1 && $c!="" {if(!s[$c]++) printf "%d ", $c}' pilot_manifest.csv)
step "input parity with D0 files (rxn $CTRL 0 1 2)"  python tests/parity_d0.py --rxn $CTRL 0 1 2
step "xtb scan output format"       python tests/xtb_scan_format.py
export D1P_MANIFEST=$PILOT_DIR/manifest_s0.csv
smoke() {
    run_stage S0 ts     python run_ts.py --job S0          || return 1
    run_stage S0 ref    python make_reference.py --job S0  || return 1
    run_stage S0 inputs python build_sp_inputs.py --job S0 || return 1
    run_stage S0 sp     run_sp S0                          || return 1
    run_stage S0 label  python assemble.py --job S0        || return 1
}
step "smoke chain HCNO + C2H4 (autodE + ORCA 6.1.1, full pipeline)"  smoke
step "smoke checks"                 python s0_check_smoke.py
unset D1P_MANIFEST
echo "" >> "$REP"; [ $FAIL -eq 0 ] && echo "**S0: PASS**" >> "$REP" || echo "**S0: FAIL — the D1 array will not start**" >> "$REP"
exit $FAIL
