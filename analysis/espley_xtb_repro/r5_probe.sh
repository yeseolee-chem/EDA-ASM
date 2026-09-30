#!/bin/bash
#SBATCH --job-name=r5_probe
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# Phase B pre-flight (no features): create the rev 5 venv on top of reactot and install tblite 0.7.0 (STOP if not
# installable), then record the real output formats the B-block parsers rely on: xtb 6.7.1 --json keys, --vipea,
# --vfukui and the D4 alpha(0) table, and the tblite result keys, for rxn 105's G1 structures (rxn 20 has no G1 geometry: OptTS not converged).
#   sbatch --partition=cpu1,cpu2 --output=$R5_SCRATCH/logs/probe.%j.out r5_probe.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
V=$R5_SCRATCH/venv
P=$R5_SCRATCH/probe; mkdir -p "$P"
if [ ! -x "$V/bin/python" ]; then python -m venv --system-site-packages "$V" || { echo "STOP: venv"; exit 2; }; fi
"$V/bin/pip" install --quiet "tblite==0.7.0" 2>&1 | tail -5
"$V/bin/python" -c "import tblite, importlib.metadata as m; print('tblite', m.version('tblite'))" \
    || { echo "STOP: tblite 0.7.0 not installable"; exit 2; }
R=$G1_ROOT/105
cp "$R/rel1.xyz" "$P/rel1.xyz"
cd "$P"
export OMP_NUM_THREADS=1
"$XTB_BIN" rel1.xyz --gfn 2 --alpb water --chrg 0 --json > sp_json.out 2>&1; cp -f xtbout.json sp.json 2>/dev/null
echo "== xtbout.json keys"; "$V/bin/python" -c "import json; d=json.load(open('sp.json')); print({k: (type(v).__name__, len(v) if hasattr(v,'__len__') else v) for k,v in d.items()})"
echo "== D4 / alpha lines"; grep -n -i -A12 "C6AA\|alpha\|α(0)" sp_json.out | head -60
"$XTB_BIN" rel1.xyz --gfn 2 --alpb water --chrg 0 --vipea > vipea.out 2>&1
echo "== vipea"; grep -n -i "delta SCC\|IP\b\|EA\b\|ionization\|electron affinity" vipea.out | head -20
"$XTB_BIN" rel1.xyz --gfn 2 --alpb water --chrg 0 --vfukui > vfukui.out 2>&1
echo "== vfukui"; grep -n -i -A14 "fukui" vfukui.out | head -40
echo "== tblite"
"$V/bin/python" - <<'EOF'
import numpy as np
from tblite.interface import Calculator
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17, "Br": 35}
L = open("rel1.xyz").read().split("\n")
n = int(L[0]); rows = [l.split() for l in L[2:2 + n]]
num = np.array([Z[r[0]] for r in rows]); xyz = np.array([[float(v) for v in r[1:4]] for r in rows]) * 1.8897259886
for solv in (None, "alpb"):
    c = Calculator("GFN2-xTB", num, xyz, charge=0, uhf=0)
    c.set("save-integrals", 1); c.set("verbosity", 0)
    if solv:
        try:
            c.add("alpb-solvation", "water")
        except Exception as e:
            print("alpb add failed:", type(e).__name__, e)
    r = c.singlepoint()
    keys = [k for k in ("energy", "orbital-energies", "orbital-occupations", "orbital-coefficients", "overlap-matrix",
                        "charges", "dipole", "quadrupole") if True]
    out = {}
    for k in keys:
        try:
            v = np.asarray(r.get(k)); out[k] = v.shape
        except Exception as e:
            out[k] = f"ERR {type(e).__name__}"
    print(solv, "E =", float(r.get("energy")), out)
    C, S = np.asarray(r.get("orbital-coefficients")), np.asarray(r.get("overlap-matrix"))
    print("  max|C^T S C - I| =", float(np.abs(C.T @ S @ C - np.eye(C.shape[1])).max()), "C shape", C.shape)
EOF
echo "== xtb total energy (ALPB)"; grep -n "TOTAL ENERGY" sp_json.out | tail -1
ls -la "$P"
