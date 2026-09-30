#!/bin/bash
#SBATCH --job-name=r5_ext_smoke
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6 --mem=8G
# Phase B-0 (docs/specs/REV5_FEATURES_FIGURES.md): engine smoke of ext_features.py on rev5_common.SMOKE_RXNS (those
# with G1 structures; a rxn whose G1 step failed is reported as skipped; all are neutral) plus one extra rxn per nonzero
# (charge1, charge2) pattern of rows_rev4 (the smallest rxn_id; the labels have (0,-2) and (0,+1)), so that gate 1 is
# also tested on ions (B0_smoke.json extra_charged). The same stop rules apply to every evaluated rxn. Per rxn:
#   gate 1  |E_tblite - E_xtb| < GATE_E_EH (tblite 0.7.0 GFN2 / ALPB water vs xtb 6.7.1) for rel1, rel2, fA, fB, TS
#   gate 2  fragment AO overlap == the TS overlap block within GATE_S_BLOCK (AO -> atom order mapping)
#   gate 3  max|C^T S C - I| < GATE_ORTHO
#   checks  xtb --json HOMO/LUMO == tblite, atomic quadrupole component order (trace test), Σ q r + Σ μ == dipole,
#           engine versions, and every block B1..B6 ok (all through the production code path, cache
#           $R5_SCRATCH/ext_smoke/<rid>/), core-s per rxn and the projected core-h for all G1-ok rxns.
# -> results_rev5/B0_smoke.json. Exit 3 = a gate / check failed: STOP (spec B-0). Always recomputes from scratch
# ($R5_SCRATCH/ext_smoke/<rid>/ is removed first) so that a rerun after a fix never reuses old records; a few
# core-minutes per rxn (one single-threaded worker per rxn: 6 rxns with the charged extras).
#   sbatch --partition=cpu1,cpu2 --cpus-per-task=6 --mem=8G --output=$R5_SCRATCH/logs/ext_smoke.%j.out r5_ext_smoke.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
PY=${PY:-$R5_SCRATCH/venv/bin/python}
[ -x "$PY" ] || { echo "STOP: $PY missing (r5_probe.sh creates the rev 5 venv)"; exit 2; }
export ESPLEY_OUT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XTB_THREADS=1 OMP_STACKSIZE=1G PYTHONWARNINGS=ignore
mkdir -p "$R5_SCRATCH/logs" "$CODE/results_rev5" || exit 2
"$XTB_BIN" --version 2>&1 | grep -i "version" | head -2
"$PY" -c "import importlib.metadata as m, tblite.interface, pyarrow, networkx, dftd3, morfeus; print('py env ok: tblite', m.version('tblite'))" \
    || { echo "STOP: rev 5 venv imports failed"; exit 2; }
echo "code $CODE  G1_ROOT $G1_ROOT  ESPLEY_OUT $ESPLEY_OUT  R5_SCRATCH $R5_SCRATCH  cpus ${SLURM_CPUS_PER_TASK:-1}"
"$PY" ext_features.py smoke --workers "${SLURM_CPUS_PER_TASK:-6}"
rc=$?
echo "ext_features.py smoke exit $rc (0 = B-0 passed, 3 = a gate / check failed: STOP)"
exit $rc
