#!/bin/bash
#SBATCH --job-name=r6_F
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/logs/%x_%j.out
# r6_F.sh — rev 6 R-3 / R-4 features + R-5 row table (r6_features.py; results_rev6/R_plan.md §5-§6).
#   MODE=all (default) runs g1feat -> ext -> ext4994 -> report; MODE=<one of them> runs one step.
# Each step skips an existing output (idempotent; ext / ext4994 also cache per rxn and block).
# Submit from analysis/espley_xtb_repro/ after r6_R.sh MODE=summary:  sbatch r6_F.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R6_SCRATCH=${R6_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev6}
PY=${PY:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/venv/bin/python}       # reactot + tblite 0.7.0 (rev 5 venv)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XTB_THREADS=1 OMP_STACKSIZE=1G PYTHONWARNINGS=ignore
"$PY" -m py_compile rev6_common.py r6_partition.py r6_features.py xtb_slice.py ext_features.py || exit 2
G1_R6=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1_rev6
G1_R4=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1
step() {
  case "$1" in
    g1feat|ext) G1_ROOT=$G1_R6 ESPLEY_PARTITION=label "$PY" r6_features.py "$1" ;;
    ext4994)    G1_ROOT=$G1_R4 ESPLEY_PARTITION=redetect R6_B1_TOL=4994:1e-5 "$PY" r6_features.py ext4994 ;;
    report)     "$PY" r6_features.py report ;;
    *) echo "unknown step $1"; return 2 ;;
  esac
}
MODE=${MODE:-all}
if [ "$MODE" = all ]; then
  for s in g1feat ext ext4994 report; do
    echo "=== $s $(date '+%F %T')"; step "$s" || { rc=$?; echo "### $s exit $rc"; exit $rc; }
  done
else
  step "$MODE"
fi
