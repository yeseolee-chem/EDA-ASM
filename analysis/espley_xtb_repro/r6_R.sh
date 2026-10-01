#!/bin/bash
#SBATCH --job-name=r6_R
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/logs/%x_%A_%a.out
# r6_R.sh — rev 6 Phase R (r6_recover.py; settings fixed in results_rev6/R_plan.md, committed before any run).
#   MODE=inventory            targets + R-1 DFT-partition diagnostic (exit 3 = STOP)
#   MODE=run                  array element i = slice i of R6_NSLICES (Espley rows first); R6_ONLY=<ids> = smoke
#   MODE=summary              R2_attempts / R_rows / R2_by_reason / R2_report (exit 3 = STOP)
# Idempotent: every rxn with a marker under $G1_ROOT_R6 is skipped; ORCA / xtb steps skip finished outputs.
# Submit from analysis/espley_xtb_repro/:
#   sbatch --export=ALL,MODE=inventory --cpus-per-task=8 --mem=16G r6_R.sh
#   sbatch --array=0-7 --export=ALL,MODE=run,R6_NSLICES=8 r6_R.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R6_SCRATCH=${R6_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev6}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m py_compile rev6_common.py r6_partition.py r6_recover.py xtb_slice.py ext_features.py || exit 2
MODE=${MODE:?MODE=inventory|run|summary}
case "$MODE" in
  inventory) python r6_recover.py inventory ;;
  run)       python r6_recover.py run "${SLURM_ARRAY_TASK_ID:-0}" "${R6_NSLICES:-1}" ;;
  summary)   python r6_recover.py summary ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
