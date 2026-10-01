#!/bin/bash
#SBATCH --job-name=r6_P
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/logs/%x_%j.out
# r6_P.sh — rev 6 Phase P (r6_espley.py). MODE = p1_ml | p1_hyp | p1_compare (Espley-pinned env, their code
# unchanged) | p3 (reactot). Idempotent through their own checkpoints / existing outputs.
# Submit from analysis/espley_xtb_repro/:  sbatch --export=ALL,MODE=p1_ml --job-name=r6_P1ml r6_P.sh
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export R6_SCRATCH=${R6_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev6}
MODE=${MODE:?MODE=p1_ml|p1_hyp|p1_compare|p3}
echo "[r6_P] host=$(hostname) mode=$MODE git=$(git rev-parse --short HEAD) start=$(date -Is)"
case "$MODE" in
  p1_ml|p1_hyp|p1_compare)
    export PYTHONNOUSERSITE=1
    PY=$R6_SCRATCH/env_espley/bin/python ;;
  p3)
    source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh; conda activate reactot
    PY=python ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
"$PY" -m py_compile rev6_common.py r6_espley.py || exit 2
"$PY" r6_espley.py "$MODE"
rc=$?
echo "[r6_P] done rc=$rc $(date -Is)"
exit $rc
