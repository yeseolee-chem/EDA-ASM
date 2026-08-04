#!/bin/bash
#SBATCH --job-name=p203_gated_ml
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/paper_reproduction_203/logs/gated_ml_%j.log
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/paper_reproduction_203/logs/gated_ml_%j.err
#
# Run the ML pipeline on the gated (n=121) cohort:
#   - ml_analysis.py × 25 seeds (Monte Carlo CV)
#   - 80_visualize.py → parity plots + REPORT.md
#
# Prereqs (must exist):
#   pipeline_work/manual_dipolar_gated.pkl   (feature filter output on 121 rxns)
#   pipeline_work/hps.pkl                    (hyp_tuning output on gated cohort)

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/paper_reproduction_203
WORK="$ROOT/pipeline_work"
GRAY="$ROOT/grayson_code"

cd "$WORK"
rm -f "$WORK/ml_results.pkl"

echo "[$(date +%H:%M:%S)] === ml_analysis × 25 seeds (gated n=121) ==="
SEEDS=(23 22 14 1 2  3 4 5 6 7  8 9 10 11 12  13 15 16 17 18  19 20 21 24 25)
for SEED in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)]   seed=$SEED"
    python "$GRAY/machine_learning/ml_analysis.py" $SEED "$WORK/manual_dipolar_gated.pkl" "$WORK/hps.pkl" >/dev/null 2>&1
done

echo "[$(date +%H:%M:%S)] === visualize + REPORT ==="
python "$ROOT/code/80_visualize.py"

echo "[$(date +%H:%M:%S)] === DONE ==="
ls -la "$WORK/ml_results.pkl" "$ROOT/results/" "$ROOT/figures/" "$ROOT/REPORT.md" 2>/dev/null
