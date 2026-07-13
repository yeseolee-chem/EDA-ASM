#!/bin/bash
#SBATCH --job-name=trig_spec
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/trig_spec.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/trig_spec.%j.err
# Fires after d26_28 array finishes → submits merge → spec5 → spec4+2
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
MERGE=$(sbatch --parsable scripts/v9_ml/run_prereq_merge.sh)
echo "[trigger] merge: $MERGE"
S5=$(sbatch --dependency=afterany:$MERGE --parsable scripts/v9_ml/run_spec05_v9.sh)
echo "[trigger] spec05: $S5"
S4=$(sbatch --dependency=afterany:$S5 --parsable scripts/v9_ml/run_spec04_v9.sh)
echo "[trigger] spec04: $S4"
S2=$(sbatch --dependency=afterany:$S5 --parsable scripts/v9_ml/run_spec02_v9.sh)
echo "[trigger] spec02: $S2"
