#!/bin/bash
#SBATCH --job-name=trig_s5
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/trig_s5.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/trig_s5.%j.err

# Wait 30 min for submit counter window to clear, then submit stage5 + stage6
sleep 1800
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
S5=$(sbatch --parsable scripts/v8_ml/stage5_train_v8.sh)
echo "[trigger] stage5: $S5"
S6=$(sbatch --dependency=afterany:$S5 --parsable scripts/v8_ml/stage6_aggregate_v8.sh)
echo "[trigger] stage6: $S6"
