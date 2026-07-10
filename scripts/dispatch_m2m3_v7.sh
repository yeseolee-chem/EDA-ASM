#!/bin/bash
#SBATCH --job-name=disp_m2m3
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/disp_m2m3.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/disp_m2m3.%j.err

# Runs after m1 v7 array completes. Submits m2 first, then m3 chained
# after m2 so only one 5-task array occupies the queue at a time.

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

JID_M2=$(sbatch --parsable scripts/train_m2_v7.sh)
echo "submitted m2: $JID_M2"
JID_M3=$(sbatch --parsable --dependency=afterany:$JID_M2 scripts/train_m3_v7.sh)
echo "submitted m3 (chained after m2): $JID_M3"
JID_AGG=$(sbatch --parsable --dependency=afterany:$JID_M3 scripts/stage6_v7.sh)
echo "submitted stage6 v7 aggregate (chained after m3): $JID_AGG"
