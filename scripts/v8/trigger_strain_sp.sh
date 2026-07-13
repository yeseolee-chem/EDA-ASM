#!/bin/bash
#SBATCH --job-name=trig_strain
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/trigger_strain.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/trigger_strain.%j.err

# When 730566 finishes, this script fires and submits strain_sp array
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
echo "[$(date)] cleanup EDA (730566) done -> submitting strain_sp"
sbatch scripts/v8/run_strain_sp.sh
echo "[$(date)] submitted"
