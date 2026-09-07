#!/bin/bash
# SPEC17rev2 failed-rxn Flask viewer.
# Runs a long-lived HTTP server on cpu2 (~2GB RAM); users connect via
# SSH tunnel or direct node URL (n0xx:8765).
#
#SBATCH --job-name=otfmVW
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train/logs/viewer_%j.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# port collision fallback: try 8765 → 8766 → ... → 8775
for p in $(seq 8765 8775); do
    if ! ss -Hln "sport = :$p" | grep -q .; then
        export VIEWER_PORT=$p
        break
    fi
done
echo "[viewer] using port $VIEWER_PORT on $(hostname)"

cd "$BASE/tools/failed_viewer"
python -u app.py
