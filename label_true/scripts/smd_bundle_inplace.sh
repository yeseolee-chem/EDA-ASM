#!/bin/bash
#SBATCH --job-name=smd_bundle2
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/logs/bundle_inplace.%j.out
set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /home1/yeseo1ee/projects/eda-asm-prediction/label_true
python scripts/smd_bundle_inplace.py
echo ""
echo "=== final layout ==="
ls -d /gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs/bundle_*/
for k in 1 2 3 4 5; do
    B=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs/bundle_$k
    N=$(ls $B/inputs 2>/dev/null | grep -c '^rxn_' || echo 0)
    SZ=$(du -sh $B 2>/dev/null | awk '{print $1}')
    echo "  bundle_$k: $N rxns, $SZ  (has: $(ls $B | tr '\n' ' '))"
done
