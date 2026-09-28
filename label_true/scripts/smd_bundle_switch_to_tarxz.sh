#!/bin/bash
#SBATCH --job-name=smd_migrate
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/logs/migrate.%j.out
set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /home1/yeseo1ee/projects/eda-asm-prediction/label_true
python scripts/smd_bundle_switch_to_tarxz.py
echo ""
echo "=== FINAL VERIFICATION ==="
for k in 1 2 3 4 5; do
    B=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs/bundle_$k
    SZ=$(du -sh $B 2>/dev/null | awk '{print $1}')
    N=$(ls $B/inputs 2>/dev/null | grep -c '^rxn_' || echo 0)
    HAS_TAR=$([ -f $B/*.tar.xz ] 2>/dev/null && echo yes || echo NO)
    HAS_ORCA_DIR=$([ -d $B/orca_6_1_1_avx2 ] && echo YES || echo no)
    HAS_LAUNCH=$([ -f $B/launch.sh ] && echo yes || echo NO)
    HAS_WORKER=$([ -f $B/worker.sh ] && echo yes || echo NO)
    printf "  bundle_%d: %d rxns, size=%s  tar.xz=%s  orca_dir=%s  launch=%s  worker=%s\n" \
        $k $N $SZ $HAS_TAR $HAS_ORCA_DIR $HAS_LAUNCH $HAS_WORKER
    # verify worker.sh has the SLURM_SUBMIT_DIR fix
    if grep -q 'SLURM_SUBMIT_DIR' $B/worker.sh; then
        echo "           worker.sh: SLURM_SUBMIT_DIR ✓"
    else
        echo "           worker.sh: SLURM_SUBMIT_DIR ✗ MISSING"
    fi
done

echo ""
echo "=== CLEAN UP TEST BUNDLE ==="
rm -rf /gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/test_bundle
echo "test_bundle deleted"
