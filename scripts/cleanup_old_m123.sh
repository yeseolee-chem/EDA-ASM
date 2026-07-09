#!/bin/bash
#SBATCH --job-name=cleanup_m123
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/cleanup_m123.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/cleanup_m123.%j.err

# Delete previous m1/m2/m3 training run artefacts to give the v7 run
# a fresh start. Keeps the code, deletes only bundles/subsamples/output/logs.

set -uo pipefail
REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features

echo "[cleanup] BEFORE:"
du -sh "$REPO/m1/code/trackB_lowlr_no_ood_geom6" 2>/dev/null || true
du -sh "$REPO/m2/code/trackB_lowlr_no_ood_xtb_geom6" 2>/dev/null || true
du -sh "$REPO/m3/code/trackB_lowlr_no_ood_xtb_geom6_plus_v2" 2>/dev/null || true
du -sh "$SCRATCH/bundles_v1" 2>/dev/null || true

# Delete previous training outputs (member*.json, checkpoints)
rm -rf "$REPO/m1/code/trackB_lowlr_no_ood_geom6"
rm -rf "$REPO/m2/code/trackB_lowlr_no_ood_xtb_geom6"
rm -rf "$REPO/m3/code/trackB_lowlr_no_ood_xtb_geom6_plus_v2"

# Delete old v1 bundles (v7 bundles are in bundles_v7/)
# NOTE: keep descriptors_v1.parquet — that's shared feature data.

# Delete old training logs from spec_v1_logs (keep new v7 logs by pattern)
find "$SCRATCH/spec_v1_logs" -maxdepth 1 -name "m[123]_*" ! -name "*_v7_*" -delete 2>/dev/null || true

echo ""
echo "[cleanup] AFTER:"
du -sh "$REPO/m1" "$REPO/m2" "$REPO/m3" 2>/dev/null || true
ls "$SCRATCH/spec_v1_logs" | wc -l
echo "cleanup done"
