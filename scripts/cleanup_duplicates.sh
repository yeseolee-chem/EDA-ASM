#!/bin/bash
#SBATCH --job-name=cleanup_dup
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/cleanup_dup.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/cleanup_dup.%j.err

# Delete duplicates in outputs/, keep only final_776_v7 + tar.gz + audit.

set -uo pipefail
OUT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs

echo "[cleanup] BEFORE:"
du -sh "$OUT"/* 2>&1 | grep -v "Not a directory"

# Delete duplicates (contents in final_776_v7)
rm -rf "$OUT/orca_eda"
rm -rf "$OUT/orca_strain"

# Delete old 789-reaction ORCA labels (per user request — v7 supersedes)
REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
rm -f "$REPO/labels/orca/orca_eda_labels.parquet"
rm -f "$REPO/labels/orca/orca_strain_labels.parquet"
rm -f "$REPO/labels/adf_vs_orca_comparison.parquet"
rm -f "$REPO/labels/adf_vs_orca_full_comparison.parquet"
# Old exports
rm -f "$OUT/frag_view.tar.gz" "$OUT/frag_view_xyz.tar.gz"
# Old ASR-V1 training results (independent from v7 work)
rm -rf "$OUT/asr_v1"
# frag_review: has cohort_v7.parquet + backups; move backups into final and delete rest
mkdir -p "$OUT/final_776_v7/backups"
cp -n "$OUT/frag_review/cohort_v7.parquet" "$OUT/final_776_v7/backups/" 2>/dev/null || true
cp -n "$OUT/frag_review/manual_partitions.json" "$OUT/final_776_v7/backups/" 2>/dev/null || true
cp -n "$OUT/frag_review/orca_inp_partitions.json" "$OUT/final_776_v7/backups/" 2>/dev/null || true
rm -rf "$OUT/frag_review"
# 2frag_audit: keep as separate small folder (already in final_776_v7? move to final)
mkdir -p "$OUT/final_776_v7/2frag_audit"
mv "$OUT/2frag_audit"/*.parquet "$OUT/final_776_v7/2frag_audit/" 2>/dev/null || true
rm -rf "$OUT/2frag_audit"

echo ""
echo "[cleanup] AFTER:"
du -sh "$OUT"/* 2>&1 | grep -v "Not a directory"
echo ""
echo "[cleanup] final structure:"
ls -la "$OUT/"
