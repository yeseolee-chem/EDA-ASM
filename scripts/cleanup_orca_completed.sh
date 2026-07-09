#!/bin/bash
#SBATCH --job-name=orca_cleanup
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_cleanup.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_cleanup.%j.err

set -euo pipefail

INP=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/orca_eda/inputs
n_cleaned=0
n_skipped=0

for out in "$INP"/*/eda.out; do
  [ -f "$out" ] || continue
  # Only clean if ORCA terminated normally
  if ! grep -q "ORCA TERMINATED NORMALLY" "$out" 2>/dev/null; then
    n_skipped=$((n_skipped + 1))
    continue
  fi
  dir=$(dirname "$out")
  rm -f "$dir"/eda.densities "$dir"/eda_frag1.densities "$dir"/eda_frag2.densities
  rm -f "$dir"/eda.gbw "$dir"/eda.nocv.gbw "$dir"/eda_frag1.gbw "$dir"/eda_frag2.gbw
  rm -f "$dir"/eda.bas0 "$dir"/eda.bas1 "$dir"/eda.bas2 "$dir"/eda.bas3 "$dir"/eda.bas4 "$dir"/eda.bas5
  rm -f "$dir"/eda_frag1.bas0 "$dir"/eda_frag1.bas1 "$dir"/eda_frag2.bas0 "$dir"/eda_frag2.bas1
  rm -f "$dir"/eda.densitiesinfo "$dir"/eda.bibtex "$dir"/eda_frag1.bibtex "$dir"/eda_frag2.bibtex
  rm -f "$dir"/eda.int.tmp "$dir"/eda.tmp
  n_cleaned=$((n_cleaned + 1))
done

echo "cleaned $n_cleaned completed reactions"
echo "skipped $n_skipped (still running or failed — kept for debugging)"
du -sh "$INP" 2>&1 || true
