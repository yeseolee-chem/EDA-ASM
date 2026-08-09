#!/bin/bash
# Renumber duplicated spec dirs so all spec numbers are unique.
# Chronological rule: older mtime keeps original number, later one shifts.
# Cascade: spec12/13/…/17 all shift by +2.
#
# Order matters for filesystem rename (reverse order to avoid conflicts).
# Text substitutions use exact strings (suffix disambiguation) so ordering
# among them is irrelevant.
#
# Idempotent: skips rename if source dir already gone (i.e., already renamed).

#SBATCH --job-name=rename_spec_dedup
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/rename_spec_dedup_%j.log
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/rename_spec_dedup_%j.log

set -euo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

echo "[$(date -Is)] rename_spec_dedup start on host=$(hostname)"

# --- rename map (in REVERSE numeric order for filesystem safety) ---
declare -a PAIRS=(
  "spec19_dataset_expansion_991                    spec19_dataset_expansion_991"
  "spec18_dipolar_selection_for_lc_extension       spec18_dipolar_selection_for_lc_extension"
  "spec17_elst_oi_grouped_prediction               spec17_elst_oi_grouped_prediction"
  "spec16_dipolar_barrier_sampling_check           spec16_dipolar_barrier_sampling_check"
  "spec15_dipolar192_xgb_channels                  spec15_dipolar192_xgb_channels"
  "spec14_cross_engine_offset_check                spec14_cross_engine_offset_check"
  "spec13_dipolar_eda_decomp_fix                   spec13_dipolar_eda_decomp_fix"
  "spec12_barrier_sign_audit                       spec12_barrier_sign_audit"
)

echo
echo "=== step 1: filesystem rename (reverse order) ==="
for pair in "${PAIRS[@]}"; do
  old=$(echo "$pair" | awk '{print $1}')
  new=$(echo "$pair" | awk '{print $2}')
  if [ -d "spec/$old" ] && [ ! -e "spec/$new" ]; then
    mv "spec/$old" "spec/$new"
    echo "  mv spec/$old  spec/$new"
  elif [ -d "spec/$new" ] && [ ! -e "spec/$old" ]; then
    echo "  [skip] spec/$new already exists (already renamed)"
  else
    echo "  [WARN] unexpected state: old=$([ -d "spec/$old" ] && echo YES || echo no)  new=$([ -e "spec/$new" ] && echo YES || echo no)"
  fi
done

echo
echo "=== step 2: sed text substitution across .py .sh .md .json .csv .txt ==="
# use grep first to enumerate files (safer than blind sed)
for pair in "${PAIRS[@]}"; do
  old=$(echo "$pair" | awk '{print $1}')
  new=$(echo "$pair" | awk '{print $2}')
  # find files matching (exclude .git/, exclude the folders' own contents to avoid re-writing already-corrected paths)
  files=$(grep -rl --include="*.py" --include="*.sh" --include="*.md" \
                 --include="*.json" --include="*.csv" --include="*.txt" \
                 "$old" . 2>/dev/null | grep -v "^\./\.git/" || true)
  if [ -z "$files" ]; then
    echo "  [$old] no references"
    continue
  fi
  n=$(echo "$files" | wc -l)
  echo "  [$old] replacing in $n files"
  # do the substitution — exact string, no regex specials
  # use sed's -i with a literal string; wrap old/new with escaping for sed
  # since old/new are plain identifiers (only letters/digits/underscore), no
  # escaping needed. Use pipe as delimiter.
  echo "$files" | xargs -r sed -i "s|${old}|${new}|g"
done

echo
echo "=== step 3: verify (no old strings should remain) ==="
remaining=0
for pair in "${PAIRS[@]}"; do
  old=$(echo "$pair" | awk '{print $1}')
  hits=$(grep -rl --include="*.py" --include="*.sh" --include="*.md" \
              --include="*.json" --include="*.csv" --include="*.txt" \
              "$old" . 2>/dev/null | grep -v "^\./\.git/" | wc -l)
  if [ "$hits" -gt 0 ]; then
    echo "  [FAIL] $hits residual references to $old"
    grep -rn --include="*.py" --include="*.sh" --include="*.md" \
             --include="*.json" --include="*.csv" --include="*.txt" \
             "$old" . 2>/dev/null | grep -v "^\./\.git/" | head -5
    remaining=$((remaining + hits))
  fi
done
if [ "$remaining" -eq 0 ]; then
  echo "  [OK] no residual old references"
fi

echo
echo "=== step 4: final spec/ listing ==="
ls -d spec/spec*/ | sort -V

echo
echo "[$(date -Is)] rename_spec_dedup end"
