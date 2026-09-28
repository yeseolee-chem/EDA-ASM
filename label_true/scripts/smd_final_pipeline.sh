#!/bin/bash
# Final SMD relabel pipeline, meant to run with --dependency=afterany:<workers>:
#   1. full audit (smd_relabel_audit.py)          -> audit/audit_full.csv
#   2. SMD uniformity scan                         -> audit/uniformity.tsv
#   3. gate (smd_relabel_gate.py pre)              -> stop here unless PASS
#   4. stage3 on the SMD outputs                   -> label_true/work_smd/*.json
#   5. postprocess (existing policy)               -> labels_all.json.tmp_smd
#   6. crosscheck vs audit (gate post)             -> only then labels_all.json
# The repo-root labels_all.json is replaced only after steps 3 and 6 pass; on
# any failure it is left as it was. Safe to re-run.
#SBATCH --job-name=smd_final
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/logs/final.%j.out
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
REPO=/home1/yeseo1ee/projects/eda-asm-prediction
LT=$REPO/label_true
cd $LT || exit 1
fail() { echo "### STOPPED at: $1 — labels_all.json NOT written"; exit 1; }

echo "### [1/6] audit  $(date -Is)"
python scripts/smd_relabel_audit.py > /dev/null || fail "audit"
sed -n '1,/^residuals/p' /gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/audit/audit_summary.txt

echo "### [2/6] SMD uniformity scan  $(date -Is)"
bash scripts/smd_relabel_uniformity.sh > /dev/null || fail "uniformity scan"

echo "### [3/6] gate  $(date -Is)"
python scripts/smd_relabel_gate.py pre || fail "gate (see audit/GATE_REPORT.txt)"

echo "### [4/6] stage3 on SMD outputs  $(date -Is)"
python scripts/smd_stage3_build.py || fail "stage3"

echo "### [5/6] postprocess  $(date -Is)"
python scripts/postprocess_labels_all.py $LT/work_smd/labels_all.json $REPO/labels_all.json.tmp_smd || fail "postprocess"

echo "### [6/6] crosscheck  $(date -Is)"
python scripts/smd_relabel_gate.py post $REPO/labels_all.json.tmp_smd || fail "crosscheck (candidate kept as labels_all.json.tmp_smd)"
mv -f $REPO/labels_all.json.tmp_smd $REPO/labels_all.json
echo "### DONE  wrote $REPO/labels_all.json  $(date -Is)"
