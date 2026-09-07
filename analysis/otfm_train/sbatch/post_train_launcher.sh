#!/bin/bash
# Runs on a compute node AFTER Step 6 (cross-fit training) completes.
# Its job: submit Steps 7, 9, 10-prep, 10-launcher, 10-parse, 11 with the
# right dependencies. Deferred submission means these tasks don't consume
# the 20-cap while Step 6's 5 array slots are pending.
#
# Per CLAUDE.md §1 rule 2 (submit the launcher itself as an sbatch job).
#
#SBATCH --job-name=otfmPL
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train/logs/postL_%j.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train

echo "=== post-train launcher started $(date -Is) ==="
squeue -u $USER -h -o "%.10i %.8j %.8T %.10M" | tail -20

JID7=$(sbatch --parsable --array=0-4%5 $BASE/sbatch/07_generate_crossfit.sh)
echo "07 generate_crossfit (array 0-4%5) : $JID7"

JID9=$(sbatch --parsable --dependency=afterok:$JID7 $BASE/sbatch/09_evaluate.sh)
echo "09 evaluate                        : $JID9  (afterok:$JID7)"

JIDp=$(sbatch --parsable --dependency=afterok:$JID7 $BASE/sbatch/10_prep.sh)
echo "10 prep                            : $JIDp  (afterok:$JID7)"

# The ORCA launcher itself spawns chained arrays; must run on a compute
# node (which it does — CLAUDE.md-compliant).
JIDL=$(sbatch --parsable --dependency=afterok:$JIDp $BASE/sbatch/10_launcher.sh)
echo "10 launcher (chains ORCA arrays)   : $JIDL  (afterok:$JIDp)"

# 10_parse depends on all ORCA runs, which the launcher submits in chunks.
# The launcher's own jid completes after all sbatch calls, not after ORCA
# runs — so we can't afterok on JIDL directly for parse. Instead defer
# 10 parse + 11 report until the user runs them after checking the ORCA
# queue is drained. Written here as a note for the user.
echo ""
echo "=== followup after ORCA runs drain (see squeue for otfm10 array) ==="
echo "sbatch $BASE/sbatch/10_parse.sh"
echo "sbatch $BASE/sbatch/11_report.sh"
echo ""
echo "=== post-train launcher done $(date -Is) ==="
