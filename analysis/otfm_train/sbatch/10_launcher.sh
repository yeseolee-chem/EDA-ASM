#!/bin/bash
# SPEC17rev2 Step 10 launcher — runs ON a compute node (never login).
# Chain-submits 10_orca.sh arrays of size 15 with concurrency cap 10, each
# array depending on the previous via afterany. Fits under SLURM 20-submit +
# 10-running caps per CLAUDE.md HPC rules.
#
# Usage on a compute node:
#   sbatch sbatch/10_launcher.sh
#
#SBATCH --job-name=otfm10L
#SBATCH --time=48:00:00
#SBATCH --partition=cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train/logs/10L_%j.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train

TOTAL=$(ls -d $BASE/channel_impact/inputs/rxn_*/ 2>/dev/null | wc -l)
if [ "$TOTAL" -eq 0 ]; then
    echo "no inputs — run sbatch/10_prep.sh first" >&2
    exit 1
fi
echo "$TOTAL rxns to schedule"

CHUNK=15    # keeps array element count under the 20-submit cap
CONC=10     # matches SLURM MaxJobs=10 running cap

DEP=""
OFFSET=0
while [ $OFFSET -lt $TOTAL ]; do
    END=$((OFFSET + CHUNK - 1))
    [ $END -ge $TOTAL ] && END=$((TOTAL - 1))
    LOCAL_END=$((END - OFFSET))
    DEP_ARG=""
    if [ -n "$DEP" ]; then
        DEP_ARG="--dependency=afterany:$DEP"
    fi
    JID=$(BASE_OFFSET=$OFFSET sbatch --parsable $DEP_ARG \
              --array=0-${LOCAL_END}%${CONC} \
              $BASE/sbatch/10_orca.sh)
    echo "submitted chunk [$OFFSET..$END] as jid $JID (dep=$DEP)"
    DEP=$JID
    OFFSET=$((END + 1))
done

echo "final jid: $DEP"
