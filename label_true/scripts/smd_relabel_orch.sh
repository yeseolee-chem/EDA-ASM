#!/bin/bash
# Self-chaining orchestrator for the SMD-keyword eda.inp re-run.
#
# Runs on cpu2 with tiny resources. Each iteration:
#   1. counts done rxns; exits if all done
#   2. submits one worker array (15 elements × dynamic claim over remaining rxns)
#   3. submits the next orchestrator via afterany:$ARR — self-chain
#   4. exits
#
# Peak queue slot use:
#   1 (this orch running) + 15 (array) + 1 (next orch pending) = 17 ≤ 20-cap.
#
# The array element itself has a 30-min wall guard — no rxn is started with
# <30 min wall left. Idempotent skip means we can re-enter forever without
# double-work.
#
#SBATCH --job-name=smd_orch
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/logs/orch_%j.out
set -euo pipefail

ROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel
STATE=$ROOT/state
mkdir -p $STATE/done
SCRIPTS=/home1/yeseo1ee/projects/eda-asm-prediction/label_true/scripts
TOTAL=5265
DONE=$(ls $STATE/done 2>/dev/null | wc -l)
PENDING=$((TOTAL - DONE))
echo "orch $(date -Is) — done=$DONE pending=$PENDING"

if [ $PENDING -le 0 ]; then
    echo "ALL DONE — stopping the chain"
    exit 0
fi

# Submit next worker array (limit 15 elements % 10 concurrent → 17 slots peak with next orch)
ARR=$(sbatch --parsable $SCRIPTS/smd_relabel_worker.sh)
echo "  submitted worker array $ARR"

# Self-chain via afterany — next orch runs once the array drains, regardless of
# per-element success/failure. Idempotent skip prevents duplication.
NEXT=$(sbatch --parsable --dependency=afterany:$ARR $SCRIPTS/smd_relabel_orch.sh)
echo "  next orch queued as $NEXT (dep: afterany:$ARR)"

echo "orch exiting cleanly $(date -Is)"
