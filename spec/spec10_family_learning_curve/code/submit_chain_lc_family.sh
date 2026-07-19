#!/bin/bash
# SPEC_10 — sbatch wrapper that runs chain_lc_family.sh on a compute node.
#
# CLAUDE.md option 2 for automated / throttled job submission: submit the
# launcher itself as an sbatch job on a small cpu partition. The launcher's
# polling + sbatch calls then run on cpu2, never on gate1.hpc.
#
# The launcher is idempotent (skips cells whose output JSON already exists),
# so hitting the 48h wall and resubmitting this same script is safe.

#SBATCH --job-name=s10_lchain
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec10_lchain_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec10_lchain_%j.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

echo "[node] $(hostname)  [start] $(date '+%F %T')"
echo "[squeue self] jobid=${SLURM_JOB_ID:-unknown}"

# No python env needed — chain_lc_family.sh only uses bash + sbatch + squeue.
# MAX_INFLIGHT=10 keeps total in-queue (this launcher + spec10 cells + any
# other jobs of ours) ≤ 10, matching CLAUDE.md's 10-concurrent rule.
exec bash spec/spec10_family_learning_curve/code/chain_lc_family.sh
