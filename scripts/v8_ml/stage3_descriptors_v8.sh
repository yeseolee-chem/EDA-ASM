#!/bin/bash
#SBATCH --job-name=stage3_v8
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/stage3_v8_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/stage3_v8_%A_%a.out

# Stage 3 (v8) -- GFN2-xTB descriptors d1..d24 for the 799-rxn v8 cohort.
#
# 8-way array on cpu2 (xTB is CPU-only via tblite). Each shard picks
# rxns where row_index % 8 == SLURM_ARRAY_TASK_ID.
#
# 48 h walltime (project rule); idempotent -- rows already committed to the
# shard parquet are skipped on resubmit. After all shards finish, shard 0 is
# responsible for merging the per-shard chunks into the final parquet.

set -euo pipefail

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
NSHARDS=8

mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v8/chunks

# Conda env
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# Keep BLAS single-threaded so 8 concurrent shards on cpu2 don't oversubscribe.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$REPO"

echo "[stage3_v8] host=$(hostname) job=${SLURM_JOB_ID} shard=${SLURM_ARRAY_TASK_ID}/${NSHARDS} start=$(date -Iseconds)"

python -u scripts/v8_ml/stage3_descriptors_v8.py \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --nshards "${NSHARDS}"

echo "[stage3_v8] shard ${SLURM_ARRAY_TASK_ID} done=$(date -Iseconds)"

# After every shard finishes, poll for the other shards' output parquets.
# Shard 0 additionally performs the merge into descriptors_v8.parquet.
CHUNK_DIR=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/descriptors_v8/chunks

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
    echo "[stage3_v8] shard 0 waiting for peer shards to finish..."
    # Wait up to ~2 h for all NSHARDS chunk files to appear. This is safe with
    # array=0-7%8: all shards start together and their runtime is bounded.
    for _ in $(seq 1 720); do
        n_have=$(ls "${CHUNK_DIR}"/shard*.parquet 2>/dev/null | wc -l)
        if [[ "${n_have}" -ge "${NSHARDS}" ]]; then
            break
        fi
        sleep 10
    done
    echo "[stage3_v8] shard 0 merging chunks -> descriptors_v8.parquet"
    python -u scripts/v8_ml/stage3_descriptors_v8.py --merge-only
    echo "[stage3_v8] merge done=$(date -Iseconds)"
fi
