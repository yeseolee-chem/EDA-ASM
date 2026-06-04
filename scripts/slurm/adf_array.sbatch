#!/bin/bash
#SBATCH --job-name=adf_800
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=ADF_800/logs/array_%A_%a.out
#SBATCH --error=ADF_800/logs/array_%A_%a.err
# Concurrency cap (%N) limits how many array tasks run at once.
# Adjust via --array=0-799%30 at sbatch invocation time.

set -eo pipefail

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
cd "$REPO"

MANIFEST="${REPO}/ADF_800/manifest.csv"
TASK_ID=${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}

# Pull the (TASK_ID+1)th data row from manifest (skip header)
LINE=$(awk -F, -v n=$((TASK_ID + 2)) 'NR==n {print; exit}' "$MANIFEST")
if [[ -z "$LINE" ]]; then
    echo "ERROR: no manifest row at task ${TASK_ID}" >&2
    exit 2
fi
REACTION_ID=$(echo "$LINE" | awk -F, '{print $1}')
OUT_DIR_REL=$(echo "$LINE" | awk -F, '{print $NF}')
# The last column "out_dir" should already be absolute; trust it
RUN_DIR=${OUT_DIR_REL}
if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: run dir does not exist: $RUN_DIR" >&2
    exit 2
fi

echo "=== task=${TASK_ID} reaction=${REACTION_ID} host=$(hostname) date=$(date -Iseconds) ==="
echo "RUN_DIR=${RUN_DIR}  NTASKS=${SLURM_NTASKS}"

# Skip if a final ASR label already exists (idempotency)
if [[ -f "${RUN_DIR}/asr_label.json" ]]; then
    echo "asr_label.json already present; skipping"
    exit 0
fi

cd "$RUN_DIR"
if ! bash run_eda.sh; then
    rc=$?
    echo "run_eda.sh failed (rc=${rc}) for ${REACTION_ID}" >&2
    echo "{\"reaction_id\":\"${REACTION_ID}\",\"stage\":\"run_eda\",\"rc\":${rc}}" \
        >> "${REPO}/ADF_800/_failures_runtime.jsonl"
    exit ${rc}
fi

cd "$REPO"
if ! python scripts/adf/parse_run.py "$RUN_DIR"; then
    rc=$?
    echo "parse_run.py failed (rc=${rc}) for ${REACTION_ID}" >&2
    echo "{\"reaction_id\":\"${REACTION_ID}\",\"stage\":\"parse\",\"rc\":${rc}}" \
        >> "${REPO}/ADF_800/_failures_runtime.jsonl"
    exit ${rc}
fi

echo "=== task=${TASK_ID} done at $(date -Iseconds) ==="
