#!/bin/bash
# Phase D — 모델 M0–M4 (SPEC_D1_ALL, analysis/d1_stats/PREREG_D1.md). 승인된 PREREG_D1.md 가 commit 되어 있어야 돈다 (d1_models.py 가 확인).
#   rev 5 venv python (sklearn 1.7.2 · numpy 1.26.4 — rev 5 lockbox 학습과 같은 버전). 저장소 루트에서 제출. 순서:
#   1) check (fast_grid vs sklearn)   sbatch -p cpu1,cpu2 --cpus-per-task=16 --mem=32G  -o analysis/d1_stats/logs/d_check_%j.out  analysis/d1_stats/jobs/phase_d.slurm check
#   2) m0                             sbatch -p cpu1,cpu2 --cpus-per-task=32 --mem=64G  -o analysis/d1_stats/logs/d_m0_%j.out     analysis/d1_stats/jobs/phase_d.slurm m0
#   3) split (M1 · M2 · M3) 배열      sbatch -p cpu1,cpu2 --array=0-8 --cpus-per-task=32 --mem=96G -o analysis/d1_stats/logs/d_split_%A_%a.out analysis/d1_stats/jobs/phase_d.slurm split
#   4) report → 결과 확인 → 5) m4 (lockbox 1회) → 6) report 다시
#   m0 은 rev 5 처럼 GridSearchCV 병렬(n_jobs = CPU, BLAS 1 스레드), 나머지는 BLAS 다중 스레드. 모두 멱등 (끝난 단위는 건너뜀).
#SBATCH --job-name=d1sD
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=96G
set -uo pipefail
CMD=${1:?check | m0 | split | m4 | report}
SUB=${SLURM_SUBMIT_DIR:?저장소 루트에서 제출}
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh; conda activate reactot
PY=/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/venv/bin/python
unset ESPLEY_LABELS ESPLEY_PROF ESPLEY_META ESPLEY_SMILES G1_ROOT ESPLEY_OUT R5_SCRATCH ESPLEY_FEAT ESPLEY_ROWS ESPLEY_ARMS ESPLEY_EXT ESPLEY_GEOM
NC=${SLURM_CPUS_PER_TASK:-1}
if [ "$CMD" = m0 ]; then T=1; else T=$NC; fi
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
cd "$SUB" || exit 2
"$PY" -c "import sklearn, numpy, scipy; print('sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
if [ "$CMD" = split ]; then
    TASK=${SLURM_ARRAY_TASK_ID:?split 은 배열로: --array=0-8}
    "$PY" analysis/d1_stats/d1_models.py split "$TASK" 9
else
    "$PY" analysis/d1_stats/d1_models.py "$CMD"
fi
rc=$?; echo "phase_d $CMD rc=$rc $(date -Is)"; exit $rc
