#!/bin/bash
# Phase C (SPEC_D1_ALL) 배열 단계 — rev 4 / rev 5 스크립트를 고치지 않고 D1 환경(phase_c_env.sh)에서 돌린다.
#   배열 원소 i = 18 조각 중 i, i+9 (--array=0-8: 9칸, MaxSubmit 20 안에서 다른 잡 자리를 남긴다). 모두 멱등이다.
#   g1   g1_geom.py run <s> 18       (r4_g1_array.sh 그대로: 16 CPU = ORCA OptTS 8개 동시 x nprocs 2)    --cpus-per-task=16 --mem=32G
#   xtb  xtb_slice.py <s> 18 <out> --geom g1 (r4_feat_array.sh 와 같은 명령; 두 조각을 동시에 돌린다)        --cpus-per-task=2 --mem=8G
#   ext  ext_features.py slice <s> 18 <out> --workers W (r5_ext_array.sh 그대로; 블록 B1..B6 전부)          --cpus-per-task=16 --mem=32G
# 저장소 루트에서 제출:
#   sbatch -p cpu1,cpu2 --array=0-8 --cpus-per-task=16 --mem=32G -o analysis/d1_stats/logs/c_g1_%A_%a.out analysis/d1_stats/jobs/phase_c_array.slurm g1
#   sbatch -p cpu1,cpu2 --array=0-8 --cpus-per-task=2  --mem=8G  -o analysis/d1_stats/logs/c_xtb_%A_%a.out analysis/d1_stats/jobs/phase_c_array.slurm xtb
#   sbatch -p cpu1,cpu2 --array=0-8 --cpus-per-task=16 --mem=32G -o analysis/d1_stats/logs/c_ext_%A_%a.out analysis/d1_stats/jobs/phase_c_array.slurm ext
# 48 h 벽에 걸리면 같은 명령을 다시 제출한다 (끝난 반응 · 조각은 건너뛴다).
#SBATCH --job-name=d1sC
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G
set -uo pipefail
STEP=${1:?g1 | xtb | ext}
TASK=${SLURM_ARRAY_TASK_ID:?배열로 제출: --array=0-8}
case "$TASK" in [0-8]) ;; *) echo "배열 번호는 0-8 (원소 i = 조각 i, i+9), 받은 값 $TASK"; exit 2 ;; esac
source "${SLURM_SUBMIT_DIR:?저장소 루트에서 제출}/analysis/d1_stats/jobs/phase_c_env.sh" || exit 2
case "$G1_ROOT" in "$D1_FEAT_ROOT"/*) ;; *) echo "STOP: G1_ROOT=$G1_ROOT 가 D1 뿌리 밖 (D0 를 건드릴 수 있음)"; exit 2 ;; esac
[ -s "$D1_FEAT_ROOT/prep.json" ] || { echo "STOP: $D1_FEAT_ROOT/prep.json 없음 — phase_c_step.slurm prep 을 먼저"; exit 2; }
N=18
rc=0
case "$STEP" in
  g1)
    for s in "$TASK" $((TASK + 9)); do
        python g1_geom.py run "$s" "$N" || rc=1
    done ;;
  xtb)
    export XTB_THREADS=1 OMP_NUM_THREADS=1
    python -c "import pyarrow, networkx, numpy, pandas, dftd3, morfeus; print('py env ok')" || exit 2
    "$XTB_BIN" --version 2>&1 | grep -i "xtb version" | head -1
    pids=()
    for s in "$TASK" $((TASK + 9)); do
        OUT=$ESPLEY_OUT/slices_g1/slice_$(printf '%02d' "$s").parquet
        if [ -e "$OUT" ]; then echo "skip: $OUT exists"; continue; fi
        mkdir -p "$(dirname "$OUT")"
        echo "labels=$ESPLEY_LABELS G1_ROOT=$G1_ROOT -> $OUT"
        python xtb_slice.py "$s" "$N" "$OUT" --geom g1 > "$D1_FEAT_ROOT/logs/xtb_slice_$(printf '%02d' "$s").log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    tail -n 3 "$D1_FEAT_ROOT"/logs/xtb_slice_$(printf '%02d' "$TASK").log "$D1_FEAT_ROOT"/logs/xtb_slice_$(printf '%02d' $((TASK + 9))).log 2>/dev/null ;;
  ext)
    [ -x "$PY" ] || { echo "STOP: $PY 없음 (rev 5 venv)"; exit 2; }
    [ -s "$ESPLEY_OUT/xtb_features_g1.parquet" ] || { echo "STOP: $ESPLEY_OUT/xtb_features_g1.parquet 없음 — phase_c_step.slurm agg 를 먼저"; exit 2; }
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XTB_THREADS=1 OMP_STACKSIZE=1G PYTHONWARNINGS=ignore
    W=${EXT_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
    RETRY_ARG=""; [ "${EXT_RETRY_FAILED:-0}" = 1 ] && RETRY_ARG="--retry-failed"
    "$PY" -c "import tblite.interface, pyarrow, networkx, dftd3, morfeus; print('py env ok')" || exit 2
    # rev 5 의 B-0 smoke (같은 xtb · tblite) 가 통과한 기록만 확인한다. D1 에서 smoke 를 다시 돌리지 않는다 (results_rev5/B0_smoke.json 을 덮어쓰므로).
    "$PY" -c "import json, sys; sys.exit(0 if json.load(open('results_rev5/B0_smoke.json')).get('passed') is True else 3)" \
        || { echo "STOP: results_rev5/B0_smoke.json 이 passed 가 아님"; exit 3; }
    "$XTB_BIN" --version 2>&1 | grep -i "xtb version" | head -1
    for s in "$TASK" $((TASK + 9)); do
        OUT=$R5_SCRATCH/ext/slices/slice_$(printf '%02d' "$s").parquet
        if [ -e "$OUT" ]; then echo "skip: $OUT exists"; continue; fi
        echo "=== slice $s/$N -> $OUT, $W workers $RETRY_ARG ($(date '+%F %T'))"
        "$PY" ext_features.py slice "$s" "$N" "$OUT" --workers "$W" $RETRY_ARG
        r=$?; [ "$r" -ne 0 ] && { echo "### slice $s exit $r"; rc=$r; }
    done ;;
  *) echo "STEP 은 g1 | xtb | ext"; exit 2 ;;
esac
echo "done $(date '+%F %T') step=$STEP task=$TASK rc=$rc"
exit $rc
