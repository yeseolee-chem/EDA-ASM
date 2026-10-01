#!/bin/bash
#SBATCH --job-name=r6_run
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/logs/%x_%A_%a.out
# r6_run.sh — rev 6 steps that run in the Espley-pinned env (env_espley; their requirements.txt pins).
#   MODE=split                       Phase S  (r6_split.py build)
#   MODE=units GROUP=B|A|Call|AppA   our-pipeline units, array element i = slice i of R6_NSLICES (r6_train.py)
#   MODE=arm ARM=ESP-C|OUR-C|ESP-C_all   Espley-protocol arm: their patched code (r6_espc.py)
#   MODE=p2_ml|p2_hyp|p2_wrap_hyp|p2_wrap_ml|p2_nn|p2_compare   P-2 (their data; TF threading at their default,
#                                    as P-1);  MODE=p2_nn_tf2  the NN re-implementation under the production threading
#   MODE=units GROUP=Nocref          OUR-O on C minus constrained_ref rows (retrained sensitivity)
#   MODE=analyze_B|analyze_A          r6_analyze.py;  MODE=figures  r6_figures.py
# Threads: BLAS 1; TensorFlow intra-op R6_TF_THREADS (default 2) / inter-op 1 for every rev 6 training step
# (all four arms alike); R6_WORKERS parallel units per allocation (default cpus / R6_TF_THREADS).
# Idempotent: units / their checkpoints / existing outputs are skipped. Submit from analysis/espley_xtb_repro/.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export R6_SCRATCH=${R6_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev6}
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
export TMPDIR=${SLURM_TMPDIR:-/tmp}
PY=$R6_SCRATCH/env_espley/bin/python
MODE=${MODE:?MODE required}
case "$MODE" in
  p2_nn_tf2) export TF_NUM_INTRAOP_THREADS=${R6_TF_THREADS:-2} TF_NUM_INTEROP_THREADS=1 R6_P2_SUFFIX=_tf2 ;;
  p2_*) ;;                                          # P-2: same threading as the P-1 runs of their code
  *) export TF_NUM_INTRAOP_THREADS=${R6_TF_THREADS:-2} TF_NUM_INTEROP_THREADS=1 ;;
esac
export R6_WORKERS=${R6_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-2} / ${R6_TF_THREADS:-2} ))}
echo "[r6_run] host=$(hostname) mode=$MODE git=$(git rev-parse --short HEAD) start=$(date -Is) workers=$R6_WORKERS"
"$PY" -m py_compile rev6_common.py r6_split.py r6_nn.py r6_train.py r6_espc.py r6_analyze.py || exit 2
case "$MODE" in
  split)      "$PY" r6_split.py build ;;
  units)      "$PY" r6_train.py run "${GROUP:?GROUP}" "${SLURM_ARRAY_TASK_ID:-0}" "${R6_NSLICES:-1}" ;;
  arm)        "$PY" r6_espc.py arm "${ARM:?ARM}" ;;
  p2_ml|p2_hyp|p2_wrap_hyp|p2_wrap_ml|p2_nn|p2_compare) "$PY" r6_espc.py "$MODE" ;;
  p2_nn_tf2)  "$PY" r6_espc.py p2_nn ;;
  analyze_B)  "$PY" r6_analyze.py B ;;
  analyze_A)  "$PY" r6_analyze.py A ;;
  figures)    "$PY" r6_figures.py ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
rc=$?
echo "[r6_run] done rc=$rc $(date -Is)"
exit $rc
