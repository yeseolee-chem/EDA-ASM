#!/bin/bash
# Phase B — 통계 B1b–B8 (SPEC_D1_ALL). 게이트 A 통과(results_final/gate_a.json) 뒤에만. B1 · B2 표 · (B3) 는 Phase A 의 run_all.py 가 이미 만들었다.
#   단계마다 화면 출력을 results_final/<단계>.txt 에 남기고 .done_<단계> 표식을 쓴다 (다시 제출하면 끝난 단계는 건너뜀; FORCE=1 이면 다시).
#   B1b  q5q7.py · ci_lowrank.py 200          B4  robustness.py (코어 부트스트랩 1000, BH-FDR) → gate_b.json
#   B2   d1_archive_scan.py → missing_cause.py  (아카이브는 읽기만; 누리온 묶음 tar + UBAI 작업 폴더)
#   B5   q2_houk_domingo.py · b5_gedt.py (Phase C 의 xTB GEDT 가 있으면 함께) · inflection.py (σ 표가 있을 때만)
#   B6   inflection.py (σ 표 필요; 없으면 보류라고 적는다)      B7  q5_methyl.py      B8  q4_missing.py
#   끝으로 stats_report.py → analysis/d1_stats/STATS_REPORT.md
#   sbatch -p cpu1,cpu2 -o analysis/d1_stats/logs/phase_b_%j.out analysis/d1_stats/jobs/phase_b.slurm
#SBATCH --job-name=d1sB
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
set -uo pipefail
source "${SLURM_SUBMIT_DIR:?저장소 루트에서 제출}/analysis/d1_stats/jobs/env.sh" || exit 2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RD=$OUT_FINAL; LAB=$FINAL_LABELS
python -c "import json,sys; g=json.load(open(sys.argv[1])); sys.exit(0 if g.get('passed') else 1)" "$RD/gate_a.json" \
    || { echo "게이트 A 가 통과하지 않았다 ($RD/gate_a.json) — Phase B 를 하지 않는다"; exit 1; }
FEXT=/gpfs/tmp_cpu2/yeseo1ee/d1_features/xtb/xtb_features_ext_d1_g1.parquet; [ -s "$FEXT" ] || FEXT=-
step() {   # step NAME cmd...
    local n=$1; shift
    if [ -f "$RD/.done_$n" ] && [ "${FORCE:-0}" != 1 ]; then echo "skip $n (끝남)"; return 0; fi
    echo "=== $n  $(date '+%F %T')"
    "$@" > "$RD/$n.txt" 2>&1; local r=$?
    tail -n 5 "$RD/$n.txt"
    if [ $r -eq 0 ]; then date -Is > "$RD/.done_$n"; else echo "### $n 실패 rc=$r ($RD/$n.txt)"; FAILED="$FAILED $n"; fi
}
FAILED=""
step b1b_q5q7      python "$ST/q5q7.py" "$LAB" "$RD"
step b1b_ci_lowrank python "$ST/ci_lowrank.py" 200 "$RD"
step b4_robustness python "$ST/robustness.py" "$LAB" "$RD" 1000
step b2_archive_scan python "$ST/d1_archive_scan.py" "$FINAL_STATUS" "$NURION_TARS" "$D1_JOBS" "$RD" "${SLURM_CPUS_PER_TASK:-8}"
step b2_missing_cause python "$ST/missing_cause.py" "$FINAL_STATUS" "$FINAL_MANIFEST" "$RD/archive_scan.csv" "$RD"
step b5_q2         python "$ST/q2_houk_domingo.py" "$LAB" "$RD"
# b5_gedt 는 Phase C 가 끝난 뒤 다시 돌리면 gedt_xtb 가 채워진다 (FORCE=1 또는 .done_b5_gedt 를 지우고)
step b5_gedt       python "$ST/b5_gedt.py" "$LAB" "$RD/archive_scan.csv" "$RD" "$FEXT" 1000
if [ -s "$SIGMA" ]; then
    step b6_inflection python "$ST/inflection.py" "$LAB" "$RD" "$SIGMA" "$RD/inflection"
else
    echo "σ 표 없음 ($SIGMA) — B3 · B6 (와 B5 의 σ 비대칭) 보류" | tee "$RD/b6_inflection.txt"
fi
step b7_methyl     python "$ST/q5_methyl.py" "$LAB" "$RD"
step b8_missing    python "$ST/q4_missing.py" "$LAB" "$RD"
python "$ST/stats_report.py" "$RD" "$ST/STATS_REPORT.md" || FAILED="$FAILED stats_report"
echo "Phase B 끝 $(date -Is); 실패: ${FAILED:-없음}"
[ -z "$FAILED" ]
