#!/bin/bash
# 시험 실행 (2026-10-14 사용자 승인): 확정 데이터 전에 Phase A · B 의 코드 경로를 중간 파일(9,459행)로 한 번 돌려 본다. 결과는 판정에 쓰지 않는다.
#   입력: 중간 라벨 (sha a2df136b…) + 같은 snapshot 의 d1_status.csv + d1_manifest.csv + σ 표.  출력: analysis/d1_stats/results_dryrun/ (git 제외)
#   run_all.py 를 결측 분석(K: --manifest/--status) · Hammett(L: --sigma) 까지 켜서 돌리고, gate_a.py, Phase B 스크립트 전부, figs.py · figs_b.py,
#   stats_report.py (→ results_dryrun/STATS_REPORT_dryrun.md) 까지. 시간을 줄이려고 부트스트랩 수를 줄이고 (ci_lowrank 20, robustness 100,
#   b5_gedt 100) 아카이브 스캔은 표본 (ok 150 + no_ts 150, seed 20261014) 으로만 한다. 단계마다 rc 를 적고 계속 간다.
#   sbatch -p cpu1,cpu2 -o analysis/d1_stats/logs/dryrun_b_%j.out analysis/d1_stats/jobs/dryrun_b.slurm
#SBATCH --job-name=d1sDry
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
set -uo pipefail
source "${SLURM_SUBMIT_DIR:?저장소 루트에서 제출}/analysis/d1_stats/jobs/env.sh" || exit 2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=$ST/results_dryrun; rm -rf "$OUT"; mkdir -p "$OUT"
LAB=$INTERIM
SNAP=$B/d1_results/snapshot_20261014_0721
STATUS=$SNAP/d1_status.csv; MAN=$B/d1_run/d1_manifest.csv
hostname; date -Is
[ "$(sha256sum "$LAB" | cut -d' ' -f1)" = "$INTERIM_SHA" ] || { echo "중간 파일 sha256 불일치"; exit 1; }
RC=""
run() {  # run NAME cmd...
    local n=$1; shift; local t0=$(date +%s)
    "$@" > "$OUT/$n.txt" 2>&1; local r=$?
    echo "=== $n rc=$r ($(( $(date +%s) - t0 )) s)"; tail -n 4 "$OUT/$n.txt"
    RC="$RC $n=$r"
}
# 미계산 행 빼기 (phase_a.slurm 과 같은 규칙) + 아카이브 표본
python - "$STATUS" "$MAN" "$OUT" <<'EOF'
import sys, pandas as pd
st, man, out = sys.argv[1:4]
S = pd.read_csv(st, dtype=str, keep_default_na=False)
op = S[S["class"].isin(("held", "pending", "running"))]
S[~S.job_id.isin(set(op.job_id))].to_csv(f"{out}/d1_status_computed.csv", index=False)
M = pd.read_csv(man, dtype=str, keep_default_na=False); M[~M.job_id.isin(set(op.job_id))].to_csv(f"{out}/d1_manifest_computed.csv", index=False)
smp = pd.concat([S[S["class"] == c].sample(n=min(150, int((S["class"] == c).sum())), random_state=20261014) for c in ("ok", "no_ts")])
smp.to_csv(f"{out}/d1_status_scan_sample.csv", index=False)
print("not computed", len(op), "| scan sample", smp["class"].value_counts().to_dict(), "| bundles", smp.harvested.replace("", "(ubai)").nunique())
EOF
run A_run_all        python "$ST/run_all.py" --labels "$LAB" --d0 "$D0_LABELS" --out "$OUT" --manifest "$OUT/d1_manifest_computed.csv" \
                     --status "$OUT/d1_status_computed.csv" --status-col "$STATUS_COL" --sigma "$SIGMA"
run A_figs           python "$ST/figs.py" "$OUT"
run A_gate_a         python "$ST/gate_a.py" "$OUT" "$LAB" "$INTERIM"
run b1b_q5q7         python "$ST/q5q7.py" "$LAB" "$OUT"
run b1b_ci_lowrank   python "$ST/ci_lowrank.py" 20 "$OUT"
run b4_robustness    python "$ST/robustness.py" "$LAB" "$OUT" 100
run b2_archive_scan  python "$ST/d1_archive_scan.py" "$OUT/d1_status_scan_sample.csv" "$BUNDLE_DIRS" "$D1_JOBS" "$OUT" 8
run b2_missing_cause python "$ST/missing_cause.py" "$OUT/d1_status_scan_sample.csv" "$MAN" "$OUT/archive_scan.csv" "$OUT"
run b5_q2            python "$ST/q2_houk_domingo.py" "$LAB" "$OUT"
run b5_gedt          python "$ST/b5_gedt.py" "$LAB" "$OUT/archive_scan.csv" "$OUT" - 100
run b6_inflection    python "$ST/inflection.py" "$LAB" "$OUT" "$SIGMA" "$OUT/inflection"
run b7_methyl        python "$ST/q5_methyl.py" "$LAB" "$OUT"
run b8_missing       python "$ST/q4_missing.py" "$LAB" "$OUT"
run figs_b           python "$ST/figs_b.py" "$OUT"
run stats_report     python "$ST/stats_report.py" "$OUT" "$OUT/STATS_REPORT_dryrun.md"
echo "시험 실행 끝 $(date -Is) — 단계별 rc:$RC"
ls -la "$OUT/fig" 2>/dev/null
case "$RC" in *=[1-9]*) exit 1 ;; esac
