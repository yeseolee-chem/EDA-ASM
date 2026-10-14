#!/bin/bash
# Phase A — 확정 데이터 동결 + QC (SPEC_D1_ALL). D1 본계산이 끝나고(RELAY_DONE) aggregate_run.py --final 이 결과를 쓴 뒤에만.
#   1. results_final/INPUTS.md: labels_d1.json · d1_status.csv · d1_manifest.csv · σ 표의 sha256 · 행 수 · 미계산 행 수 · 시각 · 저장소 commit
#   2. run_all.py (확정 파일 + --manifest --status --status-col class [+ --sigma, 표가 있으면]) → qc.json 외 B1 · B2 출력
#      미계산 행 (class held · pending · running — 예: parked 된 D'' 행) 은 실패가 아니다: status · manifest 에서 함께 뺀 사본을 준다
#      run_all.py 가 결측 분석(K: --manifest/--status) 에서 죽으면 K 없이 한 번 더 돌리고 그 사실을 INPUTS.md 에 적는다 (부록 코드는 고치지 않음)
#   3. gate_a.py → gate_a.json · inspect_large_ddbarrier_alt.csv (사람이 판단만 적는다)
#   sbatch -p cpu1,cpu2 -o analysis/d1_stats/logs/phase_a_%j.out analysis/d1_stats/jobs/phase_a.slurm
#SBATCH --job-name=d1sA
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G
set -uo pipefail
source "${SLURM_SUBMIT_DIR:?저장소 루트에서 제출}/analysis/d1_stats/jobs/env.sh" || exit 2
OUT=$ST/results_final; mkdir -p "$OUT"
python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('passed') is True else 1)" "$ST/results_interim/gate_r.json" \
    || { echo "게이트 R 통과 기록이 없다 (results_interim/gate_r.json) — Phase R 를 먼저 (Phase A 를 하지 않는다)"; exit 1; }
[ -f "$B/d1_scratch/queue/RELAY_DONE" ] || { echo "D1 본계산이 아직 끝나지 않았다 (RELAY_DONE 없음) — Phase A 를 하지 않는다"; exit 1; }
for f in "$FINAL_LABELS" "$FINAL_STATUS" "$FINAL_MANIFEST"; do [ -s "$f" ] || { echo "없음: $f (aggregate_run.py --final 을 먼저)"; exit 1; }; done
# 미계산 행 (held · pending · running): 결측 분석에 넣지 않는다 — 뺀 사본 d1_status_computed.csv · d1_manifest_computed.csv, 분류 · 사유별 건수 표
ST_IN=$FINAL_STATUS; MAN_IN=$FINAL_MANIFEST
NOPEN=$(python - "$FINAL_STATUS" "$FINAL_MANIFEST" "$OUT" "$STATUS_COL" <<'EOF'
import sys, pandas as pd
st, man, out, col = sys.argv[1:5]
OPEN = ("held", "pending", "running")                  # aggregate_run.py: OPEN = (running, pending) "통계에서 뺀다"; held = 미계산
S = pd.read_csv(st, dtype=str, keep_default_na=False)
op = S[S[col].isin(OPEN)]
reason = op["reason"].str.replace(r" —.*$", "", regex=True) if "reason" in op else ""
op.assign(reason_short=reason).groupby([col, "reason_short", "dipole"]).size().rename("n").reset_index() \
  .to_csv(f"{out}/not_computed_rows.csv", index=False)
if len(op):
    S[~S.job_id.isin(set(op.job_id))].to_csv(f"{out}/d1_status_computed.csv", index=False)
    M = pd.read_csv(man, dtype=str, keep_default_na=False)
    M[~M.job_id.isin(set(op.job_id))].to_csv(f"{out}/d1_manifest_computed.csv", index=False)
print(len(op))
EOF
) || exit 1
[ "$NOPEN" -gt 0 ] && { ST_IN=$OUT/d1_status_computed.csv; MAN_IN=$OUT/d1_manifest_computed.csv; echo "미계산 $NOPEN 행을 결측 분석에서 뺐다 (not_computed_rows.csv)"; }
SIG=(); [ -s "$SIGMA" ] && SIG=(--sigma "$SIGMA")
K_NOTE="결측 분석(K) 포함"
python "$ST/run_all.py" --labels "$FINAL_LABELS" --d0 "$D0_LABELS" --out "$OUT" \
    --manifest "$MAN_IN" --status "$ST_IN" --status-col "$STATUS_COL" "${SIG[@]}" > "$OUT/run_all.log" 2>&1
if [ $? -ne 0 ]; then
    mv "$OUT/run_all.log" "$OUT/run_all_with_K_failed.log"
    echo "run_all.py 가 --manifest/--status 와 함께 실패 ($OUT/run_all_with_K_failed.log) — 결측 분석(K) 없이 다시 (부록 코드 문제로 보고할 것)"
    python "$ST/run_all.py" --labels "$FINAL_LABELS" --d0 "$D0_LABELS" --out "$OUT" "${SIG[@]}" > "$OUT/run_all.log" 2>&1 \
        || { echo "run_all.py 실패 ($OUT/run_all.log)"; exit 1; }
    K_NOTE="결측 분석(K) 실패 — run_all_with_K_failed.log (K 없이 다시 돌림; 보고 대상)"
fi
{
  echo "# D1 확정 입력 (Phase A 동결)"; echo
  echo "- 동결 시각: $(date -Is) · 저장소 $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
  echo "- D1 실행 트리: $(readlink -f "$REPO/D1_build/d1_run") · 마지막 relay 상태: $(cat "$B/d1_scratch/queue/RELAY_STATUS" 2>/dev/null)"
  echo; echo "| 파일 | sha256 | 행 수 |"; echo "|---|---|---|"
  for f in "$FINAL_LABELS" "$FINAL_STATUS" "$FINAL_MANIFEST"; do
    n=$(case "$f" in *.json) python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$f";; *) echo $(( $(wc -l < "$f") - 1 ));; esac)
    echo "| \`${f#$REPO/}\` | \`$(sha256sum "$f" | cut -d' ' -f1)\` | $n |"
  done
  echo; echo "- 미계산 행 (held · pending · running): $NOPEN — 결측 분석(run_all.py K · missing_cause.py)에는 넣지 않았다. 분류 · 사유 · dipole 별: \`not_computed_rows.csv\`"
  echo "- run_all.py: $K_NOTE"
  if [ -s "$SIGMA" ]; then echo "- σ 표: \`${SIGMA#$REPO/}\` sha256 \`$(sha256sum "$SIGMA" | cut -d' ' -f1)\` (사용자 제공 f042f133…, B3 · B5 σ 비대칭 · B6 에 쓰임; Ts 는 원표에 없어 빈칸)"; else echo "- σ 표 없음 — B3 · B6 보류"; fi
} > "$OUT/INPUTS.md"
cat "$OUT/INPUTS.md"
python "$ST/figs.py" "$OUT" || echo "figs.py 실패"
python "$ST/gate_a.py" "$OUT" "$FINAL_LABELS" "$INTERIM"; rc=$?
[ $rc -eq 0 ] && echo "게이트 A 통과" || echo "게이트 A 실패 — STOP (gate_a.json 을 보고한다)"
date -Is; exit $rc
