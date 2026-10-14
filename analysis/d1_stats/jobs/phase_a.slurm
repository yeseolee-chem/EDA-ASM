#!/bin/bash
# Phase A — 확정 데이터 동결 + QC (SPEC_D1_ALL). D1 본계산이 끝나고(RELAY_DONE) aggregate_run.py --final 이 결과를 쓴 뒤에만.
#   1. results_final/INPUTS.md: labels_d1.json · d1_status.csv · d1_manifest.csv 의 sha256 · 행 수 · 시각 · 저장소 commit
#   2. run_all.py (확정 파일 + --manifest --status --status-col class [+ --sigma, 표가 있으면]) → qc.json 외 B1 · B2 출력
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
# held (미계산) 행은 실패가 아니다: run_all.py 의 결측 분석(K)에 넣지 않도록 status · manifest 에서 함께 뺀 사본을 준다 (없으면 원본 그대로)
ST_IN=$FINAL_STATUS; MAN_IN=$FINAL_MANIFEST
NHELD=$(python - "$FINAL_STATUS" "$FINAL_MANIFEST" "$OUT" "$STATUS_COL" <<'EOF'
import sys, pandas as pd
st, man, out, col = sys.argv[1:5]
S = pd.read_csv(st, dtype=str, keep_default_na=False); held = set(S.loc[S[col] == "held", "job_id"])
if held:
    S[~S.job_id.isin(held)].to_csv(f"{out}/d1_status_noheld.csv", index=False)
    M = pd.read_csv(man, dtype=str, keep_default_na=False); M[~M.job_id.isin(held)].to_csv(f"{out}/d1_manifest_noheld.csv", index=False)
print(len(held))
EOF
) || exit 1
[ "$NHELD" -gt 0 ] && { ST_IN=$OUT/d1_status_noheld.csv; MAN_IN=$OUT/d1_manifest_noheld.csv; echo "held $NHELD 행을 결측 분석에서 뺐다"; }
{
  echo "# D1 확정 입력 (Phase A 동결)"; echo
  echo "- 동결 시각: $(date -Is) · 저장소 $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
  echo "- D1 실행 트리: $(readlink -f "$REPO/D1_build/d1_run") · 마지막 relay 상태: $(cat "$B/d1_scratch/queue/RELAY_STATUS" 2>/dev/null)"
  echo; echo "| 파일 | sha256 | 행 수 |"; echo "|---|---|---|"
  for f in "$FINAL_LABELS" "$FINAL_STATUS" "$FINAL_MANIFEST"; do
    n=$(case "$f" in *.json) python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$f";; *) echo $(( $(wc -l < "$f") - 1 ));; esac)
    echo "| \`${f#$REPO/}\` | \`$(sha256sum "$f" | cut -d' ' -f1)\` | $n |"
  done
  echo; echo "- held (미계산) 행: $NHELD — 결측 분석(run_all.py K)에는 넣지 않았다"
} > "$OUT/INPUTS.md"
cat "$OUT/INPUTS.md"
SIG=(); [ -s "$SIGMA" ] && SIG=(--sigma "$SIGMA")
python "$ST/run_all.py" --labels "$FINAL_LABELS" --d0 "$D0_LABELS" --out "$OUT" \
    --manifest "$MAN_IN" --status "$ST_IN" --status-col "$STATUS_COL" "${SIG[@]}" > "$OUT/run_all.log" 2>&1 \
    || { echo "run_all.py 실패 ($OUT/run_all.log)"; exit 1; }
python "$ST/figs.py" "$OUT" || echo "figs.py 실패"
python "$ST/gate_a.py" "$OUT" "$FINAL_LABELS" "$INTERIM"; rc=$?
[ $rc -eq 0 ] && echo "게이트 A 통과" || echo "게이트 A 실패 — STOP (gate_a.json 을 보고한다)"
date -Is; exit $rc
