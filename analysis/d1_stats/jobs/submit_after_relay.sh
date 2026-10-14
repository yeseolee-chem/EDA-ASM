#!/bin/bash
# RELAY_DONE 이 생긴 뒤 로그인 노드에서 한 번 실행한다: sbatch 세 번(확정 export → Phase A → Phase B, afterok 사슬)만 하고 끝난다.
#   상주 · 대기 · 폴링 없음 (CLAUDE.md). 범위(2026-10-14 사용자): 통계 판정만 — Phase C · D 는 보류, 여기서 내지 않는다.
#   bash analysis/d1_stats/jobs/submit_after_relay.sh
#   막히는 경우: RELAY_DONE 없음, 게이트 R 통과 기록(results_interim/gate_r.json) 없음, 대기열에 3칸이 없음 (MaxSubmit 20, 배열 원소 단위),
#                같은 사슬 잡(d1fin · d1sA · d1sB)이 이미 대기열에 있음 (두 번 실행 방지).
#   앞 잡이 실패하면 (export 실패 · 게이트 A STOP) 뒤 잡은 --kill-on-invalid-dep=yes 로 SLURM 이 지운다 → 원인을 본 뒤 이 스크립트를 다시 실행.
#   두 번째 · 세 번째 sbatch 가 실패하면 이미 낸 잡을 지우고 끝난다 (반쯤 낸 사슬을 남기지 않음).
#   parked 행 (D'' 등, 계산하지 않은 행) 은 Phase A 가 결측 분석에서 뺀다 — 건수만 알린다.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../../.."          # 저장소 루트 (= 각 잡의 SLURM_SUBMIT_DIR)
B=/gpfs/tmp_cpu2/yeseo1ee/d1; Q=$B/d1_scratch/queue
[ -f "$Q/RELAY_DONE" ] || { echo "RELAY_DONE 없음 — D1 본계산이 아직 끝나지 않았다"; tail -n 1 "$Q/RELAY_STATUS" 2>/dev/null; exit 1; }
grep -q '"passed": true' analysis/d1_stats/results_interim/gate_r.json 2>/dev/null \
    || { echo "게이트 R 통과 기록 없음 (analysis/d1_stats/results_interim/gate_r.json) — Phase R 결과를 먼저 본다"; exit 1; }
dup=$(squeue -u "$USER" -h -n d1fin,d1sA,d1sB -o "%i %j %T" || true)
[ -z "$dup" ] || { echo "같은 사슬 잡이 이미 대기열에 있다 — 두 번 내지 않는다:"; echo "$dup"; exit 1; }
n=$(squeue -u "$USER" -h -r | wc -l)
[ $((n + 3)) -le 20 ] || { echo "대기열 $n 칸 (배열 원소 단위) — 3칸이 더 필요하다 (MaxSubmit 20)"; exit 1; }
np=$(comm -23 <(ls "$Q/parked" 2>/dev/null | sort) <(ls "$Q/done" 2>/dev/null | sort) | wc -l)
echo "parked (끝나지 않은 · 계산하지 않은) 행: $np — Phase A 가 결측 분석에서 빼고 INPUTS.md · not_computed_rows.csv 에 적는다"
j1=""; j2=""
trap 'echo "제출이 중간에 실패 — 이미 낸 잡을 지운다: ${j1} ${j2}"; [ -n "${j1}${j2}" ] && scancel ${j1} ${j2}' ERR
j1=$(sbatch --parsable -p cpu1,cpu2 -o analysis/d1_stats/logs/d1_final_export_%j.out analysis/d1_stats/jobs/d1_final_export.slurm)
echo "제출: 확정 export $j1"
j2=$(sbatch --parsable -p cpu1,cpu2 --dependency=afterok:"$j1" --kill-on-invalid-dep=yes -o analysis/d1_stats/logs/phase_a_%j.out analysis/d1_stats/jobs/phase_a.slurm)
echo "제출: Phase A $j2 (afterok:$j1)"
j3=$(sbatch --parsable -p cpu1,cpu2 --dependency=afterok:"$j2" --kill-on-invalid-dep=yes -o analysis/d1_stats/logs/phase_b_%j.out analysis/d1_stats/jobs/phase_b.slurm)
echo "제출: Phase B $j3 (afterok:$j2)"
trap - ERR
squeue -u "$USER" -h -o "%.10i %.8j %.2t %R" | grep -E "^ *($j1|$j2|$j3) " || true
