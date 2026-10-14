#!/bin/bash
# RELAY_DONE 이 생긴 뒤 로그인 노드에서 한 번 실행한다: sbatch 세 번(확정 export → Phase A → Phase B, afterok 사슬)만 하고 끝난다.
#   상주 · 대기 · 폴링 없음 (CLAUDE.md). 범위(2026-10-14 사용자): 통계 판정만 — Phase C · D 는 보류, 여기서 내지 않는다.
#   bash analysis/d1_stats/jobs/submit_after_relay.sh
#   막히는 경우: RELAY_DONE 없음, 게이트 R 통과 기록(results_interim/gate_r.json) 없음, 대기열에 3칸이 없음 (MaxSubmit 20).
#   export 가 실패하면 A · B 는 DependencyNeverSatisfied 로 남는다 → scancel 하고 원인을 본 뒤 이 스크립트를 다시 실행한다.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../../.."          # 저장소 루트 (= 각 잡의 SLURM_SUBMIT_DIR)
B=/gpfs/tmp_cpu2/yeseo1ee/d1
[ -f "$B/d1_scratch/queue/RELAY_DONE" ] || { echo "RELAY_DONE 없음 — D1 본계산이 아직 끝나지 않았다"; tail -n 1 "$B/d1_scratch/queue/RELAY_STATUS" 2>/dev/null; exit 1; }
grep -q '"passed": true' analysis/d1_stats/results_interim/gate_r.json 2>/dev/null \
    || { echo "게이트 R 통과 기록 없음 (analysis/d1_stats/results_interim/gate_r.json) — Phase R 결과를 먼저 본다"; exit 1; }
n=$(squeue -u "$USER" -h | wc -l)
[ $((n + 3)) -le 20 ] || { echo "대기열 $n 칸 — 3칸이 더 필요하다 (MaxSubmit 20)"; exit 1; }
j1=$(sbatch --parsable -p cpu1,cpu2 -o analysis/d1_stats/logs/d1_final_export_%j.out analysis/d1_stats/jobs/d1_final_export.slurm)
j2=$(sbatch --parsable -p cpu1,cpu2 --dependency=afterok:"$j1" -o analysis/d1_stats/logs/phase_a_%j.out analysis/d1_stats/jobs/phase_a.slurm)
j3=$(sbatch --parsable -p cpu1,cpu2 --dependency=afterok:"$j2" -o analysis/d1_stats/logs/phase_b_%j.out analysis/d1_stats/jobs/phase_b.slurm)
echo "제출: 확정 export $j1 → Phase A $j2 → Phase B $j3 (afterok)"
squeue -u "$USER" -h -o "%.10i %.8j %.2t %R" | grep -E "^ *($j1|$j2|$j3) " || true
