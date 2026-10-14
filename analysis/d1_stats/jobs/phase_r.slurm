#!/bin/bash
# Phase R — 재현 (SPEC_D1_ALL). 중간 파일(9,459행)에 run_all.py · figs.py 를 돌려 headline.json 을 부록 B 기준값과 비교한다.
#   sbatch -p cpu1,cpu2 -o analysis/d1_stats/logs/phase_r_%j.out analysis/d1_stats/jobs/phase_r.slurm
#   게이트 R 실패면 rc 1 (STOP: 차이 표 results_interim/GATE_R.md 를 보고한다).
#SBATCH --job-name=d1sR
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G
set -uo pipefail
source "${SLURM_SUBMIT_DIR:?저장소 루트에서 제출}/analysis/d1_stats/jobs/env.sh" || exit 2
OUT=$ST/results_interim
hostname; date -Is; python -c 'import pandas, numpy, scipy, sklearn, statsmodels, rdkit, matplotlib as m; print("pandas", pandas.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__, "sklearn", sklearn.__version__, "statsmodels", statsmodels.__version__, "rdkit", rdkit.__version__, "matplotlib", m.__version__)'
got=$(sha256sum "$INTERIM" | cut -d' ' -f1)
[ "$got" = "$INTERIM_SHA" ] || { echo "중간 파일 sha256 불일치: $got (기대 $INTERIM_SHA) — STOP"; exit 1; }
( cd "$ST" && sha256sum -c --quiet <(sed "s|analysis/d1_stats/||" SHA256SUMS) ) || { echo "부록 파일 sha256 불일치 — STOP (게이트 S)"; exit 1; }
# 게이트 S (스펙 절차 그대로): 노드 /tmp 에서 추출 스크립트를 돌려 'files written: 11' 과 저장소 파일 동일성 확인
T=/tmp/d1sS.$USER.${SLURM_JOB_ID:-$$}; mkdir -p "$T" && trap 'rm -rf "$T"' EXIT
( cd "$T" && python "$ST/extract_spec_files.py" "$REPO/docs/specs/SPEC_D1_ALL.md" ) | tee "$T/s.log"
grep -qx 'files written: 11' "$T/s.log" && ( cd "$T" && sha256sum -c --quiet "$ST/SHA256SUMS" ) || { echo "게이트 S 실패 — STOP"; exit 1; }
echo "게이트 S 통과 (files written: 11, sha256 일치)"
python "$ST/run_all.py" --labels "$INTERIM" --d0 "$D0_LABELS" --out "$OUT" > "$OUT.run_all.log" 2>&1 || { echo "run_all.py 실패 ($OUT.run_all.log)"; exit 1; }
mv "$OUT.run_all.log" "$OUT/run_all.log"
python "$ST/figs.py" "$OUT" || echo "figs.py 실패 (게이트 R 과 무관)"
python "$ST/gate_r.py" "$OUT/headline.json" "$ST/reference_interim/headline.json" "$OUT/GATE_R.md"; rc=$?
date -Is; exit $rc
