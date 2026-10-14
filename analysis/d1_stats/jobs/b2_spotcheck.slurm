#!/bin/bash
# Phase B 전 점검 (2026-10-14): d1_archive_scan.py v3 의 no_ts 판정을 표본 34건(checks/b2_v3_sample.tsv)에 그대로 돌린다.
#   기존 아카이브만 읽는다 (누리온 묶음 tar + UBAI 작업 폴더). 출력 checks/b2_v3_spotcheck{.csv,_paths.csv,.json}. rc 1 = 파일럿 T02894 불일치 또는 읽기 실패.
#   sbatch -p cpu1,cpu2 -o analysis/d1_stats/logs/b2_spotcheck_%j.out analysis/d1_stats/jobs/b2_spotcheck.slurm
#SBATCH --job-name=d1sB2c
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
set -uo pipefail
source "${SLURM_SUBMIT_DIR:?저장소 루트에서 제출}/analysis/d1_stats/jobs/env.sh" || exit 2
hostname; date -Is
python "$ST/b2_spotcheck.py" "$ST/checks/b2_v3_sample.tsv" "$NURION_TARS" "$D1_JOBS" "$ST/checks"; rc=$?
date -Is; exit $rc
