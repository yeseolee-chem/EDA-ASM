GEO1의 R-user·R-once·R-select·R-lpt·U-0·D-1·I-1 결정을 그대로 이어받음

# GEO1b 결정 기록 (docs/specs/GEO1B.md)

- **0-user** (2026-10-30): 사용자 "이 문서를 읽고 이대로 실행해줘. claude.md 준수. 모든 분기마다 추천 안으로 진행. 최대 병렬 유지해줘."
- **0-GEO2** (명령서 1-1): GEO2 job은 그 전에 이미 사용자 지시로 모두 취소되어 있었다(Y final 1053846–48). 브랜치 geo2에 STOPPED.md를 쓰고, 검토 전 작업 코드와 함께 commit했다(0a912e78). origin/geo2로 push했다.
- **0-branch** (1-2): `git switch -c geo1b origin/geo1`(00eea4f0), `git worktree add …/eda-asm-prediction-g2c -b g2c origin/geo1`.
- **0-mkdir** (1-3): 폴더 만들기, df, du, sinfo는 job으로 돌렸다(1054017, 1054022). 노드: cpu1 48코어 × 10, cpu2 256코어 × 10, gpu1/6 48코어, gpu2/3 56, gpu5 64.
- **0-2 (records.pkl 없음)**: 사용자 요청(2026-10-30 11:10, "기존 geo1,2 결과들 삭제. 저장 공간 확보하기")으로 $GEO(/gpfs/tmp_cpu2/yeseo1ee/geo) 전체와 $GEO2를 이미 지웠다. 그래서 `$GEO/data/records.pkl`이 없다. 명령서상 이 파일은 "GEO1 D 보고와 개수를 비교하는 데만" 쓴다. 비교 기준인 `analysis/geo/results/D_dataset_report.json`(git)이 남아 있으므로 그것과 비교한다(추천안).
- **0-D0 (d0_g2 일부 삭제 → 삭제된 것만 다시 계산)**: 같은 정리 job(1054016)이 d0_g2를 지우던 중(11:10:42–11:12:13) 사용자가 취소했다. 사용자 지시: "D0-G2는 필요할 것으로 보여서 삭제 멈춤. 필요하다면 삭제된것만 다시 계산. 필요없다면 삭제 계속".
  - 필요하다: GEO1b D0 학습 입력(data.py가 $D0G2/g2_final을 읽음)이고, G2c의 D0 'orig' run이다.
  - 남은 것: g2_final 4,057/5,260, jobs 5,260(전부), ts_templates. 지워진 것: d0_manifest.csv, G/, labjobs/, logs/, g2_final 약 1,200개.
  - 복구 방법(analysis/g2c/d0g2_restore.py, g2c worktree):
    1. GEO1 `d0g2.py manifest`를 다시 돌려 d0_manifest.csv, G/labels_d0_g2.json, labjobs/를 만든다. D0 라벨과 profile에서 정해진 대로 만들어지므로 원래와 같다.
    2. scan: 남은 폴더를 검사한다.
       - 완전한 것(.done + ts.xyz, rel1, rel2, result.json, ts.hess, work/ 또는 .fail_*)만 남긴다.
       - 11:00 뒤에 바뀐 폴더(rm이 지우던 중이던 폴더)는 옮겨 둔다(지우지 않음).
       - 남긴 폴더는 GEO1 Z_status.csv와 상태가 같아야 한다.
    3. 지워진 반응만 `d0g2.py pool restore_list.csv`로 다시 계산한다(같은 코드, 같은 절차; 끝난 반응은 건너뜀).
  - 다시 계산한 반응은 autodE의 새 난수로 나온 표본이다. 그래서 D0 G2 성공 수가 GEO1의 4,488과 조금 다를 수 있다. 3-1 표의 `by_src.D0.g2` 기대값은 복구 결과로 바꿔 읽는다(D 단계에서 기록).
- **0-4 (디스크)**: 할당량 여유 35 GB(265/300). 명령서 필요량 2 × (d1_g2 7.8 + d0_g2 약 8.7) + 40 = 약 73 GB.
  - 실제 G2c run root 크기: D1 약 5.3 GB(g2_final 4.7 + jobs 0.27 + logs 0.37), D0 약 8.6 GB.
  - G2c 합계: r1·r2 4개 root 약 28 GB + build 복사본 약 13 GB = 약 41 GB. 여기에 GEO1b·GEO2b 수 GB.
  - → 1-4 STOP 조건이다. 삭제가 필요한 선택지(G1 폴더 등)는 사용자에게 묻는다.
  - 그동안 공간에 맞는 단계(D0 복구, GEO1b, G2c D1 r1·r2, D0 r1)는 진행한다. D0 r2와 G2c build는 공간이 정리된 뒤에 낸다.
- **2-1**: 복사(cp -r), 정리(rm), mkdir, 코드 블록 추출은 job 1054023(/gpfs/home1/yeseo1ee/geo1b_extract_spec.py)으로 했다. `git apply --check docs/specs/geo1b.patch` 통과 후 적용했다. G1 관련 grep 0줄. 모든 .py compile 통과(job 1054029).
- **R-select-primary** (3-3): select의 주 지표는 이제 **모든 para P_main 짝** 비율이다(metrics.py primary_share, GEO1의 같은 형태 짝 840개가 아님). 11 epoch 창 규칙(R-select)은 그대로다.
