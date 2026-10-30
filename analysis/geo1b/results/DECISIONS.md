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
- **0-4 사용자 답** (2026-10-30 11:3x, AskUserQuestion): "G1 폴더 삭제 (권장)"만 골랐다. 하드링크 build와 tar 압축은 고르지 않았다.
  - job 1054098: espley_xtb_g1(6.5 GB)과 d1_pair/g1_final(9.5 MB, d1_pair/g1/<rxn>을 가리키는 symlink 8,659개)을 지웠다. 기록: /gpfs/home1/yeseo1ee/g1_cleanup_20261030.txt.
  - d1_pair/g1(실제 D1 G1 데이터)은 고르지 않았으므로 남겼다.
  - 남는 부족분: 여유 약 41.5 GB 대 예상 약 48 GB. 마지막 G2c build 복사본(약 13 GB) 때 다시 부족해진다. 그 전(4개 root가 끝날 때)에 실제 사용량을 다시 재서 사용자에게 다시 묻는다.
  - D0 r2는 G1 삭제로 생긴 공간에 들어가므로 낸다.
- **0-job-size** (명령서 5-3 ② "노드 하나 전체"): 실행 불가. cpu2 QOS cpu2_limit가 job당 64 CPU로 막는다(128코어 job이 QOSMaxCpuPerNode로 대기). 계정 GrpTRES는 cpu=640, gpu=12, node=10이다. 그래서 CPU job은 64코어, pool 8 × 8코어로 낸다(10 job × 64 = 640 = 계정 상한).
  - D0 복구는 6 × 64(1054050), G2c 2–3 × 64로 냈다.
  - slice 수: D1 root n = 2(3,113코어시간 ÷ 64 ≈ 49 h → slice당 약 24 h), D0 root n = 3(7,201 ÷ 64 ≈ 113 h → slice당 약 38 h).
- **0-4 추가 공간** (사용자 "11gb 폴더 전부 삭제" → 확인 질문 답 "필요한 5개만 남기고 삭제 (권장)"): 예전 Espley rev 폴더를 지웠다(job 1054107). 기록은 /gpfs/home1/yeseo1ee/espley_cleanup_20261030.txt.
  - 남긴 것 5개는 GEO2b와 LADDER 코드가 쓴다: espley_rev6/env_espley(PYENV=espley), espley_rev5/venv(PYENV=venv5, F 단계), espley_xtb/xtb_features_g1.parquet(d1_features 열 순서), espley_rev6/xtb_r6, espley_rev6/S.
  - 지운 것: espley_xtb_g1_rev6(6.6 GB) 등 약 7.6 GB.
  - 확인한 것: 앞서 지운 d1_pair/g1_final(symlink)과 espley_xtb_g1은 GEO2b F 경로(d1_features runroot)에서 쓰지 않는다. runroot는 G1_ROOT를 geo 루트로 바꿔 넘긴다. check_env는 환경 변수 문자열만 비교하고, g1_geom은 import 때 G1_ROOT를 읽지 않는다. espley_xtb_g1은 cmd_f0check에서만 쓴다.
- **A-G2c (사용자 결정 2026-10-30 13:4x, "2번으로 바꿔줘. 1번은 필요없음.")**: GEO1b의 입력 G2를 **G2c**로 바꾼다. G2c는 D1·D0 반응마다 autodE를 3번 돌린 결과(orig, r1, r2) 가운데 e_ts_freq_eh가 가장 낮은 것이고, 명령서 5-1 규칙 그대로다. 단일 run G2로 GEO1b를 학습하는 원래 A 트랙(①)은 하지 않는다.
  - 지금 G2로 학습하려고 걸어 둔 연쇄 job을 취소했다: gb_data 1054062, gb_u_data 1054063, gb_rlaunch 1054064. 셋 다 시작 전이었다.
  - GEO1b의 D, U, R, E는 G2c build(4개 root 완료 후 select_g2c.py build)가 끝난 뒤 시작한다.
  - 슬롯: A 트랙이 G2c를 기다리므로 명령서 6절 표(R 동안 G2c 3개 등)는 이제 해당하지 않는다. G2c가 10자리를 모두 쓴다(D0 r1·r2 slice 0–2, job 1054187–92).
  - 코드 바꿀 곳(G2c build 전에 고치고 검토한다):
    - geo_common: D1_G2_FINAL → $G2C/d1/g2_final, D0G2 → $G2C/d0(data.py가 D0G2/g2_final을 읽음).
    - tests.py U-8b와 geo_common G2RAW_PRIMARY: 명령서 값(1,667짝, 0.1338, 0.0218 Å, 시험 8,425반응)은 단일 run G2 값이다. G2c 그대로 기준값으로 바꾼다. 학습 전에 D 기록에서 계산해 고정하고 기록한다.
    - U-8b 형성 거리 대 C_compare 검사는 orig가 뽑힌 반응에서만 그대로 적용한다.
  - GEO2b: rung geo는 GEO1(G2c 입력)의 예측 구조에 G2c 폴더의 rel1/rel2를 쓴다(한 run에서 나온 TS·기준 구조, 명령서 5-1). 대체 구조도 G2c TS다.
    - 판정 비교 대상은 GEO2.md 10절 그대로 LADDER rung g2(단일 run G2)다. 따라서 판정은 "G2c + 구조 보정" 전체 파이프라인과 지금 G2를 비교한다.
    - G2c 자체만의 효과(rung g2c)는 따로 내지 않는다. 한계로 적는다.
- **0-4 해결** (사용자 2026-10-30 14:2x "ams2026.103, ORCA 설치 압축본, d1_pair/g1, d1_pair/g1_rec, d1/d1_scratch 삭제"): 다섯 개를 지웠다(약 44 GB). 지우기 전에 지금 파이프라인이 읽지 않는 것을 확인했다(기록 /gpfs/home1/yeseo1ee/cleanup_20261030_b.txt).
  - 할당량 254 → 210/300 GB, 여유 약 90 GB.
  - 이제 G2c build는 명령서대로 byte copy로 해도 공간이 충분하다. 하드링크 이탈은 하지 않는다.
  - 삭제는 job 자리가 모두 차 있어, 돌고 있는 job(1054056) 할당 안에서 srun --overlap으로 나눠 실행했다(각 10분 안, 계산 노드).
