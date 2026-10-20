# D1PAIR — 이탈 기록 (DEVIATIONS)

사양(`docs/specs/D1PAIR_REV9.md`)과 사전 등록(`results/PREREG_D1PAIR.md`)에서 벗어난 것을 적는다. 사전 등록 전의 사용자 결정은
`prereg_d1pair.json` `user_decisions`에 있다(D1_JOBS 추출, P-4 v2 규칙 29짝 제외, Si 원소 항목, F-0 마지막 자리 차이 허용, 44 core,
A_feat 규칙, G1 MPI 시작 오류 재실행).

## D-1 T2 중단 (사용자 결정 2026-10-20, `docs/specs/D1PAIR_FU1.md` F1)

- 이유: 잠정 T1 결과(`results/interim_T1_noSVR/`)에서 다른 계열의 V를 학습에 넣은 T1에서도 숨은 V 계열이 거의 V로 판정되지 않아, V를
  뺀 T2는 정보가 없다고 판단.
- 처리: `$D1PAIR_SCRATCH/M/SKIP_GROUPS` = T2(`d1_train.skipped`), T2를 계산 중이던 task 취소(2026-10-20 08:16, 목록과 끝난 T2 unit 수는
  `results/followup/K_report.json`), SVR helper를 T2 없이 다시 제출. 끝난 T2 unit은 scratch에 보관하고 분석하지 않는다
  (`M_report.json` `skipped_groups`, `E.json` `t2`, `figures/fig7_vblind_NOT_RUN.txt`).
- 영향: 사양 §4의 보조 분할 T2, §5-4의 T2 대 T1 비교, fig7이 없다.

## FU1 X STOP 처리 (2026-10-21, 추천 옵션)

- `docs/specs/D1PAIR_FU1.md` X-3: 넓힌 XGB grid(XGBW)에서 unit의 90 %가 n_estimators 최댓값(4000)을 골라 STOP 조건(> 50 %)에 걸렸다.
- 사양은 사용자에게 묻게 되어 있으나, 2026-10-20 사용자 지시("모든 선택은 추천 옵션으로 진행")에 따라 추천 옵션 **"더 넓히지 않고 X를
  끝낸다"**를 택했다. 이유와 수치는 `results/followup/SUMMARY_FU1.md` X 절.
- 영향: X는 탐색이며 PREREG 판정에 쓰이지 않는다. 더 넓힌 grid는 돌리지 않았다.
