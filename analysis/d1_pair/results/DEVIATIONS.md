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
