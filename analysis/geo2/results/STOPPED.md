# GEO2 중단 (STOPPED)

- 날짜: 2026-10-30
- 사유: 사용자 결정 2026-10-30: G1 미사용. GEO1(G1 학습) 구조 기반이라 중단, GEO2b로 대체 (docs/specs/GEO1B.md 1-1).
  - 실제 중단 시각은 10:00경이다. 사용자 지시("모든 작업 멈추기. 토큰 소모도 멈추기. 잡 전부 취소")로 GEO1B 명령서보다 먼저 멈췄다.
- 취소한 job: geo2_y_s0 1053846, geo2_y_s1 1053847, geo2_y_s2 1053848 (Phase Y final, 약 1.6시간 실행; final_fm1 pkl 없음).
  - 그 전에 끝난 job: 1053809(명령서 코드 추출), 1053810(selftest), 1053812(LADDER 원본 사본과 unit 목록), 1053845(Y plan).
  - 코드 작성 workflow wl4r5nyym도 중단했다.
- 끝난 단계: 0(명령서·그대로 코드·selftest all_pass), Y plan(c0, 196 epoch). Y final, Z 이후는 시작하지 않았다.
- 이 commit에는 중단 때 작업 폴더에 있던 **검토가 끝나지 않은** 코드도 함께 넣었다(사용 금지, GEO2b에서 다시 검토 후 참고만):
  - analysis/geo2/{feat_geo.py, g3c.py, prereg_geo2.py, g2_launch.py, jobs/, RUNBOOK*.md}, results/DECISIONS_{feat,lad,orch,tests}.md
  - analysis/d1_ladder/{train_atoms.py, train_xgb.py, inputs.py, evaluate.py, tests/run_tests.py}의 rung geo 추가 수정
- 사용자 요청(11:10)으로 $GEO2 scratch와 GEO1 scratch $GEO를 이미 지웠다. D0-G2 삭제는 사용자가 중간에 멈췄다(GEO1B DECISIONS 참고).
