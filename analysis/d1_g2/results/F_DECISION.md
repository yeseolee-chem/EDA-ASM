# D1-G2 Phase F STOP 판정 (2026-10-23)

- `results/F_report.json`: G2 ok 8,616반응 중 전역 182 feature 8,501, FU2 토큰 8,499반응.
  gate 실패 117 / 8,616 = 1.36 % > 1 %(명령서 14절 F) → `stop: true`.
  - feature 실패 115: `$G2/F/mergeroot_report.json` 기준으로 `ext:B1:engine_gate1:*`(tblite 대 xtb 엔진 교차 검사 차이
    1.0e-6–5.8e-6, 기준 1e-6)이 대부분이고, `ext:B1:parse`(원자 쌍극자 합 대 분자 쌍극자 1.2e-3–1.4e-3 a.u.)와
    `g1feat:spe_fail`(xtb Wiberg 표 parse 실패)이 일부다.
  - Q 제외 2(확장 self-check), T-a 실패 0.
- 같은 feature 코드가 D1PAIR G1에서 1.14 %, D1LADDER F0(DFT 구조)에서 1.42 %(B6 제외 0.81 %)를 냈다. 실패 종류는
  계산 경로 오류가 아니라 엔진 교차 검사 허용 오차와 parse 검사다.
- 결정: 사용자 지시("모든 판단은 너의 추천대로 진행", 2026-10-22·23)에 따라, 실패한 117반응을 g2 rung에서 제외하고
  나머지 8,499반응의 feature·토큰으로 진행한다. 이탈로 기록한다(D1LADDER 2단계 입력의 사용 가능 반응은 7절 규칙대로
  토큰 + 182 + H가 모두 있는 반응).
