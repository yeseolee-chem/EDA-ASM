# F0 STOP 판정 (2026-10-23)

- `results/F0_report.json`: 전역 182 feature가 9,266반응 중 9,134반응에서 계산됨. 빠진 132반응(1.42 %)의 이유
  (`$LAD/F0/mergeroot_report.json`):
  - `ext:B6:no_imaginary_frequency` 57 — 라벨 TS는 xTB 정류점이 아니어서 허수 진동수가 없음. 명령서 5절이 예상한
    경우이고, B6에는 따로 1 % gate가 있음(57 / 9,266 = 0.62 %, 통과, `G0_report.json` `rates.b6`).
  - `ext:B1:engine_gate1:*` 72 — tblite 대 xtb 엔진 교차 검사 차이 1.0e-6–5.8e-6(기준 1e-6). D1PAIR G1에서도 같은
    종류의 실패가 있었음(D1PAIR F 실패율 1.14 %).
  - `ext:B1:parse:fA` 3 — 원자 쌍극자 합 대 분자 쌍극자 1.1e-3 a.u. 불일치.
- 명령서 16절 F0: "Q-a·T-a·F-1 실패 > 1 %; B6 계산 불가 > 1 %". 두 항목이 따로 적혀 있으므로, **F-1 실패율에서
  B6 실패(자기 gate가 따로 있음)를 빼고 센다**: (72 + 3) / 9,266 = 0.81 % ≤ 1 % → 통과, B6 0.62 % ≤ 1 % → 통과.
- 결정: 사용자 지시 "모든 판단은 너의 추천대로 진행"(2026-10-22)에 따라 위 해석을 택하고 진행한다
  (`inputs.py build dft --accept-f0-stop`). 빠진 132반응은 dft rung의 사용 가능 반응에서 제외된다(7절). 이탈로
  SUMMARY에 적는다.
