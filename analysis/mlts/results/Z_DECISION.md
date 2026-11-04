# Phase Z 결정 (zero_shot.py report)

작성 2026-11-04T20:04:43. 시험 반응 200개(`results/Z_list.csv`: fold 0 학습 쪽, dipole별 40, seed 20261104). 모든 기하 지표는 라벨 TS에서 출발한 ORCA `ExtOpt OptTS`(D1 블록, MLIP 초기 Hessian; xtb는 G2 블록)

## 등록한 규칙 (Z 결과 전에 코드에 고정)

- 대상: D1 원소 전부(Si 포함)를 다루는 MLIP만 순위에 넣는다. MACE-OFF24(Si 없음)는 Si 없는 반응의 참고값, xtb(GFN2-xTB ALPB(water) = G2 수준)는 기준선이다.
- 순서: 같은 TS 비율(높을수록) → 차이 0.025 이내면 |Δd| 중앙값(낮을수록) → 차이 0.002 Å 이내면 GPU E+F 호출 시간(빠를수록). 상위 2개로 fold 0 FT 시범(명령서 4절).
- 사전 예상(명령서 4절): 미세조정 전 |Δd|는 g3r(0.063 Å)과 비슷하게 부족하다(FT-2 (a): 중앙값 ≤ 0.010 Å).
- 추가 후보: MACE-MH-1(omol head) — 명령서 목록에 없고 lead 지시(11-04)로 더했다. UMA-S는 설치 불가(토큰 없음).

## 결과

| 후보 | 사용 | 반응 | 같은 TS | \|Δd\| 중앙값 Å | \|Δd\| 90% Å | RMSD 중앙값 Å | 허수 1개 | 수렴 | 힘 MAE eV/Å (TS 영역) | 상대 E MAE kcal/mol | GPU ms/E+F | Hessian s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MACE-OMOL-0 | 예 | 200 | 0.01 | 0.116 | 0.290 | 0.103 | 0.98 | 0.99 | 0.255 | 1.09 | 111.8 | 4.65 |
| AIMNet2 | 예 | 200 | 0.00 | 0.198 | 0.439 | 0.201 | 0.88 | 0.80 | 0.311 | 1.99 | 9.7 | 0.02 |
| MACE-MH-1 | 예 | 200 | 0.00 | 0.255 | 0.823 | 0.211 | 0.99 | 0.98 | 0.270 | 1.68 | 44.6 | 2.83 |
| MACE-OFF24 | 예 | 194 | 0.00 | 0.895 | 1.226 | 0.464 | 0.78 | 0.51 | 0.399 | 3.63 | 40.8 | 1.91 |
| xtb | 예 | 200 | 0.01 | 0.123 | 0.511 | 0.122 | 0.96 | 0.99 | 0.243 | 2.10 | — | — |

- 순위(대상만): MACE-OMOL-0, AIMNet2, MACE-MH-1
- **FT 시범 상위 2개: MACE-OMOL-0, AIMNet2**
- 사전 예상(미세조정 전 부족) 확인: True
- 힘·에너지 지표는 수준이 달라(후보는 ωB97M 계열·기체상, 라벨은 B3LYP-D3BJ/def2-SVP SMD water) 참고값이다.

## 속도 변형 (dtype / cuEquivariance)

| 변형 | GPU | ms/E+F | Hessian s | f64 대비 max \|ΔF\| eV/Å | 오류 |
|---|---|---|---|---|---|
| MACE-OMOL-0 float64 | NVIDIA A10 | 115.4 | 4.54 | — |  |
| MACE-OMOL-0 float32 | NVIDIA A10 | 64.3 | 4.26 | 0.00002 |  |
| MACE-OMOL-0 float64 cueq | NVIDIA A10 | 80.9 | 2.92 | 0.00000 |  |
| MACE-OMOL-0 float32 cueq | NVIDIA A10 | 61.6 | 2.88 | 0.00002 |  |
| MACE-MH-1 float64 | NVIDIA A10 | 48.8 | 2.19 | — |  |
| MACE-MH-1 float32 | NVIDIA A10 | 39.0 | 2.18 | 0.00002 |  |
| MACE-MH-1 float64 cueq | NVIDIA A10 | 45.4 | 2.07 | 0.00000 |  |
| MACE-MH-1 float32 cueq | NVIDIA A10 | 45.8 | 2.07 | 0.00002 |  |
| MACE-MH-1 float64 fdhess | NVIDIA A10 | 49.0 | 1.44 | 0.00000 |  |
| MACE-OMOL-0 float64 fdhess |  | — | — | — | RuntimeError: The following operation failed in the TorchScr |
| AIMNet2 | NVIDIA A10 | 9.3 | 0.02 | — |  |
| MACE-OFF24 float64 | NVIDIA A10 | 50.1 | 1.41 | — |  |
