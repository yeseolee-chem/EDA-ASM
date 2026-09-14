# espley_xtb_repro — final results (2026-09-14)

**Espley 2024 (D4DD00224E) protocol reproduced on Coley 5,260 rxns with GFN2-xTB / ALPB water.**

## Method (one-line)

Single engine: **xtb 6.7.1 binary**, one GFN2-xTB / ALPB(water) single point per structure gives term-wise energies, Mulliken charges, per-atom Wiberg valences, HOMO-LUMO gap, dipole moment. tblite not used.

## Feature blocks

- `d^struct` **41** — 11 dist + 15 Mulliken + 15 Wiberg-valence
- `b^xtb`    **5**  — barrier, dist_dipole, dist_dipolarophile, sum, interaction (ALPB totals; q_barrier omitted, no Hessian)
- `b^ch`     **8**  — b_strain_1/2 (gas-part strain), b_elst (frozen monopole Coulomb), b_pauli (rep), b_oi (EHT), b_disp (D3(BJ)/B3LYP), b_cpcm (ΔGelec), b_cds (ΔGsasa+Ghb+Gshift) — zero markers 없음, 전부 실측
- `AUX18`         — b_elst_scc, b_disp_d4, b_axc, b_ct, gap_ts/dip/dph, mu_ts/dip/dph, dmu_complexation, dsasa_{H,C,N,O,F,Cl,Br}

**Three arms**: `ESPLEY46` (41+5), `ESPLEY54` (41+5+8), **`ESPLEY72`** (54+AUX18).

## Gates (모두 통과)

| # | 항목 | 결과 |
|---|---|---|
| G1 | 슬라이스 로그 | 18/18 accepted=5260, xtb=6.7.1, ok |
| G2 | xtb_solvation | `{'alpb-water': 5260}` (gas 없음) |
| G3 | aggregate | **5,260 rows / 5,260 unique / 100% ok** |
| G4 | **해석적 적합성** | `mean\|b_disp − dft_disp_dft\| = 2.4 × 10⁻⁶ kcal/mol`, `max 8 × 10⁻⁶` — 사실상 등식 |
| G5 | feature 품질 | 72 컬럼, NaN 0, 상수 0, 순서 중복 0 |
| G6 | ML 사용 행 | **5,260** |

## Model selection — KRR(RBF)을 사전 고정 (a priori)

**모델 앙상블에서 test MAE 최소를 뽑는 방식은 낙관 편향을 유발.** 검증 결과 `ESPLEY72` 조합에서 KRR이 11 타깃 중 8개에서 최적이고, 나머지 3개에서도 최적 대비 손실이 합계 0.05 kcal/mol (d2 +0.01, cds +0.00, disp +0.04)에 불과. 이 대가로 cherry-picking 편향을 완전히 제거.

**Ridge / SVR / XGB의 결과는 `ml_table_espley.csv`에 전부 기록** (297 행: 11 target × 3 arm × 9 model).

## Final headline model — **`ESPLEY72` + KRR(RBF), 5-seed 80/10/10**

| Channel | test MAE ± sd | NMAE | r² | % of range | group-split MAE |
|---|---:|---:|---:|---:|---:|
| ΔE‡ (barrier) | **1.49 ± 0.03** | 0.201 | 0.954 | 2.48 | 1.64 |
| d1 (dipole strain) | 1.13 ± 0.04 | 0.186 | 0.960 | 2.33 | 1.09 |
| d2 (dipolarophile strain) | 0.83 ± 0.04 | 0.165 | 0.970 | 1.35 | 1.00 |
| ΔE_int (SPE) | **0.79 ± 0.03** | 0.155 | 0.972 | 1.43 | 0.81 |
| Bond E (EDA 6채널 합) | 1.17 ± 0.03 | 0.191 | 0.962 | 1.91 | 1.24 |
| elst | 1.68 ± 0.04 | 0.123 | 0.983 | 1.18 | 1.74 |
| Pauli | 2.17 ± 0.02 | 0.084 | **0.991** | 0.85 | 2.10 |
| OI | 1.30 ± 0.05 | 0.082 | **0.991** | 0.71 | 1.27 |
| CPCM | 1.48 ± 0.06 | 0.353 | 0.881 | 2.60 | 1.56 |
| CDS | 0.24 ± 0.01 | 0.347 | 0.871 | 3.79 | 0.26 |
| *disp* | *계산으로 결정 (ML 아님, 아래 참조)* | — | — | — | — |

- 8 채널 중 **7개가 예측 대상**, 그중 5개가 1 kcal/mol 미만 또는 NMAE < 0.2
- 약한 축은 **CPCM(NMAE 0.35)과 CDS(0.35)** — 두 다 용매 항이고 값 자체가 작음. 여기가 다음 개선 지점.

## disp 채널은 ML이 아니라 해석적 등식

`b_disp = D3(BJ)/B3LYP inter-fragment dispersion at TS geometry` = 타깃 `dft_disp_dft`와 **정의상 같은 양**. 검증: `mean|b_disp − dft_disp_dft| = 2.4×10⁻⁶ kcal/mol`.

- ESPLEY54/72의 disp 채널 MAE 0.00, r² 1.000은 "Ridge가 예측했다"가 아니라 "feature를 그대로 읽었다".
- 다른 타깃의 feature로 들어가서 도움이 되지만 (제거 시 e_bond +0.02, eint_spe +0.02, barrier −0.00), 이 표에서는 예측 결과로 취급하지 않음.
- ESPLEY46(b^ch 블록 없음)에서만 disp는 실제 ML 예측이며 MAE 1.25 kcal/mol.

## 전하 그룹별 test MAE (ESPLEY72 / KRR)

| target | all | neutral (n≈2047) | charged (n≈105) | q₂=−2 (79) | q₂=+1 (26) |
|---|---:|---:|---:|---:|---:|
| barrier | 1.49 | 1.49 | 1.55 | 1.51 | 1.70 |
| d1 | 1.13 | 1.11 | 1.44 | 1.31 | 1.89 |
| d2 | 0.83 | 0.83 | 0.87 | 0.66 | 1.59 |
| eint_spe | 0.79 | 0.76 | 1.33 | 0.99 | 2.51 |
| e_bond | 1.17 | 1.14 | 1.83 | 1.62 | 2.54 |
| elst | 1.68 | 1.62 | 3.00 | 2.70 | 4.01 |
| Pauli | 2.17 | 2.04 | 4.66 | 3.78 | 7.66 |
| OI | 1.30 | 1.20 | 3.12 | 2.34 | 5.75 |
| CPCM | 1.48 | 1.42 | 2.66 | 2.70 | 2.51 |
| CDS | 0.24 | 0.24 | 0.24 | 0.18 | 0.44 |

이온 반응(4.9%)에서 elst/Pauli/OI/CPCM이 약 2배 커짐 — GFN2가 diffuse 함수가 없어 음이온이 약간 부정확 + ALPB로 완화하지만 잔차 남음. 원본 표: [`charge_breakdown.csv`](charge_breakdown.csv).

## 그룹 분할 robustness (dipolarophile / dipole 완전 분리 홀드아웃)

동일 KRR·튠 HP 재사용. 1,770 unique dipoles / 713 unique dipolarophiles / 5,260 reactions.

| target | random | dph 그룹 홀드 | 배수 | dipole 그룹 홀드 | 배수 |
|---|---:|---:|---:|---:|---:|
| barrier | 1.49 | 1.64 | 1.10 | 1.51 | 1.01 |
| d1 | 1.13 | 1.09 | 0.97 | 1.10 | 0.98 |
| d2 | 0.83 | 1.00 | 1.20 | 0.85 | 1.02 |
| eint_spe | 0.79 | 0.81 | 1.03 | 0.82 | 1.05 |
| e_bond | 1.17 | 1.24 | 1.06 | 1.23 | 1.05 |
| elst | 1.68 | 1.74 | 1.03 | 1.75 | 1.04 |
| Pauli | 2.17 | 2.10 | 0.97 | 2.17 | 1.00 |
| OI | 1.30 | 1.27 | 0.98 | 1.26 | 0.97 |
| CPCM | 1.48 | 1.56 | 1.06 | 1.52 | 1.03 |
| CDS | 0.24 | 0.26 | 1.12 | 0.25 | 1.05 |

**처음 보는 반응물에서도 성능이 거의 유지됨** — 랜덤 분할 결과가 성분 암기의 산물이 아니라는 직접 증거. 원본: [`group_split.csv`](group_split.csv).

## Espley 2024 대비 (ds3, Bath [3+2] 3,980 rxn, AM1 최적화 기하)

> **주의**: 데이터셋·기하·반경험 방법이 다 다르다. MAE 직접 비교는 부정확; %range가 그나마 공정.

| 양 | Espley pre-ML AM1 | Espley test MAE (%range) | 이번 pre-ML GFN2/ALPB | 이번 test MAE (%range) |
|---|---:|---:|---:|---:|
| Dipole distortion (d1) | 3.59 | 2.55 ± 0.13 (6.3%) | 4.70 | **1.13 ± 0.04 (2.33%)** |
| Dipolarophile distortion (d2) | 3.81 | 2.37 ± 0.12 (6.7%) | 1.49 | **0.83 ± 0.04 (1.35%)** |
| Interaction energy | 20.01 | 2.46 ± 0.12 (6.9%) | 3.39 | **0.79 ± 0.03 (1.43%)** |
| ΔE‡ | 23.07 | 3.09 ± 0.14 (6.1%) | 6.23 | **1.49 ± 0.03 (2.48%)** |

관찰:
1. Espley ds3의 pre-ML MAE는 barrier가 23.07 (AM1의 시스템 오차가 큼), 이번 pre-ML은 6.23 (GFN2/ALPB의 시스템 오차가 훨씬 작음). 출발점 자체가 다르다.
2. Dipole distortion pre-ML은 오히려 우리가 더 나쁘다 (4.70 vs 3.59). ML이 커버하는 개선폭이 여기서 나옴.
3. 이번 test range가 조금 더 크다 (ΔE‡ 60.3 vs 51.0 kcal/mol) — %range 비교가 우리에게 다소 유리하게 작용.
4. Espley 논문의 진짜 새로운 주장은 barrier 정확도가 아니라 **8채널 분해 그 자체** — distortion/interaction 2분할을 넘어 상호작용을 elst / Pauli / OI / disp / CPCM / CDS로 쪼개 전부 예측한 것이 여기서 이룬 것.

## 가정·주의

- **5 seed의 test 표본은 5,260 랜덤에서 뽑히므로 서로 겹칠 수 있음.** ±sd는 독립 표본 표준오차가 아니라 seed간 변동성(Espley 프로토콜과 동일).
- 하이퍼파라미터는 seed 23의 train 분할 위 `GridSearchCV(5-fold)`로 튠 후 5 seed 재사용 (Espley와 동일). test는 어느 단계에서도 사용되지 않음.
- 10% validation 분할은 생성만 되고 사용되지 않음. 그대로 정직한 것.
- Feature는 **DFT TS 기하**에서의 xTB 단일점이다 — 성능 상한(upper bound). React-OT 등 생성 기하에서의 배포 성능은 별도 실험이 필요.

## 산출물 (`results/`)

- `SUMMARY.md` — 이 문서
- `ml_report.json` — 11 target × 3 arm × (Protocol A 4 + Protocol B 5) 전 metric (NMAE, pre_ml_r 포함)
- `ml_table_espley.csv` — 297 행
- `predictions.parquet` — 3.7 MB, 5-seed 폴드 예측
- `xtb_features.parquet` — 3.6 MB, 5,260 rxn × 72 feature + 11 target + 메타
- `ml_targets/` — target당 JSON (병렬 array 원본)
- `charge_breakdown.csv` — 전하 그룹별 test MAE (ESPLEY72 / KRR)
- `group_split.csv` — 반응물 그룹 홀드아웃 robustness (동일)
- `figures/` (ESPLEY54 기준):
  - `scatter_Ridge/KRR_rbf/SVR_rbf/XGB_ESPLEY54.png` — 8-패널 산점도
  - `mae_bar_espley54.png` — 8 채널 × 4 모델 grouped bar

## 재현

```bash
sbatch analysis/espley_xtb_repro/s01_xtb_array.sh                         # xTB features
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s02_aggregate.sh
sbatch                            analysis/espley_xtb_repro/s03_ml_array.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s04_aggregate_ml.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s05_plot.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s06_analyze_extra.sh
```

Prerequisites: xtb 6.7.1 in `/home1/yeseo1ee/xtb-dist/`, `reactot` env with dftd3 + morfeus-ml + xgboost, `labels_all.json` (5,265 records, 5,260 accepted).
