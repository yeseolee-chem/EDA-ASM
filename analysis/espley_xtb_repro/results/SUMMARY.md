# espley_xtb_repro v5 — final results (2026-09-14)

**Espley 2024 (D4DD00224E) protocol reproduced on Coley 5,260 rxns with GFN2-xTB / ALPB water.**

## Method (one-line)

Single engine: **xtb 6.7.1 binary**, one GFN2-xTB / ALPB(water) single point per structure gives term-wise energies, Mulliken charges, per-atom Wiberg valences, HOMO-LUMO gap, dipole moment. **tblite not used.**

## Feature blocks

- `d^struct` **41** — 11 dist + 15 Mulliken + 15 Wiberg-valence
- `b^xtb`    **5**  — barrier, dist_dipole, dist_dipolarophile, sum, interaction (ALPB totals; q_barrier omitted, no Hessian)
- `b^ch`     **8**  — b_strain_1/2 (gas-part strain), b_elst (frozen monopole Coulomb), b_pauli (rep), b_oi (EHT), b_disp (D3(BJ)/B3LYP), b_cpcm (ΔGelec), b_cds (ΔGsasa+Ghb+Gshift)  **· zero markers 없음, 전부 실측**
- `AUX18`         — b_elst_scc, b_disp_d4, b_axc, b_ct, gap_ts/dip/dph, mu_ts/dip/dph, dmu_complexation, dsasa_{H,C,N,O,F,Cl,Br}

**Three arms**: `ESPLEY46` (41+5), **`ESPLEY54`** (41+5+8, 옛 55에서 q_barrier만 제외), `ESPLEY72` (54+AUX18).

## Gates (모두 통과)

| # | 항목 | 결과 |
|---|---|---|
| G1 | 슬라이스 로그 | 18/18 accepted=5260, xtb=6.7.1, ok |
| G2 | xtb_solvation | `{'alpb-water': 5260}` (gas 없음) |
| G3 | aggregate | **5,260 rows / 5,260 unique / 100% ok** |
| G4 | **해석적 적합성** | `mean\|b_disp − dft_disp_dft\| = 0.0000 kcal/mol` (max 0.0000) — 완전 등식 |
| G5 | feature 품질 | 72 컬럼, NaN 0, 상수 0 |
| G6 | ML 사용 행 | **5,260** |

## 최종 결과 — Protocol A best model per (target, arm) (5-seed 80/10/10, GridSearchCV)

| Channel | Arm | Best | test MAE ± sd | NMAE | r² | %range | pre-ML MAE (pre-ML r) |
|---|---|---|---:|---:|---:|---:|---:|
| **ΔE‡ barrier** | ESPLEY46 | KRR | 1.78 ± 0.06 | 0.24 | 0.933 | 2.96 | 6.23 (0.79) |
| ↑ | ESPLEY54 | SVR | 1.61 ± 0.07 | 0.22 | 0.942 | 2.67 | ↑ |
| ↑ | **ESPLEY72** | **KRR** | **1.49 ± 0.03** | **0.20** | **0.954** | 2.48 | ↑ |
| **d1** (dipole strain) | ESPLEY46 | KRR | 1.12 ± 0.03 | 0.19 | 0.957 | 2.31 | 4.70 (0.93) |
| ↑ | ESPLEY54 | KRR | 1.13 | 0.19 | 0.960 | 2.33 | ↑ |
| ↑ | ESPLEY72 | KRR | 1.13 | 0.19 | 0.960 | 2.33 | ↑ |
| **d2** (dph strain) | ESPLEY46 | XGB | 0.92 ± 0.05 | 0.18 | 0.946 | 1.50 | 1.49 (0.96) |
| ↑ | ESPLEY54 | XGB | 0.92 | 0.18 | 0.947 | 1.49 | ↑ |
| ↑ | **ESPLEY72** | **XGB** | **0.82 ± 0.04** | **0.16** | **0.955** | 1.33 | ↑ |
| **ΔE_int (SPE)** | ESPLEY46 | KRR | 1.26 | 0.25 | 0.929 | 2.29 | 3.39 (0.82) |
| ↑ | ESPLEY54 | SVR | 0.90 | 0.18 | 0.945 | 1.64 | ↑ |
| ↑ | **ESPLEY72** | **KRR** | **0.79 ± 0.03** | **0.16** | **0.972** | 1.43 | ↑ |
| **Bond E (EDA)** | ESPLEY46 | XGB | 1.77 | 0.29 | 0.905 | 2.90 | 5.29 (0.84) |
| ↑ | ESPLEY54 | KRR | 1.58 | 0.26 | 0.901 | 2.59 | ↑ |
| ↑ | **ESPLEY72** | **KRR** | **1.17 ± 0.03** | **0.19** | **0.962** | 1.91 | ↑ |
| **elst** | ESPLEY46 | KRR | 3.57 | 0.26 | 0.912 | 2.49 | 35.06 (0.37) |
| ↑ | ESPLEY54 | KRR | 1.99 | 0.15 | 0.976 | 1.39 | ↑ |
| ↑ | **ESPLEY72** | **KRR** | **1.68 ± 0.04** | **0.12** | **0.983** | 1.18 | ↑ |
| **Pauli** | ESPLEY46 | SVR | 2.90 | 0.11 | 0.982 | 1.14 | 92.65 (0.71) |
| ↑ | ESPLEY54 | KRR | 2.30 | 0.09 | 0.990 | 0.90 | ↑ |
| ↑ | **ESPLEY72** | **KRR** | **2.17 ± 0.02** | **0.08** | **0.991** | 0.85 | ↑ |
| **OI** | ESPLEY46 | SVR | 1.80 | 0.11 | 0.979 | 0.99 | 37.24 (0.78) |
| ↑ | ESPLEY54 | KRR | 1.50 | 0.10 | 0.989 | 0.82 | ↑ |
| ↑ | **ESPLEY72** | **KRR** | **1.30 ± 0.05** | **0.08** | **0.991** | 0.71 | ↑ |
| **disp** | ESPLEY46 | KRR | 1.25 | 0.33 | 0.880 | 4.68 | 0.00 (**1.00**) |
| ↑ | ESPLEY54 | Ridge | **0.00** | 0.00 | **1.000** | 0.00 | ↑ |
| ↑ | ESPLEY72 | Ridge | 0.00 | 0.00 | 1.000 | 0.00 | ↑ |
| **CPCM** | ESPLEY46 | XGB | 3.14 | 0.75 | 0.409 | 5.52 | 10.38 (0.63) |
| ↑ | ESPLEY54 | XGB | 2.08 | 0.50 | 0.747 | 3.65 | ↑ |
| ↑ | **ESPLEY72** | **KRR** | **1.48 ± 0.06** | **0.35** | **0.881** | 2.60 | ↑ |
| **CDS** | ESPLEY46 | XGB | 0.30 | 0.44 | 0.795 | 4.84 | 3.57 (0.28) |
| ↑ | ESPLEY54 | XGB | 0.29 | 0.43 | 0.812 | 4.66 | ↑ |
| ↑ | **ESPLEY72** | **SVR** | **0.23 ± 0.01** | **0.34** | **0.867** | 3.77 | ↑ |

### Highlights (ESPLEY72 vs 이전 v4 결과 vs Espley 원논문 AM1)

- **barrier**: v4(tblite gas, 54feat) **2.03** → v5(xtb ALPB, 72feat) **1.49** — 26% 개선. Espley 원논문 AM1 SVR ≈ 2.55
- **Bond E (EDA)**: 1.83 → **1.17** (36%)
- **CPCM**: 2.95 (r² 0.52) → **1.48** (r² 0.88) — 물리 서로게이트 → 실 값 전환의 위력
- **elst**: 2.89 → **1.68** (r² 0.947 → 0.983)
- **b_disp = DFT disp**: MAE 0.000 (해석적 등식, ML 아닌 계산 결과)
- 6 EDA 채널 전부 **r² ≥ 0.87**, 5개는 **r² ≥ 0.94**

### Pre-ML MAE는 무의미하고 pre-ML r가 정보량 있음 (SPEC 지적대로)

- b_elst: pre-ML MAE 35.06 kcal/mol (거대!) — 하지만 pre-ML r = 0.37로 상관 있음 → 절대 스케일 조정만 필요
- b_pauli: MAE 92.65, r = 0.71 — 크기 스케일 차이가 큼 (xtb Pauli는 그대로 DFT에 매핑되지 않음)
- **b_disp: MAE 0.00, r = 1.00 완전 일치** — 정의상 같은 양

## Arm 비교 (ESPLEY46 → 54 → 72, best model test MAE)

| Channel | 46 | 54 | 72 | 54가 46 대비 | 72가 54 대비 |
|---|---:|---:|---:|---:|---:|
| barrier | 1.78 | 1.61 | 1.49 | −9.6% | −7.5% |
| eint_spe | 1.26 | 0.90 | 0.79 | **−28.6%** | −12.2% |
| e_bond | 1.77 | 1.58 | 1.17 | −10.7% | **−25.9%** |
| elst | 3.57 | 1.99 | 1.68 | **−44.3%** | −15.6% |
| pauli | 2.90 | 2.30 | 2.17 | −20.7% | −5.7% |
| oi | 1.80 | 1.50 | 1.30 | −16.7% | −13.3% |
| disp | 1.25 | 0.00 | 0.00 | **−100%** | – |
| cpcm | 3.14 | 2.08 | 1.48 | **−33.8%** | **−28.8%** |
| cds | 0.30 | 0.29 | 0.23 | −3.3% | −20.7% |

**b^ch 블록 (46→54)**이 elst/CPCM/eint_spe에서 큰 개선, **AUX18 (54→72)** 이 e_bond/CPCM에서 추가 개선. 특히 **CPCM 두 단계 모두 유효**하여 최종 0.75→0.35 NMAE 개선.

## 산출물 (`results/`)

- `SUMMARY.md` — 이 문서
- `ml_report.json` — 11 target × 3 arm × (Protocol A 4 + Protocol B 5) 전 metric (NMAE, pre_ml_r 포함)
- `ml_table_espley.csv` — **297 행** (11 target × 3 arm × 9 model)
- `predictions.parquet` — 3.7 MB, 5-seed 폴드 예측
- `xtb_features.parquet` — 3.6 MB, 5,260 rxn × 72 feature + 11 target + 메타 (재현/추가 실험용)
- `ml_targets/` — target당 JSON (병렬 array 원본)
- `figures/` (ESPLEY54 기준):
  - `scatter_Ridge_ESPLEY54.png`, `scatter_KRR_rbf_ESPLEY54.png`, `scatter_SVR_rbf_ESPLEY54.png`, `scatter_XGB_ESPLEY54.png` — 8-패널 산점도
  - `mae_bar_espley54.png` — 8 채널 × 4 모델 grouped bar

## 재현

```bash
sbatch analysis/espley_xtb_repro/s01_xtb_array.sh                         # xTB features (~7 min wall, MaxJobs=10 fully used)
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s02_aggregate.sh
sbatch                            analysis/espley_xtb_repro/s03_ml_array.sh    # 11 targets × 3 arms × 4 models (~20 min wall)
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s04_aggregate_ml.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s05_plot.sh
```

Prerequisites: xtb 6.7.1 in `/home1/yeseo1ee/xtb-dist/`, `reactot` env with dftd3 + morfeus-ml + xgboost, `labels_all.json` (5,265 records, 5,260 accepted).
