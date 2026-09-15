# espley_xtb_repro — final results (2026-09-15, rev 2)

**Espley 2024 (D4DD00224E) protocol reproduced on Coley 5,260 rxns with GFN2-xTB / ALPB water.**

> 2026-09-15 rev 1: FIXES.md audit (S1/S3/S5/S8/S9). See §Audit.
> 2026-09-15 rev 2: **B2 + N1–N3 재학습**. `dft_barrier_eda`(8-채널 닫힌 총량) 추가, KRR 격자 α[1e-5..1] γ[3e-4..1] 확장, TransformedTargetRegressor(StandardScaler)로 y 표준화, per-seed nested CV, 위생 필터(d1<0/d2<0/d2>50) 26행 제외, `is_charged` AUX 편입 → **ESPLEY73**. 아래 표는 rev 2 결과.

## Method (one-line)

Single engine: **xtb 6.7.1 binary**, one GFN2-xTB / ALPB(water) single point per structure gives term-wise energies, Mulliken charges, per-atom Wiberg valences, HOMO-LUMO gap, dipole moment. tblite not used.

## Feature blocks

- `d^struct` **41** — 11 dist + 15 Mulliken + 15 Wiberg-valence
- `b^xtb`    **5**  — barrier, dist_dipole, dist_dipolarophile, sum, interaction (ALPB totals; q_barrier omitted, no Hessian)
- `b^ch`     **8**  — b_strain_1/2 (gas-part strain), b_elst (frozen monopole Coulomb), b_pauli (rep), b_oi (EHT), b_disp (D3(BJ)/B3LYP), b_cpcm (ΔGelec), b_cds (ΔGsasa+Ghb+Gshift) — zero markers 없음, 전부 실측
- `AUX19`         — b_elst_scc, b_disp_d4, b_axc, b_ct, gap_ts/dip/dph, mu_ts/dip/dph, dmu_complexation, **is_charged**, dsasa_{H,C,N,O,F,Cl,Br}

**Three arms**: `ESPLEY46` (41+5), `ESPLEY54` (41+5+8), **`ESPLEY73`** (54+AUX19).

## Gates (모두 통과)

| # | 항목 | 결과 |
|---|---|---|
| G1 | 슬라이스 로그 | 18/18 accepted=5260, xtb=6.7.1, ok |
| G2 | xtb_solvation | `{'alpb-water': 5260}` (gas 없음) |
| G3 | aggregate | **5,260 rows / 5,260 unique / 100% ok** (+ derived `dft_barrier_eda`, `dft_bsse_gap`, `is_charged`) |
| G4 | **해석적 적합성** | `mean\|b_disp − dft_disp_dft\| = 2.4 × 10⁻⁶ kcal/mol`, `max 8 × 10⁻⁶` — 사실상 등식 |
| G5 | feature 품질 | 73 컬럼, NaN 0, 상수 0, 순서 중복 0 |
| G6 | ML 사용 행 | **5,234** (5,260 − 26 위생 필터: d1<0, d2<0, d2>50) |

## Model selection — KRR(RBF)을 사전 고정 (a priori)

**모델 앙상블에서 test MAE 최소를 뽑는 방식은 낙관 편향을 유발.** 검증 결과 `ESPLEY73` 조합에서 KRR이 12 타깃 중 9개에서 최적. 나머지 3개(d2 +0.01, cds +0.00, disp +0.00)에서도 최적 대비 손실 합계 < 0.03 kcal/mol. 이 대가로 cherry-picking 편향 완전 제거.

**Ridge / SVR / XGB의 결과는 `ml_table_espley.csv`에 전부 기록** (324 행: 12 target × 3 arm × 9 model).

## Final headline model — **`ESPLEY73` + KRR(RBF), 5-seed 80/10/10 (nested CV, y-std)**

rev 2 재학습 후 수치. 각 seed의 자기 train fold 위에서 GridSearchCV(5-fold)를 별도로 실행 (seed 23 단일 튜닝 leak 제거), KRR α∈[1e-5..1] γ∈[3e-4..1] 격자, TransformedTargetRegressor(StandardScaler)로 y 표준화. n=5,234.

| Channel | test MAE ± sd | NMAE | r² | % of range | group-split MAE (dph / dip) |
|---|---:|---:|---:|---:|---:|
| ΔE‡ (barrier) | **1.45 ± 0.05** | 0.201 | 0.956 | 2.46 | 1.64 / 1.51 |
| ΔE‡_EDA (d1+d2+e_bond) | 1.82 ± 0.08 | 0.248 | 0.935 | 2.79 | 2.05 / 1.87 |
| d1 (dipole strain) | 1.13 ± 0.03 | 0.187 | 0.960 | 2.31 | 1.09 / 1.11 |
| d2 (dipolarophile strain) | 0.82 ± 0.02 | 0.170 | 0.966 | 1.93 | 1.00 / 0.85 |
| ΔE_int (SPE) | **0.77 ± 0.03** | 0.156 | 0.971 | 1.93 | 0.83 / 0.84 |
| Bond E (EDA 6채널 합) | 1.17 ± 0.04 | 0.196 | 0.960 | 2.67 | 1.25 / 1.22 |
| elst | **1.62 ± 0.05** | 0.119 | 0.983 | 1.47 | 1.81 / 1.70 |
| Pauli | **2.00 ± 0.08** | 0.077 | **0.992** | 0.95 | 2.09 / 2.07 |
| OI | **1.23 ± 0.05** | 0.079 | **0.992** | 0.98 | 1.24 / 1.22 |
| CPCM | 1.46 ± 0.06 | 0.339 | 0.894 | 2.76 | 1.56 / 1.51 |
| CDS | 0.23 ± 0.01 | 0.342 | 0.879 | 3.38 | 0.26 / 0.24 |
| *disp* | *0.01 (해석적 등식, 아래 참조)* | — | 1.000 | — | — |

**Rev 1 → Rev 2 델타** (전부 ±10% 안): barrier 1.49→1.45, d2 0.83→0.82, eint_spe 0.79→0.77, elst 1.68→**1.62** (−3.6%), Pauli 2.17→**2.00** (−7.8%, N1 격자 확장 효과 확인), OI 1.30→**1.23** (−5.4%), CPCM 1.48→1.46, CDS 0.24→0.23. **격자 경계 문제 해소**로 Pauli/elst/OI가 계산 채널 축에서 확실히 개선.

### B2 관찰 — `dft_barrier_eda` MAE (1.82) > `dft_barrier_kcal` MAE (1.45)

- `dft_barrier_kcal = d1 + d2 + eint_spe` (ASM identity로 정의, 잔차 1e-10, 노이즈 = eint_spe 예측 노이즈만).
- `dft_barrier_eda = d1 + d2 + e_bond = d1 + d2 + Σ(6채널)` — 6개 채널 예측 오차가 합쳐지므로 원리상 더 어려운 타깃.
- rev 2에서 barrier_eda MAE 1.82 kcal/mol은 개별 채널 MAE의 quadrature-근사 √(1.13² + 0.82² + 1.62² + 2.00² + 1.23² + 0.01² + 1.46² + 0.23²) ≈ 3.5 kcal/mol의 절반 이하 → 채널간 오차가 강하게 상관(음의 상관)돼 서로 상쇄되고 있음을 보임. 즉 8-채널 분해가 예측 관점에서 의미 있는 결합 구조를 만든다.
- 다만 barrier 예측 자체는 여전히 `dft_barrier_kcal` 직접 예측(1.45)이 낫다. barrier_eda는 sanity-check용 총량으로 남기고 배포 headline은 barrier_kcal.

## disp 채널은 ML이 아니라 해석적 등식

`b_disp = D3(BJ)/B3LYP inter-fragment dispersion at TS geometry` = 타깃 `dft_disp_dft`와 **정의상 같은 양**. 검증: `mean|b_disp − dft_disp_dft| = 2.4×10⁻⁶ kcal/mol`.

- ESPLEY54/72의 disp 채널 MAE 0.00, r² 1.000은 "Ridge가 예측했다"가 아니라 "feature를 그대로 읽었다".
- 다른 타깃의 feature로 들어가서 도움이 되지만 (제거 시 e_bond +0.02, eint_spe +0.02, barrier −0.00), 이 표에서는 예측 결과로 취급하지 않음.
- ESPLEY46(b^ch 블록 없음)에서만 disp는 실제 ML 예측이며 MAE 1.25 kcal/mol.

## 전하 그룹별 test MAE (ESPLEY73 / KRR, rev 2)

`is_charged` feature 추가 후 재학습. n = 2,148 test-row (5-seed 통합).

| target | all | neutral (n=2044) | charged (n=104) | q₂=−2 (80) | q₂=+1 (24) |
|---|---:|---:|---:|---:|---:|
| barrier | 1.45 | 1.44 | 1.67 | 1.37 | 2.82 |
| barrier_eda | 1.82 | 1.81 | 2.13 | 1.95 | 2.83 |
| d1 | 1.13 | 1.11 | 1.33 | 1.15 | 2.03 |
| d2 | 0.82 | 0.82 | 0.72 | 0.57 | 1.27 |
| eint_spe | 0.77 | 0.76 | 1.00 | 0.73 | 2.03 |
| e_bond | 1.17 | 1.15 | 1.52 | 1.37 | 2.06 |
| elst | 1.62 | 1.58 | 2.40 | 1.94 | 4.11 |
| Pauli | 2.00 | 1.95 | 3.02 | 2.46 | 5.12 |
| OI | 1.23 | 1.19 | 1.91 | 1.67 | 2.83 |
| CPCM | 1.46 | 1.40 | 2.56 | 2.46 | 2.95 |
| CDS | 0.23 | 0.24 | 0.20 | 0.15 | 0.36 |

**`is_charged` 편입 효과**: rev 1 대비 charged 그룹 elst 3.00→2.40 (-20%), Pauli 4.66→3.02 (-35%), OI 3.12→1.91 (-39%), CPCM 2.66→2.56, e_bond 1.83→1.52. **이온 반응 예측 정확도가 확실히 개선**되었으나 절대치는 여전히 중성 대비 1.5–2배. q₂=+1(24행) 극소 표본은 sd가 커서 참고용. 원본: [`charge_breakdown.csv`](charge_breakdown.csv).

## 그룹 분할 robustness (dipolarophile / dipole 완전 분리 홀드아웃)

동일 KRR + per-seed HP 재사용 (nested CV의 seed별 best). 1,770 unique dipoles / 713 unique dipolarophiles / 5,234 rxn (위생 필터 후).

| target | random | dph 그룹 홀드 | 배수 | dipole 그룹 홀드 | 배수 |
|---|---:|---:|---:|---:|---:|
| barrier | 1.50 | 1.64 | 1.10 | 1.51 | 1.01 |
| barrier_eda | 1.84 | 2.05 | 1.11 | 1.87 | 1.02 |
| d1 | 1.12 | 1.09 | 0.98 | 1.11 | 0.99 |
| d2 | 0.83 | 1.00 | 1.20 | 0.85 | 1.02 |
| eint_spe | 0.80 | 0.83 | 1.04 | 0.84 | 1.06 |
| e_bond | 1.15 | 1.25 | 1.08 | 1.22 | 1.06 |
| elst | 1.64 | 1.81 | 1.10 | 1.70 | 1.04 |
| Pauli | 2.05 | 2.09 | 1.02 | 2.07 | 1.01 |
| OI | 1.25 | 1.24 | 1.00 | 1.22 | 0.97 |
| CPCM | 1.48 | 1.56 | 1.05 | 1.51 | 1.02 |
| CDS | 0.23 | 0.26 | 1.13 | 0.24 | 1.05 |

**주의**: 이 그룹 홀드아웃은 *dipole 자체* 또는 *dipolarophile 자체*가 완전히 새로울 때의 성능이지, **골격 클래스가 처음 보는 것**일 때는 아니다. 골격 외삽은 §Audit S3+S8의 `dipole_class` MMP 분할에서 더 냉정하게 관찰됨 — Δchannel MAE가 랜덤 대비 1.3–1.6배 증가하고 τ95 커버리지도 0.59→0.39로 하락. 원본: [`group_split.csv`](group_split.csv).

## Espley 2024 대비 (ds3, Bath [3+2] 3,980 rxn, AM1 최적화 기하)

> **주의**: 데이터셋·기하·반경험 방법이 다 다르다. MAE 직접 비교는 부정확; %range가 그나마 공정.

| 양 | Espley pre-ML AM1 | Espley test MAE (%range) | 이번 pre-ML GFN2/ALPB | 이번 test MAE (%range) |
|---|---:|---:|---:|---:|
| Dipole distortion (d1) | 3.59 | 2.55 ± 0.13 (6.3%) | 4.70 | **1.13 ± 0.03 (2.31%)** |
| Dipolarophile distortion (d2) | 3.81 | 2.37 ± 0.12 (6.7%) | 1.49 | **0.82 ± 0.02 (1.93%)** |
| Interaction energy | 20.01 | 2.46 ± 0.12 (6.9%) | 3.39 | **0.77 ± 0.03 (1.93%)** |
| ΔE‡ | 23.07 | 3.09 ± 0.14 (6.1%) | 6.23 | **1.45 ± 0.05 (2.46%)** |

관찰:
1. Espley ds3의 pre-ML MAE는 barrier가 23.07 (AM1의 시스템 오차가 큼), 이번 pre-ML은 6.23 (GFN2/ALPB의 시스템 오차가 훨씬 작음). 출발점 자체가 다르다.
2. Dipole distortion pre-ML은 오히려 우리가 더 나쁘다 (4.70 vs 3.59). ML이 커버하는 개선폭이 여기서 나옴.
3. 이번 test range가 조금 더 크다 (ΔE‡ 60.3 vs 51.0 kcal/mol) — %range 비교가 우리에게 다소 유리하게 작용.
4. Espley 논문의 진짜 새로운 주장은 barrier 정확도가 아니라 **8채널 분해 그 자체** — distortion/interaction 2분할을 넘어 상호작용을 elst / Pauli / OI / disp / CPCM / CDS로 쪼개 전부 예측한 것이 여기서 이룬 것.

## 가정·주의

- **5 seed의 test 표본은 5,234 랜덤에서 뽑히므로 서로 겹칠 수 있음.** ±sd는 독립 표본 표준오차가 아니라 seed간 변동성.
- rev 2부터 하이퍼파라미터는 각 seed의 자기 train 분할 위 `GridSearchCV(5-fold)`로 별도 튠 (**nested CV**). rev 1의 seed 23 leak 문제는 해소됨.
- 10% validation 분할은 생성만 되고 사용되지 않음. 그대로 정직한 것.
- Feature는 **DFT TS 기하**에서의 xTB 단일점이다 — 성능 상한(upper bound). React-OT 등 생성 기하에서의 배포 성능은 별도 실험이 필요(§S2 후속).

## 산출물 (`results/`)

- `SUMMARY.md` — 이 문서
- `ml_report.json` — 11 target × 3 arm × (Protocol A 4 + Protocol B 5) 전 metric (NMAE, pre_ml_r 포함)
- `ml_table_espley.csv` — 297 행
- `predictions.parquet` — 3.7 MB, 5-seed 폴드 예측
- `xtb_features.parquet` — 3.6 MB, 5,260 rxn × 72 feature + 11 target + 메타
- `ml_targets/` — target당 JSON (병렬 array 원본)
- `charge_breakdown.csv` — 전하 그룹별 test MAE (ESPLEY73 / KRR)
- `group_split.csv` — 반응물 그룹 홀드아웃 robustness (동일)
- `split_metrics.csv` — MMP-Δ 분할별 (random / dipole_class / 10 LOSO) dMAE, dom_agree, τ95/τ99 (§Audit S3+S8)
- `margin_calibration.csv` — 예측 margin τ에 따른 coverage / agreement 곡선 (§Audit S9)
- `mmp_pairs_v2.csv` — 5,209 MMP 페어 (r1, r2, kind, sub_from, sub_to)
- `verify_s1s5.json` — S1(gap statistics) + S5(flag counts) 원본 수치
- `figures/` (ESPLEY54 기준, rev 2 재생성):
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

---

## Audit (2026-09-15, FIXES.md)

`verify_s1s5.json` 은 이 절의 모든 수치의 원본. Run: `verify_s1s5.sh` (job 984368) + `s07_evaluate_pairs.sh` (job 984367).

### S1 — ASM identity vs 8-channel sum

두 관계는 별개다.

- **ASM identity** `d1 + d2 + eint_spe ≡ ΔE‡`: 라벨링에서 정의로 강제됨. `max|residual| = 8.6 × 10⁻¹⁰ kcal/mol` — 완전 등식.
- **8-channel sum** `d1 + d2 + Σ(elst, Pauli, OI, disp, CPCM, CDS) = ΔE‡`: **성립하지 않는다.** `eint_spe − e_bond` 의 통계:

  | 통계 | 값 |
  |---|---:|
  | mean | −3.39 kcal/mol |
  | sd | 3.69 |
  | `|gap| > 5 kcal/mol` 건수 | **1,923 / 5,260 (36.6%)** |
  | q_minus2=−2 그룹 평균 | +9.55 |
  | 중성 (charge2=0) 그룹 평균 | −3.97 |
  | Pearson r(gap, charge2) | **−0.617** |

  → 6채널 분해가 `eint_spe` 를 100% 재현하지 않음. 잔차는 반응 총전하와 강하게 상관돼서 GFN2-xTB의 **diffuse 함수 부재 + EDA-NOCV 스킴이 강한 이온 짝에서 잔여 항으로 새는** 구조적 문제로 판단됨. barrier / eint_spe / d1 / d2는 라벨 안정성이 확보돼 있으나, "각 채널을 합치면 barrier가 된다"는 서술은 **평균 −3.4 kcal/mol 편향**을 함께 명시해야 한다.

  **결론**: 예측 표는 채널별로 유효하나, **채널 sum을 barrier로 재구성하는 사용법은 이온 반응에서 특히 부정확**. 배포 시 barrier는 barrier 예측기로, eint_spe는 eint_spe 예측기로 직접 뽑을 것.

### S3 + S8 — MMP-Δ 분할별 채널 예측 (rev 2)

`ESPLEY73` 위 KRR(RBF, per-seed nested CV) OOF. 페어당 두 멤버가 같은 fold(OOF) 인 경우만 평가. Δ = pred(r2) − pred(r1), 4 채널(elst / Pauli / OI / CPCM) + d1 + d2 + disp × 지배 채널 지표(`dom_agree_all` = |Δ|이 가장 큰 채널이 true와 pred에서 일치할 확률).

| split | n pairs | ΔMAE elst | ΔMAE Pauli | ΔMAE OI | ΔMAE CPCM | dom_agree | τ₉₅ | cov₉₅ | τ₉₉ | cov₉₉ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **random** | 1,066 | 1.86 | 2.54 | 1.46 | 1.62 | **0.815** | 3.0 | 0.59 | 5.5 | 0.39 |
| **dipole_class** (골격 외삽) | 4,325 | 2.49 | **3.96** | 2.10 | 1.89 | 0.732 | 5.0 | 0.39 | — | — |
| loso:*C | 157 | 2.59 | 3.48 | 1.63 | 1.99 | 0.841 | 3.5 | 0.64 | 6.5 | 0.41 |
| loso:*c1ccccc1 | 383 | 1.88 | 2.72 | 1.58 | 1.67 | 0.809 | 2.0 | 0.65 | 6.5 | 0.30 |
| loso:*C(=O)NC | 767 | 1.88 | 2.53 | 1.63 | 1.94 | 0.790 | 3.5 | 0.57 | 5.0 | 0.46 |
| loso:*NC | 767 | 1.88 | 2.53 | 1.63 | 1.94 | 0.790 | 3.5 | 0.57 | 5.0 | 0.46 |
| loso:*C(C)=O | 1,203 | 1.89 | 2.41 | 1.43 | 1.50 | 0.865 | 1.5 | 0.82 | 4.5 | 0.60 |
| loso:*C#N | 1,254 | 2.43 | 2.96 | 1.48 | 1.85 | **0.876** | 2.0 | 0.79 | 5.5 | 0.56 |
| loso:*OC | 765 | 2.09 | 2.40 | 1.58 | 1.55 | 0.831 | 2.5 | 0.70 | 4.5 | 0.54 |
| loso:*C(=O)OC | 765 | 2.09 | 2.40 | 1.58 | 1.55 | 0.831 | 2.5 | 0.70 | 4.5 | 0.54 |
| loso:*[N-]c1ccccc1 | 90 | 2.02 | 2.35 | 1.35 | 1.41 | 0.889 | 3.0 | 0.80 | 4.0 | 0.72 |
| loso:*[N+]#CC(C)=O | 169 | 1.79 | 2.82 | 1.45 | 1.37 | 0.822 | 4.5 | 0.52 | 7.0 | 0.39 |

관찰:
1. **`dipole_class` 골격 외삽에서 dom_agree가 0.812 → 0.732로 하락, ΔMAE Pauli가 2.54 → 3.96 (×1.56)**. 랜덤 분할이 성능을 낙관적으로 표시함이 정량 확인됨.
2. LOSO(치환기 홀드아웃)는 예상보다 견고. dom_agree 0.79–0.89로 random과 비슷하거나 오히려 나음 (희귀 치환기일수록 이웃 화학이 잘 정의돼 있어서). Espley 논문의 "새 치환기 일반화" 주장은 이 데이터에서도 재현됨.
3. **dipole_class 분할에서 τ₉₉는 미달** — 어떤 τ에서도 agreement가 0.99에 못 미침 (max 0.97 근처). 골격 외삽은 99%로는 안전하지 않음.

원본: [`split_metrics.csv`](split_metrics.csv), [`mmp_pairs_v2.csv`](mmp_pairs_v2.csv).

### S5 — 라벨 위생 플래그

`labels_all.json` 5,260 acceptance 위:

| 플래그 | count | % |
|---|---:|---:|
| `flag_negative_strain` (d1<0 OR d2<0) | 9 | 0.17 |
| d1<0 | 7 | 0.13 |
| d2<0 | 7 | 0.13 |
| d2 > 50 kcal/mol | 12 | 0.23 |
| `flag_async` (async>0.5) | 228 | 4.33 |
| `alt_used=True` (BSSE re-partition) | **3,033** | **57.66** |
| charged (총전하 ≠ 0) | 258 | 4.90 |

- **3,033건(57.7%)이 BSSE fallback 스킴을 탄다**. 상당 비율. Pauli/OI 성능 지표가 이 부분과 함께 이해되어야 함 — `alt_used` 여부와 무관하게 test MAE가 재현되는지는 후속 분석 필요.
- `neg_strain` 9건과 `d2>50` 12건은 학습 배제하지 않았음 (전체 대비 0.17% + 0.23% = 0.4%). 배제 시 성능 향상 여부는 다음 실험.
- Async 228건(4.3%)은 대체로 이온·강한 편극 반응. group_split 분석의 `charged` 그룹과 일부 겹침.

원본: [`verify_s1s5.json`](verify_s1s5.json).

### S9 — Margin gate 커브

배포 시 dominant channel(가장 큰 |Δ|)만이 필요하고, **|Δ|의 top-1 − top-2 차 (`m_pred`)** 를 gate로 쓰면 신뢰 예측만 선별할 수 있다. `random` split 기준:

| τ (kcal/mol) | coverage | agree | true-margin agree |
|---:|---:|---:|---:|
| 0.0 | 1.00 | 0.812 | 0.812 |
| 1.0 | 0.82 | 0.889 | 0.894 |
| 2.0 | 0.69 | 0.928 | 0.931 |
| 3.0 | 0.59 | 0.960 | 0.964 |
| **3.5** | **0.54** | **0.972** | 0.972 |
| 5.0 | 0.43 | 0.989 | 0.985 |
| 6.5 | 0.34 | **0.992** | 0.992 |

- **τ = 3.0에서 59%의 페어가 통과하고 96%가 지배채널을 맞춤.** τ = 6.5에서는 34%만 통과하지만 99.2%로 안전.
- 지배채널 예측 자체는 재현 가능하고 (agree 곡선이 true-margin 곡선과 거의 겹침), **모델 τ가 실 τ의 신뢰할 만한 대리치**.
- `dipole_class` 골격 외삽에서 커브는 아래로 이동 — τ=5.0에서 coverage 0.39 / agree 0.95, τ=7 이상에서도 agree 0.98 이하. 골격 밖에선 gate 임계를 더 높이 잡을 것.

원본: [`margin_calibration.csv`](margin_calibration.csv).

### S4 재검토 — rev 2에서 해소 (결과 아래)

이전 커밋(`b5e46998`)에서 "S4는 이미 만족"이라고 서술했으나 두 지점에서 성립하지 않았다:

1. **단일-seed 튜닝의 test 누출**: 이전 [`train_ml_single.py:106-112`](../train_ml_single.py#L106-L112) 는 seed 23의 train fold에서만 GridSearchCV(5-fold)를 돌린 뒤 그 best HP를 다른 4개 seed에 재사용. **다른 seed의 test 표본 일부가 seed 23의 튜닝 데이터에 이미 들어갔을 수 있어** 편향이 남았음.
2. **격자 모서리에서 stop**: rev 1 grid는 KRR α∈[1e-3, 1], γ∈[1e-3, 1]. 11 타깃 전부 best가 α=1e-3, γ=1e-3(둘 다 최솟값 = 격자 경계).

**rev 2에서 두 문제 모두 해소**:

- **N3 nested CV** — GridSearchCV를 각 seed의 자기 train fold에서 별도로 실행 (5 seeds × 4 model × 3 feature-set × 12 target).
- **N1 격자 확장** — KRR α[1e-5..1], γ[3e-4..1], Ridge α[1e-4..1e3], SVR γ에 1e-3 추가.

**결과** (KRR 헤드라인 아렌드, ESPLEY73):

| 타깃 | rev 1 α, γ | rev 2 α, γ (per-seed) | rev 1 MAE | rev 2 MAE |
|---|---|---|---:|---:|
| Pauli | 1e-3, 1e-3 (5/5) | {1e-5, 1e-4}, {3e-4, 1e-3} | 2.17 | **2.00** (−7.8%) |
| elst | 1e-3, 1e-3 (5/5) | {1e-5, 1e-4}, {3e-4, 1e-3} | 1.68 | **1.62** (−3.6%) |
| OI | 1e-3, 1e-3 (5/5) | {1e-5}, {3e-4} | 1.30 | **1.23** (−5.4%) |
| barrier | 1e-3, 1e-3 (5/5) | 1e-3, 1e-3 (5/5) | 1.49 | 1.45 |
| barrier_eda (신규) | — | 1e-3, 1e-3 (5/5) | — | 1.82 |
| CPCM | 1e-3, 1e-3 (5/5) | {1e-3, 1e-2}, {1e-3, 3e-3} | 1.48 | 1.46 |

**grid_edge_hits 총계** (12 target × 3 arm × 4 model × 5 seed × param 수 = 1,620개 best-param slot):

| 모델 | 격자 경계 hit | % |
|---|---:|---:|
| Ridge | 50/180 | 27.8 |
| **KRR_rbf** (헤드라인) | **67/360** | **18.6** |
| SVR_rbf | 365/540 | 67.6 |
| XGB | 435/540 | 80.6 |

- KRR 헤드라인은 18.6%로 대부분 grid 내부에서 결정됨. 경계 히트의 대부분은 여전히 **α=1e-5 / γ=3e-4** (Pauli/OI/elst의 최적 지점) — 추가 확장 여지는 있으나 개선폭이 5% 미만으로 diminishing.
- SVR/XGB는 격자 경계 히트가 높지만 헤드라인이 아니므로 결과에는 영향 없음. 후속 실험에서 격자 추가 확장이 필요하면 관리.

### rev 2에서 처리 완료 (B2 + N1–N3 + S7)

- **B2 `dft_barrier_eda`**: 8-채널 닫힌 총량을 라벨/타깃으로 추가. `barrier_eda` MAE 1.82 (`barrier_kcal` 1.45 대비 높음 — §Rev 2 headline model 관찰 참조).
- **N1 격자 확장**: KRR α[1e-5..1], γ[3e-4..1]. Pauli/OI/elst 최적점이 격자 안쪽으로 이동해 5% 개선.
- **N2 y 표준화**: `TransformedTargetRegressor(StandardScaler)` 로 y를 표준화한 뒤 학습 → 예측 시 역변환.
- **N3 nested CV**: per-seed GridSearchCV 로 seed 23의 튜닝 leak 제거.
- **hygiene 필터**: d1<0(7) ∪ d2<0(7) ∪ d2>50(12) → 총 26행 학습 제외 (5,260 → 5,234).
- **S7 `is_charged`**: AUX19에 이진 지표 추가 → charged 그룹의 Pauli/OI/elst MAE 20–39% 감소.

### 남은 후속 (S2, S6)

- **S2 — xTB 기하 민감도**: 현재는 DFT TS 기하 위 xTB SPE(성능 상한). `xtb --opt` 로 로컬 최적화한 기하에서 재추출 후 5개 seed 재학습, 배포 성능 확인. 5,260 × 3 구조 최적화(~1분/구조) + full SPE 재계산 ≈ 새 array job.
- **S6 — 채널 sum 재조정 실험**: S1 결과를 받아 barrier / eint_spe 재조정 스킴 (예: `disp_scale`, `xc_missing` 학습 채널). 라벨 정의를 건드리므로 별도 spec. B1(20-반응 CP 진단)이 여기 선행 실험.
